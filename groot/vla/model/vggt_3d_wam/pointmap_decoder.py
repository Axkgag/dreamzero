"""Ray-based robot-centric PointMap decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .geometry import normalize_metric_points, rays_in_frame, scale_intrinsics


class MLPResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.input = nn.Linear(channels, 2 * channels)
        self.output = nn.Linear(2 * channels, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.output(F.silu(self.input(self.norm(inputs))))
        return inputs + hidden


class RayPredictionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        residual_layers: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.residual_blocks = nn.Sequential(
            *[MLPResidualBlock(hidden_dim) for _ in range(residual_layers)]
        )
        self.output_projection = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.input_projection(inputs))
        hidden = self.residual_blocks(hidden)
        return self.output_projection(F.silu(hidden))


class PointMapUpsampler(nn.Module):
    """Learned 2x residual upsampling of a coarse metric PointMap."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(3, hidden_dim, 3, padding=1)
        self.residual_blocks = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
        )
        self.output_projection = nn.Conv2d(
            hidden_dim, 3 * 4, 3, padding=1
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, coarse: torch.Tensor) -> torch.Tensor:
        baseline = F.interpolate(
            coarse, scale_factor=2, mode="bilinear", align_corners=False
        )
        hidden = F.silu(self.input_projection(coarse))
        hidden = hidden + self.residual_blocks(hidden)
        residual = F.pixel_shuffle(self.output_projection(hidden), 2)
        return baseline + residual


class PointMapDecoder(nn.Module):
    def __init__(
        self,
        token_dim: int,
        image_size: tuple[int, int],
        output_size: tuple[int, int],
        ray_size: tuple[int, int] | None,
        depth_bins: int,
        min_range: float,
        max_range: float,
        ray_chunk_size: int,
        hidden_dim: int,
        residual_layers: int,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        z_range: tuple[float, float],
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.output_size = output_size
        self.ray_size = output_size if ray_size is None else ray_size
        self.ray_chunk_size = ray_chunk_size
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        if self.ray_size == self.output_size:
            self.pointmap_upsampler = nn.Identity()
        elif (
            self.output_size[0] == 2 * self.ray_size[0]
            and self.output_size[1] == 2 * self.ray_size[1]
        ):
            self.pointmap_upsampler = PointMapUpsampler()
        else:
            raise ValueError(
                "PointMap output_size must equal ray_size or be exactly 2x; "
                f"got ray_size={self.ray_size}, output_size={self.output_size}"
            )
        self.register_buffer(
            "depth_values",
            torch.linspace(min_range, max_range, depth_bins),
            persistent=True,
        )
        hidden = max(64, hidden_dim)
        self.surface_head = RayPredictionHead(
            token_dim + 6,
            hidden,
            residual_layers,
        )
        # View-independent occupancy query. Unlike surface_head, this head does
        # not receive the ray direction, so free/surface supervision must be
        # represented by the metric voxel field rather than a camera prior.
        self.occupancy_head = RayPredictionHead(
            token_dim + 3,
            hidden,
            residual_layers,
        )

    def _render_ray_chunk(
        self,
        volume: torch.Tensor,
        origin: torch.Tensor,
        direction: torch.Tensor,
        depths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        points = (
            origin[:, :, None]
            + direction[:, :, None] * depths[None, None, :, None]
        )
        sample_grid = normalize_metric_points(
            points, self.x_range, self.y_range, self.z_range
        )
        sampled = F.grid_sample(
            volume,
            sample_grid[:, None],
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled[:, :, 0].permute(0, 2, 3, 1)
        ray_features = direction[:, :, None].expand(
            -1, -1, len(depths), -1
        )
        logits = self.surface_head(
            torch.cat((sampled, sample_grid, ray_features), dim=-1)
        )[..., 0]
        occupancy_logits = self.occupancy_head(
            torch.cat((sampled, sample_grid), dim=-1)
        )[..., 0]
        sample_valid = (sample_grid.abs() <= 1).all(dim=-1)
        masked_logits = logits.masked_fill(
            ~sample_valid, torch.finfo(logits.dtype).min
        )
        # Some camera rays never intersect the configured metric volume.
        # Return zero probability instead of NaN or an arbitrary surface.
        any_valid = sample_valid.any(dim=-1, keepdim=True)
        safe_logits = torch.where(
            any_valid, masked_logits, torch.zeros_like(masked_logits)
        )
        probability = safe_logits.softmax(dim=-1)
        probability = probability * sample_valid.to(probability.dtype)
        probability = probability / probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        pointmap = (probability[..., None] * points).sum(dim=2)
        return pointmap, safe_logits, occupancy_logits, sample_valid

    def forward(
        self,
        token_grid: torch.Tensor,
        camera_k: torch.Tensor,
        base_from_camera: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, time, channels, z_size, y_size, x_size = token_grid.shape
        views = camera_k.shape[2]
        render_k = scale_intrinsics(camera_k, self.image_size, self.ray_size)
        height, width = self.ray_size
        origins, directions = rays_in_frame(
            render_k, base_from_camera, height, width
        )
        flat_origins = origins.reshape(batch * time * views, height * width, 3)
        flat_directions = directions.reshape(batch * time * views, height * width, 3)
        volume = token_grid[:, :, None].expand(
            -1, -1, views, -1, -1, -1, -1
        ).reshape(batch * time * views, channels, z_size, y_size, x_size)
        depths = self.depth_values.to(dtype=token_grid.dtype)

        point_chunks = []
        logit_chunks = []
        occupancy_logit_chunks = []
        sample_valid_chunks = []
        for start in range(0, height * width, self.ray_chunk_size):
            end = min(start + self.ray_chunk_size, height * width)
            origin = flat_origins[:, start:end]
            direction = flat_directions[:, start:end]
            if self.training and token_grid.requires_grad:
                (
                    pointmap,
                    safe_logits,
                    occupancy_logits,
                    sample_valid,
                ) = checkpoint(
                    self._render_ray_chunk,
                    volume,
                    origin,
                    direction,
                    depths,
                    use_reentrant=False,
                )
            else:
                (
                    pointmap,
                    safe_logits,
                    occupancy_logits,
                    sample_valid,
                ) = self._render_ray_chunk(
                    volume,
                    origin,
                    direction,
                    depths,
                )
            point_chunks.append(pointmap)
            logit_chunks.append(safe_logits)
            occupancy_logit_chunks.append(occupancy_logits)
            sample_valid_chunks.append(sample_valid)

        coarse_pointmap = torch.cat(point_chunks, dim=1).reshape(
            batch, time, views, height, width, 3
        ).permute(0, 1, 2, 5, 3, 4)
        flat_pointmap = coarse_pointmap.reshape(
            batch * time * views, 3, height, width
        )
        pointmap = self.pointmap_upsampler(flat_pointmap).reshape(
            batch,
            time,
            views,
            3,
            *self.output_size,
        )
        depth_logits = torch.cat(logit_chunks, dim=1).reshape(
            batch, time, views, height, width, len(depths)
        ).permute(0, 1, 2, 5, 3, 4)
        occupancy_logits = torch.cat(occupancy_logit_chunks, dim=1).reshape(
            batch, time, views, height, width, len(depths)
        ).permute(0, 1, 2, 5, 3, 4)
        sample_valid = torch.cat(sample_valid_chunks, dim=1).reshape(
            batch, time, views, height, width, len(depths)
        ).permute(0, 1, 2, 5, 3, 4)
        return (
            pointmap.contiguous(),
            depth_logits.contiguous(),
            occupancy_logits.contiguous(),
            sample_valid.contiguous(),
        )
