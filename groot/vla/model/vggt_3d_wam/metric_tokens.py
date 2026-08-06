"""Metric voxel queries with projected multi-view deformable attention."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import metric_grid, project_points, sinusoidal_encoding
from .temporal_codec import WanTemporalDecoder, WanTemporalEncoder


def _masked_softmax(
    logits: torch.Tensor,
    valid: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Softmax that returns zero when every entry is invalid."""
    masked = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    weights = torch.softmax(masked, dim=dim)
    weights = weights * valid.to(dtype=weights.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-6)


class MultiViewDeformableCrossAttention(nn.Module):
    """Project metric queries into multi-view, multi-level image features.

    Sampling offsets and level/point weights are predicted per attention head
    from the current voxel query. The first layer is geometry-conditioned; in
    later layers the residual query already contains sampled image evidence,
    making the sampling locations input-dependent as in iterative deformable
    attention encoders.
    """

    def __init__(
        self,
        input_dim: int,
        token_dim: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
        offset_scale: float,
    ) -> None:
        super().__init__()
        if token_dim % num_heads:
            raise ValueError(
                f"token_dim={token_dim} must be divisible by num_heads={num_heads}"
            )
        if num_levels < 1 or num_points < 1:
            raise ValueError("num_levels and num_points must be positive")
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = token_dim // num_heads
        self.offset_scale = float(offset_scale)

        self.value_projection = nn.Linear(input_dim, token_dim)
        self.sampling_offsets = nn.Linear(
            token_dim,
            num_heads * num_levels * num_points * 2,
        )
        self.attention_weights = nn.Linear(
            token_dim,
            num_heads * num_levels * num_points,
        )
        self.output_projection = nn.Linear(token_dim, token_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.value_projection.weight)
        nn.init.zeros_(self.value_projection.bias)
        nn.init.zeros_(self.sampling_offsets.weight)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

        # Start from a small radial pattern instead of sending all heads to
        # exactly the same image position.
        angles = torch.arange(self.num_heads, dtype=torch.float32)
        angles = angles * (2.0 * math.pi / self.num_heads)
        directions = torch.stack((angles.cos(), angles.sin()), dim=-1)
        directions = directions[:, None, None].expand(
            -1, self.num_levels, self.num_points, -1
        )
        radii = torch.arange(1, self.num_points + 1, dtype=torch.float32)
        radii = radii / self.num_points
        radial_bias = directions * radii[None, None, :, None]
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(radial_bias.reshape(-1))

    def _project_values(self, feature: torch.Tensor) -> torch.Tensor:
        batch, time, views, channels, height, width = feature.shape
        projected = self.value_projection(
            feature.permute(0, 1, 2, 4, 5, 3)
        )
        return projected.permute(0, 1, 2, 5, 3, 4).reshape(
            batch * time * views,
            self.num_heads,
            self.head_dim,
            height,
            width,
        )

    def forward(
        self,
        query: torch.Tensor,
        multi_level_features: list[torch.Tensor],
        reference_points: torch.Tensor,
        camera_visible: torch.Tensor,
        valid_grid_max: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(multi_level_features) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} feature levels, "
                f"got {len(multi_level_features)}"
            )
        batch, time, points, channels = query.shape
        views = reference_points.shape[2]
        flat_query = query.reshape(batch * time * points, channels)
        num_queries = flat_query.shape[0]

        offsets = self.sampling_offsets(flat_query).reshape(
            num_queries,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )
        offsets = torch.tanh(offsets) * self.offset_scale
        logits = self.attention_weights(flat_query).reshape(
            num_queries,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )
        references = reference_points.permute(0, 1, 3, 2, 4).reshape(
            num_queries, views, 2
        )
        camera_valid = camera_visible.permute(0, 1, 3, 2).reshape(
            num_queries, views
        )
        if valid_grid_max is None:
            valid_grid_max = references.new_ones(2)
        valid_grid_max = valid_grid_max.to(
            device=references.device, dtype=references.dtype
        )

        grids: list[torch.Tensor] = []
        valid_per_level: list[torch.Tensor] = []
        for level, feature in enumerate(multi_level_features):
            feature_h, feature_w = feature.shape[-2:]
            normalizer = offsets.new_tensor(
                [feature_w / 2.0, feature_h / 2.0]
            )
            # reference_points use [-1, 1], while offsets are predicted in
            # feature-cell units.
            level_grid = (
                references[:, None, :, None, :]
                + offsets[:, :, level, None] / normalizer
            )
            level_valid = (
                camera_valid[:, None, :, None]
                & (level_grid >= -1).all(dim=-1)
                & (level_grid <= valid_grid_max).all(dim=-1)
            )
            grids.append(level_grid)
            valid_per_level.append(level_valid)

        # [Q,H,V,L,P], normalized jointly across every visible camera, level,
        # and sampling point.
        valid = torch.stack(valid_per_level, dim=3)
        expanded_logits = logits[:, :, None].expand(
            -1, -1, views, -1, -1
        )
        flat_valid = valid.reshape(
            num_queries, self.num_heads, views * self.num_levels * self.num_points
        )
        flat_logits = expanded_logits.reshape_as(flat_valid)
        weights = _masked_softmax(flat_logits, flat_valid, dim=-1).reshape(
            num_queries,
            self.num_heads,
            views,
            self.num_levels,
            self.num_points,
        )

        output = query.new_zeros(num_queries, self.num_heads, self.head_dim)
        for level, feature in enumerate(multi_level_features):
            feature_h, feature_w = feature.shape[-2:]
            value = self._project_values(feature).reshape(
                batch * time * views * self.num_heads,
                self.head_dim,
                feature_h,
                feature_w,
            )
            level_grid = grids[level].reshape(
                batch,
                time,
                points,
                self.num_heads,
                views,
                self.num_points,
                2,
            ).permute(0, 1, 4, 3, 2, 5, 6)
            level_grid = level_grid.reshape(
                batch * time * views * self.num_heads,
                points,
                self.num_points,
                2,
            )
            sampled = F.grid_sample(
                value,
                level_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = sampled.reshape(
                batch,
                time,
                views,
                self.num_heads,
                self.head_dim,
                points,
                self.num_points,
            ).permute(0, 1, 5, 3, 2, 6, 4)
            sampled = sampled.reshape(
                num_queries,
                self.num_heads,
                views,
                self.num_points,
                self.head_dim,
            )
            level_weight = weights[:, :, :, level, :, None]
            output = output + (sampled * level_weight).sum(dim=(2, 3))

        output = output.reshape(num_queries, channels)
        output = self.output_projection(output).reshape_as(query)
        return query + output


class DenseVoxelLocalAggregation(nn.Module):
    """Exchange information between neighboring cells of the metric grid."""

    def __init__(
        self,
        token_dim: int,
        grid_size: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.input_norm = nn.LayerNorm(token_dim)
        self.depthwise = nn.Conv3d(
            token_dim,
            token_dim,
            kernel_size=3,
            padding=1,
            groups=token_dim,
        )
        self.pointwise = nn.Conv3d(token_dim, token_dim, kernel_size=1)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        batch, time, points, channels = query.shape
        z_size, y_size, x_size = self.grid_size
        if points != z_size * y_size * x_size:
            raise ValueError("Voxel query count does not match grid_size")
        hidden = self.input_norm(query).reshape(
            batch * time, z_size, y_size, x_size, channels
        ).permute(0, 4, 1, 2, 3)
        hidden = self.pointwise(F.gelu(self.depthwise(hidden)))
        hidden = hidden.permute(0, 2, 3, 4, 1).reshape_as(query)
        return query + hidden


class MetricDeformableLayer(nn.Module):
    """Deformable image injection followed by 3D local fusion and an FFN."""

    def __init__(
        self,
        input_dim: int,
        token_dim: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
        offset_scale: float,
        grid_size: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.cross_attention = MultiViewDeformableCrossAttention(
            input_dim=input_dim,
            token_dim=token_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            offset_scale=offset_scale,
        )
        self.cross_norm = nn.LayerNorm(token_dim)
        self.spatial_fusion = DenseVoxelLocalAggregation(token_dim, grid_size)
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, 4 * token_dim),
            nn.GELU(),
            nn.Linear(4 * token_dim, token_dim),
        )

    def forward(
        self,
        query: torch.Tensor,
        multi_level_features: list[torch.Tensor],
        reference_points: torch.Tensor,
        camera_visible: torch.Tensor,
        valid_grid_max: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.cross_attention(
            query,
            multi_level_features,
            reference_points,
            camera_visible,
            valid_grid_max,
        )
        query = self.cross_norm(query)
        query = self.spatial_fusion(query)
        return query + self.ffn(self.ffn_norm(query))


class MetricTokenEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        token_dim: int,
        num_heads: int,
        temporal_layers: int,
        deformable_layers: int,
        deformable_levels: int,
        deformable_points: int,
        deformable_offset_scale: float,
        temporal_stride: int,
        grid_size: tuple[int, int, int],
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        z_range: tuple[float, float],
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.deformable_levels = deformable_levels
        self.deformable_points = deformable_points
        self.grid_size = grid_size
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        if temporal_stride != 4:
            raise ValueError(
                "VGGT 3D latents must use Wan's temporal stride 4; "
                f"got {temporal_stride}"
            )
        if deformable_levels != 2:
            raise ValueError(
                "The true-depth geometry path requires exactly two feature levels"
            )
        self.coarse_adapter = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=2, stride=2),
            nn.GroupNorm(math.gcd(8, input_dim), input_dim),
            nn.SiLU(),
        )
        centers = metric_grid(x_range, y_range, z_range, grid_size)
        self.register_buffer("grid_centers", centers, persistent=True)
        self.query_features = nn.Parameter(torch.zeros(len(centers), token_dim))
        nn.init.normal_(self.query_features, std=0.02)
        self.position_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.deformable_encoder = nn.ModuleList(
            MetricDeformableLayer(
                input_dim=input_dim,
                token_dim=token_dim,
                num_heads=num_heads,
                num_levels=deformable_levels,
                num_points=deformable_points,
                offset_scale=deformable_offset_scale,
                grid_size=grid_size,
            )
            for _ in range(deformable_layers)
        )
        temporal_layer = nn.TransformerEncoderLayer(
            token_dim,
            num_heads,
            dim_feedforward=4 * token_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer, temporal_layers
        )
        self.output_norm = nn.LayerNorm(token_dim)
        self.temporal_encoder = WanTemporalEncoder(token_dim, spatial_kernel=1)

    def _base_queries(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        ranges = torch.tensor(
            [self.x_range, self.y_range, self.z_range],
            dtype=dtype,
            device=device,
        )
        normalized = 2 * (self.grid_centers.to(dtype) - ranges[:, 0]) / (
            ranges[:, 1] - ranges[:, 0]
        ) - 1
        position = sinusoidal_encoding(normalized, self.token_dim)
        return self.query_features.to(dtype) + self.position_mlp(position)

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
        camera_k: torch.Tensor,
        base_from_camera: torch.Tensor,
        image_size: tuple[int, int],
        valid_image_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(features) != 2:
            raise ValueError(
                f"MetricTokenEncoder expects layer-11/layer-23 features, got {len(features)}"
            )
        fine_features, deep_features = features
        if fine_features.shape != deep_features.shape:
            raise ValueError("Geometry feature levels must share one input grid")
        batch, time, views, channels, feature_h, feature_w = fine_features.shape
        points = self.grid_centers.to(
            dtype=fine_features.dtype, device=fine_features.device
        )
        projected, visible = project_points(
            points,
            camera_k,
            base_from_camera,
            image_size,
            valid_image_size=valid_image_size,
        )
        valid_height, valid_width = valid_image_size or image_size
        padded_height, padded_width = image_size
        valid_grid_max = fine_features.new_tensor(
            [
                2 * valid_width / padded_width - 1,
                2 * valid_height / padded_height - 1,
            ]
        )
        coarse = self.coarse_adapter(deep_features.reshape(
            batch * time * views, channels, feature_h, feature_w
        ))
        multi_level_features = [
            fine_features,
            coarse.reshape(
                batch,
                time,
                views,
                channels,
                coarse.shape[-2],
                coarse.shape[-1],
            ),
        ]
        base_query = self._base_queries(
            fine_features.dtype, fine_features.device
        )
        fused = base_query[None, None].expand(
            batch, time, -1, -1
        )
        for layer in self.deformable_encoder:
            fused = layer(
                fused,
                multi_level_features,
                projected,
                visible,
                valid_grid_max,
            )

        temporal = fused.permute(0, 2, 1, 3).reshape(
            batch * len(points), time, self.token_dim
        )
        temporal_mask = torch.triu(
            torch.ones(time, time, dtype=torch.bool, device=temporal.device),
            diagonal=1,
        )
        temporal = self.temporal_transformer(temporal, mask=temporal_mask)
        frame_tokens = self.output_norm(temporal).reshape(
            batch, len(points), time, self.token_dim
        ).permute(0, 2, 1, 3)
        compressed = self.temporal_encoder(
            frame_tokens.permute(0, 2, 3, 1).reshape(
                batch * len(points),
                self.token_dim,
                time,
                1,
                1,
            )
        )
        latent_time = compressed.shape[2]
        tokens = compressed.reshape(
            batch,
            len(points),
            self.token_dim,
            latent_time,
        ).permute(0, 3, 1, 2)
        z_size, y_size, x_size = self.grid_size
        grid = tokens.reshape(
            batch, latent_time, z_size, y_size, x_size, self.token_dim
        ).permute(0, 1, 5, 2, 3, 4)
        frame_visible = visible.any(dim=2)
        voxel_visible = torch.cat(
            (
                frame_visible[:, :1],
                frame_visible[:, 1:].reshape(
                    batch,
                    latent_time - 1,
                    4,
                    len(points),
                ).any(dim=2),
            ),
            dim=1,
        )
        return tokens, grid.contiguous(), voxel_visible


class MetricTokenDecoder(nn.Module):
    """Learned temporal decoder from compressed 3D latents to per-frame grids."""

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        temporal_layers: int,
        grid_size: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.grid_size = grid_size
        self.temporal_decoder = WanTemporalDecoder(token_dim, spatial_kernel=1)
        temporal_layer = nn.TransformerEncoderLayer(
            token_dim,
            num_heads,
            dim_feedforward=4 * token_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_refiner = nn.TransformerEncoder(
            temporal_layer,
            temporal_layers,
        )
        self.output_norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        latent_tokens: torch.Tensor,
        output_time: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latent_tokens.ndim != 4:
            raise ValueError(
                "MetricTokenDecoder expects [B,T_latent,N,C], "
                f"got {tuple(latent_tokens.shape)}"
            )
        batch, latent_time, points, channels = latent_tokens.shape
        expected_points = (
            self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        )
        if points != expected_points or channels != self.token_dim:
            raise ValueError(
                "3D latent shape does not match the configured metric grid: "
                f"got N={points}, C={channels}; expected "
                f"N={expected_points}, C={self.token_dim}"
            )
        expanded = self.temporal_decoder(
            latent_tokens.permute(0, 2, 3, 1).reshape(
                batch * points,
                channels,
                latent_time,
                1,
                1,
            ),
            output_time=output_time,
        )
        frame_time = expanded.shape[2]
        frame_tokens = expanded.reshape(
            batch,
            points,
            channels,
            frame_time,
        ).permute(0, 3, 1, 2)
        temporal_mask = torch.triu(
            torch.ones(
                frame_time,
                frame_time,
                dtype=torch.bool,
                device=frame_tokens.device,
            ),
            diagonal=1,
        )
        refined = self.temporal_refiner(
            frame_tokens.permute(0, 2, 1, 3).reshape(
                batch * points,
                frame_time,
                channels,
            ),
            mask=temporal_mask,
        )
        frame_tokens = self.output_norm(refined).reshape(
            batch,
            points,
            frame_time,
            channels,
        ).permute(0, 2, 1, 3)
        z_size, y_size, x_size = self.grid_size
        grid = frame_tokens.reshape(
            batch,
            frame_time,
            z_size,
            y_size,
            x_size,
            channels,
        ).permute(0, 1, 5, 2, 3, 4)
        return frame_tokens, grid.contiguous()
