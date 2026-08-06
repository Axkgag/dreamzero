"""Wan-compatible variational 2D video bottleneck and learned video decoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_codec import WanTemporalDecoder, WanTemporalEncoder


class SpatialResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = math.gcd(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return inputs + hidden


class FusionResidualBlock(nn.Module):
    """Residual channel reduction used before spatial query resampling."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.skip = nn.Conv2d(input_channels, output_channels, 1)
        groups = math.gcd(8, output_channels)
        self.norm1 = nn.GroupNorm(groups, output_channels)
        self.conv1 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.skip(inputs)
        residual = self.conv1(F.silu(self.norm1(hidden)))
        residual = self.conv2(F.silu(self.norm2(residual)))
        return hidden + residual


class LearnedSpatialQueryResampler(nn.Module):
    """Cross-attend fixed 10x20 latent queries to a 12x23 feature grid."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        source_size: tuple[int, int] = (12, 23),
        target_size: tuple[int, int] = (10, 20),
    ) -> None:
        super().__init__()
        self.source_size = source_size
        self.target_size = target_size
        self.source_position = nn.Parameter(
            torch.zeros(1, source_size[0] * source_size[1], channels)
        )
        self.query = nn.Parameter(
            torch.zeros(1, target_size[0] * target_size[1], channels)
        )
        nn.init.normal_(self.source_position, std=0.02)
        nn.init.normal_(self.query, std=0.02)
        self.query_norm = nn.LayerNorm(channels)
        self.source_norm = nn.LayerNorm(channels)
        self.cross_attention = nn.MultiheadAttention(
            channels,
            num_heads,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        target_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        batch, channels, height, width = inputs.shape
        target_size = target_size or self.target_size
        source_position = self.source_position.transpose(1, 2).reshape(
            1, channels, *self.source_size
        )
        if (height, width) != self.source_size:
            source_position = F.interpolate(
                source_position,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        source = inputs.flatten(2).transpose(1, 2) + source_position.flatten(
            2
        ).transpose(1, 2)
        query = self.query.transpose(1, 2).reshape(
            1, channels, *self.target_size
        )
        if target_size != self.target_size:
            query = F.interpolate(
                query,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        query = query.flatten(2).transpose(1, 2).expand(batch, -1, -1)
        attended, _ = self.cross_attention(
            self.query_norm(query),
            self.source_norm(source),
            self.source_norm(source),
            need_weights=False,
        )
        query = query + attended
        query = query + self.ffn(self.output_norm(query))
        return query.transpose(1, 2).reshape(
            batch, channels, *target_size
        )


class TemporalResidualBlock(nn.Module):
    """Mix adjacent decoded frames without changing the latent contract."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = math.gcd(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
        )
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return inputs + hidden


class LearnedUpsampleBlock(nn.Module):
    """Learned 2x upsampling followed by spatial residual refinement."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(input_channels, 4 * output_channels, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.refine = nn.Sequential(
            SpatialResidualBlock(output_channels),
            SpatialResidualBlock(output_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.refine(self.upsample(inputs))


class VideoLatentBranch(nn.Module):
    """Create stochastic 2D latents on Wan's temporal and spatial lattice."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        spatial_stride: int,
        temporal_stride: int,
        temporal_layers: int,
        temporal_heads: int,
        fusion_dim: int,
        query_heads: int,
    ) -> None:
        super().__init__()
        if temporal_stride != 4:
            raise ValueError(
                "VGGT 2D latents must use Wan's temporal stride 4; "
                f"got {temporal_stride}"
            )
        if spatial_stride != 16:
            raise ValueError(
                "VGGT 2D latents must use Wan2.2's spatial stride 16; "
                f"got {spatial_stride}"
            )
        self.spatial_stride = spatial_stride
        self.level_count = 4
        self.level_fusion = FusionResidualBlock(
            self.level_count * input_dim,
            fusion_dim,
        )
        self.spatial_resampler = LearnedSpatialQueryResampler(
            fusion_dim,
            query_heads,
        )
        self.latent_projection = nn.Sequential(
            nn.GroupNorm(math.gcd(8, fusion_dim), fusion_dim),
            nn.SiLU(),
            nn.Conv2d(fusion_dim, latent_dim, 1),
            SpatialResidualBlock(latent_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=temporal_heads,
            dim_feedforward=4 * latent_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(layer, temporal_layers)
        self.temporal_encoder = WanTemporalEncoder(latent_dim, spatial_kernel=3)
        self.mu_head = nn.Conv3d(latent_dim, latent_dim, 1)
        self.logvar_head = nn.Conv3d(latent_dim, latent_dim, 1)

    def forward(
        self,
        features: tuple[torch.Tensor, ...] | list[torch.Tensor],
        video_size: tuple[int, int],
        *,
        sample_posterior: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(features) != self.level_count:
            raise ValueError(
                f"VideoLatentBranch expects four feature levels, got {len(features)}"
            )
        batch, time, views, channels, height, width = features[0].shape
        if any(level.shape != features[0].shape for level in features[1:]):
            raise ValueError("All four VGGT feature levels must share one shape")
        video_height, video_width = video_size
        if video_height % self.spatial_stride or video_width % self.spatial_stride:
            raise ValueError(
                "Wan-compatible video dimensions must be divisible by 16; "
                f"got {video_height}x{video_width}"
            )
        latent_size = (
            video_height // self.spatial_stride,
            video_width // self.spatial_stride,
        )
        fused = torch.cat(features, dim=3)
        spatial = self.level_fusion(
            fused.reshape(
                batch * time * views,
                self.level_count * channels,
                height,
                width,
            )
        )
        spatial = self.spatial_resampler(spatial, latent_size)
        spatial = self.latent_projection(spatial)
        latent_h, latent_w = spatial.shape[-2:]
        spatial = spatial.reshape(
            batch, time, views, -1, latent_h, latent_w
        ).permute(0, 2, 4, 5, 1, 3)
        temporal_mask = torch.triu(
            torch.ones(time, time, dtype=torch.bool, device=spatial.device),
            diagonal=1,
        )
        temporal = self.temporal_transformer(
            spatial.reshape(batch * views * latent_h * latent_w, time, -1),
            mask=temporal_mask,
        )
        temporal = temporal.reshape(
            batch, views, latent_h, latent_w, time, -1
        ).permute(0, 1, 5, 4, 2, 3)
        compressed = self.temporal_encoder(
            temporal.reshape(
                batch * views,
                temporal.shape[2],
                time,
                latent_h,
                latent_w,
            )
        )
        mu = self.mu_head(compressed).reshape(
            batch, views, -1, *compressed.shape[-3:]
        )
        logvar = self.logvar_head(compressed).clamp(-12, 8).reshape_as(mu)
        if sample_posterior:
            latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            latent = mu
        return latent, mu, logvar


class VideoDecoder(nn.Module):
    """Learned ``T'→4T'-3`` and ``H'W'→16H'16W'`` RGB decoder."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        spatial_stride: int = 16,
    ) -> None:
        super().__init__()
        if spatial_stride != 16:
            raise ValueError(
                "The Wan2.2-compatible video decoder requires spatial stride 16"
            )
        hidden_dim = max(32, hidden_dim)
        self.spatial_stride = spatial_stride
        self.temporal_decoder = WanTemporalDecoder(latent_dim, spatial_kernel=3)
        self.input_projection = nn.Conv2d(latent_dim, hidden_dim, 3, padding=1)
        decoder_channels = (
            [192, 128, 96, 64]
            if hidden_dim >= 256
            else [
                max(32, hidden_dim // 2),
                max(32, hidden_dim // 4),
                32,
                32,
            ]
        )
        stages: list[nn.Module] = []
        channels = hidden_dim
        for next_channels in decoder_channels:
            stages.append(LearnedUpsampleBlock(channels, next_channels))
            channels = next_channels
        self.spatial_decoder = nn.Sequential(*stages)
        self.output_projection = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(
        self,
        latent: torch.Tensor,
        output_time: int | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if latent.ndim != 6:
            raise ValueError(
                "VideoDecoder expects [B,V,C,T,H,W], "
                f"got {tuple(latent.shape)}"
            )
        batch, views, channels, latent_time, height, width = latent.shape
        temporal = self.temporal_decoder(
            latent.reshape(batch * views, channels, latent_time, height, width),
            output_time=output_time,
        )
        decoded_time = temporal.shape[2]
        frames = temporal.permute(0, 2, 1, 3, 4).reshape(
            batch * views * decoded_time,
            channels,
            height,
            width,
        )
        frames = self.input_projection(frames)
        frames = self.spatial_decoder(frames)
        frames = self.output_projection(F.silu(frames)).tanh()
        expected_size = (height * self.spatial_stride, width * self.spatial_stride)
        if output_size is not None and tuple(output_size) != expected_size:
            raise ValueError(
                "The requested output size violates the Wan spatial contract: "
                f"latent {height}x{width} decodes to {expected_size}, "
                f"not {tuple(output_size)}"
            )
        return frames.reshape(
            batch, views, decoded_time, 3, *expected_size
        ).permute(0, 2, 1, 3, 4, 5)
