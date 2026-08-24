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


class SpaceToDepthResidualDownsample(nn.Module):
    """Preserve sub-pixel layout before learning a residual downsample.

    The packed channel mean exactly matches integer-ratio area downsampling.
    A zero-initialized residual path can then learn position-sensitive detail
    without changing a pretrained v4.1 RGB pyramid at initialization.
    """

    def __init__(self, channels: int, factor: int) -> None:
        super().__init__()
        if factor < 2 or factor & (factor - 1):
            raise ValueError(
                "Space-to-depth factor must be a power of two >= 2, "
                f"got {factor}"
            )
        self.channels = channels
        self.factor = factor
        packed_channels = channels * factor * factor
        self.residual = nn.Sequential(
            nn.Conv2d(packed_channels, channels, 1),
            SpatialResidualBlock(channels),
            nn.Conv2d(channels, channels, 1),
        )
        # Preserve the exact v4.1 area-downsample result before fine-tuning.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        height, width = inputs.shape[-2:]
        if height % self.factor or width % self.factor:
            raise ValueError(
                "RGB pyramid feature size must be divisible by its "
                f"space-to-depth factor {self.factor}, got {height}x{width}"
            )
        packed = F.pixel_unshuffle(inputs, self.factor)
        batch, _, target_height, target_width = packed.shape
        area_base = packed.reshape(
            batch,
            self.channels,
            self.factor * self.factor,
            target_height,
            target_width,
        ).mean(dim=2)
        return area_base + self.residual(packed)


class LightweightRGBEncoder(nn.Module):
    """Encode a four-level residual RGB pyramid onto the latent lattice.

    The original v4 path is retained as the deepest ``H/16`` branch.  Three
    zero-initialized lateral projections add the ``H/2``, ``H/4`` and ``H/8``
    levels, so a v4 checkpoint produces exactly the same output before the
    newly added pyramid paths start learning.
    """

    def __init__(self, output_channels: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            # Start with an exact spatial-to-channel rearrangement, following
            # Wan VAE's information-preserving first compression step.
            nn.PixelUnshuffle(2),
            nn.Conv2d(12, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        level_channels = (32, 64, 96, 128)
        self.pyramid_refine = nn.ModuleList(
            SpatialResidualBlock(channels) for channels in level_channels
        )
        for block in self.pyramid_refine:
            # Keep every new residual stage an exact identity when a v4
            # checkpoint is used to initialize the model.
            nn.init.zeros_(block.conv2.weight)
            nn.init.zeros_(block.conv2.bias)
        self.pyramid_projections = nn.ModuleList(
            nn.Conv2d(channels, output_channels, 1)
            for channels in level_channels[:-1]
        )
        self.pyramid_downsamplers = nn.ModuleList(
            SpaceToDepthResidualDownsample(channels, factor)
            for channels, factor in zip(level_channels[:-1], (8, 4, 2))
        )
        for projection in self.pyramid_projections:
            # The old deepest RGB path remains solely responsible for the
            # initial output; shallower levels are introduced gradually.
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.output_projection = nn.Conv2d(128, output_channels, 1)
        # Preserve pre-RGB-path behavior exactly at initialization. The final
        # projection learns first, then starts updating the RGB encoder.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(
                "LightweightRGBEncoder expects [N,3,H,W], "
                f"got {tuple(rgb.shape)}"
            )
        if rgb.shape[-2] % 16 or rgb.shape[-1] % 16:
            raise ValueError(
                "RGB dimensions must be divisible by 16, "
                f"got {tuple(rgb.shape[-2:])}"
            )
        hidden = self.encoder[0](rgb)
        hidden = self.encoder[2](self.encoder[1](hidden))
        level_0 = self.pyramid_refine[0](hidden)
        hidden = self.encoder[4](self.encoder[3](level_0))
        level_1 = self.pyramid_refine[1](hidden)
        hidden = self.encoder[6](self.encoder[5](level_1))
        level_2 = self.pyramid_refine[2](hidden)
        hidden = self.encoder[8](self.encoder[7](level_2))
        level_3 = self.pyramid_refine[3](hidden)

        output = self.output_projection(level_3)
        target_size = level_3.shape[-2:]
        for level, downsampler, projection in zip(
            (level_0, level_1, level_2),
            self.pyramid_downsamplers,
            self.pyramid_projections,
        ):
            lateral = downsampler(level)
            if lateral.shape[-2:] != target_size:
                raise RuntimeError(
                    "RGB pyramid downsample produced an unexpected size: "
                    f"expected {target_size}, got {lateral.shape[-2:]}"
                )
            output = output + projection(lateral)
        return output


class LearnedSpatialQueryResampler(nn.Module):
    """Cross-attend fixed 10x20 latent queries to a 12x23 feature grid."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        source_size: tuple[int, int] = (12, 23),
        target_size: tuple[int, int] = (10, 20),
        use_local_residual: bool = False,
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
        self.local_projection = (
            nn.Conv2d(channels, channels, 1)
            if use_local_residual
            else None
        )
        if self.local_projection is not None:
            # Preserve the pretrained v3 resampler exactly at initialization.
            # The local path is learned gradually during v3.1 fine-tuning.
            nn.init.zeros_(self.local_projection.weight)
            nn.init.zeros_(self.local_projection.bias)

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
        if self.local_projection is not None:
            local = F.interpolate(
                inputs,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            local = self.local_projection(local)
            query = query + local.flatten(2).transpose(1, 2)
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


class LatentPyramidUpsampleBlock(nn.Module):
    """Build one latent scale and expose a zero-init decoder injection."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(input_channels, 4 * output_channels, 1),
            nn.PixelShuffle(2),
        )
        self.refine = SpatialResidualBlock(output_channels)
        self.injection_projection = nn.Conv2d(
            output_channels,
            output_channels,
            1,
        )
        # The v4.1 decoder remains bit-identical until this path learns.
        nn.init.zeros_(self.injection_projection.weight)
        nn.init.zeros_(self.injection_projection.bias)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.refine(self.upsample(inputs))
        return features, self.injection_projection(features)


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
        query_local_residual: bool = False,
        use_rgb_path: bool = True,
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
            use_local_residual=query_local_residual,
        )
        self.rgb_encoder = (
            LightweightRGBEncoder(fusion_dim) if use_rgb_path else None
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
        rgb_video: torch.Tensor | None = None,
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
        if self.rgb_encoder is not None:
            expected_video_shape = (
                batch,
                time,
                views,
                3,
                video_height,
                video_width,
            )
            if rgb_video is None or tuple(rgb_video.shape) != expected_video_shape:
                actual_shape = None if rgb_video is None else tuple(rgb_video.shape)
                raise ValueError(
                    "RGB local path expects canonical video "
                    f"{expected_video_shape}, got {actual_shape}"
                )
            rgb_local = self.rgb_encoder(
                rgb_video.reshape(
                    batch * time * views,
                    3,
                    video_height,
                    video_width,
                )
            )
            if rgb_local.shape != spatial.shape:
                raise ValueError(
                    "RGB and VGGT feature lattices must match, got "
                    f"{tuple(rgb_local.shape)} and {tuple(spatial.shape)}"
                )
            spatial = spatial + rgb_local
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
        latent_residual_blocks: int = 0,
    ) -> None:
        super().__init__()
        if spatial_stride != 16:
            raise ValueError(
                "The Wan2.2-compatible video decoder requires spatial stride 16"
            )
        if latent_residual_blocks < 0:
            raise ValueError("latent_residual_blocks must be non-negative")
        hidden_dim = max(32, hidden_dim)
        self.spatial_stride = spatial_stride
        self.temporal_decoder = WanTemporalDecoder(latent_dim, spatial_kernel=3)
        self.input_projection = nn.Conv2d(latent_dim, hidden_dim, 3, padding=1)
        latent_refine: list[nn.Module] = []
        for _ in range(latent_residual_blocks):
            block = SpatialResidualBlock(hidden_dim)
            # A zero-initialized final convolution makes each newly added
            # residual block an exact identity before v3.1 fine-tuning.
            nn.init.zeros_(block.conv2.weight)
            nn.init.zeros_(block.conv2.bias)
            latent_refine.append(block)
        self.latent_refine = nn.Sequential(*latent_refine)
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
        latent_pyramid: list[nn.Module] = []
        channels = hidden_dim
        for next_channels in decoder_channels:
            latent_pyramid.append(
                LatentPyramidUpsampleBlock(channels, next_channels)
            )
            channels = next_channels
        self.latent_pyramid = nn.ModuleList(latent_pyramid)
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
        frames = self.latent_refine(frames)
        latent_features = frames
        for decoder_stage, latent_stage in zip(
            self.spatial_decoder,
            self.latent_pyramid,
        ):
            frames = decoder_stage(frames)
            latent_features, injection = latent_stage(latent_features)
            frames = frames + injection
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
