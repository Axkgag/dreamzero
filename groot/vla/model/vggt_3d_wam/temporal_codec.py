"""Wan-compatible learned temporal compression shared by 2D and 3D latents."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


WAN_TEMPORAL_STRIDE = 4


def wan_latent_time(video_time: int) -> int:
    """Return Wan's latent length for a ``4k + 1`` frame clip."""
    if video_time < 1:
        raise ValueError(f"video_time must be positive, got {video_time}")
    if (video_time - 1) % WAN_TEMPORAL_STRIDE:
        raise ValueError(
            "Wan-compatible clips must contain 4k+1 frames; "
            f"received {video_time}"
        )
    return 1 + (video_time - 1) // WAN_TEMPORAL_STRIDE


def wan_video_time(latent_time: int) -> int:
    """Return the decoded video length for a Wan-compatible latent."""
    if latent_time < 1:
        raise ValueError(f"latent_time must be positive, got {latent_time}")
    return 1 + WAN_TEMPORAL_STRIDE * (latent_time - 1)


class TemporalResidualBlock(nn.Module):
    """A lightweight residual block operating only along time."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=0,
            groups=channels,
        )
        self.pointwise = nn.Conv3d(channels, channels, 1)
        self.activation = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.norm(inputs.permute(0, 2, 3, 4, 1)).permute(
            0, 4, 1, 2, 3
        )
        residual = self.activation(residual)
        residual = F.pad(residual, (0, 0, 0, 0, 2, 0))
        residual = self.depthwise(residual)
        residual = self.pointwise(self.activation(residual))
        return inputs + residual


class WanTemporalEncoder(nn.Module):
    """Compress ``1 + 4k`` frames into ``1 + k`` learned latent steps.

    The first frame has its own projection. Every following non-overlapping
    four-frame chunk is compressed by a learned convolution. This gives 2D
    and 3D latents the same temporal lattice as Wan's video VAE without
    constraining either branch to Wan's latent values.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pre = TemporalResidualBlock(channels)
        self.first = nn.Conv3d(channels, channels, 1)
        self.chunk = nn.Conv3d(
            channels,
            channels,
            kernel_size=(WAN_TEMPORAL_STRIDE, 1, 1),
            stride=(WAN_TEMPORAL_STRIDE, 1, 1),
        )
        self.post = TemporalResidualBlock(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError(
                "WanTemporalEncoder expects [B,C,T,H,W], "
                f"got {tuple(inputs.shape)}"
            )
        wan_latent_time(inputs.shape[2])
        hidden = self.pre(inputs)
        first = self.first(hidden[:, :, :1])
        if hidden.shape[2] == 1:
            return self.post(first)
        chunks = self.chunk(hidden[:, :, 1:])
        return self.post(torch.cat((first, chunks), dim=2))


class WanTemporalDecoder(nn.Module):
    """Expand ``1 + k`` latent steps into ``1 + 4k`` learned frame features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pre = TemporalResidualBlock(channels)
        self.first = nn.Conv3d(channels, channels, 1)
        self.chunk = nn.Conv3d(
            channels,
            channels * WAN_TEMPORAL_STRIDE,
            1,
        )
        self.post = TemporalResidualBlock(channels)

    def forward(
        self,
        inputs: torch.Tensor,
        output_time: int | None = None,
    ) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError(
                "WanTemporalDecoder expects [B,C,T_latent,H,W], "
                f"got {tuple(inputs.shape)}"
            )
        expected_time = wan_video_time(inputs.shape[2])
        if output_time is not None and output_time != expected_time:
            raise ValueError(
                "The requested output time violates the Wan temporal contract: "
                f"latent T={inputs.shape[2]} decodes to {expected_time}, "
                f"not {output_time}"
            )
        hidden = self.pre(inputs)
        first = self.first(hidden[:, :, :1])
        if hidden.shape[2] == 1:
            return self.post(first)
        batch, channels, tail_time, height, width = (
            hidden.shape[0],
            hidden.shape[1],
            hidden.shape[2] - 1,
            hidden.shape[3],
            hidden.shape[4],
        )
        chunks = self.chunk(hidden[:, :, 1:])
        chunks = (
            chunks.reshape(
                batch,
                channels,
                WAN_TEMPORAL_STRIDE,
                tail_time,
                height,
                width,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(
                batch,
                channels,
                tail_time * WAN_TEMPORAL_STRIDE,
                height,
                width,
            )
        )
        return self.post(torch.cat((first, chunks), dim=2))
