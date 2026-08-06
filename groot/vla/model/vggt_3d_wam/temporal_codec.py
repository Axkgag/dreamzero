"""Wan-style, first-frame-aware causal temporal compression.

Both tokenizer branches use the same ``1 + 4k <-> 1 + k`` temporal lattice.
The implementation deliberately processes ``[frame 0], [1:5], [5:9], ...``
as chunks so full-clip and incremental execution share exactly the same path.
"""

from __future__ import annotations

from typing import Any

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


class RMSNorm3d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = inputs * torch.rsqrt(
            inputs.float().square().mean(dim=1, keepdim=True) + self.eps
        ).to(inputs.dtype)
        return normalized * self.weight[None, :, None, None, None]


class CachedCausalConv3d(nn.Module):
    """Causal convolution whose temporal input cache is explicit."""

    def __init__(
        self,
        channels: int,
        *,
        spatial_kernel: int,
        temporal_stride: int = 1,
    ) -> None:
        super().__init__()
        if spatial_kernel not in (1, 3):
            raise ValueError("spatial_kernel must be 1 or 3")
        self.temporal_context = 2
        self.spatial_padding = spatial_kernel // 2
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, spatial_kernel, spatial_kernel),
            stride=(temporal_stride, 1, 1),
            groups=channels,
            bias=False,
        )
        self.channel_mixing = nn.Conv3d(channels, channels, 1)

    def forward_chunk(
        self,
        inputs: torch.Tensor,
        cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cache is None:
            hidden = F.pad(inputs, (0, 0, 0, 0, self.temporal_context, 0))
        else:
            hidden = torch.cat((cache, inputs), dim=2)
            missing_context = max(0, self.temporal_context - cache.shape[2])
            if missing_context:
                hidden = F.pad(hidden, (0, 0, 0, 0, missing_context, 0))
        next_cache = hidden[:, :, -self.temporal_context :]
        if self.spatial_padding:
            hidden = F.pad(
                hidden,
                (
                    self.spatial_padding,
                    self.spatial_padding,
                    self.spatial_padding,
                    self.spatial_padding,
                ),
            )
        return self.channel_mixing(self.conv(hidden)), next_cache


class CausalResidualBlock(nn.Module):
    """Two cached causal convolutions with a lightweight residual path."""

    def __init__(self, channels: int, spatial_kernel: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm3d(channels)
        self.conv1 = CachedCausalConv3d(
            channels, spatial_kernel=spatial_kernel
        )
        self.norm2 = RMSNorm3d(channels)
        self.conv2 = CachedCausalConv3d(
            channels, spatial_kernel=spatial_kernel
        )

    def forward_chunk(
        self,
        inputs: torch.Tensor,
        state: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
    ]:
        hidden, cache1 = self.conv1.forward_chunk(
            F.silu(self.norm1(inputs)), state[0]
        )
        hidden, cache2 = self.conv2.forward_chunk(
            F.silu(self.norm2(hidden)), state[1]
        )
        return inputs + hidden, (cache1, cache2)


class CausalTemporalDownsample(nn.Module):
    """First-frame bypass followed by a cached stride-2 causal convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = CachedCausalConv3d(
            channels,
            spatial_kernel=1,
            temporal_stride=2,
        )

    def forward_chunk(
        self,
        inputs: torch.Tensor,
        state: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not state["seen"]:
            if inputs.shape[2] != 1:
                raise ValueError("The first temporal chunk must contain frame 0 only")
            return inputs, {"seen": True, "cache": inputs[:, :, -2:]}
        # A stride-2 stage carries one input position across chunk boundaries.
        # ``1 cached + 4 current`` produces two outputs without zero padding.
        hidden = torch.cat((state["cache"][:, :, -1:], inputs), dim=2)
        hidden = self.conv.channel_mixing(self.conv.conv(hidden))
        return hidden, {"seen": True, "cache": inputs[:, :, -1:]}


class CausalTemporalUpsample(nn.Module):
    """First-frame bypass followed by causal feature-to-time rearrangement."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.causal = CachedCausalConv3d(channels, spatial_kernel=1)
        self.expand = nn.Conv3d(channels, 2 * channels, 1)
        self.channels = channels

    def forward_chunk(
        self,
        inputs: torch.Tensor,
        state: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not state["seen"]:
            if inputs.shape[2] != 1:
                raise ValueError("The first latent chunk must contain one step")
            return inputs, {"seen": True, "cache": inputs[:, :, -2:]}
        hidden, cache = self.causal.forward_chunk(inputs, state["cache"])
        hidden = self.expand(F.silu(hidden))
        batch, _, time, height, width = hidden.shape
        hidden = (
            hidden.reshape(batch, self.channels, 2, time, height, width)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, self.channels, 2 * time, height, width)
        )
        return hidden, {"seen": True, "cache": cache}


def _empty_residual_state() -> tuple[None, None]:
    return None, None


class WanTemporalEncoder(nn.Module):
    """Two stride-2 causal stages mapping ``1 + 4k`` to ``1 + k``."""

    def __init__(self, channels: int, spatial_kernel: int = 3) -> None:
        super().__init__()
        self.pre = CausalResidualBlock(channels, spatial_kernel)
        self.downsample1 = CausalTemporalDownsample(channels)
        self.middle = CausalResidualBlock(channels, spatial_kernel)
        self.downsample2 = CausalTemporalDownsample(channels)
        self.post = CausalResidualBlock(channels, spatial_kernel)

    @staticmethod
    def reset_state() -> dict[str, Any]:
        return {
            "pre": _empty_residual_state(),
            "downsample1": {"seen": False, "cache": None},
            "middle": _empty_residual_state(),
            "downsample2": {"seen": False, "cache": None},
            "post": _empty_residual_state(),
        }

    def forward_chunk(
        self,
        inputs: torch.Tensor,
        state: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        hidden, state["pre"] = self.pre.forward_chunk(inputs, state["pre"])
        hidden, state["downsample1"] = self.downsample1.forward_chunk(
            hidden, state["downsample1"]
        )
        hidden, state["middle"] = self.middle.forward_chunk(
            hidden, state["middle"]
        )
        hidden, state["downsample2"] = self.downsample2.forward_chunk(
            hidden, state["downsample2"]
        )
        hidden, state["post"] = self.post.forward_chunk(hidden, state["post"])
        return hidden, state

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError(
                "WanTemporalEncoder expects [B,C,T,H,W], "
                f"got {tuple(inputs.shape)}"
            )
        wan_latent_time(inputs.shape[2])
        state = self.reset_state()
        outputs = []
        boundaries = [inputs[:, :, :1]]
        boundaries.extend(
            inputs[:, :, start : start + WAN_TEMPORAL_STRIDE]
            for start in range(1, inputs.shape[2], WAN_TEMPORAL_STRIDE)
        )
        for chunk in boundaries:
            output, state = self.forward_chunk(chunk, state)
            outputs.append(output)
        return torch.cat(outputs, dim=2)


class WanTemporalDecoder(nn.Module):
    """Two causal upsample stages mapping ``1 + k`` to ``1 + 4k``."""

    def __init__(self, channels: int, spatial_kernel: int = 3) -> None:
        super().__init__()
        self.pre = CausalResidualBlock(channels, spatial_kernel)
        self.upsample1 = CausalTemporalUpsample(channels)
        self.middle = CausalResidualBlock(channels, spatial_kernel)
        self.upsample2 = CausalTemporalUpsample(channels)
        self.post = CausalResidualBlock(channels, spatial_kernel)

    @staticmethod
    def reset_state() -> dict[str, Any]:
        return {
            "pre": _empty_residual_state(),
            "upsample1": {"seen": False, "cache": None},
            "middle": _empty_residual_state(),
            "upsample2": {"seen": False, "cache": None},
            "post": _empty_residual_state(),
        }

    def forward_chunk(
        self,
        inputs: torch.Tensor,
        state: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        hidden, state["pre"] = self.pre.forward_chunk(inputs, state["pre"])
        hidden, state["upsample1"] = self.upsample1.forward_chunk(
            hidden, state["upsample1"]
        )
        hidden, state["middle"] = self.middle.forward_chunk(
            hidden, state["middle"]
        )
        hidden, state["upsample2"] = self.upsample2.forward_chunk(
            hidden, state["upsample2"]
        )
        hidden, state["post"] = self.post.forward_chunk(hidden, state["post"])
        return hidden, state

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
        state = self.reset_state()
        outputs = []
        for index in range(inputs.shape[2]):
            output, state = self.forward_chunk(inputs[:, :, index : index + 1], state)
            outputs.append(output)
        return torch.cat(outputs, dim=2)
