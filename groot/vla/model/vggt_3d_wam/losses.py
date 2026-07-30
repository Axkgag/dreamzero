"""Differentiable visual-quality losses for the VGGT video tokenizer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _flatten_video_frames(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 6:
        raise ValueError(
            "Visual losses expect [B,T,V,C,H,W], "
            f"got {tuple(video.shape)}"
        )
    batch, time, views, channels, height, width = video.shape
    return video.reshape(batch * time * views, channels, height, width)


def charbonnier_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    return torch.sqrt((prediction - target).square() + epsilon**2).mean()


def ssim_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 7,
) -> torch.Tensor:
    """Return `1 - SSIM` for videos represented in `[-1, 1]`."""
    prediction = (_flatten_video_frames(prediction).float() + 1.0) * 0.5
    target = (_flatten_video_frames(target).float() + 1.0) * 0.5
    padding = kernel_size // 2
    mu_prediction = F.avg_pool2d(
        prediction, kernel_size, stride=1, padding=padding
    )
    mu_target = F.avg_pool2d(
        target, kernel_size, stride=1, padding=padding
    )
    prediction_variance = (
        F.avg_pool2d(
            prediction.square(), kernel_size, stride=1, padding=padding
        )
        - mu_prediction.square()
    ).clamp_min(0)
    target_variance = (
        F.avg_pool2d(
            target.square(), kernel_size, stride=1, padding=padding
        )
        - mu_target.square()
    ).clamp_min(0)
    covariance = (
        F.avg_pool2d(
            prediction * target, kernel_size, stride=1, padding=padding
        )
        - mu_prediction * mu_target
    )
    c1 = 0.01**2
    c2 = 0.03**2
    score = (
        (2 * mu_prediction * mu_target + c1)
        * (2 * covariance + c2)
        / (
            (mu_prediction.square() + mu_target.square() + c1)
            * (prediction_variance + target_variance + c2)
        ).clamp_min(1e-8)
    )
    return 1.0 - score.mean()


def spatial_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    prediction_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    return 0.5 * (
        charbonnier_loss(prediction_x, target_x)
        + charbonnier_loss(prediction_y, target_y)
    )


def temporal_difference_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape[1] <= 1:
        return prediction.new_zeros(())
    prediction_delta = prediction[:, 1:] - prediction[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    return charbonnier_loss(prediction_delta, target_delta)


class ChunkedLPIPSLoss(nn.Module):
    """Frozen LPIPS network with bounded activation memory."""

    def __init__(self, net: str = "alex", chunk_size: int = 8) -> None:
        super().__init__()
        try:
            import lpips
        except ImportError as error:
            raise ImportError(
                "LPIPS visual supervision is enabled, but package `lpips` "
                "is unavailable. Install the project dependencies or run "
                "`pip install lpips==0.1.4` in the DreamZero environment."
            ) from error
        self.metric = lpips.LPIPS(net=net, verbose=False)
        self.metric.requires_grad_(False)
        self.metric.eval()
        self.chunk_size = max(1, int(chunk_size))

    def train(self, mode: bool = True):
        super().train(mode)
        self.metric.eval()
        return self

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction_frames = _flatten_video_frames(prediction).float()
        target_frames = _flatten_video_frames(target).float()
        total = prediction_frames.new_zeros(())
        count = 0
        for start in range(0, len(prediction_frames), self.chunk_size):
            end = min(start + self.chunk_size, len(prediction_frames))
            prediction_chunk = prediction_frames[start:end]
            target_chunk = target_frames[start:end]
            if self.training and prediction_chunk.requires_grad:
                values = checkpoint(
                    self._metric_forward,
                    prediction_chunk,
                    target_chunk,
                    use_reentrant=False,
                )
            else:
                values = self._metric_forward(
                    prediction_chunk,
                    target_chunk,
                )
            total = total + values.sum()
            count += values.numel()
        return total / max(count, 1)

    def _metric_forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.metric(prediction, target, normalize=False)
