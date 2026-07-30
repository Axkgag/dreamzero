from __future__ import annotations

import torch

from groot.vla.model.vggt_3d_wam.losses import (
    charbonnier_loss,
    spatial_gradient_loss,
    ssim_loss,
    temporal_difference_loss,
)


def test_visual_quality_losses_are_finite_and_backpropagate():
    prediction = torch.randn(
        1, 3, 2, 3, 16, 24, requires_grad=True
    ).tanh()
    target = torch.randn_like(prediction).tanh()

    losses = (
        charbonnier_loss(prediction, target),
        ssim_loss(prediction, target),
        spatial_gradient_loss(prediction, target),
        temporal_difference_loss(prediction, target),
    )
    assert all(torch.isfinite(loss) for loss in losses)
    sum(losses).backward()
    assert prediction.grad is not None


def test_visual_quality_losses_are_small_for_identical_video():
    video = torch.randn(1, 3, 1, 3, 16, 24).tanh()

    assert charbonnier_loss(video, video) < 0.002
    assert ssim_loss(video, video) < 1e-5
    assert spatial_gradient_loss(video, video) < 0.002
    assert temporal_difference_loss(video, video) < 0.002
