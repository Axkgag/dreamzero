from __future__ import annotations

import unittest

import torch

from groot.vla.model.vggt_3d_wam.losses import (
    ChunkedLPIPSLoss,
    charbonnier_loss,
    spatial_gradient_loss,
    ssim_loss,
    temporal_difference_loss,
)
from groot.vla.model.vggt_3d_wam.pointmap_decoder import PointMapDecoder


class VGGTTokenizerV2SmokeTest(unittest.TestCase):
    def test_visual_quality_losses_backpropagate(self) -> None:
        prediction_logits = torch.randn(
            1, 3, 1, 3, 64, 64, requires_grad=True
        )
        prediction = prediction_logits.tanh()
        target = torch.randn_like(prediction).tanh()
        lpips_loss = ChunkedLPIPSLoss(net="alex", chunk_size=2)
        losses = (
            charbonnier_loss(prediction, target),
            ssim_loss(prediction, target),
            spatial_gradient_loss(prediction, target),
            temporal_difference_loss(prediction, target),
            lpips_loss(prediction, target),
        )

        self.assertTrue(all(torch.isfinite(loss) for loss in losses))
        sum(losses).backward()
        self.assertIsNotNone(prediction_logits.grad)

    def test_pointmap_decoder_upsamples_without_expanding_ray_logits(
        self,
    ) -> None:
        decoder = PointMapDecoder(
            token_dim=8,
            image_size=(32, 64),
            output_size=(8, 16),
            ray_size=(4, 8),
            depth_bins=4,
            min_range=0.1,
            max_range=1.0,
            ray_chunk_size=16,
            hidden_dim=16,
            residual_layers=1,
            x_range=(0.1, 1.0),
            y_range=(-0.5, 0.5),
            z_range=(0.1, 1.0),
        )
        token_grid = torch.randn(1, 1, 8, 1, 2, 2, requires_grad=True)
        intrinsics = torch.tensor(
            [[32.0, 0.0, 31.5], [0.0, 32.0, 15.5], [0.0, 0.0, 1.0]]
        ).view(1, 1, 1, 3, 3)
        transforms = torch.eye(4).view(1, 1, 1, 4, 4)

        pointmap, depth_logits, occupancy_logits, sample_valid = decoder(
            token_grid, intrinsics, transforms
        )

        self.assertEqual(tuple(pointmap.shape), (1, 1, 1, 3, 8, 16))
        self.assertEqual(tuple(depth_logits.shape), (1, 1, 1, 4, 4, 8))
        self.assertEqual(
            tuple(occupancy_logits.shape), (1, 1, 1, 4, 4, 8)
        )
        self.assertEqual(tuple(sample_valid.shape), (1, 1, 1, 4, 4, 8))
        pointmap.square().mean().backward()
        self.assertIsNotNone(token_grid.grad)


if __name__ == "__main__":
    unittest.main()
