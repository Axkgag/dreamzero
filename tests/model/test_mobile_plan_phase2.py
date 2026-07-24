from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from groot.vla.model.dreamzero.action_head.mobile_plan_flow_matching import (
    MobilePlanFlowMatchingActionHead,
)
from groot.vla.model.dreamzero.modules.wan_video_dit_dual_plan import (
    DualPlanActionDecoder,
    DualPlanActionEncoder,
)


class _UnitWeightScheduler:
    @staticmethod
    def training_weight(timestep: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(timestep, dtype=torch.float32)


class MobilePlanPhase2Test(unittest.TestCase):
    def test_dual_projection_shapes_and_gradients(self) -> None:
        encoder = DualPlanActionEncoder(
            base_action_dim=4,
            manipulator_action_dim=21,
            hidden_size=32,
            num_embodiments=1,
            plan_time_offsets=[1, 4, 8, 12, 16, 24],
            control_fps=30.0,
        )
        decoder = DualPlanActionDecoder(
            base_action_dim=4,
            manipulator_action_dim=21,
            hidden_size=16,
            model_dim=32,
            num_embodiments=1,
            plan_horizon=6,
        )
        packed = torch.randn(2, 12, 21)
        timestep = torch.randint(0, 1000, (2, 12))
        category = torch.zeros(2, dtype=torch.long)
        prediction = decoder(encoder(packed, timestep, category), category)
        self.assertEqual(tuple(prediction.shape), (2, 12, 21))
        self.assertTrue(torch.equal(prediction[:, :6, 4:], torch.zeros(2, 6, 17)))

        loss = prediction[:, :6, :4].square().mean()
        loss = loss + prediction[:, 6:].square().mean()
        loss.backward()
        self.assertGreater(
            encoder.base_encoder.W1.W.grad.abs().sum().item(), 0
        )
        self.assertGreater(
            encoder.manipulator_encoder.W1.W.grad.abs().sum().item(), 0
        )
        self.assertGreater(
            decoder.base_decoder.layer2.W.grad.abs().sum().item(), 0
        )
        self.assertGreater(
            decoder.manipulator_decoder.layer2.W.grad.abs().sum().item(), 0
        )

    def test_branch_loss_ignores_padding_and_invalid_waypoints(self) -> None:
        head = MobilePlanFlowMatchingActionHead.__new__(
            MobilePlanFlowMatchingActionHead
        )
        nn.Module.__init__(head)
        head.plan_horizon = 6
        head.base_action_dim = 4
        head.manipulator_action_dim = 21
        head.scheduler = _UnitWeightScheduler()
        head._device = "cpu"
        head.config = SimpleNamespace(
            base_flow_loss_weight=1.0,
            manipulator_flow_loss_weight=1.0,
        )

        prediction = torch.ones(1, 12, 21)
        target = torch.zeros_like(prediction)
        mask = torch.zeros_like(prediction, dtype=torch.bool)
        mask[:, :6, :4] = True
        mask[:, 6:, :10] = True
        # Last future waypoint is invalid in both streams.
        mask[:, 5] = False
        mask[:, 11] = False
        timestep = torch.zeros(1, 12, dtype=torch.long)
        losses = head.compute_action_losses(
            prediction,
            target,
            mask,
            torch.ones(1, dtype=torch.bool),
            timestep,
        )
        self.assertAlmostEqual(losses["base_flow_loss"].item(), 1.0)
        self.assertAlmostEqual(losses["manipulator_flow_loss"].item(), 1.0)

        changed = prediction.clone()
        changed[:, :6, 4:] = 1000
        changed[:, 6:, 10:] = 1000
        changed[:, 5] = 1000
        changed[:, 11] = 1000
        changed_losses = head.compute_action_losses(
            changed,
            target,
            mask,
            torch.ones(1, dtype=torch.bool),
            timestep,
        )
        torch.testing.assert_close(
            losses["action_loss"], changed_losses["action_loss"]
        )

    def test_base_and_manipulator_share_timestep(self) -> None:
        head = MobilePlanFlowMatchingActionHead.__new__(
            MobilePlanFlowMatchingActionHead
        )
        nn.Module.__init__(head)
        head.plan_horizon = 6
        source = torch.arange(12).reshape(1, 12)
        aligned = head.align_action_timestep_ids(source)
        torch.testing.assert_close(aligned[:, :6], aligned[:, 6:])

    def test_one_plan_window_timestep_expands_to_all_tokens(self) -> None:
        head = MobilePlanFlowMatchingActionHead.__new__(
            MobilePlanFlowMatchingActionHead
        )
        nn.Module.__init__(head)
        timestep_blocks = torch.tensor([[[37] * 8]])
        actions = torch.zeros(1, 12, 21)
        noise = torch.zeros(1, 9, 16, 44, 80)
        result = head.build_coupled_action_timestep_ids(
            timestep_blocks, actions, noise
        )
        self.assertEqual(tuple(result.shape), (1, 12))
        torch.testing.assert_close(result, torch.full((1, 12), 37))


if __name__ == "__main__":
    unittest.main()
