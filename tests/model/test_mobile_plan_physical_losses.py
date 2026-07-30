from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from groot.vla.model.dreamzero.action_head.mobile_plan_flow_matching import (
    MobilePlanFlowMatchingActionHead,
)
from groot.vla.model.dreamzero.action_head.mobile_plan_physical_losses import (
    MobilePlanPhysicalConsistencyLosses,
    eef_current_to_future_base,
    eef_future_to_current_base,
    rotation6d_rows_to_matrix,
)
from groot.vla.model.dreamzero.modules.flow_match_scheduler import (
    FlowMatchScheduler,
)


def _stats(path: Path, hand_dim: int = 1) -> None:
    value = {
        "fit_split": "train",
        "hand_dim": hand_dim,
        "statistics": {
            "base_xy": {"q01": [-2.0, -1.0], "q99": [2.0, 1.0]},
            "eef_xyz": {
                "q01": [-1.0, -2.0, -3.0],
                "q99": [1.0, 2.0, 3.0],
            },
            "hand": {
                "q01": [0.0] * hand_dim,
                "q99": [2.0] * hand_dim,
            },
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def _identity_actions(batch: int = 1) -> torch.Tensor:
    action = torch.zeros(batch, 12, 21)
    action[:, :6, 3] = 1.0
    action[:, 6:, 3] = 1.0
    action[:, 6:, 7] = 1.0
    return action


def _identity_prior_actions(batch: int = 1) -> torch.Tensor:
    action = torch.zeros(batch, 6, 21)
    action[:, :3, 3] = 1.0
    action[:, 3:, 3] = 1.0
    action[:, 3:, 7] = 1.0
    return action


class MobilePlanPhysicalLossesTest(unittest.TestCase):
    def test_clean_action_recovery_matches_flow_parameterization(self) -> None:
        scheduler = FlowMatchScheduler(
            num_inference_steps=100,
            shift=5,
            sigma_min=0.0,
            extra_one_step=True,
        )
        scheduler.set_timesteps(1000, training=True)
        clean = torch.randn(2, 12, 21)
        noise = torch.randn_like(clean)
        ids = torch.tensor([[0, 123, 999] * 4, [700, 333, 5] * 4])
        timestep = scheduler.timesteps[ids]
        noisy = scheduler.add_noise(clean, noise, timestep)
        velocity = scheduler.training_target(clean, noise, timestep)

        head = MobilePlanFlowMatchingActionHead.__new__(
            MobilePlanFlowMatchingActionHead
        )
        torch.nn.Module.__init__(head)
        head.scheduler = scheduler
        recovered = head.recover_clean_actions(noisy, velocity, timestep)
        torch.testing.assert_close(recovered, clean, atol=2e-6, rtol=2e-6)

    def test_rotation6d_is_right_handed(self) -> None:
        value = torch.randn(64, 6, requires_grad=True)
        rotation = rotation6d_rows_to_matrix(value)
        identity = rotation @ rotation.transpose(-1, -2)
        torch.testing.assert_close(
            identity,
            torch.eye(3).expand_as(identity),
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            torch.linalg.det(rotation),
            torch.ones(64),
            atol=2e-5,
            rtol=2e-5,
        )
        rotation.square().mean().backward()
        self.assertTrue(torch.isfinite(value.grad).all())

    def test_future_base_eef_transform_round_trip(self) -> None:
        base = torch.tensor([[[1.0, -0.5, 1.0, 0.0]]])
        eef_current = torch.tensor(
            [[[1.25, 0.5, 0.75, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]]
        )
        eef_future = eef_current_to_future_base(base, eef_current)
        reconstructed = eef_future_to_current_base(base, eef_future)
        torch.testing.assert_close(
            reconstructed, eef_current, atol=1e-6, rtol=1e-6
        )

    def test_joint_prior_terms_use_dynamic_future_base_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            _stats(path)
            module = MobilePlanPhysicalConsistencyLosses(path, plan_horizon=3)
            target = _identity_prior_actions()
            target[:, :3, 0] = 0.5
            target[:, 3:, 0] = 0.75
            mask = torch.zeros_like(target, dtype=torch.bool)
            mask[:, :3, :4] = True
            mask[:, 3:, :9] = True
            base_gt, manip_gt = module.physical_plans(target)
            eef_future = eef_current_to_future_base(base_gt, manip_gt)
            eef_prediction = eef_future.clone()
            eef_prediction[..., :3] = (
                2.0
                * (eef_future[..., :3] - module.eef_xyz_q01)
                / (module.eef_xyz_q99 - module.eef_xyz_q01)
                - 1.0
            )
            terms = module.prior_terms(
                base_prediction=target[:, :3, :4],
                eef_prediction=eef_prediction,
                clean_target=target,
                action_mask=mask,
                has_real_action=torch.ones(1).bool(),
                eef_frame="future_base",
            )
            perturbed_base = target[:, :3, :4].clone()
            perturbed_base[..., 0] += 0.1
            perturbed_terms = module.prior_terms(
                base_prediction=perturbed_base,
                eef_prediction=eef_prediction,
                clean_target=target,
                action_mask=mask,
                has_real_action=torch.ones(1).bool(),
                eef_frame="future_base",
            )
        for key in (
            "base_prior_xy_loss",
            "base_prior_yaw_loss",
            "eef_prior_position_loss",
            "eef_prior_rotation_loss",
            "joint_prior_consistency_position_loss",
            "joint_prior_consistency_rotation_loss",
        ):
            self.assertAlmostEqual(terms[key].item(), 0.0, places=6, msg=key)
        self.assertGreater(
            perturbed_terms["joint_prior_consistency_position_loss"].item(),
            0.0,
        )

    def test_physical_losses_are_zero_for_exact_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            _stats(path)
            module = MobilePlanPhysicalConsistencyLosses(path)
            target = _identity_actions()
            mask = torch.zeros_like(target, dtype=torch.bool)
            mask[:, :6, :4] = True
            mask[:, 6:, :10] = True
            losses = module(target.clone(), target, mask, torch.ones(1).bool())
        for key in (
            "base_xy_loss",
            "base_yaw_loss",
            "base_yaw_unit_loss",
            "eef_position_loss",
            "eef_rotation_loss",
            "hand_loss",
            "base_eef_consistency_position_loss",
            "base_eef_consistency_rotation_loss",
        ):
            self.assertAlmostEqual(losses[key].item(), 0.0, places=6, msg=key)

    def test_invalid_horizon_and_padding_have_zero_gradient(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            _stats(path)
            module = MobilePlanPhysicalConsistencyLosses(path)
            target = _identity_actions()
            prediction = target.clone().requires_grad_(True)
            mask = torch.zeros_like(target, dtype=torch.bool)
            mask[:, :5, :4] = True
            mask[:, 6:11, :10] = True
            losses = module(prediction, target, mask, torch.ones(1).bool())
            total = sum(
                value
                for name, value in losses.items()
                if name.endswith("_loss")
            )
            total.backward()
        self.assertEqual(prediction.grad[:, 5].abs().sum().item(), 0.0)
        self.assertEqual(prediction.grad[:, 11].abs().sum().item(), 0.0)
        self.assertEqual(prediction.grad[:, :6, 4:].abs().sum().item(), 0.0)
        self.assertEqual(prediction.grad[:, 6:, 10:].abs().sum().item(), 0.0)

    def test_loss_ramp(self) -> None:
        self.assertEqual(
            MobilePlanFlowMatchingActionHead._ramped_weight(0.2, 99, 100, 50),
            0.0,
        )
        self.assertAlmostEqual(
            MobilePlanFlowMatchingActionHead._ramped_weight(0.2, 125, 100, 50),
            0.1,
        )
        self.assertAlmostEqual(
            MobilePlanFlowMatchingActionHead._ramped_weight(0.2, 200, 100, 50),
            0.2,
        )


if __name__ == "__main__":
    unittest.main()
