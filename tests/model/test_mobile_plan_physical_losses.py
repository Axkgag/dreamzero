from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from groot.vla.model.dreamzero.action_head.mobile_plan_flow_matching import (
    MobilePlanFlowMatchingActionHead,
)
from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (
    WANPolicyHead,
)
from groot.vla.model.dreamzero.action_head.mobile_plan_physical_losses import (
    MobilePlanPhysicalConsistencyLosses,
    eef_current_to_future_base,
    eef_future_to_current_base,
    rotation6d_rows_to_matrix,
    rotation_vector_to_matrix,
    yaw_matrix,
)
from groot.vla.model.dreamzero.modules.flow_match_scheduler import (
    FlowMatchScheduler,
)


def _stats(
    path: Path,
    hand_dim: int = 1,
    eef_rotation_representation: str | None = None,
) -> None:
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
    if eef_rotation_representation is not None:
        value["eef_rotation_representation"] = eef_rotation_representation
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
    def test_stratified_action_sampler_can_force_high_noise(self) -> None:
        scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        scheduler.set_timesteps(1000, training=True)
        head = WANPolicyHead.__new__(WANPolicyHead)
        torch.nn.Module.__init__(head)
        head.scheduler = scheduler
        head.config = SimpleNamespace(
            action_high_noise_fraction=1.0,
            action_high_noise_min_sigma=0.7,
        )
        torch.manual_seed(7)
        ids = head.sample_decoupled_action_timestep_ids((128, 12))
        sigma = scheduler.sigma_from_timestep(scheduler.timesteps[ids])
        self.assertTrue(torch.all(sigma >= 0.7))

    def test_action_flow_weight_floor_keeps_sigma_one_supervision(self) -> None:
        scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        scheduler.set_timesteps(1000, training=True)
        head = MobilePlanFlowMatchingActionHead.__new__(
            MobilePlanFlowMatchingActionHead
        )
        torch.nn.Module.__init__(head)
        head.scheduler = scheduler
        head._device = "cpu"
        head.config = SimpleNamespace(action_flow_weight_floor=0.1)
        timestep = scheduler.timesteps[torch.zeros(1, 2, dtype=torch.long)]
        loss = head._masked_branch_loss(
            torch.ones(1, 2, 1),
            torch.zeros(1, 2, 1),
            torch.ones(1, 2, 1, dtype=torch.bool),
            torch.ones(1, dtype=torch.bool),
            timestep,
        )
        self.assertAlmostEqual(loss.item(), 0.1, places=6)

    def test_rotvec_exponential_map_has_finite_gradients(self) -> None:
        value = torch.tensor(
            [[0.0, 0.0, 0.0], [0.1, -0.2, 0.3]], requires_grad=True
        )
        rotation = rotation_vector_to_matrix(value)
        identity = rotation @ rotation.transpose(-1, -2)
        torch.testing.assert_close(
            identity,
            torch.eye(3).expand_as(identity),
            atol=2e-6,
            rtol=2e-6,
        )
        rotation.square().sum().backward()
        self.assertTrue(torch.isfinite(value.grad).all())

    def test_delta_rotvec_loss_uses_anchor_eef_and_sigma_weight(self) -> None:
        representation = "current_eef_delta_rotvec"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            _stats(path, eef_rotation_representation=representation)
            module = MobilePlanPhysicalConsistencyLosses(
                path,
                plan_horizon=1,
                eef_rotation_representation=representation,
                eef_rotation_sigma_weight_base=0.5,
                eef_rotation_sigma_weight_scale=1.5,
            )
            target = torch.zeros(1, 2, 21)
            target[:, 0, 3] = 1.0
            target[:, 1, 5] = 0.1
            prediction = target.clone()
            prediction[:, 1, 5] = 0.2
            mask = torch.zeros_like(target, dtype=torch.bool)
            mask[:, 0, :4] = True
            mask[:, 1, :7] = True
            anchor = torch.tensor([[0.0, 0.0, 0.0, 0.2, -0.1, 1.5]])
            low = module(
                prediction,
                target,
                mask,
                torch.ones(1).bool(),
                action_sigma=torch.zeros(1, 2),
                anchor_state=anchor,
            )
            high = module(
                prediction,
                target,
                mask,
                torch.ones(1).bool(),
                action_sigma=torch.ones(1, 2),
                anchor_state=anchor,
            )
        self.assertAlmostEqual(
            high["eef_rotation_loss"].item(),
            4.0 * low["eef_rotation_loss"].item(),
            places=5,
        )
        self.assertAlmostEqual(
            high["eef_rotation_error_deg"].item(),
            np.degrees(0.1),
            places=4,
        )

    def test_delta_rotvec_eef_prior_joint_composition_is_exact(self) -> None:
        representation = "current_eef_delta_rotvec"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            _stats(path, eef_rotation_representation=representation)
            module = MobilePlanPhysicalConsistencyLosses(
                path,
                plan_horizon=1,
                eef_rotation_representation=representation,
            )
            target = torch.zeros(1, 2, 21)
            target[:, 0, 3] = 1.0
            target[:, 1, 3:6] = torch.tensor([0.05, -0.02, 0.1])
            mask = torch.zeros_like(target, dtype=torch.bool)
            mask[:, 0, :4] = True
            mask[:, 1, :7] = True
            terms = module.prior_terms(
                base_prediction=target[:, :1, :4],
                eef_prediction=target[:, 1:, :6],
                clean_target=target,
                action_mask=mask,
                has_real_action=torch.ones(1).bool(),
                eef_frame="current_eef_delta",
                anchor_state=torch.tensor(
                    [[0.0, 0.0, 0.0, 0.3, -0.2, 1.4]]
                ),
            )
        for name in (
            "eef_prior_position_loss",
            "eef_prior_rotation_loss",
            "joint_prior_consistency_position_loss",
            "joint_prior_consistency_rotation_loss",
        ):
            self.assertAlmostEqual(terms[name].item(), 0.0, places=6, msg=name)

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
            target[:, 5] = 0.0
            target[:, 11] = 0.0
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
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(prediction.grad[:, 5].abs().sum().item(), 0.0)
        self.assertEqual(prediction.grad[:, 11].abs().sum().item(), 0.0)
        self.assertEqual(prediction.grad[:, :6, 4:].abs().sum().item(), 0.0)
        self.assertEqual(prediction.grad[:, 6:, 10:].abs().sum().item(), 0.0)

    def test_zero_and_near_zero_yaw_have_finite_gradients(self) -> None:
        for initial in ([0.0, 0.0], [1e-8, -1e-8]):
            sincos = torch.tensor([initial], requires_grad=True)
            rotation = yaw_matrix(sincos)
            rotation[..., 0, 1].sum().backward()
            self.assertTrue(torch.isfinite(rotation).all())
            self.assertTrue(torch.isfinite(sincos.grad).all())

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
