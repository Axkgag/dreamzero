from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from groot.vla.model.dreamzero.modules.wan_video_dit_dual_plan_prior import (
    CleanPriorDirectedCausalWanSelfAttention,
    CleanPriorDualPlanActionDecoder,
    CleanPriorDualPlanActionEncoder,
    MobilePlanPriorConfig,
    WanVideoDiTDualPlanPrior,
    coerce_mobile_plan_prior_config,
    resolve_prior_flow_indices,
)
from groot.vla.model.dreamzero.modules.wan_video_dit_dual_plan import (
    DualPlanActionDecoder,
    DualPlanActionEncoder,
)


class _MeanValueAttention(nn.Module):
    def forward(self, query, key, value):
        del key
        return query + value.mean(dim=1, keepdim=True)


def _attention() -> CleanPriorDirectedCausalWanSelfAttention:
    module = CleanPriorDirectedCausalWanSelfAttention(
        dim=8,
        num_heads=2,
        frame_seqlen=2,
        num_frame_per_block=2,
        num_action_per_block=15,
        num_state_per_block=1,
        num_base_prior_tokens=3,
    )
    module.attn = _MeanValueAttention()
    return module


def _streams(batch=2):
    shape = (batch, 2, 2, 4)
    clean_k = torch.randn(batch, 6, 2, 4)
    clean_v = torch.randn(batch, 6, 2, 4)
    noisy_k = torch.randn(batch, 6, 2, 4)
    noisy_v = torch.randn(batch, 6, 2, 4)
    action_q = torch.randn(batch, 15, 2, 4)
    action_k = torch.randn(batch, 15, 2, 4)
    action_v = torch.randn(batch, 15, 2, 4)
    state_k = torch.randn(*shape[:1], 1, *shape[2:])
    state_v = torch.randn(*shape[:1], 1, *shape[2:])
    return (
        action_q,
        action_k,
        action_v,
        clean_k,
        clean_v,
        noisy_k,
        noisy_v,
        state_k,
        state_v,
    )


class MobilePlanCleanPriorTest(unittest.TestCase):
    def test_structured_prior_config_defaults_to_three_base_offsets(self):
        config = coerce_mobile_plan_prior_config(
            {
                "time_offsets": [8, 16, 24],
                "predict_base": True,
                "predict_eef": False,
                "eef_frame": "future_base",
            }
        )
        self.assertEqual(config.time_offsets, (8, 16, 24))
        self.assertTrue(config.predict_base)
        self.assertFalse(config.predict_eef)
        self.assertEqual(config.eef_frame, "future_base")
        with self.assertRaisesRegex(ValueError, "At least one"):
            MobilePlanPriorConfig(
                predict_base=False,
                predict_eef=False,
            )
        with self.assertRaisesRegex(ValueError, "eef_frame"):
            MobilePlanPriorConfig(eef_frame="world")

    def test_prior_offsets_are_configurable_flow_subset(self):
        self.assertEqual(
            resolve_prior_flow_indices(
                [1, 4, 8, 12, 16, 24],
                [8, 16, 24],
            ),
            (2, 4, 5),
        )
        self.assertEqual(
            resolve_prior_flow_indices([1, 4, 8, 12, 16, 24], [24]),
            (5,),
        )
        with self.assertRaisesRegex(ValueError, "subset"):
            resolve_prior_flow_indices(
                [1, 4, 8, 12, 16, 24],
                [6, 24],
            )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            resolve_prior_flow_indices(
                [1, 4, 8, 12, 16, 24],
                [16, 8],
            )

    def test_dual_plan_projection_keys_are_clean_prior_compatible(self):
        common = dict(
            base_action_dim=4,
            manipulator_action_dim=21,
            hidden_size=32,
            num_embodiments=1,
            plan_time_offsets=[1, 4, 8, 12, 16, 24],
            control_fps=30.0,
        )
        dual_plan_encoder = DualPlanActionEncoder(**common)
        clean_prior_encoder = CleanPriorDualPlanActionEncoder(
            **common, prior_flow_indices=[2, 4, 5]
        )
        missing, unexpected = clean_prior_encoder.load_state_dict(
            dual_plan_encoder.state_dict(), strict=False
        )
        self.assertFalse(unexpected)
        self.assertEqual(
            set(missing), {"prior_query", "prior_type_embedding"}
        )

        decoder_common = dict(
            base_action_dim=4,
            manipulator_action_dim=21,
            hidden_size=16,
            model_dim=32,
            num_embodiments=1,
            plan_horizon=6,
        )
        dual_plan_decoder = DualPlanActionDecoder(**decoder_common)
        clean_prior_decoder = CleanPriorDualPlanActionDecoder(
            **decoder_common, prior_flow_indices=[2, 4, 5]
        )
        missing, unexpected = clean_prior_decoder.load_state_dict(
            dual_plan_decoder.state_dict(), strict=False
        )
        self.assertFalse(unexpected)
        self.assertTrue(missing)
        self.assertTrue(
            all(
                key.startswith(("base_prior_head.", "eef_prior_head."))
                for key in missing
            )
        )

    def test_prior_encoder_has_three_clean_queries_and_twelve_flow_tokens(self):
        encoder = CleanPriorDualPlanActionEncoder(
            base_action_dim=4,
            manipulator_action_dim=21,
            hidden_size=32,
            num_embodiments=1,
            plan_time_offsets=[1, 4, 8, 12, 16, 24],
            prior_flow_indices=[2, 4, 5],
            control_fps=30.0,
        )
        packed_a = torch.randn(2, 12, 21)
        packed_b = torch.randn(2, 12, 21)
        timestep_a = torch.randint(0, 1000, (2, 12))
        timestep_b = torch.randint(0, 1000, (2, 12))
        category = torch.zeros(2, dtype=torch.long)
        output_a = encoder(packed_a, timestep_a, category)
        output_b = encoder(packed_b, timestep_b, category)
        self.assertEqual(tuple(output_a.shape), (2, 15, 32))
        torch.testing.assert_close(output_a[:, :3], output_b[:, :3])
        self.assertFalse(torch.allclose(output_a[:, 3:], output_b[:, 3:]))

    def test_single_prior_offset_uses_one_clean_query(self):
        encoder = CleanPriorDualPlanActionEncoder(
            base_action_dim=4,
            manipulator_action_dim=21,
            hidden_size=32,
            num_embodiments=1,
            plan_time_offsets=[1, 4, 8, 12, 16, 24],
            prior_flow_indices=[5],
            control_fps=30.0,
        )
        packed = torch.randn(2, 12, 21)
        timestep = torch.randint(0, 1000, (2, 12))
        category = torch.zeros(2, dtype=torch.long)
        output = encoder(packed, timestep, category)
        self.assertEqual(tuple(output.shape), (2, 13, 32))

    def test_prior_decoder_keeps_flow_shape_and_packs_direct_coarse_output(self):
        decoder = CleanPriorDualPlanActionDecoder(
            base_action_dim=4,
            manipulator_action_dim=21,
            hidden_size=16,
            model_dim=32,
            num_embodiments=1,
            plan_horizon=6,
            prior_flow_indices=[2, 4, 5],
        )
        hidden = torch.randn(2, 15, 32, requires_grad=True)
        category = torch.zeros(2, dtype=torch.long)
        output = decoder(hidden, category)
        self.assertEqual(tuple(output.shape), (2, 12, 21))
        self.assertGreater(output[:, [2, 4, 5], 4:8].abs().sum().item(), 0)
        self.assertGreater(output[:, [2, 4, 5], 8:17].abs().sum().item(), 0)
        torch.testing.assert_close(
            output[:, [0, 1, 3], 4:8], torch.zeros(2, 3, 4)
        )
        torch.testing.assert_close(
            output[:, [0, 1, 3], 8:17], torch.zeros(2, 3, 9)
        )
        torch.testing.assert_close(output[:, :6, 17:], torch.zeros(2, 6, 4))
        output[:, :6, :17].square().mean().backward()
        self.assertGreater(hidden.grad[:, :3].abs().sum().item(), 0)
        self.assertGreater(hidden.grad[:, 3:9].abs().sum().item(), 0)
        self.assertGreater(
            decoder.base_prior_head.layer2.W.grad.abs().sum().item(), 0
        )
        self.assertGreater(
            decoder.eef_prior_head.layer2.W.grad.abs().sum().item(), 0
        )

    def test_prior_register_uses_clean_zero_timestep(self):
        model = WanVideoDiTDualPlanPrior.__new__(WanVideoDiTDualPlanPrior)
        nn.Module.__init__(model)
        model.plan_horizon = 6
        model.prior_horizon = 3
        timestep = torch.arange(12).reshape(1, 12).float() + 10
        action_features = torch.zeros(1, 15, 8)
        state_features = torch.zeros(1, 1, 8)
        register, state = model._action_register_timesteps(
            timestep, action_features, state_features
        )
        self.assertEqual(tuple(register.shape), (1, 15))
        torch.testing.assert_close(register[:, :3], torch.zeros(1, 3))
        torch.testing.assert_close(register[:, 3:], timestep)
        torch.testing.assert_close(state, timestep[:, :1])

    def test_prior_cannot_read_noisy_video_or_flow_values(self):
        module = _attention()
        streams = list(_streams())
        output = module._process_noisy_action_blocks(
            *streams, half_frames=3, action_horizon=15, state_horizon=1
        )
        changed = [value.clone() for value in streams]
        changed[2][:, 3:] += 1000
        changed[6] += 1000
        changed_output = module._process_noisy_action_blocks(
            *changed, half_frames=3, action_horizon=15, state_horizon=1
        )
        torch.testing.assert_close(output[:, :3], changed_output[:, :3])
        self.assertFalse(torch.allclose(output[:, 3:], changed_output[:, 3:]))

    def test_masked_prior_has_no_direct_or_video_mediated_path(self):
        module = _attention()
        module.prior_condition_mode = "masked"
        streams = list(_streams())
        action_output = module._process_noisy_action_blocks(
            *streams, half_frames=3, action_horizon=15, state_horizon=1
        )
        image_output = module._process_noisy_image_blocks(
            streams[5],
            streams[6],
            streams[5],
            streams[3],
            streams[4],
            streams[1],
            streams[2],
            streams[7],
            streams[8],
            half_frames=3,
            action_horizon=15,
            state_horizon=1,
        )
        changed = [value.clone() for value in streams]
        changed[1][:, :3] += 1000
        changed[2][:, :3] += 1000
        changed_action = module._process_noisy_action_blocks(
            *changed, half_frames=3, action_horizon=15, state_horizon=1
        )
        changed_image = module._process_noisy_image_blocks(
            changed[5],
            changed[6],
            changed[5],
            changed[3],
            changed[4],
            changed[1],
            changed[2],
            changed[7],
            changed[8],
            half_frames=3,
            action_horizon=15,
            state_horizon=1,
        )
        torch.testing.assert_close(action_output[:, 3:], changed_action[:, 3:])
        torch.testing.assert_close(image_output, changed_image)

    def test_normal_prior_changes_refinement_and_shuffled_requires_batch(self):
        module = _attention()
        streams = list(_streams())
        output = module._process_noisy_action_blocks(
            *streams, half_frames=3, action_horizon=15, state_horizon=1
        )
        streams[2][:, :3] += 10
        changed = module._process_noisy_action_blocks(
            *streams, half_frames=3, action_horizon=15, state_horizon=1
        )
        self.assertFalse(torch.allclose(output[:, 3:], changed[:, 3:]))

        module.prior_condition_mode = "shuffled"
        single = list(_streams(batch=1))
        with self.assertRaisesRegex(ValueError, "batch_size >= 2"):
            module._process_noisy_action_blocks(
                *single, half_frames=3, action_horizon=15, state_horizon=1
            )


if __name__ == "__main__":
    unittest.main()
