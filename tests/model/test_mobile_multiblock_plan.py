from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from groot.vla.model.dreamzero.action_head.mobile_plan_multiblock_flow_matching import (
    MobilePlanMultiBlockFlowMatchingActionHead,
)
from groot.vla.model.dreamzero.modules.wan_video_dit_dual_plan_multiblock import (
    MultiBlockCleanPriorActionDecoder,
    MultiBlockCleanPriorActionEncoder,
    MultiBlockDualPlanActionDecoder,
    MultiBlockDualPlanActionEncoder,
    WanVideoDiTMultiBlockDualPlanPrior,
)
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
    CausalWanSelfAttention,
)
from groot.vla.model.dreamzero.transform.mobile_plan_cotrain import (
    MobileBlockPlanDataCollator,
)


class _MeanValueAttention(nn.Module):
    def forward(self, query, key, value):
        del key
        return query + value.mean(dim=1, keepdim=True)


class MobileMultiBlockPlanTest(unittest.TestCase):
    def test_collator_shape_contract_supports_three_waypoints(self) -> None:
        collator = MobileBlockPlanDataCollator.__new__(MobileBlockPlanDataCollator)
        collator.num_plan_blocks = 4
        collator.plan_waypoints_per_block = 3
        collator.base_action_dim = 4
        collator.manipulator_action_dim = 21
        collator.max_state_dim = 64
        batch_size = 2
        batch = {
            "base_action": torch.zeros(batch_size, 4, 3, 4),
            "manipulator_action": torch.zeros(batch_size, 4, 3, 21),
            "base_action_mask": torch.zeros(batch_size, 4, 3, 4),
            "manipulator_action_mask": torch.zeros(batch_size, 4, 3, 21),
            "plan_local_offsets": torch.tensor([[2, 4, 8], [2, 4, 8]]),
            "block_anchor_offsets": torch.tensor(
                [[0, 8, 16, 24], [0, 8, 16, 24]]
            ),
            "global_plan_offsets": torch.zeros(batch_size, 12),
            "action": torch.zeros(batch_size, 24, 21),
            "state": torch.zeros(batch_size, 4, 64),
        }
        collator._validate_batch_shapes(batch, batch_size)
        with self.assertRaisesRegex(ValueError, "packed action"):
            invalid = dict(batch)
            invalid["action"] = torch.zeros(batch_size, 16, 21)
            collator._validate_batch_shapes(invalid, batch_size)

    def test_three_waypoint_encoder_decoder_and_prior_shapes(self) -> None:
        flow_encoder = MultiBlockDualPlanActionEncoder(
            base_action_dim=4,
            manipulator_action_dim=21,
            hidden_size=32,
            num_embodiments=1,
            plan_local_offsets=[2, 4, 8],
            control_fps=30.0,
        )
        flow_decoder = MultiBlockDualPlanActionDecoder(
            base_action_dim=4,
            manipulator_action_dim=21,
            hidden_size=16,
            model_dim=32,
            num_embodiments=1,
            waypoints_per_block=3,
        )
        action = torch.randn(2, 24, 21)
        timestep = torch.randint(0, 1000, (2, 24))
        category = torch.zeros(2, dtype=torch.long)
        prediction = flow_decoder(flow_encoder(action, timestep, category), category)
        self.assertEqual(tuple(prediction.shape), (2, 24, 21))

        prior_encoder = MultiBlockCleanPriorActionEncoder(
            base_action_dim=4,
            manipulator_action_dim=21,
            hidden_size=32,
            num_embodiments=1,
            plan_local_offsets=[2, 4, 8],
            prior_flow_indices=[2],
            control_fps=30.0,
        )
        prior_decoder = MultiBlockCleanPriorActionDecoder(
            base_action_dim=4,
            manipulator_action_dim=21,
            hidden_size=16,
            model_dim=32,
            num_embodiments=1,
            waypoints_per_block=3,
            prior_flow_index=2,
        )
        hidden = prior_encoder(action, timestep, category)
        self.assertEqual(tuple(hidden.shape), (2, 28, 32))
        self.assertEqual(tuple(prior_decoder(hidden, category).shape), (2, 24, 21))

    def test_teacher_forcing_blocks_only_read_earlier_clean_video(self) -> None:
        attention = CausalWanSelfAttention(
            dim=8,
            num_heads=2,
            frame_seqlen=2,
            num_frame_per_block=2,
            num_action_per_block=4,
            num_state_per_block=1,
        )
        attention.attn = _MeanValueAttention()
        batch, heads, head_dim = 1, 2, 4
        action_q = torch.randn(batch, 8, heads, head_dim)
        action_k = torch.randn_like(action_q)
        action_v = torch.randn_like(action_q)
        clean_k = torch.randn(batch, 10, heads, head_dim)
        clean_v = torch.randn_like(clean_k)
        noisy_k = torch.randn_like(clean_k)
        noisy_v = torch.randn_like(clean_k)
        state_k = torch.randn(batch, 2, heads, head_dim)
        state_v = torch.randn_like(state_k)

        def run(current_clean_v):
            return attention._process_noisy_action_blocks(
                action_q,
                action_k,
                action_v,
                clean_k,
                current_clean_v,
                noisy_k,
                noisy_v,
                state_k,
                state_v,
                half_frames=5,
                action_horizon=8,
                state_horizon=2,
            )

        original = run(clean_v)
        future_changed = clean_v.clone()
        future_changed[:, 6:10] += 1000
        torch.testing.assert_close(original, run(future_changed))
        earlier_changed = clean_v.clone()
        earlier_changed[:, 2:6] += 1000
        changed = run(earlier_changed)
        torch.testing.assert_close(original[:, :4], changed[:, :4])
        self.assertFalse(torch.allclose(original[:, 4:], changed[:, 4:]))

    def test_encoder_decoder_preserve_block_major_shape(self) -> None:
        encoder = MultiBlockDualPlanActionEncoder(
            base_action_dim=4,
            manipulator_action_dim=21,
            hidden_size=32,
            num_embodiments=1,
            plan_local_offsets=[4, 8],
            control_fps=30.0,
        )
        decoder = MultiBlockDualPlanActionDecoder(
            base_action_dim=4,
            manipulator_action_dim=21,
            hidden_size=16,
            model_dim=32,
            num_embodiments=1,
            waypoints_per_block=2,
        )
        action = torch.randn(2, 16, 21)
        timestep = torch.randint(0, 1000, (2, 16))
        category = torch.zeros(2, dtype=torch.long)
        prediction = decoder(encoder(action, timestep, category), category)
        self.assertEqual(tuple(prediction.shape), (2, 16, 21))
        block = prediction.reshape(2, 4, 4, 21)
        torch.testing.assert_close(block[:, :, :2, 4:], torch.zeros(2, 4, 2, 17))
        prediction.square().mean().backward()
        self.assertIsNotNone(encoder.base_encoder.W1.W.grad)
        self.assertIsNotNone(encoder.manipulator_encoder.W1.W.grad)

    def test_prior_is_inserted_at_start_of_every_block(self) -> None:
        encoder = MultiBlockCleanPriorActionEncoder(
            base_action_dim=4,
            manipulator_action_dim=21,
            hidden_size=32,
            num_embodiments=1,
            plan_local_offsets=[4, 8],
            prior_flow_indices=[1],
            control_fps=30.0,
        )
        decoder = MultiBlockCleanPriorActionDecoder(
            base_action_dim=4,
            manipulator_action_dim=21,
            hidden_size=16,
            model_dim=32,
            num_embodiments=1,
            waypoints_per_block=2,
            prior_flow_index=1,
        )
        action = torch.randn(2, 12, 21)
        timestep = torch.randint(0, 1000, (2, 12))
        category = torch.zeros(2, dtype=torch.long)
        hidden = encoder(action, timestep, category)
        self.assertEqual(tuple(hidden.shape), (2, 15, 32))
        prediction = decoder(hidden, category)
        self.assertEqual(tuple(prediction.shape), (2, 12, 21))
        block = prediction.reshape(2, 3, 4, 21)
        self.assertGreater(block[:, :, 1, 4:17].abs().sum().item(), 0)
        torch.testing.assert_close(block[:, :, 0, 4:17], torch.zeros(2, 3, 13))

    def test_prior_timestep_is_clean_per_block(self) -> None:
        model = WanVideoDiTMultiBlockDualPlanPrior.__new__(
            WanVideoDiTMultiBlockDualPlanPrior
        )
        nn.Module.__init__(model)
        model.flow_tokens_per_block = 4
        timestep = torch.tensor([[11, 11, 11, 11, 27, 27, 27, 27]])
        action_features = torch.zeros(1, 10, 8)
        state_features = torch.zeros(1, 2, 8)
        action_time, state_time = model._action_register_timesteps(
            timestep, action_features, state_features
        )
        torch.testing.assert_close(
            action_time,
            torch.tensor([[0, 11, 11, 11, 11, 0, 27, 27, 27, 27]]),
        )
        torch.testing.assert_close(state_time, torch.tensor([[11, 27]]))

    def test_coupled_timestep_repeats_inside_each_block(self) -> None:
        head = MobilePlanMultiBlockFlowMatchingActionHead.__new__(
            MobilePlanMultiBlockFlowMatchingActionHead
        )
        nn.Module.__init__(head)
        head.flow_tokens_per_block = 4
        actions = torch.zeros(1, 8, 21)
        blocks = torch.tensor([[[13, 13], [41, 41]]])
        result = head.build_coupled_action_timestep_ids(
            blocks, actions, torch.zeros(1, 5, 2, 2, 2)
        )
        torch.testing.assert_close(
            result, torch.tensor([[13, 13, 13, 13, 41, 41, 41, 41]])
        )


if __name__ == "__main__":
    unittest.main()
