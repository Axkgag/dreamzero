from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

EVAL_DIR = Path(__file__).resolve().parents[2] / "scripts" / "eval"
sys.path.insert(0, str(EVAL_DIR))

from evaluate_mobilemanibench_multiblock_plan import (  # noqa: E402
    MetricAccumulator,
    compose_block_plans,
    prepare_inference_batch,
)


def _identity_manipulator(blocks: int, waypoints: int) -> np.ndarray:
    value = np.zeros((blocks, waypoints, 9), dtype=np.float32)
    value[..., 3] = 1.0
    value[..., 7] = 1.0
    return value


class MobileManiBenchMultiBlockEvalTest(unittest.TestCase):
    def test_compose_block_plans_uses_predicted_endpoint_as_next_anchor(self) -> None:
        base = np.zeros((2, 2, 4), dtype=np.float32)
        base[..., 3] = 1.0
        base[0, :, 0] = [1.0, 2.0]
        base[1, :, 0] = [1.0, 3.0]
        manipulator = _identity_manipulator(2, 2)
        manipulator[..., 0] = base[..., 0] + 0.5
        valid = np.ones((2, 2), dtype=bool)

        global_base, global_manipulator = compose_block_plans(
            base, manipulator, valid
        )
        np.testing.assert_allclose(global_base[0, :, 0], [1.0, 2.0])
        np.testing.assert_allclose(global_base[1, :, 0], [3.0, 5.0])
        np.testing.assert_allclose(global_manipulator[1, :, 0], [3.5, 5.5])

    def test_perfect_prediction_has_zero_local_and_composed_metrics(self) -> None:
        base = np.zeros((2, 2, 4), dtype=np.float32)
        base[..., 3] = 1.0
        base[..., 0] = [[1.0, 2.0], [0.5, 1.0]]
        manipulator = _identity_manipulator(2, 2)
        valid = np.ones((2, 2), dtype=bool)
        accumulator = MetricAccumulator([0, 2], [1, 2])
        for block_index in range(2):
            accumulator.add_block(
                episode_index=1,
                root_frame_index=0,
                anchor_frame_index=2 * block_index,
                task="task",
                block_index=block_index,
                base_pred=base[block_index],
                base_gt=base[block_index],
                manip_pred=manipulator[block_index],
                manip_gt=manipulator[block_index],
                valid=valid[block_index],
                hand_dim=0,
            )
        accumulator.add_composed_window(
            task="task",
            base_pred=base,
            base_gt=base,
            manip_pred=manipulator,
            manip_gt=manipulator,
            valid=valid,
            hand_dim=0,
        )
        summary = accumulator.summary()
        self.assertEqual(summary["num_windows"], 1)
        self.assertEqual(summary["num_block_predictions"], 2)
        self.assertEqual(summary["primary_metrics"]["base_ade_m"], 0.0)
        self.assertEqual(
            summary["primary_metrics"]["composed_eef_position_l2_m"], 0.0
        )

    def test_prepare_inference_batch_removes_future_state_and_action_blocks(self) -> None:
        class IdentityTransform:
            def __call__(self, value):
                return value

        class FakeCollator:
            def __call__(self, values):
                self.values = values
                return {
                    "state": torch.zeros(1, 4, 64),
                    "action": torch.zeros(1, 24, 21),
                    "action_mask": torch.ones(1, 24, 21),
                }

        batch = prepare_inference_batch(
            {"sample": 1}, IdentityTransform(), FakeCollator(), 6
        )
        self.assertEqual(tuple(batch["state"].shape), (1, 1, 64))
        self.assertEqual(tuple(batch["action"].shape), (1, 6, 21))
        self.assertEqual(tuple(batch["action_mask"].shape), (1, 6, 21))


if __name__ == "__main__":
    unittest.main()
