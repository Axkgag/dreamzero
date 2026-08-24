from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

EVAL_DIR = Path(__file__).resolve().parents[2] / "scripts" / "eval"
sys.path.insert(0, str(EVAL_DIR))

from evaluate_mobilemanibench_multiblock_plan import (  # noqa: E402
    MetricAccumulator,
    build_transform_and_collator,
    canonicalize_manipulator_rotation,
    compose_block_plans,
    parse_args,
    prepare_inference_batch,
    teacher_forced_block_observation,
)


def _identity_manipulator(blocks: int, waypoints: int) -> np.ndarray:
    value = np.zeros((blocks, waypoints, 9), dtype=np.float32)
    value[..., 3] = 1.0
    value[..., 7] = 1.0
    return value


class MobileManiBenchMultiBlockEvalTest(unittest.TestCase):
    def test_public_cli_uses_one_fixed_complete_rollout_protocol(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["evaluate", "--dataset-root", "/tmp/dataset"],
        ):
            args = parse_args()
        self.assertEqual(args.mode, "teacher_forced_open_loop")
        self.assertFalse(hasattr(args, "num_rollout_blocks"))

        for removed_argument in ("--mode", "--num-rollout-blocks"):
            with self.subTest(argument=removed_argument):
                with patch.object(
                    sys,
                    "argv",
                    [
                        "evaluate",
                        "--dataset-root",
                        "/tmp/dataset",
                        removed_argument,
                        "1",
                    ],
                ), self.assertRaises(SystemExit):
                    parse_args()

    def test_teacher_forced_block_observation_slices_arrived_history(self) -> None:
        root = {
            "video.head": np.arange(33, dtype=np.int64)[:, None],
            "video.wrist": np.arange(100, 133, dtype=np.int64)[:, None],
            "state.eef_position": np.arange(12, dtype=np.float32).reshape(4, 3),
            "state.eef_rotation_rpy": np.arange(12, 24, dtype=np.float32).reshape(4, 3),
            "physical_block_state": np.arange(24, dtype=np.float32).reshape(4, 6),
            "annotation.task": np.asarray(["task"]),
        }
        anchors = [0, 8, 16, 24]

        block0 = teacher_forced_block_observation(root, 0, anchors, 8)
        block1 = teacher_forced_block_observation(root, 1, anchors, 8)
        block2 = teacher_forced_block_observation(root, 2, anchors, 8)
        block3 = teacher_forced_block_observation(root, 3, anchors, 8)

        np.testing.assert_array_equal(block0["video.head"][:, 0], [0])
        np.testing.assert_array_equal(block1["video.head"][:, 0], np.arange(9))
        np.testing.assert_array_equal(block2["video.head"][:, 0], np.arange(8, 17))
        np.testing.assert_array_equal(block3["video.head"][:, 0], np.arange(16, 25))
        np.testing.assert_array_equal(
            block3["video.wrist"][:, 0], np.arange(116, 125)
        )
        for block_index, observation in enumerate(
            (block0, block1, block2, block3)
        ):
            self.assertEqual(observation["state.eef_position"].shape, (4, 3))
            np.testing.assert_array_equal(
                observation["state.eef_position"],
                np.repeat(
                    root["state.eef_position"][block_index : block_index + 1],
                    4,
                    axis=0,
                ),
            )
            np.testing.assert_array_equal(
                observation["physical_block_state"],
                np.repeat(
                    root["physical_block_state"][block_index : block_index + 1],
                    4,
                    axis=0,
                ),
            )
        self.assertEqual(root["video.head"].shape[0], 33)

    def test_transform_keeps_supervised_per_sample_schema(self) -> None:
        class FakeTransform:
            def __init__(self):
                self.training = False

            def train(self):
                self.training = True

        transform = FakeTransform()
        collator = object()
        cfg = OmegaConf.create(
            {
                "train_dataset": {
                    "plan_transform": {
                        "transforms": [{"stats_path": "old-stats.json"}]
                    }
                },
                "data_collator": {"_target_": "unused.FakeCollator"},
            }
        )
        stats_path = Path("new-stats.json")

        with patch(
            "evaluate_mobilemanibench_multiblock_plan.instantiate",
            side_effect=[transform, collator],
        ) as instantiate_mock:
            actual_transform, actual_collator = build_transform_and_collator(
                cfg, stats_path
            )

        self.assertIs(actual_transform, transform)
        self.assertIs(actual_collator, collator)
        self.assertTrue(transform.training)
        transform_config = instantiate_mock.call_args_list[0].args[0]
        self.assertEqual(
            transform_config.transforms[0].stats_path, str(stats_path)
        )

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

    def test_delta_rotvec_is_decoded_with_anchor_eef_orientation(self) -> None:
        manipulator = np.zeros((2, 21), dtype=np.float32)
        manipulator[:, 5] = np.deg2rad([10.0, 20.0])
        manipulator[:, 6] = [0.25, 0.5]
        anchor = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(90.0)])
        decoded = canonicalize_manipulator_rotation(
            manipulator,
            anchor,
            "current_eef_delta_rotvec",
            hand_dim=1,
        )
        yaw = np.arctan2(decoded[:, 6], decoded[:, 3])
        np.testing.assert_allclose(
            yaw,
            np.deg2rad([100.0, 110.0]),
            atol=1e-6,
        )
        np.testing.assert_allclose(decoded[:, 9], [0.25, 0.5])

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
        self.assertEqual(summary["primary_metrics"]["composed_base_ade_m"], 0.0)
        self.assertEqual(summary["primary_metrics"]["composed_base_fde_m"], 0.0)
        self.assertEqual(summary["primary_metrics"]["composed_eef_ade_m"], 0.0)
        self.assertEqual(summary["primary_metrics"]["composed_eef_fde_m"], 0.0)

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
