from __future__ import annotations

import unittest

import numpy as np

from groot.vla.data.plan_geometry import build_dynamic_block_plan_labels
from groot.vla.data.dataset.mobilemanibench_block_plan import (
    coherent_block_masks,
    variable_block_layout,
)
from groot.vla.utils.mobile_plan_spec import (
    EEF_ROTATION_CURRENT_EEF_DELTA_ROTVEC,
    block_plan_spec_hash,
    canonical_block_plan_spec,
    dynamic_block_plan_stats_path,
)
from scripts.data.convert_mobilemanibench_to_gear import build_block_plan_labels


class MobileManiBenchBlockPlanLabelsTest(unittest.TestCase):
    @staticmethod
    def _trajectory(length: int = 12):
        rng = np.random.default_rng(7)
        base = rng.normal(size=(length, 6)).cumsum(axis=0) * 0.05
        eef = rng.normal(size=(length, 6)).cumsum(axis=0) * 0.03
        joints = rng.normal(size=(length, 5, 3))
        return base, eef, joints

    def test_dynamic_geometry_matches_materialized_converter(self) -> None:
        base, eef, joints = self._trajectory()
        expected = build_block_plan_labels(
            base,
            eef,
            joints,
            hand_joint_indices=[1, 4],
            block_anchor_offsets=(0, 4),
            local_waypoint_offsets=(2, 4),
        )
        actual = build_dynamic_block_plan_labels(
            base,
            eef,
            joints,
            hand_joint_indices=[1, 4],
            block_anchor_offsets=(0, 4),
            local_waypoint_offsets=(2, 4),
        )
        for expected_value, actual_value in zip(expected, actual):
            np.testing.assert_allclose(actual_value, expected_value, atol=1e-6)

    def test_statistics_cache_is_keyed_by_label_spec(self) -> None:
        first = dynamic_block_plan_stats_path("/data/g1", [0, 8], [4, 8])
        second = dynamic_block_plan_stats_path("/data/g1", [0, 8], [2, 8])
        self.assertNotEqual(first, second)
        self.assertEqual(first.parent.as_posix(), "/data/g1/meta/dynamic_plan_stats")
        spec = canonical_block_plan_spec([0, 8], [4, 8])
        self.assertEqual(spec["global_waypoint_offsets"], [4, 8, 12, 16])
        self.assertIn(block_plan_spec_hash([0, 8], [4, 8]), first.name)
        delta = dynamic_block_plan_stats_path(
            "/data/g1",
            [0, 8],
            [4, 8],
            eef_rotation_representation=EEF_ROTATION_CURRENT_EEF_DELTA_ROTVEC,
        )
        self.assertNotEqual(first, delta)

    def test_current_eef_delta_rotvec_is_small_relative_motion(self) -> None:
        length = 12
        base = np.zeros((length, 6), dtype=np.float64)
        eef = np.zeros((length, 6), dtype=np.float64)
        eef[:, 2] = 0.5
        eef[:, 5] = np.deg2rad(90.0) + np.arange(length) * np.deg2rad(1.0)
        joints = np.zeros((length, 2, 3), dtype=np.float64)
        _, manipulator, valid, _, _ = build_dynamic_block_plan_labels(
            base,
            eef,
            joints,
            hand_joint_indices=[1],
            block_anchor_offsets=(0, 4),
            local_waypoint_offsets=(2, 4),
            eef_rotation_representation=EEF_ROTATION_CURRENT_EEF_DELTA_ROTVEC,
        )
        self.assertEqual(manipulator.shape, (length, 2, 2, 7))
        self.assertTrue(valid[0].all())
        np.testing.assert_allclose(manipulator[0, 0, :, 3:5], 0.0, atol=1e-8)
        np.testing.assert_allclose(
            manipulator[0, 0, :, 5],
            np.deg2rad([2.0, 4.0]),
            atol=1e-7,
        )

    def test_each_block_uses_its_own_base_anchor(self) -> None:
        length = 12
        base = np.zeros((length, 6), dtype=np.float64)
        base[:, 0] = np.arange(length)
        eef = np.zeros((length, 6), dtype=np.float64)
        eef[:, 0] = np.arange(length) + 0.5
        joints = np.zeros((length, 3, 3), dtype=np.float64)
        joints[:, 2, 0] = np.arange(length) * 0.1
        base_plan, manipulator, valid, state, state_valid = build_block_plan_labels(
            base,
            eef,
            joints,
            hand_joint_indices=[2],
            block_anchor_offsets=(0, 4),
            local_waypoint_offsets=(2, 4),
        )
        self.assertEqual(base_plan.shape, (length, 2, 2, 4))
        self.assertEqual(manipulator.shape, (length, 2, 2, 10))
        np.testing.assert_allclose(base_plan[0, :, :, 0], [[2, 4], [2, 4]])
        np.testing.assert_allclose(manipulator[0, :, :, 0], [[2.5, 4.5], [2.5, 4.5]])
        np.testing.assert_allclose(state[0, :, 0], [0.5, 0.5])
        self.assertTrue(valid[0].all())
        self.assertTrue(state_valid[0].all())

    def test_tail_slots_are_zero_and_masked(self) -> None:
        length = 6
        base = np.zeros((length, 6), dtype=np.float64)
        eef = np.zeros((length, 6), dtype=np.float64)
        joints = np.zeros((length, 2, 3), dtype=np.float64)
        base_plan, manipulator, valid, state, state_valid = build_block_plan_labels(
            base,
            eef,
            joints,
            hand_joint_indices=[1],
            block_anchor_offsets=(0, 4),
            local_waypoint_offsets=(2, 4),
        )
        self.assertFalse(valid[-1].any())
        self.assertFalse(state_valid[-1, 1])
        np.testing.assert_array_equal(base_plan[-1], 0)
        np.testing.assert_array_equal(manipulator[-1], 0)
        np.testing.assert_array_equal(state[-1, 1], 0)

    def test_terminal_may_end_inside_any_later_block(self) -> None:
        self.assertEqual(
            variable_block_layout(171, 0, 48, 4, True),
            (3, 3, 26, 22),
        )
        self.assertEqual(
            variable_block_layout(171, 74, 48, 4, True),
            (2, -1, 0, 0),
        )
        self.assertEqual(
            variable_block_layout(171, 100, 48, 4, True),
            (1, 1, 22, 26),
        )
        self.assertEqual(
            variable_block_layout(171, 100, 48, 4, False),
            (1, -1, 22, 0),
        )

    def test_root_must_have_one_complete_block(self) -> None:
        with self.assertRaisesRegex(ValueError, "no complete"):
            variable_block_layout(171, 123, 48, 4, True)

    def test_partial_supervision_is_reduced_to_complete_block_prefix(self) -> None:
        plan_valid = np.asarray(
            [
                [True, True],
                [True, False],
                [True, True],
                [True, True],
            ]
        )
        state_valid = np.ones(4, dtype=bool)
        latent_valid = np.ones(9, dtype=bool)
        plan, state, latent, blocks = coherent_block_masks(
            plan_valid, state_valid, latent_valid
        )
        np.testing.assert_array_equal(blocks, [True, False, False, False])
        self.assertTrue(plan[0].all())
        self.assertFalse(plan[1:].any())
        np.testing.assert_array_equal(state, [True, False, False, False])
        np.testing.assert_array_equal(
            latent, [True, True, True, False, False, False, False, False, False]
        )


if __name__ == "__main__":
    unittest.main()
