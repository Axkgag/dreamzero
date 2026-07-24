from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from groot.vla.data.dataset.mobilemanibench_plan import MobileManiBenchPlanDataset


SMOKE_ROOT = Path(
    os.environ.get(
        "MOBILEMANIBENCH_SMOKE_ROOT",
        "/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_smoke_v2",
    )
)
REPO_ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(SMOKE_ROOT.exists(), f"smoke dataset not found: {SMOKE_ROOT}")
class MobileManiBenchPlanDatasetTest(unittest.TestCase):
    def test_g1_shapes_padding_and_terminal_mask(self) -> None:
        dataset = MobileManiBenchPlanDataset(SMOKE_ROOT / "g1", load_videos=False)
        sample = dataset[0]
        self.assertEqual(sample["base_plan"].shape, (6, 4))
        self.assertEqual(sample["manipulator_plan"].shape, (6, 21))
        self.assertEqual(sample["plan_valid"].shape, (6,))
        self.assertEqual(sample["plan_time_offsets"].tolist(), [1, 4, 8, 12, 16, 24])
        self.assertTrue(sample["manipulator_dim_mask"][:, :10].all())
        self.assertFalse(sample["manipulator_dim_mask"][:, 10:].any())
        self.assertTrue(np.allclose(sample["manipulator_plan"][:, 10:], 0))
        self.assertEqual(sample["base_plan"].ndim, 2, "plan horizon was expanded twice")

        terminal = dataset[len(dataset) - 1]
        self.assertFalse(terminal["plan_valid"].any())

    def test_xhand_shapes_and_mask(self) -> None:
        dataset = MobileManiBenchPlanDataset(SMOKE_ROOT / "xhand", load_videos=False)
        sample = dataset[0]
        self.assertEqual(sample["base_plan"].shape, (6, 4))
        self.assertEqual(sample["manipulator_plan"].shape, (6, 21))
        self.assertTrue(sample["manipulator_dim_mask"].all())
        self.assertEqual(int(sample["hand_dim"]), 12)

    def test_stored_labels_reconstruct_from_preserved_realized_state(self) -> None:
        converter_path = REPO_ROOT / "scripts/data/convert_mobilemanibench_to_gear.py"
        spec = importlib.util.spec_from_file_location("mobile_converter", converter_path)
        converter = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = converter
        spec.loader.exec_module(converter)

        for robot in ("g1", "xhand"):
            root = SMOKE_ROOT / robot
            extensions = json.loads((root / "meta/extensions.json").read_text())
            schema = json.loads((root / "meta/robot_schema.json").read_text())
            parquet = sorted((root / "data").glob("*/*.parquet"))[0]
            frame = pd.read_parquet(parquet)
            base_world = np.stack(frame["observation.base.world"])
            eef_world = np.stack(frame["observation.eef.world"])
            robot_joint = np.stack(frame["observation.robot_joint"]).reshape(
                len(frame), -1, 3
            )
            expected = converter.build_plan_labels(
                base_world,
                eef_world,
                robot_joint,
                schema["hand_joint_indices"],
                np.asarray(
                    extensions["action_plan"]["waypoint_offsets"], dtype=np.int64
                ),
            )
            actual_base = np.stack(frame["action.plan.base_waypoints"]).reshape(
                expected[0].shape
            )
            actual_manipulator = np.stack(frame["action.plan.manipulator"]).reshape(
                expected[1].shape
            )
            actual_valid = np.stack(frame["action.plan.valid"])
            np.testing.assert_allclose(actual_base, expected[0], atol=2e-5)
            np.testing.assert_allclose(actual_manipulator, expected[1], atol=2e-5)
            np.testing.assert_array_equal(actual_valid, expected[2])


if __name__ == "__main__":
    unittest.main()
