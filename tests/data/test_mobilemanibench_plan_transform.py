from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
import torch

from groot.vla.data.dataset.mobilemanibench_plan import MobileManiBenchPlanDataset
from groot.vla.data.transform.mobile_plan import MobilePlanTransform


SMOKE_ROOT = Path(
    os.environ.get(
        "MOBILEMANIBENCH_SMOKE_ROOT",
        "/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_smoke_v2",
    )
)


@unittest.skipUnless(SMOKE_ROOT.exists(), f"smoke dataset not found: {SMOKE_ROOT}")
class MobilePlanTransformTest(unittest.TestCase):
    def test_normalization_masks_and_inverse_for_both_robots(self) -> None:
        for robot in ("g1", "xhand"):
            with self.subTest(robot=robot):
                root = SMOKE_ROOT / robot
                dataset = MobileManiBenchPlanDataset(root, load_videos=False)
                raw = dataset[0]
                transform = MobilePlanTransform(
                    stats_path=root / "meta/plan_stats.json"
                )
                transformed = transform(dict(raw))

                self.assertEqual(tuple(transformed["base_action"].shape), (6, 4))
                self.assertEqual(
                    tuple(transformed["manipulator_action"].shape), (6, 21)
                )
                self.assertTrue(
                    torch.all(torch.abs(transformed["base_action"][:, :2]) <= 1)
                )
                self.assertTrue(
                    torch.all(
                        torch.abs(transformed["manipulator_action"][:, :3]) <= 1
                    )
                )
                np.testing.assert_allclose(
                    transformed["base_action"][:, 2:4].numpy(),
                    raw["base_plan"][:, 2:4],
                )
                np.testing.assert_allclose(
                    transformed["manipulator_action"][:, 3:9].numpy(),
                    raw["manipulator_plan"][:, 3:9],
                )
                expected_mask = (
                    torch.as_tensor(raw["manipulator_dim_mask"])
                    & torch.as_tensor(raw["plan_valid"]).unsqueeze(-1)
                )
                torch.testing.assert_close(
                    transformed["manipulator_action_mask"], expected_mask
                )

                restored = transform.unapply(dict(transformed))
                stats = dataset.plan_stats["statistics"]
                expected_base = torch.as_tensor(raw["base_plan"]).clone()
                expected_manipulator = torch.as_tensor(
                    raw["manipulator_plan"]
                ).clone()
                expected_base[:, :2] = torch.maximum(
                    torch.minimum(
                        expected_base[:, :2], torch.tensor(stats["base_xy"]["q99"])
                    ),
                    torch.tensor(stats["base_xy"]["q01"]),
                )
                expected_manipulator[:, :3] = torch.maximum(
                    torch.minimum(
                        expected_manipulator[:, :3],
                        torch.tensor(stats["eef_xyz"]["q99"]),
                    ),
                    torch.tensor(stats["eef_xyz"]["q01"]),
                )
                hand_dim = int(raw["hand_dim"])
                expected_manipulator[:, 9 : 9 + hand_dim] = torch.maximum(
                    torch.minimum(
                        expected_manipulator[:, 9 : 9 + hand_dim],
                        torch.tensor(stats["hand"]["q99"]),
                    ),
                    torch.tensor(stats["hand"]["q01"]),
                )
                torch.testing.assert_close(
                    restored["base_plan"], expected_base, atol=1e-6, rtol=0
                )
                torch.testing.assert_close(
                    restored["manipulator_plan"],
                    expected_manipulator,
                    atol=1e-6,
                    rtol=0,
                )


if __name__ == "__main__":
    unittest.main()
