from __future__ import annotations

import os
import unittest
from pathlib import Path

import torch

from groot.vla.data.dataset.mobilemanibench_vggt import (
    MobileManiBenchVGGTDataCollator,
    MobileManiBenchVGGTDataset,
)


SMOKE_ROOT = Path(
    os.environ.get(
        "MOBILEMANIBENCH_SMOKE_ROOT",
        "/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_smoke_v2",
    )
)


@unittest.skipUnless(SMOKE_ROOT.exists(), f"smoke dataset not found: {SMOKE_ROOT}")
class MobileManiBenchVGGTDatasetTest(unittest.TestCase):
    def test_rgb_camera_and_pointmap_contract(self) -> None:
        dataset = MobileManiBenchVGGTDataset(
            SMOKE_ROOT / "g1",
            video_delta_indices=[0, 4],
            image_size=[32, 32],
            pointmap_size=[8, 8],
            split="all",
            sample_stride=100,
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["video"].shape), (2, 2, 3, 32, 32))
        self.assertEqual(sample["video"].dtype, torch.uint8)
        self.assertEqual(tuple(sample["camera_K"].shape), (2, 2, 3, 3))
        self.assertEqual(tuple(sample["T_b0_camera"].shape), (2, 2, 4, 4))
        self.assertEqual(
            tuple(sample["pseudo_pointmap_b0"].shape), (2, 2, 3, 8, 8)
        )
        self.assertEqual(tuple(sample["pointmap_valid"].shape), (2, 2, 8, 8))
        self.assertTrue(torch.isfinite(sample["pseudo_pointmap_b0"]).all())
        self.assertGreater(float(sample["pointmap_valid"].sum()), 0)
        self.assertLessEqual(float(sample["pointmap_valid"].max()), 0.25)

        batch = MobileManiBenchVGGTDataCollator()([sample, sample])
        self.assertEqual(tuple(batch["video"].shape), (2, 2, 2, 3, 32, 32))


if __name__ == "__main__":
    unittest.main()
