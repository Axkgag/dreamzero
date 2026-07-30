from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from groot.vla.model.vggt_3d_wam.visualization import (
    CONTACT_SHEET_COLUMNS,
    save_vggt_visualization,
)


class VGGTVisualizationTest(unittest.TestCase):
    def test_contact_sheet_contains_only_core_prediction_columns(self) -> None:
        self.assertEqual(
            CONTACT_SHEET_COLUMNS,
            (
                "RGB GT",
                "RGB reconstruction",
                "PointMap GT range",
                "PointMap predicted range",
            ),
        )

    def test_reconstruction_geometry_and_metrics_are_saved_together(self) -> None:
        batch, time, views = 1, 2, 2
        video = torch.randint(
            0, 256, (batch, time, views, 3, 16, 16), dtype=torch.uint8
        )
        pointmap = torch.randn(batch, time, views, 3, 4, 4)
        transforms = torch.eye(4).expand(batch, time, views, 4, 4).clone()
        inputs = {
            "video": video,
            "pseudo_pointmap_b0": pointmap,
            "pointmap_valid": torch.full((batch, time, views, 4, 4), 0.25),
            "T_b0_camera": transforms,
        }
        outputs = {
            "reconstructed_video": video.float() / 255 * 0.9,
            "predicted_pointmap_b0": pointmap + 0.1,
            "loss": torch.tensor(1.25),
            "video_recon_loss": torch.tensor(0.25),
            "pointmap_loss": torch.tensor(1.0),
        }

        with tempfile.TemporaryDirectory() as directory:
            paths = save_vggt_visualization(
                inputs,
                outputs,
                directory,
                split="train",
                step=50,
                max_time_steps=2,
                max_views=2,
                scatter_max_points=16,
            )
            self.assertEqual(
                set(paths), {"contact_sheet", "pointmap_3d_scatter", "metadata"}
            )
            for path in paths.values():
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

            metadata = json.loads(Path(paths["metadata"]).read_text())
            self.assertEqual(metadata["step"], 50)
            self.assertEqual(metadata["split"], "train")
            self.assertAlmostEqual(metadata["losses"]["loss"], 1.25)
            self.assertEqual(metadata["visualized_time_indices"], [0, 1])
            self.assertEqual(metadata["visualized_view_indices"], [0, 1])


if __name__ == "__main__":
    unittest.main()
