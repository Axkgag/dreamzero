from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("DS_ACCELERATOR", "cuda")

import torch
from safetensors import safe_open
from transformers import TrainingArguments

from groot.vla.experiment.vggt_3d_wam import VGGTTrainer
from groot.vla.model.vggt_3d_wam import VGGT3DWAMConfig, VGGT3DWAMModel
from groot.vla.model.vggt_3d_wam.backbone import _adapt_checkpoint_state
from groot.vla.model.vggt_3d_wam.geometry import (
    range_to_pointmap,
    scale_intrinsics,
)


def tiny_config() -> VGGT3DWAMConfig:
    return VGGT3DWAMConfig(
        image_size=(32, 32),
        patch_size=8,
        patch_embed_type="conv",
        backbone_dim=32,
        backbone_depth=1,
        backbone_heads=4,
        init_random=True,
        freeze_backbone=False,
        lora_rank=0,
        latent_dim=16,
        latent_spatial_stride=16,
        latent_temporal_stride=4,
        video_temporal_layers=1,
        video_temporal_heads=4,
        geometry_dim=24,
        geometry_heads=4,
        geometry_temporal_layers=1,
        deformable_points=2,
        grid_size=(2, 3, 3),
        pointmap_size=(8, 8),
        pointmap_ray_size=(4, 4),
        pointmap_depth_bins=8,
        pointmap_ray_chunk_size=32,
        pointmap_loss_warmup_steps=0,
        ssim_loss_weight=0.2,
        spatial_gradient_loss_weight=0.1,
        temporal_difference_loss_weight=0.1,
        temporal_geometry_loss_weight=0.1,
        surface_normal_loss_weight=0.1,
        depth_gradient_loss_weight=0.1,
    )


def tiny_batch() -> dict[str, torch.Tensor]:
    batch, time, views, height, width = 1, 9, 2, 32, 32
    intrinsics = torch.tensor(
        [[20.0, 0.0, 15.5], [0.0, 20.0, 15.5], [0.0, 0.0, 1.0]]
    ).expand(batch, time, views, 3, 3).clone()
    extrinsics = torch.eye(4).expand(batch, time, views, 4, 4).clone()
    extrinsics[..., 2, 3] = 0.2
    pointmap_intrinsics = scale_intrinsics(
        intrinsics, (height, width), (8, 8)
    )
    distance = torch.full((batch, time, views, 8, 8), 0.8)
    pointmap = range_to_pointmap(
        distance, pointmap_intrinsics, extrinsics
    ).permute(0, 1, 2, 5, 3, 4)
    return {
        "video": torch.randint(
            0, 256, (batch, time, views, 3, height, width), dtype=torch.uint8
        ),
        "camera_K": intrinsics,
        "T_b0_camera": extrinsics,
        "pseudo_pointmap_b0": pointmap,
        "pointmap_valid": torch.ones(batch, time, views, 8, 8),
    }


class VGGT3DWAMModelTest(unittest.TestCase):
    def test_official_dino_checkpoint_layout_is_adapted(self) -> None:
        source_position = torch.arange(10).reshape(1, 5, 2)
        source_register = torch.randn(1, 4, 2)
        expected = {
            "patch_embed.pos_embed": torch.empty(1, 4, 2),
            "patch_embed.reg_token": torch.empty(1, 4, 2),
        }
        adapted = _adapt_checkpoint_state(
            {
                "patch_embed.pos_embed": source_position,
                "patch_embed.register_tokens": source_register,
            },
            expected,
        )
        torch.testing.assert_close(
            adapted["patch_embed.pos_embed"], source_position[:, 1:]
        )
        torch.testing.assert_close(
            adapted["patch_embed.reg_token"], source_register
        )

    def test_forward_backward_contract(self) -> None:
        model = VGGT3DWAMModel(tiny_config())
        outputs = model(tiny_batch())
        self.assertEqual(tuple(outputs["z_2d_video"].shape), (1, 2, 16, 3, 2, 2))
        self.assertEqual(tuple(outputs["z_3d_video"].shape), (1, 3, 18, 24))
        self.assertEqual(
            tuple(outputs["reconstructed_video"].shape), (1, 9, 2, 3, 32, 32)
        )
        self.assertEqual(
            tuple(outputs["predicted_pointmap_b0"].shape), (1, 9, 2, 3, 8, 8)
        )
        self.assertGreaterEqual(
            float(outputs["pointmap_raw_valid_count"]), 1
        )
        self.assertGreaterEqual(
            float(outputs["ray_total_sample_count"]),
            float(outputs["ray_valid_sample_count"]),
        )
        self.assertTrue(torch.isfinite(outputs["loss"]))
        self.assertTrue(torch.isfinite(outputs["video_quality_loss"]))
        self.assertTrue(torch.isfinite(outputs["video_ssim_loss"]))
        self.assertTrue(torch.isfinite(outputs["surface_normal_loss"]))
        self.assertTrue(torch.isfinite(outputs["depth_gradient_loss"]))
        self.assertTrue(torch.isfinite(outputs["weighted_geometry_loss"]))
        outputs["loss"].backward()
        self.assertIsNotNone(
            model.video_decoder.output_projection.weight.grad
        )
        self.assertIsNotNone(model.metric_encoder.query_features.grad)

    def test_saved_checkpoint_does_not_require_source_vggt_file(self) -> None:
        model = VGGT3DWAMModel(tiny_config()).eval()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = VGGT3DWAMModel.from_pretrained(directory)
            self.assertTrue(restored.config.init_random)
            self.assertIsNone(restored.config.vggt_checkpoint_path)

    def test_frozen_shared_lpips_tensors_are_excluded_from_checkpoint(
        self,
    ) -> None:
        model = VGGT3DWAMModel(tiny_config()).eval()
        shared = torch.nn.Linear(2, 1, bias=False)
        shared.requires_grad_(False)
        metric = torch.nn.Module()
        metric.first_alias = shared
        metric.second_alias = shared
        model.lpips_loss = metric

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            checkpoint = os.path.join(directory, "model.safetensors")
            with safe_open(checkpoint, framework="pt", device="cpu") as source:
                saved_keys = set(source.keys())

        self.assertFalse(
            any(key.startswith("lpips_loss.") for key in saved_keys)
        )

    def test_invalid_ray_target_bins_do_not_make_loss_nonfinite(self) -> None:
        model = VGGT3DWAMModel(tiny_config())
        batch = tiny_batch()
        batch["pseudo_pointmap_b0"] = torch.randn_like(
            batch["pseudo_pointmap_b0"]
        )
        outputs = model(batch)
        self.assertTrue(torch.isfinite(outputs["ray_surface_loss"]))
        self.assertTrue(torch.isfinite(outputs["loss"]))

    def test_geometry_quality_weight_scales_the_whole_branch(self) -> None:
        config = tiny_config()
        config.geometry_quality_weight = 0.25
        model = VGGT3DWAMModel(config)
        self.assertAlmostEqual(model._geometry_weight(), 0.025)

    def test_frozen_backbone_only_trains_lora_parameters(self) -> None:
        config = tiny_config()
        config.freeze_backbone = True
        config.lora_rank = 2
        model = VGGT3DWAMModel(config)
        trainable_backbone = [
            name
            for name, parameter in model.backbone.named_parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable_backbone)
        self.assertTrue(
            all(
                name.endswith(("down.weight", "up.weight"))
                for name in trainable_backbone
            )
        )

    def test_optimizer_uses_separate_backbone_and_head_learning_rates(
        self,
    ) -> None:
        model = VGGT3DWAMModel(tiny_config())
        with tempfile.TemporaryDirectory() as directory:
            args = TrainingArguments(
                output_dir=directory,
                learning_rate=5e-5,
                weight_decay=1e-2,
                report_to=[],
            )
            trainer = VGGTTrainer(
                model=model,
                args=args,
                backbone_learning_rate=2e-5,
                visualization_config={},
            )
            optimizer = trainer.create_optimizer()
        learning_rates = {
            group["group_name"]: group["lr"]
            for group in optimizer.param_groups
        }
        self.assertAlmostEqual(learning_rates["backbone"], 2e-5)
        self.assertAlmostEqual(learning_rates["heads"], 5e-5)


if __name__ == "__main__":
    unittest.main()
