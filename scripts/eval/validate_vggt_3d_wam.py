#!/usr/bin/env python3
"""Evaluate RGB reconstruction and coarse PointMap accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import torch
from torch.utils.data import DataLoader

from groot.vla.data.dataset.mobilemanibench_vggt import (
    ISAAC_X_FORWARD_FROM_OPENCV,
    MobileManiBenchVGGTDataCollator,
    MobileManiBenchVGGTDataset,
)
from groot.vla.model.vggt_3d_wam.geometry import points_in_metric_grid
from groot.vla.model.vggt_3d_wam.model import VGGT3DWAMModel
from groot.vla.model.vggt_3d_wam.visualization import save_vggt_visualization


EXPECTED_VIDEO_OFFSETS = tuple(range(33))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--video-delta-indices",
        default=",".join(str(value) for value in EXPECTED_VIDEO_OFFSETS),
    )
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--visualization-root", type=Path)
    parser.add_argument("--max-visualizations", type=int, default=4)
    return parser.parse_args()


def _checkpoint_step(checkpoint: Path) -> int:
    match = re.search(r"checkpoint-(\d+)", checkpoint.name)
    if match:
        return int(match.group(1))
    state_path = checkpoint / "trainer_state.json"
    if state_path.is_file():
        return int(json.loads(state_path.read_text())["global_step"])
    return 0


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model = VGGT3DWAMModel.from_pretrained(args.checkpoint).to(device).eval()
    offsets = [int(value) for value in args.video_delta_indices.split(",")]
    if tuple(offsets) != EXPECTED_VIDEO_OFFSETS:
        raise ValueError(
            "VGGT validation must use the same contiguous 33-frame window as "
            f"training; expected {list(EXPECTED_VIDEO_OFFSETS)}, got {offsets}"
        )
    latent_frames = 1 + (len(offsets) - 1) // model.config.latent_temporal_stride
    print(
        "VGGT validation temporal contract: "
        f"video_frames={len(offsets)}, latent_frames={latent_frames}, "
        f"temporal_stride={model.config.latent_temporal_stride}"
    )
    dataset = MobileManiBenchVGGTDataset(
        args.dataset_root,
        video_delta_indices=offsets,
        image_size=model.config.image_size,
        pointmap_size=model.config.pointmap_size,
        split=args.split,
        validation_fraction=args.validation_fraction,
        camera_optical_transform=ISAAC_X_FORWARD_FROM_OPENCV,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=MobileManiBenchVGGTDataCollator(),
    )

    totals = {
        "video_absolute_error": 0.0,
        "video_values": 0,
        "pointmap_l1_m": 0.0,
        "pointmap_l2_m": 0.0,
        "pointmap_raw_valid_count": 0.0,
        "pointmap_inside_grid_count": 0.0,
        "pointmap_raw_valid_weight": 0.0,
        "pointmap_inside_grid_weight": 0.0,
        "ray_total_sample_count": 0.0,
        "ray_valid_sample_count": 0.0,
        "ray_supervised_pixel_count": 0.0,
        "free_space_sample_count": 0.0,
        "surface_sample_count": 0.0,
        "multiview_candidate_count": 0.0,
        "multiview_correspondence_count": 0.0,
        "multiview_correspondence_weight": 0.0,
        "samples": 0,
    }
    checkpoint_step = _checkpoint_step(args.checkpoint)
    visualizations_saved = 0
    with torch.inference_mode():
        for batch in loader:
            if totals["samples"] >= args.max_samples:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                outputs = model(batch)
            if (
                args.visualization_root is not None
                and visualizations_saved < args.max_visualizations
            ):
                save_vggt_visualization(
                    batch,
                    outputs,
                    args.visualization_root,
                    split="val",
                    step=checkpoint_step,
                    sample_index=visualizations_saved,
                )
                visualizations_saved += 1
            target_video = batch["video"].float() / 255
            reconstructed_video = (
                (outputs["reconstructed_video"].float() + 1.0) * 0.5
            ).clamp(0.0, 1.0)
            video_error = (reconstructed_video - target_video).abs()
            totals["video_absolute_error"] += float(video_error.sum())
            totals["video_values"] += video_error.numel()

            point_error = outputs["predicted_pointmap_b0"] - batch[
                "pseudo_pointmap_b0"
            ]
            valid = batch["pointmap_valid"]
            target_xyz = batch["pseudo_pointmap_b0"].permute(
                0, 1, 2, 4, 5, 3
            )
            inside = points_in_metric_grid(
                target_xyz,
                model.config.grid_x_range,
                model.config.grid_y_range,
                model.config.grid_z_range,
            )
            metric_valid = valid * inside
            totals["pointmap_l1_m"] += float(
                (point_error.abs().sum(dim=3) * metric_valid).sum()
            )
            totals["pointmap_l2_m"] += float(
                (
                    torch.linalg.vector_norm(point_error, dim=3)
                    * metric_valid
                ).sum()
            )
            for key in (
                "pointmap_raw_valid_count",
                "pointmap_inside_grid_count",
                "pointmap_raw_valid_weight",
                "pointmap_inside_grid_weight",
                "ray_total_sample_count",
                "ray_valid_sample_count",
                "ray_supervised_pixel_count",
                "free_space_sample_count",
                "surface_sample_count",
                "multiview_candidate_count",
                "multiview_correspondence_count",
                "multiview_correspondence_weight",
            ):
                totals[key] += float(outputs[key])
            totals["samples"] += batch["video"].shape[0]

    valid_weight = max(1.0, totals["pointmap_inside_grid_weight"])

    def ratio(numerator: str, denominator: str) -> float:
        return totals[numerator] / max(1.0, totals[denominator])

    metrics = {
        "video_mae": totals["video_absolute_error"] / max(1, totals["video_values"]),
        "pointmap_coordinate_mae_m": totals["pointmap_l1_m"] / (3 * valid_weight),
        "pointmap_euclidean_error_m": totals["pointmap_l2_m"] / valid_weight,
        "pointmap_raw_valid_count": totals["pointmap_raw_valid_count"],
        "pointmap_inside_grid_count": totals["pointmap_inside_grid_count"],
        "pointmap_raw_valid_weight": totals["pointmap_raw_valid_weight"],
        "pointmap_inside_grid_weight": totals["pointmap_inside_grid_weight"],
        "pointmap_inside_grid_ratio": ratio(
            "pointmap_inside_grid_count", "pointmap_raw_valid_count"
        ),
        "pointmap_inside_grid_weight_ratio": ratio(
            "pointmap_inside_grid_weight", "pointmap_raw_valid_weight"
        ),
        "ray_valid_ratio": ratio(
            "ray_valid_sample_count", "ray_total_sample_count"
        ),
        "ray_supervised_pixel_ratio": ratio(
            "ray_supervised_pixel_count", "pointmap_inside_grid_count"
        ),
        "free_space_sample_ratio": ratio(
            "free_space_sample_count", "ray_valid_sample_count"
        ),
        "surface_sample_ratio": ratio(
            "surface_sample_count", "ray_valid_sample_count"
        ),
        "multiview_correspondence_ratio": ratio(
            "multiview_correspondence_count",
            "multiview_candidate_count",
        ),
        "ray_total_sample_count": totals["ray_total_sample_count"],
        "ray_valid_sample_count": totals["ray_valid_sample_count"],
        "ray_supervised_pixel_count": totals[
            "ray_supervised_pixel_count"
        ],
        "free_space_sample_count": totals["free_space_sample_count"],
        "surface_sample_count": totals["surface_sample_count"],
        "multiview_candidate_count": totals["multiview_candidate_count"],
        "multiview_correspondence_count": totals[
            "multiview_correspondence_count"
        ],
        "multiview_correspondence_weight": totals[
            "multiview_correspondence_weight"
        ],
        "samples": totals["samples"],
        "split": args.split,
        "video_frames": len(offsets),
        "latent_frames": latent_frames,
        "visualizations_saved": visualizations_saved,
        "geometry_quality": "lossy_h264_pseudo_robot_centric_pointmap",
        "calibration_status": dataset.calibration.get("status", "unknown"),
    }
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
