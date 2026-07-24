#!/usr/bin/env python3
"""Render one Phase-1 MobileManiBench plan sample for manual inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from groot.vla.data.dataset import MobileManiBenchPlanDataset
from groot.vla.data.transform import MobilePlanTransform


def as_image(value: Any) -> np.ndarray:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.dtype != np.uint8:
        if array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def trajectory_panel(base: np.ndarray, eef: np.ndarray, valid: np.ndarray) -> np.ndarray:
    panel = np.full((520, 520, 3), 248, dtype=np.uint8)
    points = np.concatenate(
        [np.zeros((1, 2), dtype=np.float32), base[valid, :2], eef[valid, :2]],
        axis=0,
    )
    span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 0.25)
    center = np.mean(points, axis=0)

    def pixel(xy: np.ndarray) -> tuple[int, int]:
        scaled = (xy - center) / span * 400
        return int(260 + scaled[0]), int(260 - scaled[1])

    origin = pixel(np.zeros(2))
    cv2.drawMarker(panel, origin, (0, 0, 0), cv2.MARKER_CROSS, 16, 2)
    previous_base = origin
    previous_eef = origin
    for horizon_index, is_valid in enumerate(valid):
        if not is_valid:
            continue
        base_pixel = pixel(base[horizon_index, :2])
        eef_pixel = pixel(eef[horizon_index, :2])
        cv2.line(panel, previous_base, base_pixel, (220, 90, 20), 2)
        cv2.line(panel, previous_eef, eef_pixel, (30, 150, 40), 2)
        cv2.circle(panel, base_pixel, 5, (220, 90, 20), -1)
        cv2.circle(panel, eef_pixel, 5, (30, 150, 40), -1)
        cv2.putText(
            panel,
            str(horizon_index),
            (base_pixel[0] + 5, base_pixel[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        previous_base, previous_eef = base_pixel, eef_pixel
    cv2.putText(panel, "base XY (blue)", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 90, 20), 2)
    cv2.putText(panel, "EEF XY (green)", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 150, 40), 2)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    transform = MobilePlanTransform(stats_path=args.dataset_root / "meta/plan_stats.json")
    dataset = MobileManiBenchPlanDataset(
        args.dataset_root,
        video_delta_indices=[0],
        load_videos=True,
        plan_transform=transform,
    )
    sample = dataset[args.index]
    raw_base = np.asarray(sample["base_plan"])
    raw_manipulator = np.asarray(sample["manipulator_plan"])
    valid = np.asarray(sample["plan_valid"], dtype=bool)

    head = cv2.resize(as_image(sample["video.head"]), (520, 520))
    wrist = cv2.resize(as_image(sample["video.wrist"]), (520, 520))
    plot = trajectory_panel(raw_base, raw_manipulator[:, :3], valid)
    canvas = np.concatenate([head[:, :, ::-1], wrist[:, :, ::-1], plot], axis=1)
    cv2.putText(canvas, "head", (18, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, "wrist", (538, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    output_dir = args.output_dir or args.dataset_root / "validation_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"phase1_plan_batch_ep{int(sample['episode_index']):06d}_frame{int(sample['frame_index']):06d}"
    image_path = output_dir / f"{stem}.png"
    json_path = output_dir / f"{stem}.json"
    cv2.imwrite(str(image_path), canvas)
    summary = {
        "dataset_root": str(args.dataset_root),
        "episode_index": int(sample["episode_index"]),
        "frame_index": int(sample["frame_index"]),
        "hand_dim": int(sample["hand_dim"]),
        "plan_time_offsets": np.asarray(sample["plan_time_offsets"]).tolist(),
        "plan_time_seconds": np.asarray(sample["plan_time_seconds"]).tolist(),
        "plan_valid": valid.tolist(),
        "shapes": {
            "base_plan": list(raw_base.shape),
            "manipulator_plan": list(raw_manipulator.shape),
            "base_action": list(sample["base_action"].shape),
            "manipulator_action": list(sample["manipulator_action"].shape),
        },
        "normalized_ranges": {
            "base_xy": [
                float(sample["base_action"][:, :2].min()),
                float(sample["base_action"][:, :2].max()),
            ],
            "eef_xyz": [
                float(sample["manipulator_action"][:, :3].min()),
                float(sample["manipulator_action"][:, :3].max()),
            ],
        },
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Wrote {image_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
