#!/usr/bin/env python3
"""Create train-split statistics for MobileManiBench realized action plans."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def summarize(values: np.ndarray) -> dict[str, list[float]]:
    if values.ndim != 2 or not len(values):
        raise ValueError(f"Expected a non-empty [N,D] array, got {values.shape}")
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def resolve_roots(dataset_root: Path) -> list[Path]:
    if (dataset_root / "meta/extensions.json").is_file():
        return [dataset_root]
    roots = [
        dataset_root / name
        for name in ("g1", "xhand")
        if (dataset_root / name / "meta/extensions.json").is_file()
    ]
    if not roots:
        raise FileNotFoundError(
            f"{dataset_root} is neither a converted robot root nor a g1/xhand parent"
        )
    return roots


def prepare(root: Path, force: bool) -> dict[str, Any]:
    output = root / "meta/plan_stats.json"
    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists; pass --force to replace it")

    schema = read_json(root / "meta/robot_schema.json")
    extensions = read_json(root / "meta/extensions.json")
    plan_meta = extensions["action_plan"]
    offsets = plan_meta["waypoint_offsets"]
    horizon = len(offsets)
    hand_dim = len(schema["hand_joint_indices"])
    manipulator_dim = 9 + hand_dim

    base_values: list[np.ndarray] = []
    manipulator_values: list[np.ndarray] = []
    valid_count = 0
    total_count = 0
    for parquet_path in sorted((root / "data").glob("*/*.parquet")):
        frame = pd.read_parquet(
            parquet_path,
            columns=[
                "action.plan.base_waypoints",
                "action.plan.manipulator",
                "action.plan.valid",
            ],
        )
        for row in frame.itertuples(index=False, name=None):
            base = np.asarray(row[0], dtype=np.float64).reshape(horizon, 4)
            manipulator = np.asarray(row[1], dtype=np.float64).reshape(
                horizon, manipulator_dim
            )
            valid = np.asarray(row[2], dtype=np.bool_).reshape(horizon)
            base_values.append(base[valid])
            manipulator_values.append(manipulator[valid])
            valid_count += int(valid.sum())
            total_count += horizon

    if not base_values:
        raise ValueError(f"No parquet plan rows found below {root}")
    base = np.concatenate(base_values, axis=0)
    manipulator = np.concatenate(manipulator_values, axis=0)
    rotation = manipulator[:, 3:9].reshape(-1, 2, 3)

    result = {
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "embodiment": schema["embodiment"],
        "fit_split": "train (all episodes selected by the converted smoke dataset)",
        "normalization_policy": {
            "base_xy": "q99",
            "base_yaw_sincos": "identity",
            "eef_xyz": "q99",
            "eef_rotation6d": "identity",
            "hand": "per_joint_q99",
            "valid_mask": "identity",
        },
        "plan_horizon": horizon,
        "plan_time_offsets": offsets,
        "control_fps": float(extensions["time"]["control_fps"]),
        "base_dim": 4,
        "manipulator_dim": manipulator_dim,
        "max_manipulator_dim": 21,
        "hand_dim": hand_dim,
        "hand_joint_names": schema["hand_joint_names"],
        "counts": {
            "valid_waypoints": valid_count,
            "all_waypoint_slots": total_count,
        },
        "statistics": {
            "base_xy": summarize(base[:, 0:2]),
            "eef_xyz": summarize(manipulator[:, 0:3]),
            "hand": summarize(manipulator[:, 9:])
            if hand_dim
            else {key: [] for key in ("mean", "std", "min", "max", "q01", "q99")},
        },
        "geometry_qa": {
            "all_finite": bool(np.isfinite(base).all() and np.isfinite(manipulator).all()),
            "base_yaw_sincos_max_unit_norm_error": float(
                np.max(np.abs(np.linalg.norm(base[:, 2:4], axis=-1) - 1.0))
            ),
            "rotation6d_max_row_unit_norm_error": float(
                np.max(np.abs(np.linalg.norm(rotation, axis=-1) - 1.0))
            ),
            "rotation6d_max_abs_row_dot": float(
                np.max(np.abs(np.sum(rotation[:, 0] * rotation[:, 1], axis=-1)))
            ),
        },
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for root in resolve_roots(args.dataset_root):
        result = prepare(root, args.force)
        print(
            f"Wrote {root / 'meta/plan_stats.json'}: "
            f"{result['counts']['valid_waypoints']} valid waypoints, "
            f"manipulator_dim={result['manipulator_dim']}"
        )


if __name__ == "__main__":
    main()

