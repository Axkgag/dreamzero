#!/usr/bin/env python3
"""Create train-split statistics for MobileManiBench realized action plans.

The implementation is intentionally streaming: full MobileManiBench contains
hundreds of millions of valid waypoint slots, so retaining every waypoint in
RAM is not practical. Mean/std/min/max are exact, while q01/q99 are estimated
from a deterministic uniform Bernoulli sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from groot.vla.data.plan_geometry import build_dynamic_block_plan_labels
from groot.vla.utils.mobile_plan_spec import (
    EEF_ROTATION_ANCHOR_BASE_6D,
    EEF_ROTATION_CURRENT_EEF_DELTA_ROTVEC,
    block_plan_spec_hash,
    canonical_block_plan_spec,
    dynamic_block_plan_stats_path,
    eef_rotation_dim,
    manipulator_plan_dim,
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


class StreamingSummary:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.count = 0
        self.total = np.zeros(dim, dtype=np.float64)
        self.total_sq = np.zeros(dim, dtype=np.float64)
        self.minimum = np.full(dim, np.inf, dtype=np.float64)
        self.maximum = np.full(dim, -np.inf, dtype=np.float64)
        self.samples: list[np.ndarray] = []

    def update(self, values: np.ndarray, sample_mask: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dim:
            raise ValueError(f"Expected [N,{self.dim}], got {values.shape}")
        if not len(values):
            return
        self.count += len(values)
        self.total += values.sum(axis=0)
        self.total_sq += np.square(values).sum(axis=0)
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))
        if sample_mask.any():
            self.samples.append(values[sample_mask].copy())

    def finish(
        self, sample_limit: int, rng: np.random.Generator
    ) -> tuple[dict[str, list[float]], int]:
        if self.count == 0:
            empty = {key: [] for key in ("mean", "std", "min", "max", "q01", "q99")}
            return empty, 0
        mean = self.total / self.count
        variance = np.maximum(self.total_sq / self.count - np.square(mean), 0.0)
        sampled = (
            np.concatenate(self.samples, axis=0)
            if self.samples
            else np.stack([self.minimum, self.maximum], axis=0)
        )
        if len(sampled) > sample_limit:
            sampled = sampled[
                rng.choice(len(sampled), size=sample_limit, replace=False)
            ]
        result = {
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "q01": np.quantile(sampled, 0.01, axis=0).tolist(),
            "q99": np.quantile(sampled, 0.99, axis=0).tolist(),
        }
        return result, len(sampled)


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


def prepare(
    root: Path,
    force: bool,
    quantile_sample_size: int,
    seed: int,
    split: str,
    split_manifest: Path | None,
    write_core_stats: bool,
    multiblock: bool,
    dynamic_multiblock: bool = False,
    block_anchor_offsets: tuple[int, ...] = (0, 8, 16, 24),
    local_waypoint_offsets: tuple[int, ...] = (4, 8),
    eef_rotation_representation: str = EEF_ROTATION_ANCHOR_BASE_6D,
    stats_output_dir: Path | None = None,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    if multiblock and dynamic_multiblock:
        raise ValueError("Choose either materialized or dynamic multiblock statistics")
    label_spec: dict[str, object] | None = None
    if dynamic_multiblock:
        label_spec = canonical_block_plan_spec(
            block_anchor_offsets,
            local_waypoint_offsets,
            eef_rotation_representation=eef_rotation_representation,
        )
        default_output = dynamic_block_plan_stats_path(
            root,
            block_anchor_offsets,
            local_waypoint_offsets,
            eef_rotation_representation=eef_rotation_representation,
        )
        output = (
            Path(stats_output_dir) / default_output.name
            if stats_output_dir is not None
            else default_output
        )
    else:
        output = root / (
            "meta/multiblock_plan_stats.json" if multiblock else "meta/plan_stats.json"
        )
    manifest_path = None
    split_manifest_sha256 = None
    if split != "all":
        manifest_path = (
            split_manifest
            if split_manifest is not None
            else root / "meta/plan_splits.json"
        )
        split_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if output.exists() and not force:
        if reuse_existing:
            existing = read_json(output)
            expected_hash = (
                block_plan_spec_hash(
                    block_anchor_offsets,
                    local_waypoint_offsets,
                    eef_rotation_representation=eef_rotation_representation,
                )
                if dynamic_multiblock
                else None
            )
            if (
                existing.get("fit_split") == split
                and existing.get("split_manifest_sha256")
                == split_manifest_sha256
                and (
                    not dynamic_multiblock
                    or existing.get("label_spec_hash") == expected_hash
                )
            ):
                existing.setdefault("statistics_path", str(output))
                return existing
        raise FileExistsError(f"{output} already exists; pass --force to replace it")

    schema = read_json(root / "meta/robot_schema.json")
    extensions = read_json(root / "meta/extensions.json")
    if dynamic_multiblock:
        assert label_spec is not None
        plan_meta = {
            "local_waypoint_offsets": label_spec["local_waypoint_offsets"],
            "block_anchor_offsets": label_spec["block_anchor_offsets"],
            "global_waypoint_offsets": label_spec["global_waypoint_offsets"],
            "coordinate_frame": label_spec["coordinate_frame"],
            "num_blocks": label_spec["num_blocks"],
        }
    else:
        plan_meta = extensions["action_plan_block" if multiblock else "action_plan"]
    offsets = (
        plan_meta["local_waypoint_offsets"]
        if multiblock or dynamic_multiblock
        else plan_meta["waypoint_offsets"]
    )
    horizon = len(offsets)
    num_plan_blocks = (
        int(plan_meta["num_blocks"])
        if multiblock or dynamic_multiblock
        else 1
    )
    slot_count = num_plan_blocks * horizon
    hand_dim = len(schema["hand_joint_indices"])
    manipulator_dim = manipulator_plan_dim(
        hand_dim, eef_rotation_representation
    )
    rotation_dim = eef_rotation_dim(eef_rotation_representation)
    hand_start = 3 + rotation_dim

    info = read_json(root / "meta/info.json")
    selected_episode_ids: set[int] | None = None
    selected_frames = int(info["total_frames"])
    if split != "all":
        assert manifest_path is not None
        manifest = read_json(manifest_path)
        if split not in manifest["splits"]:
            raise ValueError(f"Split {split!r} is missing from {manifest_path}")
        split_meta = manifest["splits"][split]
        selected_episode_ids = {
            int(value) for value in split_meta["episode_indices"]
        }
        selected_frames = int(split_meta["num_frames"])
    total_slots_upper_bound = selected_frames * slot_count
    sample_probability = min(
        1.0, quantile_sample_size / max(total_slots_upper_bound, 1)
    )
    rng = np.random.default_rng(seed)
    base_summary = StreamingSummary(2)
    eef_summary = StreamingSummary(3)
    hand_summary = StreamingSummary(hand_dim)
    state_summary = StreamingSummary(6)
    action_dim = int(info["features"]["action"]["shape"][0])
    action_summary = StreamingSummary(action_dim)
    core_sample_probability = min(
        1.0, quantile_sample_size / max(selected_frames, 1)
    )
    valid_count = 0
    total_count = 0
    all_finite = True
    base_yaw_sincos_max_unit_norm_error = 0.0
    rotation6d_max_row_unit_norm_error = 0.0
    rotation6d_max_abs_row_dot = 0.0
    rotvec_max_angle_rad = 0.0
    parquet_paths = sorted((root / "data").glob("*/*.parquet"))
    if selected_episode_ids is not None:
        parquet_paths = [
            path
            for path in parquet_paths
            if int(path.stem.rsplit("_", 1)[1]) in selected_episode_ids
        ]
    for file_index, parquet_path in enumerate(parquet_paths, start=1):
        if dynamic_multiblock:
            columns = [
                "observation.base.world",
                "observation.eef.world",
                "observation.robot_joint",
            ]
        else:
            prefix = "action.plan.block" if multiblock else "action.plan"
            columns = [
                f"{prefix}.base_waypoints",
                f"{prefix}.manipulator",
                f"{prefix}.valid",
            ]
        if write_core_stats:
            columns.extend(["observation.state", "action"])
        frame = pd.read_parquet(parquet_path, columns=columns)
        if dynamic_multiblock:
            base_world = np.stack(
                frame["observation.base.world"].to_numpy()
            ).astype(np.float64, copy=False)
            eef_world = np.stack(
                frame["observation.eef.world"].to_numpy()
            ).astype(np.float64, copy=False)
            joint_flat = np.stack(
                frame["observation.robot_joint"].to_numpy()
            ).astype(np.float64, copy=False)
            if joint_flat.shape[1] % 3:
                raise ValueError(
                    f"{parquet_path}: joint width {joint_flat.shape[1]} is not "
                    "divisible by three"
                )
            labels = build_dynamic_block_plan_labels(
                base_world,
                eef_world,
                joint_flat.reshape(len(frame), -1, 3),
                schema["hand_joint_indices"],
                block_anchor_offsets,
                local_waypoint_offsets,
                eef_rotation_representation,
            )
            base = labels[0].reshape(-1, slot_count, 4).astype(
                np.float64, copy=False
            )
            manipulator = labels[1].reshape(
                -1, slot_count, manipulator_dim
            ).astype(np.float64, copy=False)
            valid = labels[2].reshape(-1, slot_count)
        else:
            base = np.stack(frame[f"{prefix}.base_waypoints"].to_numpy()).astype(
                np.float64, copy=False
            ).reshape(-1, slot_count, 4)
            manipulator = np.stack(
                frame[f"{prefix}.manipulator"].to_numpy()
            ).astype(np.float64, copy=False).reshape(
                -1, slot_count, manipulator_dim
            )
            valid = np.stack(frame[f"{prefix}.valid"].to_numpy()).astype(
                np.bool_, copy=False
            ).reshape(-1, slot_count)
        base_valid = base[valid]
        manipulator_valid = manipulator[valid]
        sample_mask = rng.random(len(base_valid)) < sample_probability

        base_summary.update(base_valid[:, 0:2], sample_mask)
        eef_summary.update(manipulator_valid[:, 0:3], sample_mask)
        if hand_dim:
            hand_summary.update(manipulator_valid[:, hand_start:], sample_mask)
        if write_core_stats:
            core_sample_mask = rng.random(len(frame)) < core_sample_probability
            state_summary.update(
                np.stack(frame["observation.state"].to_numpy()), core_sample_mask
            )
            action_summary.update(
                np.stack(frame["action"].to_numpy()), core_sample_mask
            )
        valid_count += len(base_valid)
        total_count += valid.size

        finite = np.isfinite(base_valid).all() and np.isfinite(manipulator_valid).all()
        all_finite = all_finite and bool(finite)
        if len(base_valid):
            base_yaw_sincos_max_unit_norm_error = max(
                base_yaw_sincos_max_unit_norm_error,
                float(
                    np.max(
                        np.abs(
                            np.linalg.norm(base_valid[:, 2:4], axis=-1) - 1.0
                        )
                    )
                ),
            )
            if eef_rotation_representation == EEF_ROTATION_ANCHOR_BASE_6D:
                rotation = manipulator_valid[:, 3:9].reshape(-1, 2, 3)
                rotation6d_max_row_unit_norm_error = max(
                    rotation6d_max_row_unit_norm_error,
                    float(
                        np.max(
                            np.abs(np.linalg.norm(rotation, axis=-1) - 1.0)
                        )
                    ),
                )
                rotation6d_max_abs_row_dot = max(
                    rotation6d_max_abs_row_dot,
                    float(
                        np.max(
                            np.abs(np.sum(rotation[:, 0] * rotation[:, 1], axis=-1))
                        )
                    ),
                )
            elif eef_rotation_representation == EEF_ROTATION_CURRENT_EEF_DELTA_ROTVEC:
                rotvec_max_angle_rad = max(
                    rotvec_max_angle_rad,
                    float(np.linalg.norm(manipulator_valid[:, 3:6], axis=-1).max()),
                )
        if file_index % 1000 == 0 or file_index == len(parquet_paths):
            print(
                f"[{root.name}] plan stats {file_index}/{len(parquet_paths)} "
                f"parquet files, {valid_count} valid waypoints",
                flush=True,
            )

    if valid_count == 0:
        raise ValueError(f"No parquet plan rows found below {root}")
    base_stats, base_sample_count = base_summary.finish(quantile_sample_size, rng)
    eef_stats, eef_sample_count = eef_summary.finish(quantile_sample_size, rng)
    hand_stats, hand_sample_count = hand_summary.finish(quantile_sample_size, rng)
    if write_core_stats:
        state_stats, _ = state_summary.finish(quantile_sample_size, rng)
        action_stats, _ = action_summary.finish(quantile_sample_size, rng)
        write_json(
            root / "meta/stats.json",
            {
                "observation.state": state_stats,
                "action": action_stats,
            },
        )

    result = {
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "embodiment": schema["embodiment"],
        "fit_split": split,
        "split_manifest_sha256": split_manifest_sha256,
        "statistics_method": {
            "mean_std_min_max": "exact_streaming",
            "q01_q99": "deterministic_uniform_bernoulli_sample",
            "quantile_sample_size_target": quantile_sample_size,
            "quantile_sample_probability": sample_probability,
            "quantile_sample_counts": {
                "base_xy": base_sample_count,
                "eef_xyz": eef_sample_count,
                "hand": hand_sample_count,
            },
            "seed": seed,
        },
        "normalization_policy": {
            "base_xy": "q99",
            "base_yaw_sincos": "identity",
            "eef_xyz": "q99",
            "eef_rotation": "identity",
            "hand": "per_joint_q99",
            "valid_mask": "identity",
        },
        "plan_horizon": horizon,
        "plan_time_offsets": offsets,
        "num_plan_blocks": num_plan_blocks,
        "block_anchor_offsets": plan_meta.get("block_anchor_offsets", [0]),
        "global_plan_offsets": plan_meta.get("global_waypoint_offsets", offsets),
        "coordinate_frame": plan_meta.get("coordinate_frame", "current_base"),
        "eef_rotation_representation": eef_rotation_representation,
        "eef_rotation_dim": rotation_dim,
        "hand_start_index": hand_start,
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
            "base_xy": base_stats,
            "eef_xyz": eef_stats,
            "hand": hand_stats,
        },
        "geometry_qa": {
            "all_finite": all_finite,
            "base_yaw_sincos_max_unit_norm_error": base_yaw_sincos_max_unit_norm_error,
            "rotation6d_max_row_unit_norm_error": rotation6d_max_row_unit_norm_error,
            "rotation6d_max_abs_row_dot": rotation6d_max_abs_row_dot,
            "rotvec_max_angle_rad": rotvec_max_angle_rad,
        },
        "label_source": (
            "dynamic_world_trajectory" if dynamic_multiblock else "materialized"
        ),
        "label_spec": label_spec,
        "label_spec_hash": (
            block_plan_spec_hash(
                block_anchor_offsets,
                local_waypoint_offsets,
                eef_rotation_representation=eef_rotation_representation,
            )
            if dynamic_multiblock
            else None
        ),
        "statistics_path": str(output),
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quantile-sample-size", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument(
        "--write-core-stats",
        action="store_true",
        help="Also replace meta/stats.json using only the selected split.",
    )
    parser.add_argument(
        "--multiblock",
        action="store_true",
        help="Fit block-major labels and write meta/multiblock_plan_stats.json.",
    )
    parser.add_argument(
        "--dynamic-multiblock",
        action="store_true",
        help="Derive block labels from canonical world-state columns.",
    )
    parser.add_argument(
        "--block-anchor-offsets",
        default="0,8,16,24",
        help="Comma-separated dynamic block anchor offsets.",
    )
    parser.add_argument(
        "--block-local-offsets",
        default="4,8",
        help="Comma-separated local offsets inside every dynamic block.",
    )
    parser.add_argument(
        "--plan-config",
        type=Path,
        help="Read block_anchor_offsets and plan_local_offsets from a YAML config.",
    )
    parser.add_argument(
        "--stats-output-dir",
        type=Path,
        help="Optional output directory; filename remains spec-hashed.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse an existing statistics file when split and spec match.",
    )
    args = parser.parse_args()
    if args.quantile_sample_size <= 0:
        parser.error("--quantile-sample-size must be positive")

    if args.multiblock and args.dynamic_multiblock:
        parser.error("--multiblock and --dynamic-multiblock are mutually exclusive")

    def parse_offsets(raw: str) -> tuple[int, ...]:
        try:
            return tuple(int(value.strip()) for value in raw.split(",") if value.strip())
        except ValueError as error:
            parser.error(f"Invalid offset list {raw!r}: {error}")

    block_anchor_offsets = parse_offsets(args.block_anchor_offsets)
    local_waypoint_offsets = parse_offsets(args.block_local_offsets)
    if args.plan_config is not None:
        with args.plan_config.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        block_anchor_offsets = tuple(int(v) for v in config["block_anchor_offsets"])
        local_waypoint_offsets = tuple(int(v) for v in config["plan_local_offsets"])
        eef_rotation_representation = str(
            config.get(
                "eef_rotation_representation",
                EEF_ROTATION_ANCHOR_BASE_6D,
            )
        )
    else:
        eef_rotation_representation = EEF_ROTATION_ANCHOR_BASE_6D

    for root in resolve_roots(args.dataset_root):
        result = prepare(
            root,
            args.force,
            args.quantile_sample_size,
            args.seed,
            args.split,
            args.split_manifest,
            args.write_core_stats,
            args.multiblock,
            args.dynamic_multiblock,
            block_anchor_offsets,
            local_waypoint_offsets,
            eef_rotation_representation,
            args.stats_output_dir,
            args.reuse_existing,
        )
        print(
            f"Ready {result['statistics_path']}: "
            f"{result['counts']['valid_waypoints']} valid waypoints, "
            f"manipulator_dim={result['manipulator_dim']}"
        )


if __name__ == "__main__":
    main()
