#!/usr/bin/env python3
"""Offline validation for MobileManiBench multiblock WAM checkpoints.

The evaluator keeps the deployment contract explicit:

* ``single_block_reset`` predicts only the first block from one current frame;
* ``episode_ordered_reset`` predicts every requested block independently;
* ``gt_history_cached`` predicts a block at a time and feeds only RGB frames
  that have actually arrived before the next prediction;
* ``teacher_forced_open_loop`` slices the complete root window into arrived
  RGB chunks, runs the full Flow solver for every block, and reports waypoint
  metrics under ground-truth history and anchor-state conditioning;
* ``oracle_four_block_teacher_forced`` evaluates the stochastic training loss
  with the complete clean 33-frame target window.  It is a diagnostic upper
  bound, not a deployment waypoint metric.

All waypoint predictions are inverse-normalized before metric computation.
Per-block labels remain in each block's anchor-Base frame; composed metrics
recursively place blocks into the first anchor frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import OmegaConf

from groot.vla.data.dataset import MobileManiBenchBlockPlanDataset
from groot.vla.data.transform import MobilePlanTransform
from groot.vla.utils.mobile_plan_spec import dynamic_block_plan_stats_path

from evaluate_mobilemanibench_plan import (
    initialize_distributed,
    load_episode_tasks,
    load_model,
    read_jsonl,
    reset_sampler_state,
    resolve_episode_split,
    rotation6d_rows_to_matrix,
    rotation_geodesic_deg,
    sample_seed,
    to_numpy,
    wrap_angle,
    write_json,
)
from mobilemanibench_sampling import (
    count_tasks_for_indices,
    select_task_balanced_indices,
)


INFERENCE_MODES = (
    "single_block_reset",
    "episode_ordered_reset",
    "gt_history_cached",
)
TEACHER_FORCED_OPEN_LOOP_MODE = "teacher_forced_open_loop"
ORACLE_MODE = "oracle_four_block_teacher_forced"
FULL_WINDOW_MODES = (TEACHER_FORCED_OPEN_LOOP_MODE, ORACLE_MODE)
ALL_MODES = (*INFERENCE_MODES, *FULL_WINDOW_MODES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--mode", choices=ALL_MODES, default="single_block_reset")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum task-balanced root windows; 0 evaluates every root window.",
    )
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1140)
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument(
        "--num-rollout-blocks",
        type=int,
        default=0,
        help="0 uses every configured block; single_block_reset always uses one.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Validate split, labels, offsets and sampling without loading a model.",
    )
    return parser.parse_args()


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
    }


def _mean(values: Iterable[float]) -> float | None:
    return _stats(values)["mean"]


def _rz(yaw: np.ndarray) -> np.ndarray:
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    result = np.zeros(yaw.shape + (3, 3), dtype=np.float64)
    result[..., 0, 0] = cosine
    result[..., 0, 1] = -sine
    result[..., 1, 0] = sine
    result[..., 1, 1] = cosine
    result[..., 2, 2] = 1.0
    return result


def compose_block_plans(
    base_blocks: np.ndarray,
    manipulator_blocks: np.ndarray,
    valid_blocks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose local block plans into the coordinate frame of block 0."""
    base = np.asarray(base_blocks, dtype=np.float64)
    manipulator = np.asarray(manipulator_blocks, dtype=np.float64)
    valid = np.asarray(valid_blocks, dtype=bool)
    if base.ndim != 3 or base.shape[-1] != 4:
        raise ValueError(f"Expected Base [blocks,waypoints,4], got {base.shape}")
    if manipulator.shape[:2] != base.shape[:2] or manipulator.shape[-1] < 9:
        raise ValueError(
            f"Expected Manipulator [blocks,waypoints,>=9], got {manipulator.shape}"
        )
    if valid.shape != base.shape[:2]:
        raise ValueError(f"Expected validity {base.shape[:2]}, got {valid.shape}")

    global_base = np.zeros_like(base)
    global_manipulator = np.zeros_like(manipulator)
    anchor_xy = np.zeros(2, dtype=np.float64)
    anchor_yaw = 0.0
    for block_index in range(base.shape[0]):
        rotation = _rz(np.asarray(anchor_yaw))
        local_xy = base[block_index, :, :2]
        global_xy = local_xy @ rotation[:2, :2].T + anchor_xy
        local_yaw = np.arctan2(
            base[block_index, :, 2], base[block_index, :, 3]
        )
        global_yaw = anchor_yaw + local_yaw
        global_base[block_index, :, :2] = global_xy
        global_base[block_index, :, 2] = np.sin(global_yaw)
        global_base[block_index, :, 3] = np.cos(global_yaw)

        local_position = manipulator[block_index, :, :3]
        anchor_xyz = np.asarray([anchor_xy[0], anchor_xy[1], 0.0])
        global_manipulator[block_index, :, :3] = (
            local_position @ rotation.T + anchor_xyz
        )
        local_eef_rotation = rotation6d_rows_to_matrix(
            manipulator[block_index, :, 3:9]
        )
        global_eef_rotation = np.einsum(
            "ij,hjk->hik", rotation, local_eef_rotation
        )
        global_manipulator[block_index, :, 3:6] = global_eef_rotation[:, 0]
        global_manipulator[block_index, :, 6:9] = global_eef_rotation[:, 1]
        global_manipulator[block_index, :, 9:] = manipulator[block_index, :, 9:]

        valid_indices = np.flatnonzero(valid[block_index])
        if valid_indices.size:
            endpoint = int(valid_indices[-1])
            anchor_xy = global_xy[endpoint].copy()
            anchor_yaw = float(global_yaw[endpoint])
    return global_base.astype(np.float32), global_manipulator.astype(np.float32)


def _waypoint_errors(
    base_pred: np.ndarray,
    base_gt: np.ndarray,
    manip_pred: np.ndarray,
    manip_gt: np.ndarray,
    hand_dim: int,
) -> dict[str, np.ndarray]:
    base_delta = np.asarray(base_pred[:, :2] - base_gt[:, :2], dtype=np.float64)
    base_yaw_pred = np.arctan2(base_pred[:, 2], base_pred[:, 3])
    base_yaw_gt = np.arctan2(base_gt[:, 2], base_gt[:, 3])
    eef_delta = np.asarray(manip_pred[:, :3] - manip_gt[:, :3], dtype=np.float64)
    eef_rotation_pred = rotation6d_rows_to_matrix(manip_pred[:, 3:9])
    eef_rotation_gt = rotation6d_rows_to_matrix(manip_gt[:, 3:9])
    result = {
        "base_xy_l1_m": np.abs(base_delta).sum(axis=-1),
        "base_xy_l2_m": np.linalg.norm(base_delta, axis=-1),
        "base_yaw_error_deg": np.degrees(
            np.abs(wrap_angle(base_yaw_pred - base_yaw_gt))
        ),
        "eef_position_l1_m": np.abs(eef_delta).sum(axis=-1),
        "eef_position_l2_m": np.linalg.norm(eef_delta, axis=-1),
        "eef_rotation_geodesic_deg": rotation_geodesic_deg(
            eef_rotation_pred, eef_rotation_gt
        ),
    }
    if hand_dim:
        result["hand_mae"] = np.abs(
            manip_pred[:, 9 : 9 + hand_dim] - manip_gt[:, 9 : 9 + hand_dim]
        ).mean(axis=-1)
    return result


class MetricAccumulator:
    def __init__(
        self,
        block_anchor_offsets: list[int],
        plan_local_offsets: list[int],
    ) -> None:
        self.block_anchor_offsets = block_anchor_offsets
        self.plan_local_offsets = plan_local_offsets
        self.overall: dict[str, list[float]] = defaultdict(list)
        self.by_block: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.by_global_offset: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.by_task: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.by_task_block: dict[
            tuple[str, int], dict[str, list[float]]
        ] = defaultdict(lambda: defaultdict(list))
        self.per_prediction: list[dict[str, Any]] = []
        self.predictions: dict[str, list[np.ndarray]] = defaultdict(list)
        self.num_windows = 0

    @staticmethod
    def _append(bucket: dict[str, list[float]], name: str, value: float) -> None:
        if np.isfinite(value):
            bucket[name].append(float(value))

    def _append_grouped(
        self,
        task: str,
        block_index: int,
        global_offset: int | None,
        name: str,
        value: float,
    ) -> None:
        self._append(self.overall, name, value)
        self._append(self.by_block[block_index], name, value)
        self._append(self.by_task[task], name, value)
        self._append(self.by_task_block[(task, block_index)], name, value)
        if global_offset is not None:
            self._append(self.by_global_offset[global_offset], name, value)

    def add_block(
        self,
        *,
        episode_index: int,
        root_frame_index: int,
        anchor_frame_index: int,
        task: str,
        block_index: int,
        base_pred: np.ndarray,
        base_gt: np.ndarray,
        manip_pred: np.ndarray,
        manip_gt: np.ndarray,
        valid: np.ndarray,
        hand_dim: int,
        base_prior_pred: np.ndarray | None = None,
        eef_prior_pred: np.ndarray | None = None,
        prior_waypoint_index: int | None = None,
    ) -> None:
        valid = np.asarray(valid, dtype=bool)
        errors = _waypoint_errors(
            base_pred, base_gt, manip_pred, manip_gt, hand_dim
        )
        for waypoint_index, local_offset in enumerate(self.plan_local_offsets):
            global_offset = self.block_anchor_offsets[block_index] + local_offset
            self._append_grouped(
                task,
                block_index,
                global_offset,
                "valid_waypoint_ratio",
                float(valid[waypoint_index]),
            )
            if not valid[waypoint_index]:
                continue
            for name, values in errors.items():
                self._append_grouped(
                    task,
                    block_index,
                    global_offset,
                    name,
                    float(values[waypoint_index]),
                )

        valid_indices = np.flatnonzero(valid)
        row: dict[str, Any] = {
            "episode_index": episode_index,
            "root_frame_index": root_frame_index,
            "anchor_frame_index": anchor_frame_index,
            "task": task,
            "block_index": block_index,
            "block_anchor_offset": self.block_anchor_offsets[block_index],
            "valid_waypoints": int(valid.sum()),
        }
        if valid_indices.size:
            selected = valid_indices.tolist()
            endpoint = int(valid_indices[-1])
            for name, values in errors.items():
                mean_value = float(np.mean(values[selected]))
                row[f"{name}_mean"] = mean_value
            self._append_grouped(
                task,
                block_index,
                None,
                "base_ade_m",
                float(np.mean(errors["base_xy_l2_m"][selected])),
            )
            self._append_grouped(
                task,
                block_index,
                None,
                "eef_ade_m",
                float(np.mean(errors["eef_position_l2_m"][selected])),
            )
            self._append_grouped(
                task,
                block_index,
                None,
                "base_fde_m",
                float(errors["base_xy_l2_m"][endpoint]),
            )
            self._append_grouped(
                task,
                block_index,
                None,
                "eef_fde_m",
                float(errors["eef_position_l2_m"][endpoint]),
            )

        if prior_waypoint_index is not None:
            prior_index = int(prior_waypoint_index)
            if valid[prior_index] and base_prior_pred is not None:
                prior_error = _waypoint_errors(
                    np.asarray(base_prior_pred).reshape(1, 4),
                    base_gt[prior_index : prior_index + 1],
                    manip_gt[prior_index : prior_index + 1],
                    manip_gt[prior_index : prior_index + 1],
                    0,
                )
                for source, target in (
                    ("base_xy_l2_m", "base_prior_xy_l2_m"),
                    ("base_yaw_error_deg", "base_prior_yaw_error_deg"),
                ):
                    value = float(prior_error[source][0])
                    row[target] = value
                    self._append_grouped(task, block_index, None, target, value)
            if valid[prior_index] and eef_prior_pred is not None:
                eef = np.asarray(eef_prior_pred).reshape(1, -1)
                target = manip_gt[prior_index : prior_index + 1]
                position_error = float(np.linalg.norm(eef[:, :3] - target[:, :3]))
                rotation_error = float(
                    rotation_geodesic_deg(
                        rotation6d_rows_to_matrix(eef[:, 3:9]),
                        rotation6d_rows_to_matrix(target[:, 3:9]),
                    )[0]
                )
                row["eef_prior_position_l2_m"] = position_error
                row["eef_prior_rotation_geodesic_deg"] = rotation_error
                self._append_grouped(
                    task,
                    block_index,
                    None,
                    "eef_prior_position_l2_m",
                    position_error,
                )
                self._append_grouped(
                    task,
                    block_index,
                    None,
                    "eef_prior_rotation_geodesic_deg",
                    rotation_error,
                )

        self.per_prediction.append(row)
        for name, value in (
            ("episode_index", episode_index),
            ("root_frame_index", root_frame_index),
            ("anchor_frame_index", anchor_frame_index),
            ("block_index", block_index),
        ):
            self.predictions[name].append(np.asarray(value, dtype=np.int64))
        self.predictions["base_pred"].append(np.asarray(base_pred, dtype=np.float32))
        self.predictions["base_gt"].append(np.asarray(base_gt, dtype=np.float32))
        self.predictions["manipulator_pred"].append(
            np.asarray(manip_pred, dtype=np.float32)
        )
        self.predictions["manipulator_gt"].append(
            np.asarray(manip_gt, dtype=np.float32)
        )
        self.predictions["plan_valid"].append(valid)

    def add_composed_window(
        self,
        *,
        task: str,
        base_pred: np.ndarray,
        base_gt: np.ndarray,
        manip_pred: np.ndarray,
        manip_gt: np.ndarray,
        valid: np.ndarray,
        hand_dim: int,
    ) -> None:
        pred_base_global, pred_manip_global = compose_block_plans(
            base_pred, manip_pred, valid
        )
        gt_base_global, gt_manip_global = compose_block_plans(
            base_gt, manip_gt, valid
        )
        for block_index in range(base_pred.shape[0]):
            errors = _waypoint_errors(
                pred_base_global[block_index],
                gt_base_global[block_index],
                pred_manip_global[block_index],
                gt_manip_global[block_index],
                hand_dim,
            )
            for waypoint_index, local_offset in enumerate(self.plan_local_offsets):
                if not valid[block_index, waypoint_index]:
                    continue
                global_offset = self.block_anchor_offsets[block_index] + local_offset
                for source, target in (
                    ("base_xy_l2_m", "composed_base_xy_l2_m"),
                    ("base_yaw_error_deg", "composed_base_yaw_error_deg"),
                    ("eef_position_l2_m", "composed_eef_position_l2_m"),
                    (
                        "eef_rotation_geodesic_deg",
                        "composed_eef_rotation_geodesic_deg",
                    ),
                ):
                    self._append_grouped(
                        task,
                        block_index,
                        global_offset,
                        target,
                        float(errors[source][waypoint_index]),
                    )

        flat_valid = np.asarray(valid, dtype=bool).reshape(-1)
        valid_indices = np.flatnonzero(flat_valid)
        if valid_indices.size:
            flat_errors = _waypoint_errors(
                pred_base_global.reshape(-1, pred_base_global.shape[-1]),
                gt_base_global.reshape(-1, gt_base_global.shape[-1]),
                pred_manip_global.reshape(-1, pred_manip_global.shape[-1]),
                gt_manip_global.reshape(-1, gt_manip_global.shape[-1]),
                hand_dim,
            )
            endpoint = int(valid_indices[-1])
            window_metrics = {
                "composed_base_ade_m": float(
                    np.mean(flat_errors["base_xy_l2_m"][valid_indices])
                ),
                "composed_base_fde_m": float(
                    flat_errors["base_xy_l2_m"][endpoint]
                ),
                "composed_eef_ade_m": float(
                    np.mean(flat_errors["eef_position_l2_m"][valid_indices])
                ),
                "composed_eef_fde_m": float(
                    flat_errors["eef_position_l2_m"][endpoint]
                ),
            }
            for name, value in window_metrics.items():
                self._append(self.overall, name, value)
                self._append(self.by_task[task], name, value)
        self.num_windows += 1

    @staticmethod
    def _summarize_bucket(
        bucket: Mapping[str, Iterable[float]],
    ) -> dict[str, dict[str, float | int | None]]:
        return {name: _stats(values) for name, values in sorted(bucket.items())}

    def summary(self) -> dict[str, Any]:
        primary_names = (
            "base_ade_m",
            "base_fde_m",
            "base_yaw_error_deg",
            "eef_ade_m",
            "eef_fde_m",
            "eef_rotation_geodesic_deg",
            "hand_mae",
            "composed_base_xy_l2_m",
            "composed_eef_position_l2_m",
            "composed_base_ade_m",
            "composed_base_fde_m",
            "composed_eef_ade_m",
            "composed_eef_fde_m",
            "base_prior_xy_l2_m",
        )
        return {
            "num_windows": self.num_windows,
            "num_block_predictions": len(self.per_prediction),
            "primary_metrics": {
                name: _mean(self.overall.get(name, [])) for name in primary_names
            },
            "metrics": self._summarize_bucket(self.overall),
            "per_block": [
                {
                    "block_index": block_index,
                    "anchor_offset": self.block_anchor_offsets[block_index],
                    "metrics": self._summarize_bucket(self.by_block[block_index]),
                }
                for block_index in sorted(self.by_block)
            ],
            "per_global_offset": [
                {
                    "offset": offset,
                    "metrics": self._summarize_bucket(
                        self.by_global_offset[offset]
                    ),
                }
                for offset in sorted(self.by_global_offset)
            ],
            "per_task": {
                task: self._summarize_bucket(bucket)
                for task, bucket in sorted(self.by_task.items())
            },
            "per_task_block": [
                {
                    "task": task,
                    "block_index": block_index,
                    "metrics": self._summarize_bucket(bucket),
                }
                for (task, block_index), bucket in sorted(
                    self.by_task_block.items()
                )
            ],
        }

    def save(self, output_dir: Path, metadata: dict[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = self.summary()
        summary["evaluation"] = metadata
        write_json(output_dir / "summary.json", summary)
        with (output_dir / "per_prediction_metrics.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in self.per_prediction:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        if self.predictions:
            np.savez_compressed(
                output_dir / "predictions.npz",
                **{
                    name: np.stack(values)
                    for name, values in self.predictions.items()
                },
            )
        with (output_dir / "per_block_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["block_index", "anchor_offset", "metric", "count", "mean", "median", "p90"])
            for block_index in sorted(self.by_block):
                for metric, values in sorted(self.by_block[block_index].items()):
                    stats = _stats(values)
                    writer.writerow(
                        [
                            block_index,
                            self.block_anchor_offsets[block_index],
                            metric,
                            stats["count"],
                            stats["mean"],
                            stats["median"],
                            stats["p90"],
                        ]
                    )


def _resolve_plan_spec(cfg: Any) -> tuple[list[int], list[int], int]:
    anchors = [int(value) for value in cfg.block_anchor_offsets]
    local = [int(value) for value in cfg.plan_local_offsets]
    if not anchors or anchors[0] != 0:
        raise ValueError(f"block_anchor_offsets must begin at 0, got {anchors}")
    if len(anchors) > 1:
        strides = np.diff(anchors)
        if not np.all(strides == strides[0]):
            raise ValueError(f"Cached evaluation requires uniform anchors, got {anchors}")
        block_stride = int(strides[0])
    else:
        block_stride = max(local)
    if max(local) > block_stride:
        raise ValueError("Local waypoint offsets exceed the video block stride")
    if int(cfg.num_plan_blocks) != len(anchors):
        raise ValueError("num_plan_blocks does not match block_anchor_offsets")
    if int(cfg.plan_waypoints_per_block) != len(local):
        raise ValueError("plan_waypoints_per_block does not match plan_local_offsets")
    return anchors, local, block_stride


def _resolve_stats_path(
    dataset_root: Path,
    cfg: Any,
    anchors: list[int],
    local_offsets: list[int],
    *,
    require_existing: bool = True,
) -> Path:
    configured = cfg.get("mobilemanibench_plan_stats_path")
    if configured is not None:
        configured_path = Path(str(configured)).expanduser()
        if configured_path.is_file():
            return configured_path
    dynamic = dynamic_block_plan_stats_path(dataset_root, anchors, local_offsets)
    if require_existing and not dynamic.is_file():
        raise FileNotFoundError(
            f"No matching multiblock plan statistics: {dynamic}. Run the "
            "multiblock metadata preparation step first."
        )
    return dynamic


def _dataset_kwargs(
    dataset_root: Path,
    cfg: Any,
    anchors: list[int],
    local_offsets: list[int],
) -> dict[str, Any]:
    return {
        "dataset_path": dataset_root,
        "video_backend": "decord",
        "max_manipulator_dim": int(cfg.max_manipulator_action_dim),
        "plan_transform": None,
        "split": "all",
        "label_source": str(cfg.get("mobilemanibench_plan_label_source", "dynamic")),
        "block_anchor_offsets": anchors,
        "plan_local_offsets": local_offsets,
    }


def _index_map(dataset: MobileManiBenchBlockPlanDataset) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for index, (episode_index, frame_index) in enumerate(dataset.all_steps):
        result[(int(episode_index), int(frame_index))] = index
    return result


def build_datasets(
    dataset_root: Path,
    cfg: Any,
    anchors: list[int],
    local_offsets: list[int],
    block_stride: int,
    mode: str,
) -> dict[str, Any]:
    common = _dataset_kwargs(dataset_root, cfg, anchors, local_offsets)
    needs_full_video = mode in FULL_WINDOW_MODES
    root_dataset = MobileManiBenchBlockPlanDataset(
        **common,
        load_videos=needs_full_video,
        video_delta_indices=(
            list(range(int(cfg.num_frames))) if needs_full_video else [0]
        ),
        require_full_video_window=True,
    )
    result: dict[str, Any] = {"root": root_dataset}
    if mode in INFERENCE_MODES:
        current = MobileManiBenchBlockPlanDataset(
            **common,
            load_videos=True,
            video_delta_indices=[0],
            require_full_video_window=False,
        )
        result["current"] = current
        result["current_map"] = _index_map(current)
        if mode == "gt_history_cached":
            history = MobileManiBenchBlockPlanDataset(
                **common,
                load_videos=True,
                video_delta_indices=list(range(block_stride + 1)),
                require_full_video_window=False,
            )
            result["history"] = history
            result["history_map"] = _index_map(history)
    return result


def _override_transform_stats(transform_cfg: Any, stats_path: Path) -> Any:
    transform_cfg = OmegaConf.create(OmegaConf.to_container(transform_cfg, resolve=False))
    transforms = transform_cfg.get("transforms")
    if transforms is None or not transforms:
        raise ValueError("Expected composed multiblock plan transform")
    transforms[0]["stats_path"] = str(stats_path)
    return transform_cfg


def build_transform_and_collator(cfg: Any, stats_path: Path):
    transform_cfg = _override_transform_stats(
        cfg.train_dataset.plan_transform, stats_path
    )
    model_transform = instantiate(transform_cfg)
    # DreamTransform.training controls the *sample schema*, not the model's
    # train/eval mode.  Its eval form omits supervised action fields and adds
    # a batch dimension internally, while this evaluator deliberately applies
    # the transform per sample and then uses the training data collator.  Keep
    # the transform in per-sample supervised form; the loaded model itself is
    # still in evaluation/inference mode.
    model_transform.train()
    collator = instantiate(cfg.data_collator)
    return model_transform, collator


def _merge_history_video(current: dict[str, Any], history: dict[str, Any]) -> None:
    video_keys = [key for key in history if str(key).startswith("video.")]
    if not video_keys:
        raise KeyError("History sample has no video.* fields")
    for key in video_keys:
        current[key] = history[key]


def teacher_forced_block_observation(
    root_sample: dict[str, Any],
    block_index: int,
    anchors: list[int],
    block_stride: int,
) -> dict[str, Any]:
    """Slice one causal inference input from a complete clean root window.

    Block 0 receives only the root RGB frame.  Later blocks receive exactly
    the preceding block's nine arrived frames, matching ``gt_history_cached``
    without requiring every anchor to be exposed as a separate dataset root.
    Only the current anchor state is retained semantically.  It is repeated
    across the four pre-collation slots required by the multiblock collator;
    ``prepare_inference_batch`` then keeps the first slot only.
    """
    if not 0 <= block_index < len(anchors):
        raise IndexError(f"Invalid block index {block_index} for {len(anchors)} blocks")
    anchor_offset = int(anchors[block_index])
    video_start = 0 if block_index == 0 else anchor_offset - block_stride
    video_stop = anchor_offset + 1
    if video_start < 0:
        raise ValueError(
            f"Block {block_index} has invalid history range "
            f"[{video_start},{video_stop})"
        )

    observation = dict(root_sample)
    video_keys = [
        key for key in root_sample if str(key).startswith("video.")
    ]
    if not video_keys:
        raise KeyError("Teacher-forced root sample has no video.* fields")
    for key in video_keys:
        value = root_sample[key]
        if int(value.shape[0]) < video_stop:
            raise ValueError(
                f"{key} has {value.shape[0]} frames, but block {block_index} "
                f"needs frames [{video_start},{video_stop})"
            )
        observation[key] = value[video_start:video_stop]

    state_keys = ("state.eef_position", "state.eef_rotation_rpy")
    for key in state_keys:
        if key not in root_sample:
            raise KeyError(f"Teacher-forced root sample is missing {key}")
        value = root_sample[key]
        if int(value.shape[0]) <= block_index:
            raise ValueError(
                f"{key} has {value.shape[0]} anchors, but block "
                f"{block_index} was requested"
            )
        current_state = np.asarray(value[block_index : block_index + 1])
        observation[key] = np.repeat(
            current_state,
            len(anchors),
            axis=0,
        )
    return observation


def prepare_inference_batch(
    raw_sample: dict[str, Any],
    model_transform: Any,
    collator: Any,
    flow_tokens_per_block: int,
) -> dict[str, Any]:
    transformed = model_transform(dict(raw_sample))
    batch = collator([transformed])
    if batch["state"].shape[1] < 1:
        raise ValueError("Inference sample has no current state block")
    batch["state"] = batch["state"][:, :1]
    for key in ("action", "action_mask"):
        if key in batch:
            batch[key] = batch[key][:, :flow_tokens_per_block]
    return batch


def _physical_predictions(
    output: Mapping[str, Any],
    plan_transform: MobilePlanTransform,
    manipulator_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    base_normalized = output["base_plan_pred"].detach().float().cpu()
    manip_normalized = output["manipulator_plan_pred"].detach().float().cpu()
    physical = plan_transform.unapply(
        {
            "base_action": base_normalized,
            "manipulator_action": manip_normalized,
        }
    )
    base = to_numpy(physical["base_plan"])[0]
    manipulator = to_numpy(physical["manipulator_plan"])[0]

    base_prior = None
    if "base_prior_pred" in output:
        normalized = output["base_prior_pred"].detach().float().cpu()
        dummy = torch.zeros(
            (*normalized.shape[:-1], manipulator_dim), dtype=normalized.dtype
        )
        prior_physical = plan_transform.unapply(
            {"base_action": normalized, "manipulator_action": dummy}
        )
        base_prior = to_numpy(prior_physical["base_plan"])[0, 0]

    eef_prior = None
    if "eef_prior_pred" in output:
        normalized = output["eef_prior_pred"].detach().float().cpu()
        dummy_base = torch.zeros(
            (*normalized.shape[:-1], 4), dtype=normalized.dtype
        )
        prior_physical = plan_transform.unapply(
            {"base_action": dummy_base, "manipulator_action": normalized}
        )
        eef_prior = to_numpy(prior_physical["manipulator_plan"])[0, 0]
    return base, manipulator, base_prior, eef_prior


def _prior_waypoint_index(action_head: Any, local_offsets: list[int]) -> int | None:
    if hasattr(action_head, "prior_flow_index"):
        index = int(action_head.prior_flow_index)
        if not 0 <= index < len(local_offsets):
            raise ValueError(f"Invalid prior_flow_index={index}")
        return index
    return None


def _selected_root_indices(
    root_dataset: MobileManiBenchBlockPlanDataset,
    episode_ids: set[int],
    episode_tasks: dict[int, str],
    sample_stride: int,
    max_samples: int,
) -> list[int]:
    return select_task_balanced_indices(
        root_dataset.all_steps,
        episode_ids,
        episode_tasks,
        stride=sample_stride,
        max_samples=max_samples,
    )


def _rollout_blocks(args: argparse.Namespace, num_plan_blocks: int) -> int:
    if args.mode == "single_block_reset":
        return 1
    if args.num_rollout_blocks <= 0:
        return num_plan_blocks
    if args.num_rollout_blocks > num_plan_blocks:
        raise ValueError(
            f"--num-rollout-blocks cannot exceed {num_plan_blocks}"
        )
    return args.num_rollout_blocks


def _output_dir(args: argparse.Namespace, checkpoint: Path) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return checkpoint / f"mobile_multiblock_plan_eval_{args.split}_{args.mode}"


def _checkpoint_step(checkpoint: Path) -> int | None:
    prefix = "checkpoint-"
    if checkpoint.name.startswith(prefix):
        suffix = checkpoint.name[len(prefix) :]
        if suffix.isdigit():
            return int(suffix)
    return None


def _evaluation_metadata(
    *,
    args: argparse.Namespace,
    checkpoint: Path,
    dataset_root: Path,
    split_source: str,
    episode_ids: set[int],
    root_indices: list[int],
    root_dataset: MobileManiBenchBlockPlanDataset,
    episode_tasks: dict[int, str],
    anchors: list[int],
    local_offsets: list[int],
    block_stride: int,
    rollout_blocks: int,
    world_size: int,
    stats_path: Path,
) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_step": _checkpoint_step(checkpoint),
        "dataset_root": str(dataset_root),
        "split": args.split,
        "split_source": split_source,
        "mode": args.mode,
        "mode_contract": {
            "single_block_reset": "one current RGB/state; cache reset; block 0 only",
            "episode_ordered_reset": "each block uses current RGB/state; cache reset for every block",
            "gt_history_cached": "block 0 reset, later blocks receive only the 9 RGB frames that have arrived",
            "teacher_forced_open_loop": "four Flow rollouts from one clean root window; GT arrived RGB and anchor state; cached across blocks",
            "oracle_four_block_teacher_forced": "full clean 33-frame training window; loss-only diagnostic",
        }[args.mode],
        "episode_ids": sorted(episode_ids),
        "num_root_windows": len(root_indices),
        "root_task_counts": count_tasks_for_indices(
            root_dataset.all_steps, root_indices, episode_tasks
        ),
        "sample_stride": args.sample_stride,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "num_rollout_blocks": rollout_blocks,
        "world_size": world_size,
        "control_fps": float(root_dataset.control_fps),
        "block_stride": block_stride,
        "block_anchor_offsets": anchors,
        "plan_local_offsets": local_offsets,
        "global_plan_offsets": [
            anchor + offset for anchor in anchors for offset in local_offsets
        ],
        "plan_stats_path": str(stats_path),
        "future_state_leakage": (
            False
            if args.mode in (*INFERENCE_MODES, TEACHER_FORCED_OPEN_LOOP_MODE)
            else None
        ),
        "deployment_metric": args.mode in INFERENCE_MODES,
        "teacher_forced_history": args.mode == TEACHER_FORCED_OPEN_LOOP_MODE,
    }


def run_oracle(
    *,
    args: argparse.Namespace,
    model: Any,
    dataset: MobileManiBenchBlockPlanDataset,
    root_indices: list[int],
    model_transform: Any,
    collator: Any,
    rank: int,
    world_size: int,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    loss_values: dict[str, list[float]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for ordinal, dataset_index in enumerate(root_indices, start=1):
        raw = dataset[dataset_index]
        episode_index = int(raw["episode_index"])
        frame_index = int(raw["frame_index"])
        current_seed = sample_seed(
            args.seed, [(episode_index, frame_index), (ordinal, 0)]
        )
        torch.manual_seed(current_seed)
        torch.cuda.manual_seed_all(current_seed)
        batch = collator([model_transform(dict(raw))])
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            output = model(batch)
        if rank == 0:
            row: dict[str, Any] = {
                "episode_index": episode_index,
                "frame_index": frame_index,
            }
            for name, value in output.items():
                if not torch.is_tensor(value) or value.numel() != 1:
                    continue
                scalar = float(value.detach().float().cpu())
                if not math.isfinite(scalar):
                    raise FloatingPointError(
                        f"Non-finite oracle metric {name} at "
                        f"episode={episode_index}, frame={frame_index}: {scalar}"
                    )
                if name == "loss" or name.endswith(("_loss", "_metric")):
                    loss_values[name].append(scalar)
                    row[name] = scalar
            rows.append(row)
            if ordinal == 1 or ordinal % 10 == 0 or ordinal == len(root_indices):
                print(f"[oracle] {ordinal}/{len(root_indices)}", flush=True)
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "summary.json",
            {
                "evaluation": metadata,
                "num_windows": len(rows),
                "teacher_forced_losses": {
                    name: _stats(values)
                    for name, values in sorted(loss_values.items())
                },
            },
        )
        with (output_dir / "per_window_losses.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False) + "\n")


def run_teacher_forced_open_loop(
    *,
    args: argparse.Namespace,
    model: Any,
    dataset: MobileManiBenchBlockPlanDataset,
    root_indices: list[int],
    episode_tasks: dict[int, str],
    model_transform: Any,
    collator: Any,
    plan_transform: MobilePlanTransform,
    anchors: list[int],
    local_offsets: list[int],
    block_stride: int,
    rollout_blocks: int,
    rank: int,
    world_size: int,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    """Run full Flow sampling with GT arrived RGB and anchor states."""
    accumulator = MetricAccumulator(anchors, local_offsets)
    action_head = model.action_head
    flow_tokens_per_block = 2 * len(local_offsets)
    if int(action_head.action_horizon) != flow_tokens_per_block:
        raise ValueError(
            f"Checkpoint action_horizon={action_head.action_horizon} does not "
            f"match 2*K={flow_tokens_per_block}"
        )
    prior_index = _prior_waypoint_index(action_head, local_offsets)

    for ordinal, root_index in enumerate(root_indices, start=1):
        root = dataset[root_index]
        episode_index = int(root["episode_index"])
        root_frame = int(root["frame_index"])
        task = episode_tasks[episode_index]
        hand_dim = int(root["hand_dim"])
        block_base_predictions: list[np.ndarray] = []
        block_manip_predictions: list[np.ndarray] = []
        block_base_gt: list[np.ndarray] = []
        block_manip_gt: list[np.ndarray] = []
        block_valid: list[np.ndarray] = []

        for block_index in range(rollout_blocks):
            anchor_frame = root_frame + anchors[block_index]
            raw_observation = teacher_forced_block_observation(
                root,
                block_index,
                anchors,
                block_stride,
            )
            inference_seed = sample_seed(
                args.seed,
                [(episode_index, root_frame), (block_index, anchor_frame)],
            )
            if block_index == 0:
                reset_sampler_state(action_head, inference_seed)
            else:
                action_head.seed = int(inference_seed)
            batch = prepare_inference_batch(
                raw_observation,
                model_transform,
                collator,
                flow_tokens_per_block,
            )
            expected_video_frames = 1 if block_index == 0 else block_stride + 1
            if int(batch["images"].shape[1]) != expected_video_frames:
                raise ValueError(
                    f"{args.mode} block {block_index} expected "
                    f"{expected_video_frames} RGB frames, got "
                    f"{batch['images'].shape[1]}"
                )
            if int(batch["state"].shape[1]) != 1:
                raise ValueError("Inference must expose exactly one current state")

            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                output = model.get_action(batch)
            if rank == 0:
                base_pred, manip_pred, base_prior, eef_prior = _physical_predictions(
                    output,
                    plan_transform,
                    int(action_head.manipulator_action_dim),
                )
                base_gt = to_numpy(root["base_plan"])[block_index]
                manip_gt = to_numpy(root["manipulator_plan"])[block_index]
                valid = to_numpy(root["plan_valid"])[block_index].astype(bool)
                accumulator.add_block(
                    episode_index=episode_index,
                    root_frame_index=root_frame,
                    anchor_frame_index=anchor_frame,
                    task=task,
                    block_index=block_index,
                    base_pred=base_pred,
                    base_gt=base_gt,
                    manip_pred=manip_pred,
                    manip_gt=manip_gt,
                    valid=valid,
                    hand_dim=hand_dim,
                    base_prior_pred=base_prior,
                    eef_prior_pred=eef_prior,
                    prior_waypoint_index=prior_index,
                )
                block_base_predictions.append(base_pred)
                block_manip_predictions.append(manip_pred)
                block_base_gt.append(base_gt)
                block_manip_gt.append(manip_gt)
                block_valid.append(valid)
            if world_size > 1:
                dist.barrier()

        if rank == 0:
            accumulator.add_composed_window(
                task=task,
                base_pred=np.stack(block_base_predictions),
                base_gt=np.stack(block_base_gt),
                manip_pred=np.stack(block_manip_predictions),
                manip_gt=np.stack(block_manip_gt),
                valid=np.stack(block_valid),
                hand_dim=hand_dim,
            )
            print(
                f"[eval:{args.mode}] {ordinal}/{len(root_indices)} root windows",
                flush=True,
            )

    if rank == 0:
        accumulator.save(output_dir, metadata)


def run_inference(
    *,
    args: argparse.Namespace,
    model: Any,
    datasets: dict[str, Any],
    root_indices: list[int],
    episode_tasks: dict[int, str],
    model_transform: Any,
    collator: Any,
    plan_transform: MobilePlanTransform,
    anchors: list[int],
    local_offsets: list[int],
    block_stride: int,
    rollout_blocks: int,
    rank: int,
    world_size: int,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    accumulator = MetricAccumulator(anchors, local_offsets)
    root_dataset = datasets["root"]
    current_dataset = datasets["current"]
    current_map = datasets["current_map"]
    history_dataset = datasets.get("history")
    history_map = datasets.get("history_map")
    action_head = model.action_head
    flow_tokens_per_block = 2 * len(local_offsets)
    if int(action_head.action_horizon) != flow_tokens_per_block:
        raise ValueError(
            f"Checkpoint action_horizon={action_head.action_horizon} does not "
            f"match 2*K={flow_tokens_per_block}"
        )
    prior_index = _prior_waypoint_index(action_head, local_offsets)

    for ordinal, root_index in enumerate(root_indices, start=1):
        root = root_dataset[root_index]
        episode_index = int(root["episode_index"])
        root_frame = int(root["frame_index"])
        task = episode_tasks[episode_index]
        hand_dim = int(root["hand_dim"])
        block_base_predictions: list[np.ndarray] = []
        block_manip_predictions: list[np.ndarray] = []
        block_base_gt: list[np.ndarray] = []
        block_manip_gt: list[np.ndarray] = []
        block_valid: list[np.ndarray] = []

        for block_index in range(rollout_blocks):
            anchor_frame = root_frame + anchors[block_index]
            key = (episode_index, anchor_frame)
            if key not in current_map:
                raise KeyError(f"Missing current observation for {key}")
            raw_observation = dict(current_dataset[current_map[key]])
            use_cache = args.mode == "gt_history_cached" and block_index > 0
            if use_cache:
                history_key = (episode_index, anchor_frame - block_stride)
                if history_map is None or history_key not in history_map:
                    raise KeyError(f"Missing arrived RGB history for {history_key}")
                _merge_history_video(
                    raw_observation,
                    dict(history_dataset[history_map[history_key]]),
                )

            inference_seed = sample_seed(
                args.seed,
                [(episode_index, root_frame), (block_index, anchor_frame)],
            )
            if block_index == 0 or args.mode != "gt_history_cached":
                reset_sampler_state(action_head, inference_seed)
            else:
                action_head.seed = int(inference_seed)
            batch = prepare_inference_batch(
                raw_observation,
                model_transform,
                collator,
                flow_tokens_per_block,
            )
            expected_video_frames = block_stride + 1 if use_cache else 1
            if int(batch["images"].shape[1]) != expected_video_frames:
                raise ValueError(
                    f"{args.mode} block {block_index} expected "
                    f"{expected_video_frames} RGB frames, got "
                    f"{batch['images'].shape[1]}"
                )
            if int(batch["state"].shape[1]) != 1:
                raise ValueError("Inference must expose exactly one current state")

            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                output = model.get_action(batch)
            if rank == 0:
                base_pred, manip_pred, base_prior, eef_prior = _physical_predictions(
                    output,
                    plan_transform,
                    int(model.action_head.manipulator_action_dim),
                )
                base_gt = to_numpy(root["base_plan"])[block_index]
                manip_gt = to_numpy(root["manipulator_plan"])[block_index]
                valid = to_numpy(root["plan_valid"])[block_index].astype(bool)
                accumulator.add_block(
                    episode_index=episode_index,
                    root_frame_index=root_frame,
                    anchor_frame_index=anchor_frame,
                    task=task,
                    block_index=block_index,
                    base_pred=base_pred,
                    base_gt=base_gt,
                    manip_pred=manip_pred,
                    manip_gt=manip_gt,
                    valid=valid,
                    hand_dim=hand_dim,
                    base_prior_pred=base_prior,
                    eef_prior_pred=eef_prior,
                    prior_waypoint_index=prior_index,
                )
                block_base_predictions.append(base_pred)
                block_manip_predictions.append(manip_pred)
                block_base_gt.append(base_gt)
                block_manip_gt.append(manip_gt)
                block_valid.append(valid)
            if world_size > 1:
                dist.barrier()

        if rank == 0:
            accumulator.add_composed_window(
                task=task,
                base_pred=np.stack(block_base_predictions),
                base_gt=np.stack(block_base_gt),
                manip_pred=np.stack(block_manip_predictions),
                manip_gt=np.stack(block_manip_gt),
                valid=np.stack(block_valid),
                hand_dim=hand_dim,
            )
            if ordinal == 1 or ordinal % 10 == 0 or ordinal == len(root_indices):
                print(
                    f"[eval:{args.mode}] {ordinal}/{len(root_indices)} root windows",
                    flush=True,
                )

    if rank == 0:
        accumulator.save(output_dir, metadata)


def main() -> int:
    args = parse_args()
    if args.sample_stride < 1:
        raise ValueError("--sample-stride must be >= 1")
    dataset_root = args.dataset_root.resolve()
    episode_ids, split_source = resolve_episode_split(dataset_root, args.split)
    episode_tasks = load_episode_tasks(dataset_root)

    if args.checkpoint is None:
        if not args.inspect_only:
            raise ValueError("--checkpoint is required unless --inspect-only is used")
        # Inspect-only still needs a resolved training config.  Use the checked-in
        # data YAML through Hydra-free OmegaConf loading.
        config_path = Path(
            "groot/vla/configs/data/dreamzero/mobilemanibench_multiblock_plan.yaml"
        )
        cfg = OmegaConf.load(config_path)
        cfg.mobilemanibench_data_root = str(dataset_root)
        cfg.max_manipulator_action_dim = int(cfg.max_manipulator_action_dim)
    else:
        checkpoint = args.checkpoint.resolve()
        config_path = checkpoint / "experiment_cfg/conf.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        cfg = OmegaConf.load(config_path)

    anchors, local_offsets, block_stride = _resolve_plan_spec(cfg)
    stats_path = _resolve_stats_path(
        dataset_root,
        cfg,
        anchors,
        local_offsets,
        require_existing=not args.inspect_only,
    )
    datasets = build_datasets(
        dataset_root,
        cfg,
        anchors,
        local_offsets,
        block_stride,
        args.mode,
    )
    root_dataset = datasets["root"]
    root_indices = _selected_root_indices(
        root_dataset,
        episode_ids,
        episode_tasks,
        args.sample_stride,
        args.max_samples,
    )
    rollout_blocks = _rollout_blocks(args, len(anchors))

    if args.inspect_only:
        episodes = read_jsonl(dataset_root / "meta/episodes.jsonl")
        selected_episode_rows = [
            row for row in episodes if int(row["episode_index"]) in episode_ids
        ]
        print(
            json.dumps(
                {
                    "dataset_root": str(dataset_root),
                    "split": args.split,
                    "split_source": split_source,
                    "mode": args.mode,
                    "num_episodes": len(selected_episode_rows),
                    "num_root_windows": len(root_indices),
                    "root_task_counts": count_tasks_for_indices(
                        root_dataset.all_steps, root_indices, episode_tasks
                    ),
                    "num_rollout_blocks": rollout_blocks,
                    "num_block_predictions": (
                        None
                        if args.mode == ORACLE_MODE
                        else len(root_indices) * rollout_blocks
                    ),
                    "block_anchor_offsets": anchors,
                    "plan_local_offsets": local_offsets,
                    "global_plan_offsets": [
                        anchor + offset
                        for anchor in anchors
                        for offset in local_offsets
                    ],
                    "block_stride": block_stride,
                    "stats_path": str(stats_path),
                    "stats_exist": stats_path.is_file(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    checkpoint = args.checkpoint.resolve()
    device, mesh, rank, world_size = initialize_distributed()
    model, cfg = load_model(
        checkpoint,
        device,
        mesh,
        args.num_inference_steps,
    )
    checkpoint_step = _checkpoint_step(checkpoint)
    if checkpoint_step is not None:
        # Loss ramps are ordinary Python attributes rather than checkpoint
        # tensors.  Restore their evaluation-time step explicitly.
        model.action_head.global_step = checkpoint_step
    model_transform, collator = build_transform_and_collator(cfg, stats_path)
    metadata = _evaluation_metadata(
        args=args,
        checkpoint=checkpoint,
        dataset_root=dataset_root,
        split_source=split_source,
        episode_ids=episode_ids,
        root_indices=root_indices,
        root_dataset=root_dataset,
        episode_tasks=episode_tasks,
        anchors=anchors,
        local_offsets=local_offsets,
        block_stride=block_stride,
        rollout_blocks=rollout_blocks,
        world_size=world_size,
        stats_path=stats_path,
    )
    output_dir = _output_dir(args, checkpoint)

    if args.mode == ORACLE_MODE:
        run_oracle(
            args=args,
            model=model,
            dataset=root_dataset,
            root_indices=root_indices,
            model_transform=model_transform,
            collator=collator,
            rank=rank,
            world_size=world_size,
            output_dir=output_dir,
            metadata=metadata,
        )
    elif args.mode == TEACHER_FORCED_OPEN_LOOP_MODE:
        plan_transform = MobilePlanTransform(stats_path=stats_path)
        run_teacher_forced_open_loop(
            args=args,
            model=model,
            dataset=root_dataset,
            root_indices=root_indices,
            episode_tasks=episode_tasks,
            model_transform=model_transform,
            collator=collator,
            plan_transform=plan_transform,
            anchors=anchors,
            local_offsets=local_offsets,
            block_stride=block_stride,
            rollout_blocks=rollout_blocks,
            rank=rank,
            world_size=world_size,
            output_dir=output_dir,
            metadata=metadata,
        )
    else:
        plan_transform = MobilePlanTransform(stats_path=stats_path)
        run_inference(
            args=args,
            model=model,
            datasets=datasets,
            root_indices=root_indices,
            episode_tasks=episode_tasks,
            model_transform=model_transform,
            collator=collator,
            plan_transform=plan_transform,
            anchors=anchors,
            local_offsets=local_offsets,
            block_stride=block_stride,
            rollout_blocks=rollout_blocks,
            rank=rank,
            world_size=world_size,
            output_dir=output_dir,
            metadata=metadata,
        )
    if rank == 0:
        print(f"Wrote evaluation to {output_dir}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
