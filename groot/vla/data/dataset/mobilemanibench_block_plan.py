"""Block-major plans derived dynamically from canonical robot trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .lerobot import LeRobotSingleDataset, ModalityConfig
from ..plan_geometry import build_dynamic_block_plan_labels
from ...utils.mobile_plan_spec import (
    EEF_ROTATION_ANCHOR_BASE_6D,
    canonical_block_plan_spec,
    dynamic_block_plan_stats_path,
    manipulator_plan_dim,
)


BLOCK_PLAN_COLUMNS = (
    "action.plan.block.base_waypoints",
    "action.plan.block.manipulator",
    "action.plan.block.valid",
    "observation.plan.block.state",
    "observation.plan.block.state_valid",
)

BLOCK_PLAN_SOURCE_COLUMNS = (
    "observation.base.world",
    "observation.eef.world",
    "observation.robot_joint",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required MobileManiBench metadata is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class MobileManiBenchBlockPlanDataset(Dataset):
    """Load four complete video/action/state blocks without tail padding."""

    def __init__(
        self,
        dataset_path: str | Path,
        video_delta_indices: list[int] | None = None,
        load_videos: bool = True,
        video_backend: str = "decord",
        max_manipulator_dim: int = 21,
        plan_transform: Any | None = None,
        split: str = "all",
        split_manifest_path: str | Path | None = None,
        max_samples: int | None = None,
        require_full_video_window: bool = True,
        label_source: str = "dynamic",
        block_anchor_offsets: list[int] | tuple[int, ...] = (0, 8, 16, 24),
        plan_local_offsets: list[int] | tuple[int, ...] = (4, 8),
        eef_rotation_representation: str = EEF_ROTATION_ANCHOR_BASE_6D,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.max_manipulator_dim = int(max_manipulator_dim)
        self.plan_transform = plan_transform
        self.robot_schema = _read_json(self.dataset_path / "meta/robot_schema.json")
        self.extensions = _read_json(self.dataset_path / "meta/extensions.json")
        if label_source not in {"dynamic", "materialized"}:
            raise ValueError(f"Unknown block-plan label_source: {label_source}")
        self.label_source = label_source
        if label_source == "materialized":
            if eef_rotation_representation != EEF_ROTATION_ANCHOR_BASE_6D:
                raise ValueError(
                    "Materialized block plans only contain anchor-base rotation6d; "
                    "use label_source=dynamic for current-EEF delta rotations"
                )
            plan_meta = self.extensions.get("action_plan_block")
            if plan_meta is None:
                raise KeyError("Materialized labels require extensions.action_plan_block")
            if plan_meta.get("packing") != "block_major_base_then_manipulator":
                raise ValueError(f"Unsupported plan packing: {plan_meta.get('packing')}")
            block_anchor_offsets = plan_meta["block_anchor_offsets"]
            plan_local_offsets = plan_meta["local_waypoint_offsets"]
        spec = canonical_block_plan_spec(
            block_anchor_offsets,
            plan_local_offsets,
            eef_rotation_representation=eef_rotation_representation,
        )
        self.eef_rotation_representation = str(
            spec["eef_rotation_representation"]
        )
        self.block_anchor_offsets = np.asarray(
            spec["block_anchor_offsets"], dtype=np.int64
        )
        self.plan_local_offsets = np.asarray(
            spec["local_waypoint_offsets"], dtype=np.int64
        )
        self.global_plan_offsets = np.asarray(
            spec["global_waypoint_offsets"], dtype=np.int64
        )
        self.num_plan_blocks = len(self.block_anchor_offsets)
        self.waypoints_per_block = len(self.plan_local_offsets)
        self.control_fps = float(self.extensions["time"]["control_fps"])
        self.hand_dim = len(self.robot_schema["hand_joint_indices"])
        self.manipulator_dim = manipulator_plan_dim(
            self.hand_dim, self.eef_rotation_representation
        )
        if label_source == "materialized":
            expected_base = (self.num_plan_blocks, self.waypoints_per_block, 4)
            expected_manipulator = (
                self.num_plan_blocks,
                self.waypoints_per_block,
                self.manipulator_dim,
            )
            if tuple(plan_meta["base_shape"]) != expected_base:
                raise ValueError(f"Unexpected block Base shape: {plan_meta['base_shape']}")
            if tuple(plan_meta["manipulator_shape"]) != expected_manipulator:
                raise ValueError(
                    "Unexpected block Manipulator shape: "
                    f"{plan_meta['manipulator_shape']}"
                )
        if self.manipulator_dim > self.max_manipulator_dim:
            raise ValueError(
                f"max_manipulator_dim={self.max_manipulator_dim} is smaller than "
                f"the native dimension {self.manipulator_dim}"
            )
        if split not in {"train", "val", "all"}:
            raise ValueError(f"Unknown split: {split}")
        episode_indices = None
        if split != "all":
            manifest_path = (
                Path(split_manifest_path)
                if split_manifest_path is not None
                else self.dataset_path / "meta/plan_splits.json"
            )
            manifest = _read_json(manifest_path)
            episode_indices = manifest["splits"][split]["episode_indices"]

        modality_configs: dict[str, ModalityConfig] = {
            "state": ModalityConfig(
                delta_indices=self.block_anchor_offsets.tolist(),
                modality_keys=["state.eef_position", "state.eef_rotation_rpy"],
            ),
            "language": ModalityConfig(
                delta_indices=[0], modality_keys=["annotation.task"]
            ),
        }
        if load_videos:
            modality_configs["video"] = ModalityConfig(
                delta_indices=video_delta_indices or list(range(33)),
                modality_keys=["video.head", "video.wrist"],
            )
        self.observation_dataset = LeRobotSingleDataset(
            dataset_path=self.dataset_path,
            modality_configs=modality_configs,
            embodiment_tag="xdof",
            use_global_metadata=False,
            video_backend=video_backend,
            discard_bad_trajectories=True,
            episode_indices=episode_indices,
        )
        if require_full_video_window:
            lengths: dict[int, int] = {}
            for trajectory_id, frame_index in self.observation_dataset._all_steps:
                trajectory_id = int(trajectory_id)
                lengths[trajectory_id] = max(
                    lengths.get(trajectory_id, 0), int(frame_index) + 1
                )
            final_offset = int(
                max(
                    self.global_plan_offsets.max(),
                    max(video_delta_indices or list(range(33))),
                )
            )
            self.observation_dataset._all_steps = [
                (trajectory_id, frame_index)
                for trajectory_id, frame_index in self.observation_dataset._all_steps
                if int(frame_index) + final_offset < lengths[int(trajectory_id)]
            ]
        if max_samples is not None:
            max_samples = int(max_samples)
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            all_steps = self.observation_dataset._all_steps
            if len(all_steps) > max_samples:
                selected = np.linspace(
                    0, len(all_steps) - 1, num=max_samples, dtype=np.int64
                )
                self.observation_dataset._all_steps = [
                    all_steps[int(index)] for index in selected
                ]
        self.merged_metadata = {"xdof": self.observation_dataset.metadata}
        self.require_full_video_window = bool(require_full_video_window)
        self._trajectory_cache: dict[
            int,
            tuple[
                pd.DataFrame,
                tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
                | None,
            ],
        ] = {}
        stats_path = (
            dynamic_block_plan_stats_path(
                self.dataset_path,
                self.block_anchor_offsets,
                self.plan_local_offsets,
                eef_rotation_representation=self.eef_rotation_representation,
            )
            if self.label_source == "dynamic"
            else self.dataset_path / "meta/multiblock_plan_stats.json"
        )
        self.plan_stats = _read_json(stats_path) if stats_path.exists() else None

    def __len__(self) -> int:
        return len(self.observation_dataset)

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        return self.observation_dataset.all_steps

    def _trajectory(
        self, trajectory_id: int
    ) -> tuple[
        pd.DataFrame,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    ]:
        if trajectory_id not in self._trajectory_cache:
            frame = self.observation_dataset.get_trajectory_data(trajectory_id)
            required = (
                BLOCK_PLAN_SOURCE_COLUMNS
                if self.label_source == "dynamic"
                else BLOCK_PLAN_COLUMNS
            )
            missing = [name for name in required if name not in frame.columns]
            if missing:
                raise KeyError(
                    f"Block-plan source columns missing from episode {trajectory_id}: "
                    f"{missing}"
                )
            labels = None
            if self.label_source == "dynamic":
                base = np.stack(
                    frame["observation.base.world"].to_numpy()
                ).astype(np.float64, copy=False)
                eef = np.stack(
                    frame["observation.eef.world"].to_numpy()
                ).astype(np.float64, copy=False)
                joint_flat = np.stack(
                    frame["observation.robot_joint"].to_numpy()
                ).astype(np.float64, copy=False)
                if joint_flat.shape[1] % 3:
                    raise ValueError(
                        f"Episode {trajectory_id} joint width {joint_flat.shape[1]} "
                        "is not divisible by three"
                    )
                joint = joint_flat.reshape(len(frame), -1, 3)
                labels = build_dynamic_block_plan_labels(
                    base,
                    eef,
                    joint,
                    self.robot_schema["hand_joint_indices"],
                    self.block_anchor_offsets,
                    self.plan_local_offsets,
                    self.eef_rotation_representation,
                )
            self._trajectory_cache = {trajectory_id: (frame, labels)}
        return self._trajectory_cache[trajectory_id]

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.observation_dataset[index]
        trajectory_id, frame_index = self.all_steps[index]
        frame, labels = self._trajectory(int(trajectory_id))
        row = frame.iloc[int(frame_index)]
        if labels is not None:
            base = labels[0][int(frame_index)]
            native_manipulator = labels[1][int(frame_index)]
            valid = labels[2][int(frame_index)]
            block_state = labels[3][int(frame_index)]
            state_valid = labels[4][int(frame_index)]
        else:
            base = np.asarray(
                row["action.plan.block.base_waypoints"], dtype=np.float32
            ).reshape(self.num_plan_blocks, self.waypoints_per_block, 4)
            native_manipulator = np.asarray(
                row["action.plan.block.manipulator"], dtype=np.float32
            ).reshape(
                self.num_plan_blocks, self.waypoints_per_block, self.manipulator_dim
            )
            valid = np.asarray(
                row["action.plan.block.valid"], dtype=np.bool_
            ).reshape(self.num_plan_blocks, self.waypoints_per_block)
            state_valid = np.asarray(
                row["observation.plan.block.state_valid"], dtype=np.bool_
            ).reshape(self.num_plan_blocks)
            block_state = np.asarray(
                row["observation.plan.block.state"], dtype=np.float32
            ).reshape(self.num_plan_blocks, 6)
        if self.require_full_video_window and (
            not valid.all() or not state_valid.all()
        ):
            raise ValueError(
                "Full-window dataset yielded invalid block supervision at "
                f"episode={trajectory_id}, frame={frame_index}"
            )
        manipulator = np.zeros(
            (
                self.num_plan_blocks,
                self.waypoints_per_block,
                self.max_manipulator_dim,
            ),
            dtype=np.float32,
        )
        manipulator[..., : self.manipulator_dim] = native_manipulator
        base_dim_mask = np.ones_like(base, dtype=np.bool_)
        manipulator_dim_mask = np.zeros_like(manipulator, dtype=np.bool_)
        manipulator_dim_mask[..., : self.manipulator_dim] = True
        sample.update(
            {
                "base_plan": base,
                "manipulator_plan": manipulator,
                "plan_valid": valid,
                "base_dim_mask": base_dim_mask,
                "manipulator_dim_mask": manipulator_dim_mask,
                "plan_local_offsets": self.plan_local_offsets.copy(),
                "plan_time_seconds": self.plan_local_offsets.astype(np.float32)
                / self.control_fps,
                "block_anchor_offsets": self.block_anchor_offsets.copy(),
                "global_plan_offsets": self.global_plan_offsets.copy(),
                "block_state_valid": state_valid,
                "physical_block_state": block_state,
                "eef_rotation_representation": self.eef_rotation_representation,
                "episode_index": np.int64(trajectory_id),
                "frame_index": np.int64(frame_index),
                "hand_dim": np.int64(self.hand_dim),
            }
        )
        return self.plan_transform(sample) if self.plan_transform else sample
