"""Phase-1 dataset adapter for MobileManiBench realized action plans.

The adapter intentionally leaves DreamZero's existing step-action loader alone.
It reads one already-materialized plan per parquet row, so the six waypoint
dimension is never sampled as an additional LeRobot time horizon.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .lerobot import LeRobotSingleDataset, ModalityConfig


PLAN_COLUMNS = (
    "action.plan.base_waypoints",
    "action.plan.manipulator",
    "action.plan.valid",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required MobileManiBench metadata is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class MobileManiBenchPlanDataset(Dataset):
    """Load observations and the two-branch realized action plan.

    Returned plan tensors are NumPy arrays. ``manipulator_plan`` is padded to
    ``max_manipulator_dim`` while ``manipulator_dim_mask`` records which
    dimensions belong to the current robot.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        video_delta_indices: list[int] | None = None,
        load_videos: bool = True,
        video_backend: str = "decord",
        max_manipulator_dim: int = 21,
        plan_transform: Any | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.max_manipulator_dim = int(max_manipulator_dim)
        self.plan_transform = plan_transform

        self.robot_schema = _read_json(self.dataset_path / "meta/robot_schema.json")
        self.extensions = _read_json(self.dataset_path / "meta/extensions.json")
        plan_meta = self.extensions["action_plan"]
        self.plan_offsets = np.asarray(plan_meta["waypoint_offsets"], dtype=np.int64)
        self.plan_horizon = len(self.plan_offsets)
        self.control_fps = float(self.extensions["time"]["control_fps"])
        self.hand_dim = len(self.robot_schema["hand_joint_indices"])
        self.manipulator_dim = 9 + self.hand_dim

        if tuple(plan_meta["base_shape"]) != (self.plan_horizon, 4):
            raise ValueError(f"Unexpected base plan shape: {plan_meta['base_shape']}")
        if tuple(plan_meta["manipulator_shape"]) != (
            self.plan_horizon,
            self.manipulator_dim,
        ):
            raise ValueError(
                "Manipulator metadata and robot hand indices disagree: "
                f"{plan_meta['manipulator_shape']} versus "
                f"[{self.plan_horizon}, {self.manipulator_dim}]"
            )
        if self.manipulator_dim > self.max_manipulator_dim:
            raise ValueError(
                f"max_manipulator_dim={self.max_manipulator_dim} is smaller than "
                f"the dataset dimension {self.manipulator_dim}"
            )

        modality_configs: dict[str, ModalityConfig] = {
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=["state.eef_position", "state.eef_rotation_rpy"],
            ),
            "language": ModalityConfig(
                delta_indices=[0],
                modality_keys=["annotation.task"],
            ),
        }
        if load_videos:
            modality_configs["video"] = ModalityConfig(
                delta_indices=video_delta_indices or [0],
                modality_keys=["video.head", "video.wrist"],
            )

        self.observation_dataset = LeRobotSingleDataset(
            dataset_path=self.dataset_path,
            modality_configs=modality_configs,
            embodiment_tag="xdof",
            use_global_metadata=False,
            video_backend=video_backend,
            discard_bad_trajectories=True,
        )
        # BaseExperiment persists this field beside checkpoints. Keep the same
        # mapping interface as LeRobot mixture datasets even though this adapter
        # represents exactly one embodiment root.
        self.merged_metadata = {"xdof": self.observation_dataset.metadata}
        self._trajectory_cache: dict[int, pd.DataFrame] = {}

        stats_path = self.dataset_path / "meta/plan_stats.json"
        self.plan_stats = _read_json(stats_path) if stats_path.exists() else None

    def __len__(self) -> int:
        return len(self.observation_dataset)

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        return self.observation_dataset.all_steps

    def _trajectory(self, trajectory_id: int) -> pd.DataFrame:
        if trajectory_id not in self._trajectory_cache:
            frame = self.observation_dataset.get_trajectory_data(trajectory_id)
            missing = [column for column in PLAN_COLUMNS if column not in frame.columns]
            if missing:
                raise KeyError(f"Plan columns missing from episode {trajectory_id}: {missing}")
            self._trajectory_cache = {trajectory_id: frame}
        return self._trajectory_cache[trajectory_id]

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.observation_dataset[index]
        trajectory_id, frame_index = self.all_steps[index]
        row = self._trajectory(int(trajectory_id)).iloc[int(frame_index)]

        base_plan = np.asarray(
            row["action.plan.base_waypoints"], dtype=np.float32
        ).reshape(self.plan_horizon, 4)
        native_manipulator = np.asarray(
            row["action.plan.manipulator"], dtype=np.float32
        ).reshape(self.plan_horizon, self.manipulator_dim)
        plan_valid = np.asarray(row["action.plan.valid"], dtype=np.bool_).reshape(
            self.plan_horizon
        )

        manipulator_plan = np.zeros(
            (self.plan_horizon, self.max_manipulator_dim), dtype=np.float32
        )
        manipulator_plan[:, : self.manipulator_dim] = native_manipulator
        base_dim_mask = np.ones((self.plan_horizon, 4), dtype=np.bool_)
        manipulator_dim_mask = np.zeros_like(manipulator_plan, dtype=np.bool_)
        manipulator_dim_mask[:, : self.manipulator_dim] = True

        sample.update(
            {
                "base_plan": base_plan,
                "manipulator_plan": manipulator_plan,
                "plan_valid": plan_valid,
                "base_dim_mask": base_dim_mask,
                "manipulator_dim_mask": manipulator_dim_mask,
                "plan_time_offsets": self.plan_offsets.copy(),
                "plan_time_seconds": self.plan_offsets.astype(np.float32)
                / self.control_fps,
                "episode_index": np.int64(trajectory_id),
                "frame_index": np.int64(frame_index),
                "hand_dim": np.int64(self.hand_dim),
            }
        )
        return self.plan_transform(sample) if self.plan_transform is not None else sample
