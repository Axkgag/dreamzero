"""Block-major plans derived dynamically from canonical robot trajectories."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
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

PHASE_TO_ID = {
    "natural": -1,
    "navigation": 0,
    "approach": 1,
    "grasp": 2,
    "manipulation": 3,
}
ID_TO_PHASE = {value: name for name, value in PHASE_TO_ID.items()}


@dataclass(frozen=True, slots=True)
class _SamplingRecord:
    episode_index: int
    frame_index: int
    phase_id: int
    sampled_block_slot: int
    sampled_horizon: int
    full_window: bool


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required MobileManiBench metadata is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def coherent_block_masks(
    plan_valid: np.ndarray,
    state_valid: np.ndarray,
    latent_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make action, state and video supervision a contiguous block prefix."""
    plan_valid = np.asarray(plan_valid, dtype=np.bool_).copy()
    state_valid = np.asarray(state_valid, dtype=np.bool_).copy()
    latent_valid = np.asarray(latent_valid, dtype=np.bool_).copy()
    if plan_valid.ndim != 2:
        raise ValueError(f"plan_valid must be [blocks, waypoints], got {plan_valid.shape}")
    num_blocks = plan_valid.shape[0]
    if state_valid.shape != (num_blocks,):
        raise ValueError(
            f"state_valid must be [{num_blocks}], got {state_valid.shape}"
        )
    if len(latent_valid) == 1:
        latents_per_block = 0
        video_block_valid = np.ones(num_blocks, dtype=np.bool_)
    elif (len(latent_valid) - 1) % num_blocks:
        raise ValueError(
            "Future video latents must divide evenly across blocks: "
            f"latents={len(latent_valid)}, blocks={num_blocks}"
        )
    else:
        latents_per_block = (len(latent_valid) - 1) // num_blocks
        video_block_valid = latent_valid[1:].reshape(
            num_blocks, latents_per_block
        ).all(axis=1)
    raw_block_valid = state_valid & plan_valid.all(axis=1) & video_block_valid
    block_valid = np.logical_and.accumulate(raw_block_valid)
    plan_valid &= block_valid[:, None]
    state_valid &= block_valid
    for block_index, valid in enumerate(block_valid.tolist()):
        if not valid and latents_per_block:
            start = 1 + block_index * latents_per_block
            latent_valid[start : start + latents_per_block] = False
    return plan_valid, state_valid, latent_valid, block_valid


def variable_block_layout(
    episode_length: int,
    root: int,
    block_stride: int,
    max_blocks: int,
    success_hold_eligible: bool,
) -> tuple[int, int, int, int]:
    """Describe complete real blocks and an optional terminal hold block.

    A legal root has at least one complete real block.  If a successful episode
    ends part-way through the following block, that block may be completed with
    an explicit absorbing hold target without moving the root.
    """
    episode_length = int(episode_length)
    root = int(root)
    block_stride = int(block_stride)
    max_blocks = int(max_blocks)
    if episode_length <= 0 or block_stride <= 0 or max_blocks <= 0:
        raise ValueError("episode_length, block_stride and max_blocks must be positive")
    terminal = episode_length - 1
    remaining = terminal - root
    if remaining < block_stride:
        raise ValueError(
            f"Root {root} has no complete {block_stride}-tick block before "
            f"terminal frame {terminal}"
        )
    num_real_blocks = min(max_blocks, remaining // block_stride)
    terminal_local_offset = remaining % block_stride
    hold_block_slot = -1
    hold_ticks = 0
    if (
        success_hold_eligible
        and num_real_blocks < max_blocks
        and terminal_local_offset > 0
    ):
        hold_block_slot = num_real_blocks
        hold_ticks = block_stride - terminal_local_offset
    return (
        int(num_real_blocks),
        int(hold_block_slot),
        int(terminal_local_offset),
        int(hold_ticks),
    )


class MobileManiBenchBlockPlanDataset(Dataset):
    """Load block plans with optional target-centred phase-balanced sampling."""

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
        require_complete_first_block: bool = False,
        success_hold: dict[str, Any] | None = None,
        label_source: str = "dynamic",
        block_anchor_offsets: list[int] | tuple[int, ...] = (0, 8, 16, 24),
        plan_local_offsets: list[int] | tuple[int, ...] = (4, 8),
        eef_rotation_representation: str = EEF_ROTATION_ANCHOR_BASE_6D,
        sampling: dict[str, Any] | None = None,
        phase_index_path: str | Path | None = None,
        sampling_seed: int = 42,
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
        if self.block_anchor_offsets[0] != 0:
            raise ValueError("The first block anchor must be zero")
        if self.num_plan_blocks > 1:
            anchor_deltas = np.diff(self.block_anchor_offsets)
            if not np.all(anchor_deltas == anchor_deltas[0]):
                raise ValueError("Variable-block sampling requires evenly spaced anchors")
            self.block_stride = int(anchor_deltas[0])
        else:
            self.block_stride = int(self.plan_local_offsets.max())
        if int(self.plan_local_offsets.max()) != self.block_stride:
            raise ValueError(
                "The final local waypoint must equal the block stride: "
                f"waypoint={int(self.plan_local_offsets.max())}, "
                f"stride={self.block_stride}"
            )
        self.success_hold_config = dict(success_hold or {})
        self.success_hold_enabled = bool(
            self.success_hold_config.get("enabled", False)
        )
        self.success_hold_min_frames = int(
            self.success_hold_config.get("min_success_frames", 4)
        )
        if self.success_hold_min_frames <= 0:
            raise ValueError("success_hold.min_success_frames must be positive")
        if self.success_hold_enabled and label_source != "dynamic":
            raise ValueError("success-hold completion requires dynamic labels")
        self.control_fps = float(self.extensions["time"]["control_fps"])
        self.video_delta_indices = np.asarray(
            video_delta_indices or list(range(33)), dtype=np.int64
        )
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
                delta_indices=self.video_delta_indices.tolist(),
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
        self._episode_lengths = {
            int(trajectory_id): int(length)
            for trajectory_id, length in zip(
                self.observation_dataset.trajectory_ids,
                self.observation_dataset.trajectory_lengths,
                strict=True,
            )
        }
        self._sample_metadata: dict[str, np.ndarray] | None = None
        if sampling:
            phase_path = Path(
                phase_index_path or self.dataset_path / "meta/phase_index.jsonl"
            )
            self._install_sampling_schedule(
                sampling, phase_path, int(sampling_seed)
            )
        elif require_full_video_window:
            final_offset = int(
                max(
                    self.global_plan_offsets.max(),
                    self.video_delta_indices.max(),
                )
            )
            self.observation_dataset._all_steps = [
                (trajectory_id, frame_index)
                for trajectory_id, frame_index in self.observation_dataset._all_steps
                if int(frame_index) + final_offset
                < self._episode_lengths[int(trajectory_id)]
            ]
        elif require_complete_first_block:
            if len(self.video_delta_indices) < 9:
                raise ValueError(
                    "require_complete_first_block needs the 33-frame WAM video grid"
                )
            first_block_offset = int(
                max(self.plan_local_offsets.max(), self.video_delta_indices[8])
            )
            self.observation_dataset._all_steps = [
                (trajectory_id, frame_index)
                for trajectory_id, frame_index in self.observation_dataset._all_steps
                if int(frame_index) + first_block_offset
                < self._episode_lengths[int(trajectory_id)]
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
                if self._sample_metadata is not None:
                    self._sample_metadata = {
                        key: value[selected] for key, value in self._sample_metadata.items()
                    }
        self.merged_metadata = {"xdof": self.observation_dataset.metadata}
        self.require_full_video_window = bool(require_full_video_window)
        self._trajectory_cache: dict[
            int,
            tuple[
                pd.DataFrame,
                tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
                | None,
                bool,
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

    @staticmethod
    def _read_phase_index(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Phase-balanced sampling requires sidecar index: {path}"
            )
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def _full_window_limit(self, episode_index: int) -> int:
        final_offset = int(
            max(self.global_plan_offsets.max(), self.video_delta_indices.max())
        )
        return self._episode_lengths[episode_index] - 1 - final_offset

    def _set_sampling_schedule(self, schedule: list[_SamplingRecord]) -> None:
        if not schedule:
            raise ValueError("No valid MobileManiBench sampling windows were built")
        self.observation_dataset._all_steps = [
            (row.episode_index, row.frame_index) for row in schedule
        ]
        self._sample_metadata = {
            "phase_id": np.asarray([row.phase_id for row in schedule], np.int8),
            "block_slot": np.asarray(
                [row.sampled_block_slot for row in schedule], np.int8
            ),
            "horizon": np.asarray(
                [row.sampled_horizon for row in schedule], np.int16
            ),
            "full_window": np.asarray(
                [row.full_window for row in schedule], np.bool_
            ),
        }

    def _install_sampling_schedule(
        self, sampling: dict[str, Any], phase_index_path: Path, seed: int
    ) -> None:
        phase_ratio = float(sampling.get("phase_balanced_ratio", 0.0))
        natural_ratio = float(sampling.get("natural_ratio", 1.0 - phase_ratio))
        if not np.isclose(phase_ratio + natural_ratio, 1.0):
            raise ValueError("phase_balanced_ratio + natural_ratio must equal 1")
        if phase_ratio < 0 or natural_ratio < 0:
            raise ValueError("Sampling mixture ratios must be non-negative")
        rng = np.random.default_rng(seed)
        allowed_episodes = set(self._episode_lengths)
        records = [
            row
            for row in self._read_phase_index(phase_index_path)
            if int(row["episode_index"]) in allowed_episodes
        ]
        if not records:
            raise ValueError("Phase index has no records for the requested split")
        episode_tasks = {
            int(row["episode_index"]): str(row["task"]) for row in records
        }

        candidate_groups: dict[
            tuple[str, str, int, int],
            dict[int, list[tuple[int, int, int, int, int]]],
        ] = defaultdict(lambda: defaultdict(list))
        phase_strides = sampling.get(
            "window_stride",
            {"navigation": 4, "approach": 2, "grasp": 1, "manipulation": 1},
        )
        for row in records:
            episode_index = int(row["episode_index"])
            phase = str(row["phase"])
            task = str(row["task"])
            stride = max(1, int(phase_strides.get(phase, 1)))
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            is_success_episode = bool(row.get("is_success_episode", False))
            first_block_limit = (
                self._episode_lengths[episode_index] - 1 - self.block_stride
            )
            pair_ranges: list[tuple[int, int, int, int, int]] = []
            for block_index, anchor in enumerate(self.block_anchor_offsets.tolist()):
                for horizon_index, horizon in enumerate(
                    self.plan_local_offsets.tolist()
                ):
                    low = max(0, start - int(anchor) - int(horizon))
                    high = min(
                        end - int(anchor) - int(horizon), first_block_limit
                    )
                    if not is_success_episode:
                        high = min(
                            high,
                            self._episode_lengths[episode_index]
                            - 1
                            - int(anchor)
                            - self.block_stride,
                        )
                    if low <= high:
                        pair_ranges.append(
                            (block_index, horizon_index, int(horizon), low, high)
                        )
            for block_index, horizon_index, horizon, low, high in pair_ranges:
                candidate_groups[(task, phase, block_index, horizon_index)][
                    episode_index
                ].append(
                    (
                        low,
                        high,
                        stride,
                        int(self.block_anchor_offsets[block_index]),
                        horizon,
                    )
                )
        if not candidate_groups:
            raise ValueError("No target-centred candidates could be built")

        natural: list[_SamplingRecord] = []
        for episode_index, frame_index in self.observation_dataset._all_steps:
            episode_index = int(episode_index)
            frame_index = int(frame_index)
            first_block_limit = (
                self._episode_lengths[episode_index] - 1 - self.block_stride
            )
            if frame_index <= first_block_limit:
                natural.append(
                    _SamplingRecord(
                        episode_index=episode_index,
                        frame_index=frame_index,
                        phase_id=-1,
                        sampled_block_slot=-1,
                        sampled_horizon=-1,
                        full_window=bool(
                            frame_index <= self._full_window_limit(episode_index)
                        ),
                    )
                )
        if not natural:
            raise ValueError("No natural root candidates could be built")

        total = len(natural)
        balanced_count = int(round(total * phase_ratio))
        natural_count = total - balanced_count
        group_keys = sorted(candidate_groups)
        task_counts: dict[str, int] = defaultdict(int)
        for row in records:
            task_counts[str(row["task"])] += 1
        combinations = sorted(
            {(phase, block, horizon) for _, phase, block, horizon in group_keys}
        )
        episode_cursor: dict[tuple[str, str, int, int], int] = defaultdict(int)

        balanced: list[_SamplingRecord] = []
        for index in range(balanced_count):
            phase, block_index, horizon_index = combinations[
                index % len(combinations)
            ]
            tasks = sorted(
                task
                for task in task_counts
                if (task, phase, block_index, horizon_index) in candidate_groups
            )
            probabilities = np.asarray(
                [
                    1.0
                    if sampling.get("task_balanced", False)
                    else task_counts[task]
                    for task in tasks
                ],
                dtype=np.float64,
            )
            probabilities /= probabilities.sum()
            task = str(rng.choice(tasks, p=probabilities))
            key = (task, phase, block_index, horizon_index)
            episodes = candidate_groups[key]
            episode_ids = sorted(episodes)
            episode_index = episode_ids[episode_cursor[key] % len(episode_ids)]
            episode_cursor[key] += 1
            options = episodes[episode_index]
            option = options[int(rng.integers(len(options)))]
            (
                root_start,
                root_end,
                stride,
                anchor,
                horizon,
            ) = option
            roots = np.arange(
                int(root_start),
                int(root_end) + 1,
                int(stride),
                dtype=np.int64,
            )
            root = int(roots[int(rng.integers(len(roots)))])
            balanced.append(
                _SamplingRecord(
                    episode_index=episode_index,
                    frame_index=root,
                    phase_id=PHASE_TO_ID[key[1]],
                    sampled_block_slot=block_index,
                    sampled_horizon=int(horizon),
                    full_window=bool(
                        root <= self._full_window_limit(episode_index)
                    ),
                )
            )
        if natural_count and sampling.get("task_balanced", False):
            natural_by_task: dict[str, list[_SamplingRecord]] = defaultdict(list)
            for row in natural:
                natural_by_task[
                    episode_tasks.get(row.episode_index, "unknown")
                ].append(row)
            natural_tasks = sorted(natural_by_task)
            selected_natural = []
            for index in range(natural_count):
                task = natural_tasks[index % len(natural_tasks)]
                options = natural_by_task[task]
                selected_natural.append(
                    options[int(rng.integers(len(options)))]
                )
        elif natural_count:
            natural_indices = rng.choice(
                len(natural), size=natural_count, replace=natural_count > len(natural)
            )
            selected_natural = [natural[int(index)] for index in natural_indices]
        else:
            selected_natural = []
        schedule = balanced + selected_natural
        rng.shuffle(schedule)
        self._set_sampling_schedule(schedule)

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
        bool,
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
            hold_eligible = False
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
                if self.success_hold_enabled:
                    if "success" not in frame.columns:
                        raise KeyError(
                            "success-hold completion requires the success column"
                        )
                    success = np.asarray(
                        [
                            float(np.asarray(value).reshape(-1)[0])
                            for value in frame["success"].to_numpy()
                        ],
                        dtype=np.float64,
                    )
                    hold_eligible = bool(
                        len(success) >= self.success_hold_min_frames
                        and np.all(success[-self.success_hold_min_frames :] > 0.5)
                    )
                label_base = base
                label_eef = eef
                label_joint = joint
                if hold_eligible:
                    label_base = np.concatenate(
                        [base, np.repeat(base[-1:], self.block_stride, axis=0)]
                    )
                    label_eef = np.concatenate(
                        [eef, np.repeat(eef[-1:], self.block_stride, axis=0)]
                    )
                    label_joint = np.concatenate(
                        [joint, np.repeat(joint[-1:], self.block_stride, axis=0)]
                    )
                labels = build_dynamic_block_plan_labels(
                    label_base,
                    label_eef,
                    label_joint,
                    self.robot_schema["hand_joint_indices"],
                    self.block_anchor_offsets,
                    self.plan_local_offsets,
                    self.eef_rotation_representation,
                )
                labels = tuple(value[: len(frame)] for value in labels)
            self._trajectory_cache = {
                trajectory_id: (frame, labels, hold_eligible)
            }
        return self._trajectory_cache[trajectory_id]

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.observation_dataset[index]
        trajectory_id, frame_index = self.all_steps[index]
        frame, labels, hold_eligible = self._trajectory(int(trajectory_id))
        row = frame.iloc[int(frame_index)]
        if labels is not None:
            base = labels[0][int(frame_index)].copy()
            native_manipulator = labels[1][int(frame_index)].copy()
            block_state = labels[3][int(frame_index)].copy()
        else:
            base = np.asarray(
                row["action.plan.block.base_waypoints"], dtype=np.float32
            ).reshape(self.num_plan_blocks, self.waypoints_per_block, 4).copy()
            native_manipulator = np.asarray(
                row["action.plan.block.manipulator"], dtype=np.float32
            ).reshape(
                self.num_plan_blocks, self.waypoints_per_block, self.manipulator_dim
            ).copy()
            block_state = np.asarray(
                row["observation.plan.block.state"], dtype=np.float32
            ).reshape(self.num_plan_blocks, 6).copy()

        episode_length = self._episode_lengths[int(trajectory_id)]
        root = int(frame_index)
        anchor_indices = root + self.block_anchor_offsets
        target_indices = (
            anchor_indices[:, None] + self.plan_local_offsets[None, :]
        )
        state_valid = anchor_indices < episode_length
        plan_real_valid = state_valid[:, None] & (target_indices < episode_length)
        plan_hold_valid = np.zeros_like(plan_real_valid, dtype=np.bool_)
        (
            num_real_blocks,
            hold_block_slot,
            terminal_local_offset,
            hold_ticks,
        ) = variable_block_layout(
            episode_length,
            root,
            self.block_stride,
            self.num_plan_blocks,
            hold_eligible,
        )
        if hold_block_slot >= 0:
            plan_hold_valid[hold_block_slot] = (
                target_indices[hold_block_slot] >= episode_length
            )
        valid = plan_real_valid | plan_hold_valid
        base[~valid] = 0.0
        native_manipulator[~valid] = 0.0

        video_real_valid = root + self.video_delta_indices < episode_length
        video_hold_valid = np.zeros_like(video_real_valid, dtype=np.bool_)
        if hold_block_slot >= 0:
            hold_block_end = (hold_block_slot + 1) * self.block_stride
            video_hold_valid = (~video_real_valid) & (
                self.video_delta_indices <= hold_block_end
            )
        video_valid = video_real_valid | video_hold_valid

        def latent_mask(frame_mask: np.ndarray) -> np.ndarray:
            latent = np.empty(
                (1 + (len(frame_mask) - 1) // 4,), dtype=np.bool_
            )
            latent[0] = bool(frame_mask[0])
            for latent_index in range(1, len(latent)):
                start = 1 + 4 * (latent_index - 1)
                latent[latent_index] = bool(
                    frame_mask[start : start + 4].all()
                )
            return latent

        latent_valid = latent_mask(video_valid)
        latent_real_valid = latent_mask(video_real_valid)
        latent_hold_valid = latent_valid & ~latent_real_valid
        valid, state_valid, latent_valid, block_valid = coherent_block_masks(
            valid,
            state_valid,
            latent_valid,
        )
        plan_real_valid &= valid
        plan_hold_valid &= valid
        if len(latent_valid) > 1:
            latents_per_block = (len(latent_valid) - 1) // self.num_plan_blocks
            expanded_blocks = np.repeat(block_valid, latents_per_block)
            latent_real_valid[1:] &= expanded_blocks
            latent_hold_valid[1:] &= expanded_blocks
            valid_video_end = int(block_valid.sum()) * self.block_stride
            video_prefix_valid = self.video_delta_indices <= valid_video_end
            video_valid &= video_prefix_valid
            video_real_valid &= video_prefix_valid
            video_hold_valid &= video_prefix_valid
        block_state[~state_valid] = 0.0

        if self.require_full_video_window and (
            not plan_real_valid.all()
            or not state_valid.all()
            or not video_real_valid.all()
        ):
            raise ValueError(
                "Full-window dataset yielded invalid real supervision at "
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
        sample_metadata = self._sample_metadata
        phase_id = (
            int(sample_metadata["phase_id"][index])
            if sample_metadata is not None
            else -1
        )
        success_hold = bool(plan_hold_valid.any() or video_hold_valid.any())
        sample.update(
            {
                "base_plan": base,
                "manipulator_plan": manipulator,
                "plan_valid": valid,
                "plan_real_valid": plan_real_valid,
                "plan_hold_valid": plan_hold_valid,
                "base_dim_mask": base_dim_mask,
                "manipulator_dim_mask": manipulator_dim_mask,
                "plan_local_offsets": self.plan_local_offsets.copy(),
                "plan_time_seconds": self.plan_local_offsets.astype(np.float32)
                / self.control_fps,
                "block_anchor_offsets": self.block_anchor_offsets.copy(),
                "global_plan_offsets": self.global_plan_offsets.copy(),
                "block_state_valid": state_valid,
                "block_valid": block_valid,
                "num_valid_blocks": np.int64(block_valid.sum()),
                "num_real_blocks": np.int64(num_real_blocks),
                "video_valid": video_valid,
                "video_real_valid": video_real_valid,
                "video_hold_valid": video_hold_valid,
                "video_latent_valid": latent_valid,
                "video_latent_real_valid": latent_real_valid,
                "video_latent_hold_valid": latent_hold_valid,
                "physical_block_state": block_state,
                "eef_rotation_representation": self.eef_rotation_representation,
                "episode_index": np.int64(trajectory_id),
                "frame_index": np.int64(frame_index),
                "hand_dim": np.int64(self.hand_dim),
                "sample_phase": ID_TO_PHASE.get(phase_id, "natural"),
                "sample_task": str(row["annotation.task"]),
                "sample_phase_id": np.int64(phase_id),
                "sample_task_id": np.int64(row["task_index"]),
                "sampled_block_slot": np.int64(
                    sample_metadata["block_slot"][index]
                    if sample_metadata is not None
                    else -1
                ),
                "sampled_horizon": np.int64(
                    sample_metadata["horizon"][index]
                    if sample_metadata is not None
                    else -1
                ),
                "full_window": np.bool_(
                    sample_metadata["full_window"][index]
                    if sample_metadata is not None
                    else video_real_valid.all()
                ),
                "terminal_block_slot": np.int64(hold_block_slot),
                "terminal_local_offset": np.int64(
                    terminal_local_offset if hold_block_slot >= 0 else -1
                ),
                "hold_ticks": np.int64(hold_ticks),
                "success_hold": np.bool_(success_hold),
            }
        )
        return self.plan_transform(sample) if self.plan_transform else sample
