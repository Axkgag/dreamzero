"""MobileManiBench RGB, camera, and pseudo-PointMap data for VGGT training."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from groot.vla.model.vggt_3d_wam.geometry import (
    invert_transform,
    pose_rpy_to_matrix,
    range_to_pointmap,
    scale_intrinsics,
)

from .lerobot import LeRobotSingleDataset, ModalityConfig


RGB_KEYS = ("video.head", "video.wrist")
DEPTH_KEYS = ("video.depth_head", "video.depth_wrist")
CAMERA_COLUMNS = (
    "observation.camera.head.pose_world",
    "observation.camera.wrist.pose_world",
)
ISAAC_X_FORWARD_FROM_OPENCV = (
    (0.0, 0.0, 1.0, 0.0),
    (-1.0, 0.0, 0.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required MobileManiBench metadata is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resize_video(video: np.ndarray, size: tuple[int, int], mode: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(video)).permute(0, 3, 1, 2).float()
    if tensor.shape[-2:] != size:
        kwargs = {"size": size, "mode": mode}
        if mode in {"bilinear", "bicubic"}:
            kwargs["align_corners"] = False
        tensor = F.interpolate(tensor, **kwargs)
    return tensor


def _edge_confidence(distance: torch.Tensor, threshold: float) -> torch.Tensor:
    dx = F.pad((distance[..., :, 1:] - distance[..., :, :-1]).abs(), (0, 1))
    dy = F.pad((distance[..., 1:, :] - distance[..., :-1, :]).abs(), (0, 0, 0, 1))
    gradient = torch.maximum(dx, dy)
    return (1 - gradient / threshold).clamp(0, 1)


class MobileManiBenchVGGTDataCollator:
    """Stack tensor-only VGGT samples without changing axis order."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        keys = features[0].keys()
        return {key: torch.stack([item[key] for item in features]) for key in keys}


class MobileManiBenchVGGTDataset(Dataset):
    """Load synchronized multi-view clips and coarse robot-centric PointMaps."""

    def __init__(
        self,
        dataset_path: str | Path,
        video_delta_indices: Sequence[int] = (0, 4, 8, 12),
        image_size: Sequence[int] = (224, 224),
        pointmap_size: Sequence[int] = (32, 32),
        video_backend: str = "decord",
        split: str = "train",
        validation_fraction: float = 0.1,
        split_seed: int = 42,
        sample_stride: int = 1,
        max_range_m: float = 5.0,
        invalid_range_margin: float = 0.04,
        edge_threshold_m: float = 0.15,
        border_pixels: int = 1,
        recover_limited_range: bool = False,
        camera_optical_transform: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.video_delta_indices = np.asarray(video_delta_indices, dtype=np.int64)
        self.image_size = tuple(int(value) for value in image_size)
        self.pointmap_size = tuple(int(value) for value in pointmap_size)
        self.max_range_m = float(max_range_m)
        self.invalid_range_margin = float(invalid_range_margin)
        self.edge_threshold_m = float(edge_threshold_m)
        self.border_pixels = int(border_pixels)
        self.recover_limited_range = bool(recover_limited_range)

        self.extensions = _read_json(self.dataset_path / "meta/extensions.json")
        self.calibration = _read_json(self.dataset_path / "meta/calibration.json")
        self.calibration_verified = self.calibration.get("status") == "verified"
        if not self.calibration_verified:
            warnings.warn(
                "MobileManiBench camera calibration is not verified; PointMap "
                "supervision must be treated as coarse.",
                stacklevel=2,
            )
        if self.extensions["depth"].get("projection_ready") is not True:
            warnings.warn(
                "Depth metadata is not projection-ready; using the configured "
                "camera optical transform and low-confidence pseudo labels.",
                stacklevel=2,
            )

        optical = (
            np.eye(4, dtype=np.float32)
            if camera_optical_transform is None
            else camera_optical_transform
        )
        self.camera_optical_transform = torch.as_tensor(optical, dtype=torch.float32)
        if self.camera_optical_transform.shape != (4, 4):
            raise ValueError("camera_optical_transform must have shape [4, 4]")

        self.observation_dataset = LeRobotSingleDataset(
            dataset_path=self.dataset_path,
            modality_configs={
                "video": ModalityConfig(
                    delta_indices=self.video_delta_indices.tolist(),
                    modality_keys=[*RGB_KEYS, *DEPTH_KEYS],
                )
            },
            embodiment_tag="xdof",
            use_global_metadata=False,
            video_backend=video_backend,
            discard_bad_trajectories=True,
        )
        self._trajectory_cache: dict[int, pd.DataFrame] = {}
        self.indices = self._build_split_indices(
            split, validation_fraction, split_seed, sample_stride
        )

        intrinsics = []
        for view in ("head", "wrist"):
            camera = self.calibration["cameras"][view]
            intrinsics.append(torch.tensor(camera["K_nominal"], dtype=torch.float32))
        self.source_intrinsics = torch.stack(intrinsics)
        head = self.calibration["cameras"]["head"]
        self.source_size = (int(head["height"]), int(head["width"]))

    def _build_split_indices(
        self,
        split: str,
        validation_fraction: float,
        split_seed: int,
        sample_stride: int,
    ) -> list[int]:
        if split not in {"train", "val", "all"}:
            raise ValueError(f"Unknown split: {split}")
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        episode_ids = sorted({int(episode) for episode, _ in self.observation_dataset.all_steps})
        generator = np.random.default_rng(split_seed)
        shuffled = np.asarray(episode_ids)
        generator.shuffle(shuffled)
        num_val = 0
        if validation_fraction > 0 and len(shuffled) > 1:
            num_val = max(1, int(round(len(shuffled) * validation_fraction)))
        validation_ids = set(shuffled[:num_val].tolist())
        trajectory_lengths = dict(
            zip(
                self.observation_dataset.trajectory_ids.tolist(),
                self.observation_dataset.trajectory_lengths.tolist(),
            )
        )
        maximum_delta = int(self.video_delta_indices.max())
        selected: list[int] = []
        for index, (episode, frame) in enumerate(self.observation_dataset.all_steps):
            in_validation = int(episode) in validation_ids
            split_matches = (
                split == "all"
                or (split == "val" and in_validation)
                or (split == "train" and not in_validation)
            )
            complete_clip = int(frame) + maximum_delta < trajectory_lengths[int(episode)]
            if split_matches and complete_clip:
                selected.append(index)
        return selected[:: max(1, int(sample_stride))]

    def __len__(self) -> int:
        return len(self.indices)

    def _trajectory(self, trajectory_id: int) -> pd.DataFrame:
        if trajectory_id not in self._trajectory_cache:
            frame = self.observation_dataset.get_trajectory_data(trajectory_id)
            required = [
                "observation.base.world",
                *CAMERA_COLUMNS,
                "frame_index",
            ]
            missing = [key for key in required if key not in frame.columns]
            if missing:
                raise KeyError(f"Geometry columns missing from episode {trajectory_id}: {missing}")
            self._trajectory_cache = {trajectory_id: frame}
        return self._trajectory_cache[trajectory_id]

    def _decode_distance(self, depth_video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gray = depth_video.median(dim=-3).values
        if self.recover_limited_range:
            gray = ((gray - 16) * (255 / 219)).clamp(0, 255)
        distance = gray * (self.max_range_m / 255)
        valid = (
            (distance > self.invalid_range_margin)
            & (distance < self.max_range_m - self.invalid_range_margin)
        )
        confidence = valid.float() * _edge_confidence(distance, self.edge_threshold_m)
        if self.border_pixels:
            border = self.border_pixels
            confidence[..., :border, :] = 0
            confidence[..., -border:, :] = 0
            confidence[..., :, :border] = 0
            confidence[..., :, -border:] = 0
        if not self.calibration_verified:
            confidence *= 0.25
        return distance, confidence

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index = self.indices[index]
        sample = self.observation_dataset[source_index]
        trajectory_id, frame_index = self.observation_dataset.all_steps[source_index]
        trajectory = self._trajectory(int(trajectory_id))
        step_indices = int(frame_index) + self.video_delta_indices
        if step_indices[-1] >= len(trajectory):
            raise RuntimeError(
                "Incomplete VGGT clip escaped dataset filtering: "
                f"episode={trajectory_id}, frame={frame_index}"
            )

        rgb_views = [
            _resize_video(sample[key], self.image_size, "bilinear") for key in RGB_KEYS
        ]
        video = torch.stack(rgb_views, dim=1).to(torch.uint8)

        base_pose = torch.tensor(
            np.asarray(trajectory.iloc[int(frame_index)]["observation.base.world"][:6]),
            dtype=torch.float32,
        )
        world_from_base = pose_rpy_to_matrix(base_pose)
        base_from_world = invert_transform(world_from_base)

        camera_transforms = []
        for column in CAMERA_COLUMNS:
            poses = torch.tensor(
                np.stack(trajectory.iloc[step_indices][column].to_numpy()),
                dtype=torch.float32,
            )
            world_from_camera = pose_rpy_to_matrix(poses)
            camera_transforms.append(
                base_from_world[None] @ world_from_camera @ self.camera_optical_transform
            )
        base_from_camera = torch.stack(camera_transforms, dim=1)

        input_intrinsics = scale_intrinsics(
            self.source_intrinsics, self.source_size, self.image_size
        )
        input_intrinsics = input_intrinsics[None].expand(len(step_indices), -1, -1, -1)
        render_intrinsics = scale_intrinsics(
            self.source_intrinsics, self.source_size, self.pointmap_size
        )
        render_intrinsics = render_intrinsics[None].expand(len(step_indices), -1, -1, -1)

        depth_views = [
            _resize_video(sample[key], self.pointmap_size, "bilinear")
            for key in DEPTH_KEYS
        ]
        distance, confidence = self._decode_distance(torch.stack(depth_views, dim=1))
        pointmap = range_to_pointmap(
            distance,
            render_intrinsics,
            base_from_camera,
        ).permute(0, 1, 4, 2, 3)

        return {
            "video": video,
            "camera_K": input_intrinsics.contiguous(),
            "T_b0_camera": base_from_camera.contiguous(),
            "pseudo_pointmap_b0": pointmap.contiguous(),
            "pointmap_valid": confidence.contiguous(),
            "episode_index": torch.tensor(int(trajectory_id), dtype=torch.long),
            "frame_index": torch.tensor(int(frame_index), dtype=torch.long),
        }
