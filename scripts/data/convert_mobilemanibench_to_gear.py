#!/usr/bin/env python3
"""Convert MobileManiBench releases into DreamZero-compatible LeRobot/GEAR data.

The converter deliberately has two layers:

1. A compatibility layer consumed by the current DreamZero LeRobot loader:
   - ``observation.state``: current EEF pose in the current base frame (xyz+rpy)
   - ``action``: the official next-recorded-step 7/18-D action, converted to the
     current base frame exactly as the official MobileManiVLA dataset loader does
   - head/wrist RGB plus optional depth/segmentation MP4 video modalities
   - standard LeRobot v2 / GEAR metadata

2. An extension layer kept as extra parquet columns and metadata. Current
   DreamZero ignores these columns, but they preserve future research labels:
   - future base waypoints
   - future realized EEF pose + hand configuration (manipulator plan)
   - full robot/camera/object state and controller targets
   - source provenance, calibration assumptions and quality flags

This script does not modify DreamZero source code. It only creates a new dataset.
It never mutates the MobileManiBench source tree.

Example smoke conversion (two episodes for each embodiment):

  python scripts/data/convert_mobilemanibench_to_gear.py convert \
    --input-root /mnt/yihao/datasets/MobileManiBench/MobileManipVLA_opensource \
    --output-root /mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_smoke \
    --embodiments g1 xhand \
    --max-episodes-per-embodiment 2 \
    --link-mode hardlink \
    --validate

Re-run validation and regenerate visual examples:

  python scripts/data/convert_mobilemanibench_to_gear.py validate \
    --output-root /path/to/MobileManipVLA_dreamzero_smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import pandas as pd


VIDEO_SOURCES = {
    "head": "rgb_image_head.mp4",
    "wrist": "rgb_image_arm.mp4",
    "depth_head": "depth_image_head.mp4",
    "depth_wrist": "depth_image_arm.mp4",
    "segmentation_head": "segment_image_head.mp4",
    "segmentation_wrist": "segment_image_arm.mp4",
}


G1_JOINT_NAMES = (
    "slider_basex", "slider_basey", "idx01_body_joint1", "idx02_body_joint2",
    "idx11_head_joint1", "idx12_head_joint2", "idx21_arm_l_joint1",
    "idx61_arm_r_joint1", "idx22_arm_l_joint2", "idx62_arm_r_joint2",
    "idx23_arm_l_joint3", "idx63_arm_r_joint3", "idx24_arm_l_joint4",
    "idx64_arm_r_joint4", "idx25_arm_l_joint5", "idx65_arm_r_joint5",
    "idx26_arm_l_joint6", "idx66_arm_r_joint6", "idx27_arm_l_joint7",
    "idx67_arm_r_joint7", "idx31_gripper_l_inner_joint1",
    "idx41_gripper_l_outer_joint1", "idx71_gripper_r_inner_joint1",
    "idx81_gripper_r_outer_joint1", "idx32_gripper_l_inner_joint3",
    "idx42_gripper_l_outer_joint3", "idx72_gripper_r_inner_joint3",
    "idx82_gripper_r_outer_joint3", "idx33_gripper_l_inner_joint4",
    "idx43_gripper_l_outer_joint4", "idx73_gripper_r_inner_joint4",
    "idx83_gripper_r_outer_joint4", "idx54_gripper_l_inner_joint0",
    "idx53_gripper_l_outer_joint0", "idx94_gripper_r_inner_joint0",
    "idx93_gripper_r_outer_joint0",
)

XHAND_JOINT_NAMES = (
    "slider_basex", "slider_basey", "left_arm_joint1", "right_arm_joint1",
    "left_arm_joint2", "right_arm_joint2", "left_arm_joint3", "right_arm_joint3",
    "left_arm_joint4", "right_arm_joint4", "left_arm_joint5", "right_arm_joint5",
    "left_arm_joint6", "right_arm_joint6", "left_arm_joint7", "right_arm_joint7",
    "left_hand_index_bend_joint", "left_hand_mid_joint1", "left_hand_pinky_joint1",
    "left_hand_ring_joint1", "left_hand_thumb_bend_joint",
    "right_hand_index_bend_joint", "right_hand_mid_joint1",
    "right_hand_pinky_joint1", "right_hand_ring_joint1",
    "right_hand_thumb_bend_joint", "left_hand_index_joint1",
    "left_hand_mid_joint2", "left_hand_pinky_joint2", "left_hand_ring_joint2",
    "left_hand_thumb_rota_joint1", "right_hand_index_joint1",
    "right_hand_mid_joint2", "right_hand_pinky_joint2", "right_hand_ring_joint2",
    "right_hand_thumb_rota_joint1", "left_hand_index_joint2",
    "left_hand_thumb_rota_joint2", "right_hand_index_joint2",
    "right_hand_thumb_rota_joint2",
)


@dataclass(frozen=True)
class RobotSchema:
    key: str
    source_dir: str
    robot_type: str
    action_dim: int
    joint_names: tuple[str, ...]
    arm_joint_indices: tuple[int, ...]
    hand_joint_indices: tuple[int, ...]
    hand_joint_names: tuple[str, ...]
    source_eef_link: str
    expected_body_count: int


ROBOT_SCHEMAS = {
    "g1": RobotSchema(
        key="g1",
        source_dir="G1_Robot",
        robot_type="mobilemanibench_g1",
        action_dim=7,
        joint_names=G1_JOINT_NAMES,
        arm_joint_indices=(0, 1, 7, 9, 11, 13, 15, 17, 19),
        hand_joint_indices=(23,),
        hand_joint_names=("idx81_gripper_r_outer_joint1",),
        source_eef_link="gripper_r_center_link",
        expected_body_count=48,
    ),
    "xhand": RobotSchema(
        key="xhand",
        source_dir="XHand_Robot",
        robot_type="mobilemanibench_xhand",
        action_dim=18,
        joint_names=XHAND_JOINT_NAMES,
        arm_joint_indices=(0, 1, 3, 5, 7, 9, 11, 13, 15),
        hand_joint_indices=(21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 38, 39),
        hand_joint_names=(
            "right_hand_index_bend_joint", "right_hand_mid_joint1",
            "right_hand_pinky_joint1", "right_hand_ring_joint1",
            "right_hand_thumb_bend_joint", "right_hand_index_joint1",
            "right_hand_mid_joint2", "right_hand_pinky_joint2",
            "right_hand_ring_joint2", "right_hand_thumb_rota_joint1",
            "right_hand_index_joint2", "right_hand_thumb_rota_joint2",
        ),
        source_eef_link="xhand_right_base_link",
        expected_body_count=47,
    ),
}


def log(message: str) -> None:
    print(message, flush=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(value), f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_fraction(raw: str | None) -> float | None:
    if not raw or raw in {"0/0", "N/A"}:
        return None
    try:
        return float(Fraction(raw))
    except (ValueError, ZeroDivisionError):
        return None


def probe_video(path: Path, count_frames: bool = True) -> dict[str, Any]:
    entries = (
        "stream=width,height,codec_name,pix_fmt,color_range,avg_frame_rate,"
        "r_frame_rate,nb_frames,nb_read_frames"
    )
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", entries, "-of", "json",
    ]
    if count_frames:
        cmd.insert(5, "-count_frames")
    cmd.append(str(path))
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found: {path}")
    stream = streams[0]
    fps = parse_fraction(stream.get("avg_frame_rate")) or parse_fraction(stream.get("r_frame_rate"))
    frame_count = stream.get("nb_read_frames") or stream.get("nb_frames")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "codec": stream.get("codec_name", "unknown"),
        "pix_fmt": stream.get("pix_fmt", "unknown"),
        "color_range": stream.get("color_range", "unknown"),
        "fps": float(fps) if fps else None,
        "frames": int(frame_count) if frame_count not in {None, "N/A"} else None,
    }


def euler_rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Convert roll-X, pitch-Y, yaw-Z angles to Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    rpy = np.asarray(rpy, dtype=np.float64)
    roll, pitch, yaw = np.moveaxis(rpy, -1, 0)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    result = np.empty(rpy.shape[:-1] + (3, 3), dtype=np.float64)
    result[..., 0, 0] = cy * cp
    result[..., 0, 1] = cy * sp * sr - sy * cr
    result[..., 0, 2] = cy * sp * cr + sy * sr
    result[..., 1, 0] = sy * cp
    result[..., 1, 1] = sy * sp * sr + cy * cr
    result[..., 1, 2] = sy * sp * cr - cy * sr
    result[..., 2, 0] = -sp
    result[..., 2, 1] = cp * sr
    result[..., 2, 2] = cp * cr
    return result


def matrix_to_euler_rpy(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    roll = np.arctan2(matrix[..., 2, 1], matrix[..., 2, 2])
    pitch = np.arctan2(
        -matrix[..., 2, 0],
        np.sqrt(matrix[..., 2, 1] ** 2 + matrix[..., 2, 2] ** 2),
    )
    yaw = np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0])
    return np.stack([roll, pitch, yaw], axis=-1)


def relative_pose(
    target_position_w: np.ndarray,
    target_rpy_w: np.ndarray,
    frame_position_w: np.ndarray,
    frame_rpy_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_rotation_w = euler_rpy_to_matrix(target_rpy_w)
    frame_rotation_w = euler_rpy_to_matrix(frame_rpy_w)
    position_local = np.einsum(
        "...ji,...j->...i", frame_rotation_w, target_position_w - frame_position_w
    )
    rotation_local = np.einsum("...ji,...jk->...ik", frame_rotation_w, target_rotation_w)
    rpy_local = matrix_to_euler_rpy(rotation_local)
    return position_local, rpy_local, rotation_local


def transform_delta_action_to_base(action_world: np.ndarray, base_pose_world: np.ndarray) -> np.ndarray:
    """Reproduce MobileManiBench's official world-action-to-local-base transform."""
    result = np.asarray(action_world, dtype=np.float64).copy()
    base_rotation = euler_rpy_to_matrix(base_pose_world[:, 3:6])
    result[:, :3] = np.einsum("tji,tj->ti", base_rotation, result[:, :3])
    action_rotation = euler_rpy_to_matrix(result[:, 3:6])
    local_rotation = np.einsum("tji,tjk->tik", base_rotation, action_rotation)
    result[:, 3:6] = matrix_to_euler_rpy(local_rotation)
    return result.astype(np.float32)


def build_core_state(robot_hand: np.ndarray, robot_base: np.ndarray) -> np.ndarray:
    position, rpy, _ = relative_pose(
        robot_hand[:, :3], robot_hand[:, 3:6], robot_base[:, :3], robot_base[:, 3:6]
    )
    return np.concatenate([position, rpy], axis=-1).astype(np.float32)


def build_plan_labels(
    robot_base: np.ndarray,
    robot_hand: np.ndarray,
    robot_joint: np.ndarray,
    hand_joint_indices: Sequence[int],
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build anchor-relative future realized base/manipulator plans."""
    length = robot_base.shape[0]
    horizon = len(offsets)
    target_indices = np.arange(length, dtype=np.int64)[:, None] + offsets[None, :]
    valid = target_indices < length
    safe_indices = np.minimum(target_indices, length - 1)

    anchor_pos = robot_base[:, :3]
    anchor_rot = euler_rpy_to_matrix(robot_base[:, 3:6])

    future_base_pos = robot_base[safe_indices, :3]
    future_base_rot = euler_rpy_to_matrix(robot_base[safe_indices, 3:6])
    base_rel_pos = np.einsum(
        "tji,thj->thi", anchor_rot, future_base_pos - anchor_pos[:, None, :]
    )
    base_rel_rot = np.einsum("tji,thjk->thik", anchor_rot, future_base_rot)
    base_yaw = np.arctan2(base_rel_rot[..., 1, 0], base_rel_rot[..., 0, 0])
    base_plan = np.stack(
        [base_rel_pos[..., 0], base_rel_pos[..., 1], np.sin(base_yaw), np.cos(base_yaw)],
        axis=-1,
    )

    future_eef_pos = robot_hand[safe_indices, :3]
    future_eef_rot = euler_rpy_to_matrix(robot_hand[safe_indices, 3:6])
    eef_rel_pos = np.einsum(
        "tji,thj->thi", anchor_rot, future_eef_pos - anchor_pos[:, None, :]
    )
    eef_rel_rot = np.einsum("tji,thjk->thik", anchor_rot, future_eef_rot)
    # The convention is the first two matrix rows, matching common rotation-6D utilities.
    eef_rotation_6d = eef_rel_rot[..., :2, :].reshape(length, horizon, 6)
    joint_position = robot_joint[..., 0]
    hand_configuration = joint_position[safe_indices][..., list(hand_joint_indices)]
    manipulator_plan = np.concatenate(
        [eef_rel_pos, eef_rotation_6d, hand_configuration], axis=-1
    )

    base_plan[~valid] = 0.0
    manipulator_plan[~valid] = 0.0
    return (
        base_plan.astype(np.float32),
        manipulator_plan.astype(np.float32),
        valid.astype(bool),
    )


def discover_episodes(root: Path, schema: RobotSchema, limit: int) -> list[Path]:
    robot_root = root / schema.source_dir
    if not robot_root.is_dir():
        raise FileNotFoundError(f"Missing embodiment directory: {robot_root}")
    episodes: list[Path] = []
    for current, dirs, files in os.walk(robot_root):
        dirs.sort()
        files.sort()
        if "state_infos.pkl" in files:
            episodes.append(Path(current) / "state_infos.pkl")
            if limit > 0 and len(episodes) >= limit:
                break
    if not episodes:
        raise ValueError(f"No state_infos.pkl files found under {robot_root}")
    return episodes


def parse_source_taxonomy(state_path: Path, input_root: Path) -> dict[str, Any]:
    relative = state_path.relative_to(input_root)
    parts = relative.parts
    train_index = next((i for i, p in enumerate(parts) if p.startswith("train_")), None)
    traj_index = next((i for i, p in enumerate(parts) if p.startswith("traj_")), None)
    episode_name = state_path.parent.name
    if train_index is None or traj_index is None or len(parts) < 5:
        raise ValueError(f"Unexpected MobileManiBench path: {relative}")
    return {
        "source_relative_path": relative.as_posix(),
        "robot": parts[0],
        "raw_task": parts[1],
        "asset_type": parts[2],
        "object_group": parts[3],
        "object_path_parts": list(parts[4:train_index]),
        "train_split": parts[train_index],
        "trajectory": parts[traj_index],
        "episode": episode_name,
    }


def normalized_task(taxonomy: dict[str, Any], scene_info: dict[str, Any]) -> str:
    raw_task = str(taxonomy["raw_task"]).lower()
    group = str(taxonomy["object_group"]).lower()
    asset_type = str(taxonomy["asset_type"]).lower()
    if asset_type == "ycb" or group == "ycb":
        skill = "pick"
    elif group in {"cart", "chair"}:
        skill = "pull" if raw_task == "open" else "push"
    else:
        skill = raw_task
    object_name = str(scene_info.get("object") or group).replace("_", " ")
    return f"{skill} {object_name}".strip()


def load_pickle_trusted(path: Path) -> dict[str, Any]:
    # Pickle is unsafe for untrusted inputs. This converter is only for the trusted official release.
    with path.open("rb") as f:
        value = pickle.load(f)
    if not isinstance(value, dict):
        raise TypeError(f"Expected dict in {path}, got {type(value)!r}")
    return value


def load_scene_info(state_path: Path) -> dict[str, Any]:
    scene_path = state_path.parent.parent / "scene_infos.json"
    if not scene_path.exists():
        return {}
    with scene_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_source_arrays(data: dict[str, Any], schema: RobotSchema, path: Path) -> int:
    required = {
        "time", "success", "action", "camera_head_pose", "camera_arm_pose", "object",
        "robot_base", "robot_hand", "robot_body", "robot_joint", "robot_joint_target", "init",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    lengths = {
        key: len(np.asarray(value))
        for key, value in data.items()
        if key != "init" and hasattr(value, "__len__")
    }
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"{path}: inconsistent array lengths {lengths}")
    length = next(iter(unique_lengths))
    action = np.asarray(data["action"])
    joints = np.asarray(data["robot_joint"])
    bodies = np.asarray(data["robot_body"])
    if action.shape != (length, schema.action_dim):
        raise ValueError(f"{path}: action shape {action.shape}, expected {(length, schema.action_dim)}")
    if joints.shape != (length, len(schema.joint_names), 3):
        raise ValueError(
            f"{path}: robot_joint shape {joints.shape}, expected {(length, len(schema.joint_names), 3)}"
        )
    if bodies.shape[1] != schema.expected_body_count:
        raise ValueError(
            f"{path}: body count {bodies.shape[1]}, expected {schema.expected_body_count}"
        )
    for key in required.difference({"init"}):
        array = np.asarray(data[key])
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError(f"{path}: non-finite values in {key}")
    return length


def create_video_link(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError as exc:
            raise OSError(
                f"Hardlink failed for {source} -> {destination}. Use --link-mode copy or symlink."
            ) from exc
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(mode)


def list_column(array: np.ndarray) -> list[list[Any]]:
    return [row.tolist() for row in np.asarray(array)]


def compute_stats(arrays: Sequence[np.ndarray]) -> dict[str, list[float]]:
    values = np.concatenate([np.asarray(a, dtype=np.float64) for a in arrays], axis=0)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def calibration_metadata(schema: RobotSchema) -> dict[str, Any]:
    return {
        "status": "nominal_unverified",
        "warning": (
            "K is derived from focal_length=18, horizontal_aperture=36 and 520px. "
            "Verify principal point and camera-to-OpenCV optical convention before projection losses."
        ),
        "cameras": {
            view: {
                "source_pose_field": f"camera_{'head' if view == 'head' else 'arm'}_pose",
                "model": "pinhole",
                "width": 520,
                "height": 520,
                "focal_length": 18.0,
                "horizontal_aperture": 36.0,
                "K_nominal": [[260.0, 0.0, 259.5], [0.0, 260.0, 259.5], [0.0, 0.0, 1.0]],
                "K_verified": False,
                "pose_semantics": "camera position xyz + world-frame roll/pitch/yaw",
                "optical_frame_transform_verified": False,
            }
            for view in ("head", "wrist")
        },
        "source_eef_link": schema.source_eef_link,
        "target_tcp": schema.source_eef_link,
        "T_source_eef_target_tcp": np.eye(4, dtype=np.float32),
        "tcp_transform_verified": False,
    }


def robot_schema_metadata(schema: RobotSchema) -> dict[str, Any]:
    return {
        "embodiment": schema.robot_type,
        "dreamzero_compatibility_tag": "xdof",
        "joint_names": list(schema.joint_names),
        "joint_count": len(schema.joint_names),
        "expected_body_count": schema.expected_body_count,
        "robot_body_width_per_link": 12,
        "arm_joint_indices": list(schema.arm_joint_indices),
        "arm_joint_names": [schema.joint_names[i] for i in schema.arm_joint_indices],
        "hand_joint_indices": list(schema.hand_joint_indices),
        "hand_joint_names": list(schema.hand_joint_names),
        "source_eef_link": schema.source_eef_link,
        "target_tcp": schema.source_eef_link,
        "indices_source": "official MobileManiBench robot environment configuration",
        "warning": "Do not infer hand joints from the final D array positions.",
    }


def extensions_metadata(schema: RobotSchema, offsets: Sequence[int]) -> dict[str, Any]:
    hand_dim = len(schema.hand_joint_indices)
    return {
        "version": 1,
        "core_compatibility": {
            "dreamzero_code_changes_required": False,
            "embodiment_tag": "xdof",
            "core_state": "current EEF pose relative to current base, xyz+rpy",
            "core_action": "official next-recorded-step local-base normalized action",
            "relative_action": False,
        },
        "time": {
            "control_fps": 30.0,
            "timestamp_semantics": "media clock for current DreamZero video seeking",
            "control_timestamp_field": "control_timestamp",
            "alignment_authority": "frame_index",
        },
        "action_plan": {
            "semantics": "future realized trajectory",
            "anchor_frame": "current base B(t)",
            "waypoint_offsets": list(offsets),
            "base_shape": [len(offsets), 4],
            "base_encoding": ["x_m", "y_m", "sin_yaw", "cos_yaw"],
            "manipulator_shape": [len(offsets), 9 + hand_dim],
            "manipulator_encoding": [
                "eef_x_m", "eef_y_m", "eef_z_m",
                "rotation_6d_first_two_rows[6]", f"hand_joint_position[{hand_dim}]",
            ],
            "valid_field": "action.plan.valid",
            "command_shift_applies_to_plan": False,
        },
        "depth": {
            "source": "depth_image_{head,arm}.mp4",
            "quality": "lossy_h264_pseudo_range_depth",
            "source_semantics": "Isaac distance_to_camera; camera-ray range, not optical-axis z",
            "nominal_decode": "D_m = 5.0 * gray / 255 after verifying full/limited color range",
            "clip_m": [0.0, 5.0],
            "quantization_step_m_before_codec": 5.0 / 255.0,
            "projection_ready": False,
        },
        "extra_columns": {
            "action.plan.base_waypoints": {"shape": [len(offsets) * 4], "role": "future_target"},
            "action.plan.manipulator": {
                "shape": [len(offsets) * (9 + hand_dim)], "role": "future_target"
            },
            "action.plan.valid": {"shape": [len(offsets)], "role": "mask"},
            "observation.robot_joint": {
                "shape": [len(schema.joint_names) * 3], "role": "auxiliary_state"
            },
            "observation.robot_body": {
                "shape": [schema.expected_body_count * 12],
                "role": "auxiliary_link_state_for_robot_geometry",
            },
            "supervision.robot_joint_target": {
                "shape": [len(schema.joint_names)], "role": "auxiliary_target"
            },
            "observation.camera.head.pose_world": {"shape": [6], "role": "calibration_state"},
            "observation.camera.wrist.pose_world": {"shape": [6], "role": "calibration_state"},
            "supervision.object": {"shape": [9], "role": "privileged_target_only"},
        },
    }


def build_modality(schema: RobotSchema) -> dict[str, Any]:
    return {
        "state": {
            "eef_position": {
                "original_key": "observation.state", "start": 0, "end": 3,
                "rotation_type": None, "absolute": True, "dtype": "float32", "range": None,
            },
            "eef_rotation_rpy": {
                "original_key": "observation.state", "start": 3, "end": 6,
                "rotation_type": "euler_angles_rpy", "absolute": True,
                "dtype": "float32", "range": None,
            },
        },
        "action": {
            "eef_delta_position_normalized": {
                "original_key": "action", "start": 0, "end": 3,
                "rotation_type": None, "absolute": False, "dtype": "float32", "range": [-1.0, 1.0],
            },
            "eef_delta_rotation_rpy_normalized": {
                "original_key": "action", "start": 3, "end": 6,
                "rotation_type": "euler_angles_rpy", "absolute": False,
                "dtype": "float32", "range": None,
            },
            "hand_target_normalized": {
                "original_key": "action", "start": 6, "end": schema.action_dim,
                "rotation_type": None, "absolute": True, "dtype": "float32", "range": [-1.0, 1.0],
            },
        },
        "video": {
            name: {"original_key": f"observation.images.{name}"} for name in VIDEO_SOURCES
        },
        "annotation": {"task": {"original_key": "task_index"}},
    }


def dataframe_for_episode(
    data: dict[str, Any],
    schema: RobotSchema,
    episode_index: int,
    global_start_index: int,
    task_index: int,
    task: str,
    media_fps: float,
    control_fps: float,
    offsets: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    length = len(data["action"])
    robot_base = np.asarray(data["robot_base"], dtype=np.float64)
    robot_hand = np.asarray(data["robot_hand"], dtype=np.float64)
    robot_joint = np.asarray(data["robot_joint"], dtype=np.float64)

    core_state = build_core_state(robot_hand, robot_base)
    raw_action = np.asarray(data["action"], dtype=np.float64)
    shifted_action = np.concatenate([raw_action[1:], raw_action[-1:]], axis=0)
    core_action = transform_delta_action_to_base(shifted_action, robot_base[:, :6])
    source_action_index = np.minimum(np.arange(length) + 1, length - 1)
    base_plan, manipulator_plan, plan_valid = build_plan_labels(
        robot_base, robot_hand, robot_joint, schema.hand_joint_indices, offsets
    )

    frame_index = np.arange(length, dtype=np.int64)
    source_time = np.asarray(data["time"]).reshape(length, -1)[:, 0]
    success = np.asarray(data["success"]).reshape(length, -1)[:, 0]
    columns: dict[str, Any] = {
        "observation.state": list_column(core_state),
        "action": list_column(core_action),
        "timestamp": frame_index.astype(np.float64) / media_fps,
        "control_timestamp": frame_index.astype(np.float64) / control_fps,
        "source_step": source_time.astype(np.int64),
        "frame_index": frame_index,
        "episode_index": np.full(length, episode_index, dtype=np.int64),
        "index": np.arange(global_start_index, global_start_index + length, dtype=np.int64),
        "task_index": np.full(length, task_index, dtype=np.int64),
        "annotation.task": [task] * length,
        "next.done": frame_index == length - 1,
        "is_terminal": frame_index == length - 1,
        "is_first": frame_index == 0,
        "success": success.astype(np.float32),
        "action.source_index": source_action_index.astype(np.int64),
        "action.raw_world_normalized": list_column(raw_action.astype(np.float32)),
        "action.plan.base_waypoints": list_column(base_plan.reshape(length, -1)),
        "action.plan.manipulator": list_column(manipulator_plan.reshape(length, -1)),
        "action.plan.valid": list_column(plan_valid),
        "observation.base.world": list_column(robot_base.astype(np.float32)),
        "observation.eef.world": list_column(robot_hand.astype(np.float32)),
        "observation.robot_joint": list_column(robot_joint.reshape(length, -1).astype(np.float32)),
        "observation.robot_body": list_column(
            np.asarray(data["robot_body"], dtype=np.float32).reshape(length, -1)
        ),
        "supervision.robot_joint_target": list_column(
            np.asarray(data["robot_joint_target"], dtype=np.float32)
        ),
        "observation.camera.head.pose_world": list_column(
            np.asarray(data["camera_head_pose"], dtype=np.float32)
        ),
        "observation.camera.wrist.pose_world": list_column(
            np.asarray(data["camera_arm_pose"], dtype=np.float32)
        ),
        "supervision.object": list_column(np.asarray(data["object"], dtype=np.float32)),
    }
    arrays = {
        "state": core_state,
        "action": core_action,
        "base_plan": base_plan,
        "manipulator_plan": manipulator_plan,
        "plan_valid": plan_valid,
    }
    return pd.DataFrame(columns), arrays


def video_feature(probe: dict[str, Any], is_depth: bool) -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [probe["height"], probe["width"], 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": probe["fps"],
            "video.codec": probe["codec"],
            "video.pix_fmt": probe["pix_fmt"],
            "video.color_range": probe["color_range"],
            "video.is_depth_map": is_depth,
            "has_audio": False,
        },
    }


def convert_embodiment(
    input_root: Path,
    output_root: Path,
    schema: RobotSchema,
    source_paths: Sequence[Path],
    offsets: np.ndarray,
    link_mode: str,
    control_fps: float,
) -> Path:
    dataset_root = output_root / schema.key
    if dataset_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {dataset_root}")
    (dataset_root / "meta").mkdir(parents=True)

    tasks: dict[str, int] = {}
    episode_rows: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []
    source_episode_rows: list[dict[str, Any]] = []
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    first_video_probes: dict[str, dict[str, Any]] = {}
    total_frames = 0

    for episode_index, state_path in enumerate(source_paths):
        log(f"[{schema.key}] episode {episode_index + 1}/{len(source_paths)}: {state_path}")
        data = load_pickle_trusted(state_path)
        length = validate_source_arrays(data, schema, state_path)
        scene_info = load_scene_info(state_path)
        taxonomy = parse_source_taxonomy(state_path, input_root)
        task = normalized_task(taxonomy, scene_info)
        task_index = tasks.setdefault(task, len(tasks))

        probes: dict[str, dict[str, Any]] = {}
        for video_name, source_name in VIDEO_SOURCES.items():
            source_video = state_path.parent / source_name
            if not source_video.exists():
                raise FileNotFoundError(source_video)
            probe = probe_video(source_video)
            probes[video_name] = probe
            if probe["frames"] != length:
                raise ValueError(
                    f"{source_video}: video frames={probe['frames']} but state length={length}"
                )
            if probe["width"] != 520 or probe["height"] != 520:
                raise ValueError(f"{source_video}: unexpected resolution {probe['width']}x{probe['height']}")
        media_fps_values = {round(p["fps"], 6) for p in probes.values() if p["fps"] is not None}
        if len(media_fps_values) != 1:
            raise ValueError(f"{state_path}: mismatched media FPS values {media_fps_values}")
        media_fps = float(next(iter(media_fps_values)))
        if not first_video_probes:
            first_video_probes = probes

        dataframe, arrays = dataframe_for_episode(
            data=data,
            schema=schema,
            episode_index=episode_index,
            global_start_index=total_frames,
            task_index=task_index,
            task=task,
            media_fps=media_fps,
            control_fps=control_fps,
            offsets=offsets,
        )
        chunk = episode_index // 1000
        parquet_path = dataset_root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(parquet_path, index=False, engine="pyarrow", compression="zstd")

        for video_name, source_name in VIDEO_SOURCES.items():
            source_video = state_path.parent / source_name
            destination = (
                dataset_root
                / f"videos/chunk-{chunk:03d}/observation.images.{video_name}"
                / f"episode_{episode_index:06d}.mp4"
            )
            create_video_link(source_video, destination, link_mode)

        source_init = data.get("init", {})
        source_episode_rows.append({
            "episode_index": episode_index,
            "source_taxonomy": taxonomy,
            "scene": scene_info,
            "init": source_init,
            "source_eef_link": schema.source_eef_link,
        })
        source_manifest.append({
            "episode_index": episode_index,
            "source_state_path": str(state_path),
            "source_relative_path": taxonomy["source_relative_path"],
            "state_size_bytes": state_path.stat().st_size,
            "state_sha256": sha256_file(state_path),
            "video_files": {
                name: {
                    "path": str(state_path.parent / source_name),
                    "size_bytes": (state_path.parent / source_name).stat().st_size,
                    "probe": probes[name],
                }
                for name, source_name in VIDEO_SOURCES.items()
            },
        })
        episode_rows.append({
            "episode_index": episode_index,
            "tasks": [task],
            "length": length,
            "source_relative_path": taxonomy["source_relative_path"],
        })
        all_states.append(arrays["state"])
        all_actions.append(arrays["action"])
        total_frames += length

    task_rows = [
        {"task_index": index, "task": task}
        for task, index in sorted(tasks.items(), key=lambda item: item[1])
    ]
    write_jsonl(dataset_root / "meta/tasks.jsonl", task_rows)
    write_jsonl(dataset_root / "meta/episodes.jsonl", episode_rows)
    write_jsonl(dataset_root / "meta/source_manifest.jsonl", source_manifest)
    write_jsonl(dataset_root / "meta/source_episodes.jsonl", source_episode_rows)

    write_json(dataset_root / "meta/modality.json", build_modality(schema))
    write_json(dataset_root / "meta/embodiment.json", {"embodiment_tag": "xdof"})
    write_json(dataset_root / "meta/robot_schema.json", robot_schema_metadata(schema))
    write_json(dataset_root / "meta/calibration.json", calibration_metadata(schema))
    write_json(dataset_root / "meta/extensions.json", extensions_metadata(schema, offsets.tolist()))
    write_json(dataset_root / "meta/relative_stats_dreamzero.json", {})
    write_json(
        dataset_root / "meta/stats.json",
        {
            "observation.state": compute_stats(all_states),
            "action": compute_stats(all_actions),
        },
    )

    info_features: dict[str, Any] = {
        f"observation.images.{name}": video_feature(
            first_video_probes[name], name.startswith("depth_")
        )
        for name in VIDEO_SOURCES
    }
    hand_dim = len(schema.hand_joint_indices)
    info_features.update({
        "observation.state": {"dtype": "float32", "shape": [6], "names": ["eef_xyz", "eef_rpy"]},
        "action": {"dtype": "float32", "shape": [schema.action_dim], "names": ["eef_delta_6d", "hand_target"]},
        "timestamp": {"dtype": "float64", "shape": [1]},
        "control_timestamp": {"dtype": "float64", "shape": [1]},
        "source_step": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
        "is_terminal": {"dtype": "bool", "shape": [1]},
        "is_first": {"dtype": "bool", "shape": [1]},
        "success": {"dtype": "float32", "shape": [1]},
        "action.raw_world_normalized": {"dtype": "float32", "shape": [schema.action_dim]},
        "action.source_index": {"dtype": "int64", "shape": [1]},
        "action.plan.base_waypoints": {"dtype": "float32", "shape": [len(offsets) * 4]},
        "action.plan.manipulator": {"dtype": "float32", "shape": [len(offsets) * (9 + hand_dim)]},
        "action.plan.valid": {"dtype": "bool", "shape": [len(offsets)]},
        "observation.base.world": {"dtype": "float32", "shape": [6]},
        "observation.eef.world": {"dtype": "float32", "shape": [6]},
        "observation.robot_joint": {
            "dtype": "float32", "shape": [len(schema.joint_names) * 3]
        },
        "observation.robot_body": {
            "dtype": "float32", "shape": [schema.expected_body_count * 12]
        },
        "supervision.robot_joint_target": {
            "dtype": "float32", "shape": [len(schema.joint_names)]
        },
        "observation.camera.head.pose_world": {"dtype": "float32", "shape": [6]},
        "observation.camera.wrist.pose_world": {"dtype": "float32", "shape": [6]},
        "supervision.object": {"dtype": "float32", "shape": [9]},
        "annotation.task": {"dtype": "string", "shape": [1]},
    })
    info = {
        "codebase_version": "v2.0",
        "robot_type": schema.robot_type,
        "total_episodes": len(episode_rows),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": len(VIDEO_SOURCES),
        "total_chunks": math.ceil(len(episode_rows) / 1000),
        "chunks_size": 1000,
        "fps": control_fps,
        "media_fps": first_video_probes["head"]["fps"],
        "splits": {"train": "0:100"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": info_features,
    }
    write_json(dataset_root / "meta/info.json", info)
    log(f"[{schema.key}] wrote {len(episode_rows)} episodes / {total_frames} frames to {dataset_root}")
    return dataset_root


def read_video_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise ValueError(f"Unable to decode frame {frame_index}: {path}")
    return frame


def labeled_panel(image: np.ndarray, label: str, width: int = 320) -> np.ndarray:
    height = int(round(image.shape[0] * width / image.shape[1]))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(resized, (0, 0), (width, 28), (0, 0, 0), thickness=-1)
    cv2.putText(resized, label, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return resized


def make_frame_montage(dataset_root: Path, episode_index: int, frame_index: int) -> Path:
    chunk = episode_index // 1000
    frames: dict[str, np.ndarray] = {}
    for name in VIDEO_SOURCES:
        path = (
            dataset_root / f"videos/chunk-{chunk:03d}/observation.images.{name}"
            / f"episode_{episode_index:06d}.mp4"
        )
        frames[name] = read_video_frame(path, frame_index)

    for name in ("depth_head", "depth_wrist"):
        gray = cv2.cvtColor(frames[name], cv2.COLOR_BGR2GRAY)
        frames[name] = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

    order = [
        ("head", "RGB head"), ("wrist", "RGB wrist"),
        ("depth_head", "pseudo-range head"), ("depth_wrist", "pseudo-range wrist"),
        ("segmentation_head", "segment head"), ("segmentation_wrist", "segment wrist"),
    ]
    panels = [labeled_panel(frames[name], label) for name, label in order]
    montage = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
    output = dataset_root / "validation_samples" / f"episode_{episode_index:06d}_frame_{frame_index:06d}.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), montage, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise ValueError(f"Failed to write montage: {output}")
    return output


def make_plan_plot(dataset_root: Path, episode_index: int, anchor: int, offsets: Sequence[int]) -> Path:
    chunk = episode_index // 1000
    parquet = dataset_root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    row = pd.read_parquet(parquet).iloc[anchor]
    base = np.asarray(row["action.plan.base_waypoints"], dtype=np.float64).reshape(len(offsets), 4)
    schema_meta = json.loads((dataset_root / "meta/robot_schema.json").read_text())
    hand_dim = len(schema_meta["hand_joint_indices"])
    manip = np.asarray(row["action.plan.manipulator"], dtype=np.float64).reshape(len(offsets), 9 + hand_dim)
    valid = np.asarray(row["action.plan.valid"], dtype=bool)

    canvas = np.full((700, 900, 3), 255, dtype=np.uint8)
    center = np.array([450.0, 350.0])
    points = np.concatenate([base[valid, :2], manip[valid, :2]], axis=0) if valid.any() else np.zeros((1, 2))
    extent = max(float(np.max(np.abs(points))), 0.25)
    scale = 280.0 / extent

    def pixel(xy: np.ndarray) -> tuple[int, int]:
        p = center + np.array([xy[0], -xy[1]]) * scale
        return int(round(p[0])), int(round(p[1]))

    cv2.line(canvas, (80, 350), (820, 350), (220, 220, 220), 1)
    cv2.line(canvas, (450, 60), (450, 640), (220, 220, 220), 1)
    cv2.circle(canvas, pixel(np.zeros(2)), 7, (0, 0, 0), thickness=-1)
    previous_base = pixel(np.zeros(2))
    previous_eef: tuple[int, int] | None = None
    for h, is_valid in enumerate(valid):
        if not is_valid:
            continue
        bp = pixel(base[h, :2])
        ep = pixel(manip[h, :2])
        cv2.line(canvas, previous_base, bp, (200, 80, 20), 2)
        cv2.circle(canvas, bp, 6, (200, 80, 20), thickness=-1)
        if previous_eef is not None:
            cv2.line(canvas, previous_eef, ep, (20, 80, 220), 2)
        cv2.circle(canvas, ep, 6, (20, 80, 220), thickness=-1)
        cv2.putText(canvas, str(offsets[h]), (bp[0] + 5, bp[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 40, 10), 1)
        previous_base, previous_eef = bp, ep
    cv2.putText(canvas, "base waypoint XY", (35, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 80, 20), 2)
    cv2.putText(canvas, "EEF XY in anchor base", (260, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 80, 220), 2)
    cv2.putText(canvas, f"episode={episode_index} anchor={anchor}", (620, 675), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    output = dataset_root / "validation_samples" / f"episode_{episode_index:06d}_plan_anchor_{anchor:06d}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise ValueError(f"Failed to write plan plot: {output}")
    return output


def validate_dataset(dataset_root: Path, make_samples: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    episode_reports: list[dict[str, Any]] = []
    required_meta = [
        "info.json", "modality.json", "embodiment.json", "stats.json", "tasks.jsonl",
        "episodes.jsonl", "extensions.json", "calibration.json", "robot_schema.json",
    ]
    for name in required_meta:
        if not (dataset_root / "meta" / name).exists():
            errors.append(f"Missing meta/{name}")
    if errors:
        return {"passed": False, "errors": errors, "warnings": warnings}

    info = json.loads((dataset_root / "meta/info.json").read_text())
    modality = json.loads((dataset_root / "meta/modality.json").read_text())
    extensions = json.loads((dataset_root / "meta/extensions.json").read_text())
    robot_schema = json.loads((dataset_root / "meta/robot_schema.json").read_text())
    episodes = [json.loads(line) for line in (dataset_root / "meta/episodes.jsonl").read_text().splitlines() if line]
    offsets = extensions["action_plan"]["waypoint_offsets"]
    hand_dim = len(robot_schema["hand_joint_indices"])
    expected_action_dim = modality["action"]["hand_target_normalized"]["end"]
    total_frames = 0

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        expected_length = int(episode["length"])
        chunk = episode_index // int(info["chunks_size"])
        parquet = dataset_root / info["data_path"].format(
            episode_chunk=chunk, episode_index=episode_index
        )
        if not parquet.exists():
            errors.append(f"Missing {parquet}")
            continue
        df = pd.read_parquet(parquet)
        report: dict[str, Any] = {"episode_index": episode_index, "length": len(df), "videos": {}}
        if len(df) != expected_length:
            errors.append(f"episode {episode_index}: parquet length {len(df)} != {expected_length}")
        required_columns = {
            "observation.state", "action", "timestamp", "control_timestamp", "frame_index",
            "episode_index", "task_index", "action.plan.base_waypoints",
            "action.plan.manipulator", "action.plan.valid", "action.source_index",
            "action.raw_world_normalized", "observation.base.world",
            "observation.eef.world", "observation.robot_joint",
            "observation.robot_body", "supervision.robot_joint_target",
            "observation.camera.head.pose_world", "observation.camera.wrist.pose_world",
            "supervision.object",
        }
        missing_columns = sorted(required_columns.difference(df.columns))
        if missing_columns:
            errors.append(f"episode {episode_index}: missing columns {missing_columns}")
            continue
        state = np.stack(df["observation.state"].to_numpy())
        action = np.stack(df["action"].to_numpy())
        base_plan = np.stack(df["action.plan.base_waypoints"].to_numpy())
        manip_plan = np.stack(df["action.plan.manipulator"].to_numpy())
        plan_valid = np.stack(df["action.plan.valid"].to_numpy())
        expected_shapes = {
            "state": (expected_length, 6),
            "action": (expected_length, expected_action_dim),
            "base_plan": (expected_length, len(offsets) * 4),
            "manipulator_plan": (expected_length, len(offsets) * (9 + hand_dim)),
            "plan_valid": (expected_length, len(offsets)),
        }
        actual_shapes = {
            "state": state.shape, "action": action.shape, "base_plan": base_plan.shape,
            "manipulator_plan": manip_plan.shape, "plan_valid": plan_valid.shape,
        }
        for key, shape in expected_shapes.items():
            if actual_shapes[key] != shape:
                errors.append(f"episode {episode_index}: {key} shape {actual_shapes[key]} != {shape}")
        for key, values in (("state", state), ("action", action), ("base_plan", base_plan), ("manipulator_plan", manip_plan)):
            if not np.isfinite(values).all():
                errors.append(f"episode {episode_index}: non-finite {key}")

        # Independently reconstruct every training/plan label from preserved
        # source-frame state. This catches coordinate-frame and off-by-one bugs
        # that dimensional checks alone cannot detect.
        raw_action = np.stack(df["action.raw_world_normalized"].to_numpy()).astype(np.float64)
        base_world = np.stack(df["observation.base.world"].to_numpy()).astype(np.float64)
        eef_world = np.stack(df["observation.eef.world"].to_numpy()).astype(np.float64)
        joint_count = len(robot_schema["joint_names"])
        robot_joint = np.stack(df["observation.robot_joint"].to_numpy()).reshape(
            expected_length, joint_count, 3
        ).astype(np.float64)
        robot_body = np.stack(df["observation.robot_body"].to_numpy())
        if robot_body.shape != (
            expected_length,
            int(robot_schema["expected_body_count"]) * 12,
        ):
            errors.append(
                f"episode {episode_index}: robot_body shape {robot_body.shape} is invalid"
            )
        for preserved_key in (
            "observation.robot_body",
            "supervision.robot_joint_target",
            "observation.camera.head.pose_world",
            "observation.camera.wrist.pose_world",
            "supervision.object",
        ):
            preserved = np.stack(df[preserved_key].to_numpy())
            if not np.isfinite(preserved).all():
                errors.append(f"episode {episode_index}: non-finite {preserved_key}")
        source_action_index = np.minimum(np.arange(expected_length) + 1, expected_length - 1)
        expected_state = build_core_state(eef_world, base_world)
        expected_action = transform_delta_action_to_base(
            raw_action[source_action_index], base_world[:, :6]
        )
        expected_base_plan, expected_manip_plan, expected_plan_valid = build_plan_labels(
            base_world,
            eef_world,
            robot_joint,
            robot_schema["hand_joint_indices"],
            np.asarray(offsets, dtype=np.int64),
        )
        expected_base_plan = expected_base_plan.reshape(expected_length, -1)
        expected_manip_plan = expected_manip_plan.reshape(expected_length, -1)
        semantic_errors = {
            "state_max_abs": float(np.max(np.abs(state - expected_state))),
            "action_max_abs": float(np.max(np.abs(action - expected_action))),
            "base_plan_max_abs": float(np.max(np.abs(base_plan - expected_base_plan))),
            "manipulator_plan_max_abs": float(
                np.max(np.abs(manip_plan - expected_manip_plan))
            ),
        }
        report["semantic_reconstruction_max_abs_error"] = semantic_errors
        for key, value in semantic_errors.items():
            if value > 2e-5:
                errors.append(
                    f"episode {episode_index}: semantic reconstruction {key}={value:.6g}"
                )
        if not np.array_equal(plan_valid, expected_plan_valid):
            errors.append(f"episode {episode_index}: plan validity mask reconstruction mismatch")
        if not np.array_equal(
            df["action.source_index"].to_numpy(), source_action_index
        ):
            errors.append(f"episode {episode_index}: shifted action source-index mismatch")
        if not np.array_equal(df["frame_index"].to_numpy(), np.arange(expected_length)):
            errors.append(f"episode {episode_index}: non-contiguous frame_index")
        if not np.allclose(df["control_timestamp"].to_numpy(), np.arange(expected_length) / info["fps"]):
            errors.append(f"episode {episode_index}: control_timestamp mismatch")
        expected_media_time = np.arange(expected_length) / info["media_fps"]
        if not np.allclose(df["timestamp"].to_numpy(), expected_media_time):
            errors.append(f"episode {episode_index}: media timestamp mismatch")

        for video_name, modality_meta in modality["video"].items():
            original_key = modality_meta["original_key"]
            path = dataset_root / info["video_path"].format(
                episode_chunk=chunk, episode_index=episode_index, video_key=original_key
            )
            if not path.exists():
                errors.append(f"episode {episode_index}: missing video {path}")
                continue
            probe = probe_video(path)
            report["videos"][video_name] = probe
            if probe["frames"] != expected_length:
                errors.append(
                    f"episode {episode_index} {video_name}: frames {probe['frames']} != {expected_length}"
                )
        report["state_range"] = [float(state.min()), float(state.max())]
        report["action_range"] = [float(action.min()), float(action.max())]
        report["valid_plan_fraction"] = float(plan_valid.mean())
        episode_reports.append(report)
        total_frames += len(df)

    if total_frames != int(info["total_frames"]):
        errors.append(f"total_frames {total_frames} != info.json {info['total_frames']}")
    if len(episodes) != int(info["total_episodes"]):
        errors.append(f"total_episodes {len(episodes)} != info.json {info['total_episodes']}")

    sample_paths: list[str] = []
    if make_samples and not errors and episodes:
        first = episodes[0]
        episode_index = int(first["episode_index"])
        length = int(first["length"])
        for frame in sorted({0, length // 2, length - 1}):
            sample_paths.append(str(make_frame_montage(dataset_root, episode_index, frame)))
        sample_paths.append(str(make_plan_plot(dataset_root, episode_index, 0, offsets)))

    calibration = json.loads((dataset_root / "meta/calibration.json").read_text())
    if calibration.get("status") != "verified":
        warnings.append("Camera K/optical convention is nominal_unverified; do not enable projection losses yet.")
    warnings.append("Depth MP4 is lossy pseudo-range; use confidence masks and coarse geometry claims only.")
    report = {
        "passed": not errors,
        "dataset_root": str(dataset_root),
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "episodes": len(episodes),
            "frames": total_frames,
            "control_fps": info.get("fps"),
            "media_fps": info.get("media_fps"),
            "core_state_dim": 6,
            "core_action_dim": expected_action_dim,
            "base_plan_dim_per_waypoint": 4,
            "manipulator_plan_dim_per_waypoint": 9 + hand_dim,
            "waypoint_offsets": offsets,
        },
        "episodes": episode_reports,
        "validation_samples": sample_paths,
    }
    write_json(dataset_root / "meta/validation_report.json", report)
    return report


def write_smoke_readme(output_root: Path, dataset_roots: Sequence[Path]) -> None:
    lines = [
        "# MobileManiBench DreamZero smoke conversion",
        "",
        "This directory was generated by `convert_mobilemanibench_to_gear.py`.",
        "It contains no modified DreamZero code. Source videos are linked/copied according to the CLI.",
        "",
        "## Dataset roots",
        "",
    ]
    for root in dataset_roots:
        report_path = root / "meta/validation_report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        lines.append(f"- `{root}`: validation passed = `{report.get('passed', 'not run')}`")
    lines.extend([
        "",
        "## Current DreamZero compatibility",
        "",
        "- Use embodiment tag `xdof` (written in `meta/embodiment.json`).",
        "- Set `relative_action: false`.",
        "- Core videos for the baseline are `video.head` and `video.wrist`.",
        "- Core state keys: `state.eef_position`, `state.eef_rotation_rpy`.",
        "- Core action keys: `action.eef_delta_position_normalized`,",
        "  `action.eef_delta_rotation_rpy_normalized`, `action.hand_target_normalized`.",
        "- The other video and parquet fields are preserved extensions and can be ignored by the current loader.",
        "",
        "## Important warnings",
        "",
        "- `timestamp` follows the MP4 media clock so the current loader seeks the correct frame;",
        "  `control_timestamp` follows the 30 Hz physical/control clock.",
        "- Camera intrinsics and optical convention are nominal, not verified.",
        "- Depth videos are lossy pseudo-range, not lossless metric ground truth.",
        "- The main research action plan is stored as extra parquet columns; current DreamZero trains",
        "  on the official 7/18-D step action unless its data configuration is later extended.",
        "",
        "See each dataset's `meta/validation_report.json` and `validation_samples/` before training.",
        "",
    ])
    (output_root / "SMOKE_README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_offsets(raw: str) -> np.ndarray:
    offsets = np.array([int(value.strip()) for value in raw.split(",") if value.strip()], dtype=np.int64)
    if offsets.size == 0 or np.any(offsets < 0) or len(np.unique(offsets)) != len(offsets):
        raise argparse.ArgumentTypeError("waypoint offsets must be unique non-negative integers")
    return np.sort(offsets)


def command_convert(args: argparse.Namespace) -> int:
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")
    output_root.mkdir(parents=True)
    offsets = parse_offsets(args.waypoint_offsets)
    dataset_roots: list[Path] = []
    for embodiment in args.embodiments:
        schema = ROBOT_SCHEMAS[embodiment]
        source_paths = discover_episodes(
            input_root, schema, args.max_episodes_per_embodiment
        )
        dataset_root = convert_embodiment(
            input_root=input_root,
            output_root=output_root,
            schema=schema,
            source_paths=source_paths,
            offsets=offsets,
            link_mode=args.link_mode,
            control_fps=args.control_fps,
        )
        dataset_roots.append(dataset_root)
        if args.validate:
            report = validate_dataset(dataset_root, make_samples=True)
            log(f"[{embodiment}] validation passed={report['passed']}")
            if not report["passed"]:
                for error in report["errors"]:
                    log(f"  ERROR: {error}")
                return 2
    write_smoke_readme(output_root, dataset_roots)
    return 0


def command_validate(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve()
    roots: list[Path] = []
    if (output_root / "meta/info.json").exists():
        roots = [output_root]
    else:
        roots = [output_root / key for key in ROBOT_SCHEMAS if (output_root / key / "meta/info.json").exists()]
    if not roots:
        raise FileNotFoundError(f"No converted dataset found under {output_root}")
    passed = True
    for root in roots:
        report = validate_dataset(root, make_samples=not args.no_samples)
        log(f"[{root.name}] validation passed={report['passed']}")
        for warning in report.get("warnings", []):
            log(f"  WARNING: {warning}")
        for error in report.get("errors", []):
            log(f"  ERROR: {error}")
        passed = passed and bool(report["passed"])
    return 0 if passed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="Convert raw MobileManiBench episodes")
    convert.add_argument("--input-root", required=True)
    convert.add_argument("--output-root", required=True)
    convert.add_argument(
        "--embodiments", nargs="+", choices=sorted(ROBOT_SCHEMAS), default=["g1", "xhand"]
    )
    convert.add_argument(
        "--max-episodes-per-embodiment", type=int, default=0,
        help="0 converts all episodes; use a tiny value for smoke/overfit data",
    )
    convert.add_argument(
        "--waypoint-offsets", default="1,4,8,12,16,24",
        help="Comma-separated control-frame offsets for extension action-plan labels",
    )
    convert.add_argument("--control-fps", type=float, default=30.0)
    convert.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    convert.add_argument("--validate", action="store_true")
    convert.set_defaults(func=command_convert)

    validate = subparsers.add_parser("validate", help="Validate converted output")
    validate.add_argument("--output-root", required=True)
    validate.add_argument("--no-samples", action="store_true")
    validate.set_defaults(func=command_validate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:
        log(f"FATAL: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
