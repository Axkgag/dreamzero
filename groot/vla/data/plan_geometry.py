"""Geometry used to derive block-local supervision from canonical trajectories."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from groot.vla.utils.mobile_plan_spec import canonical_block_plan_spec


def euler_rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
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
    rotation_local = np.einsum(
        "...ji,...jk->...ik", frame_rotation_w, target_rotation_w
    )
    rpy_local = matrix_to_euler_rpy(rotation_local)
    return position_local, rpy_local, rotation_local


def build_dynamic_block_plan_labels(
    robot_base: np.ndarray,
    robot_hand: np.ndarray,
    robot_joint: np.ndarray,
    hand_joint_indices: Sequence[int],
    block_anchor_offsets: Sequence[int] = (0, 8, 16, 24),
    local_waypoint_offsets: Sequence[int] = (4, 8),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build labels for every possible start row of one canonical trajectory."""
    spec = canonical_block_plan_spec(
        block_anchor_offsets, local_waypoint_offsets
    )
    anchors = np.asarray(spec["block_anchor_offsets"], dtype=np.int64)
    local = np.asarray(spec["local_waypoint_offsets"], dtype=np.int64)
    robot_base = np.asarray(robot_base, dtype=np.float64)
    robot_hand = np.asarray(robot_hand, dtype=np.float64)
    robot_joint = np.asarray(robot_joint, dtype=np.float64)
    if robot_base.ndim != 2 or robot_base.shape[1] < 6:
        raise ValueError(f"Expected Base trajectory [T,>=6], got {robot_base.shape}")
    if robot_hand.ndim != 2 or robot_hand.shape[1] < 6:
        raise ValueError(f"Expected EEF trajectory [T,>=6], got {robot_hand.shape}")
    if robot_joint.ndim != 3 or robot_joint.shape[-1] < 1:
        raise ValueError(f"Expected joint trajectory [T,J,D], got {robot_joint.shape}")
    if not (len(robot_base) == len(robot_hand) == len(robot_joint)):
        raise ValueError("Base, EEF, and joint trajectories must have equal length")
    hand_indices = tuple(int(value) for value in hand_joint_indices)
    if hand_indices and max(hand_indices) >= robot_joint.shape[1]:
        raise ValueError(
            f"Hand joint index {max(hand_indices)} exceeds {robot_joint.shape[1]} joints"
        )

    length = robot_base.shape[0]
    num_blocks = len(anchors)
    waypoints_per_block = len(local)
    row = np.arange(length, dtype=np.int64)[:, None]
    anchor_indices = row + anchors[None, :]
    anchor_valid = anchor_indices < length
    safe_anchors = np.minimum(anchor_indices, length - 1)
    target_indices = anchor_indices[:, :, None] + local[None, None, :]
    valid = anchor_valid[:, :, None] & (target_indices < length)
    safe_targets = np.minimum(target_indices, length - 1)

    anchor_position = robot_base[safe_anchors, :3]
    anchor_rotation = euler_rpy_to_matrix(robot_base[safe_anchors, 3:6])
    future_base_position = robot_base[safe_targets, :3]
    future_base_rotation = euler_rpy_to_matrix(robot_base[safe_targets, 3:6])
    base_relative_position = np.einsum(
        "tbji,tbwj->tbwi",
        anchor_rotation,
        future_base_position - anchor_position[:, :, None, :],
    )
    base_relative_rotation = np.einsum(
        "tbji,tbwjk->tbwik", anchor_rotation, future_base_rotation
    )
    base_yaw = np.arctan2(
        base_relative_rotation[..., 1, 0], base_relative_rotation[..., 0, 0]
    )
    base_plan = np.stack(
        [
            base_relative_position[..., 0],
            base_relative_position[..., 1],
            np.sin(base_yaw),
            np.cos(base_yaw),
        ],
        axis=-1,
    )

    future_eef_position = robot_hand[safe_targets, :3]
    future_eef_rotation = euler_rpy_to_matrix(robot_hand[safe_targets, 3:6])
    eef_relative_position = np.einsum(
        "tbji,tbwj->tbwi",
        anchor_rotation,
        future_eef_position - anchor_position[:, :, None, :],
    )
    eef_relative_rotation = np.einsum(
        "tbji,tbwjk->tbwik", anchor_rotation, future_eef_rotation
    )
    eef_rotation_6d = eef_relative_rotation[..., :2, :].reshape(
        length, num_blocks, waypoints_per_block, 6
    )
    joint_position = robot_joint[..., 0]
    hand_configuration = joint_position[safe_targets][..., list(hand_indices)]
    manipulator_plan = np.concatenate(
        [eef_relative_position, eef_rotation_6d, hand_configuration], axis=-1
    )

    anchor_eef_position = robot_hand[safe_anchors, :3]
    anchor_eef_rpy = robot_hand[safe_anchors, 3:6]
    anchor_base_rpy = robot_base[safe_anchors, 3:6]
    state_position, state_rpy, _ = relative_pose(
        anchor_eef_position,
        anchor_eef_rpy,
        anchor_position,
        anchor_base_rpy,
    )
    block_state = np.concatenate([state_position, state_rpy], axis=-1)

    base_plan[~valid] = 0.0
    manipulator_plan[~valid] = 0.0
    block_state[~anchor_valid] = 0.0
    return (
        base_plan.astype(np.float32),
        manipulator_plan.astype(np.float32),
        valid.astype(bool),
        block_state.astype(np.float32),
        anchor_valid.astype(bool),
    )
