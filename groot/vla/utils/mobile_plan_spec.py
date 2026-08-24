"""Canonical specification and cache paths for dynamic block plans."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence


BLOCK_PLAN_SPEC_VERSION = 2
BLOCK_PLAN_COORDINATE_FRAME = "each_block_anchor_base"
EEF_ROTATION_ANCHOR_BASE_6D = "anchor_base_rotation6d"
EEF_ROTATION_CURRENT_EEF_DELTA_ROTVEC = "current_eef_delta_rotvec"
EEF_ROTATION_REPRESENTATIONS = frozenset(
    {
        EEF_ROTATION_ANCHOR_BASE_6D,
        EEF_ROTATION_CURRENT_EEF_DELTA_ROTVEC,
    }
)


def eef_rotation_dim(representation: str) -> int:
    if representation == EEF_ROTATION_ANCHOR_BASE_6D:
        return 6
    if representation == EEF_ROTATION_CURRENT_EEF_DELTA_ROTVEC:
        return 3
    raise ValueError(f"Unsupported EEF rotation representation: {representation}")


def manipulator_plan_dim(hand_dim: int, representation: str) -> int:
    return 3 + eef_rotation_dim(representation) + int(hand_dim)


def canonical_block_plan_spec(
    block_anchor_offsets: Sequence[int],
    local_waypoint_offsets: Sequence[int],
    coordinate_frame: str = BLOCK_PLAN_COORDINATE_FRAME,
    eef_rotation_representation: str = EEF_ROTATION_ANCHOR_BASE_6D,
) -> dict[str, object]:
    anchors = tuple(int(value) for value in block_anchor_offsets)
    local = tuple(int(value) for value in local_waypoint_offsets)
    if not anchors or not local:
        raise ValueError("Block anchors and local waypoint offsets must be non-empty")
    if anchors[0] != 0 or any(value < 0 for value in anchors):
        raise ValueError("Block anchors must begin at zero and be non-negative")
    if any(value <= 0 for value in local):
        raise ValueError("Local waypoint offsets must be positive")
    if tuple(sorted(anchors)) != anchors or len(set(anchors)) != len(anchors):
        raise ValueError("Block anchors must be sorted and unique")
    if tuple(sorted(local)) != local or len(set(local)) != len(local):
        raise ValueError("Local waypoint offsets must be sorted and unique")
    if coordinate_frame != BLOCK_PLAN_COORDINATE_FRAME:
        raise ValueError(f"Unsupported block-plan coordinate frame: {coordinate_frame}")
    rotation_dim = eef_rotation_dim(eef_rotation_representation)
    global_offsets = tuple(
        anchor + offset for anchor in anchors for offset in local
    )
    return {
        "version": BLOCK_PLAN_SPEC_VERSION,
        "source": "dynamic_world_trajectory",
        "coordinate_frame": coordinate_frame,
        "eef_position_frame": "block_anchor_base",
        "eef_rotation_representation": eef_rotation_representation,
        "eef_rotation_frame": (
            "block_anchor_base"
            if eef_rotation_representation == EEF_ROTATION_ANCHOR_BASE_6D
            else "block_anchor_eef_delta"
        ),
        "eef_rotation_dim": rotation_dim,
        "block_anchor_offsets": list(anchors),
        "local_waypoint_offsets": list(local),
        "global_waypoint_offsets": list(global_offsets),
        "num_blocks": len(anchors),
        "waypoints_per_block": len(local),
    }


def block_plan_spec_hash(
    block_anchor_offsets: Sequence[int],
    local_waypoint_offsets: Sequence[int],
    coordinate_frame: str = BLOCK_PLAN_COORDINATE_FRAME,
    eef_rotation_representation: str = EEF_ROTATION_ANCHOR_BASE_6D,
) -> str:
    payload = canonical_block_plan_spec(
        block_anchor_offsets,
        local_waypoint_offsets,
        coordinate_frame,
        eef_rotation_representation,
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def dynamic_block_plan_stats_path(
    dataset_root: str | Path,
    block_anchor_offsets: Sequence[int],
    local_waypoint_offsets: Sequence[int],
    coordinate_frame: str = BLOCK_PLAN_COORDINATE_FRAME,
    eef_rotation_representation: str = EEF_ROTATION_ANCHOR_BASE_6D,
) -> Path:
    digest = block_plan_spec_hash(
        block_anchor_offsets,
        local_waypoint_offsets,
        coordinate_frame,
        eef_rotation_representation,
    )
    return Path(dataset_root) / "meta/dynamic_plan_stats" / f"block_plan_{digest}.json"
