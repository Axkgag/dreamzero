"""Canonical specification and cache paths for dynamic block plans."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence


BLOCK_PLAN_SPEC_VERSION = 1
BLOCK_PLAN_COORDINATE_FRAME = "each_block_anchor_base"


def canonical_block_plan_spec(
    block_anchor_offsets: Sequence[int],
    local_waypoint_offsets: Sequence[int],
    coordinate_frame: str = BLOCK_PLAN_COORDINATE_FRAME,
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
    global_offsets = tuple(
        anchor + offset for anchor in anchors for offset in local
    )
    return {
        "version": BLOCK_PLAN_SPEC_VERSION,
        "source": "dynamic_world_trajectory",
        "coordinate_frame": coordinate_frame,
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
) -> str:
    payload = canonical_block_plan_spec(
        block_anchor_offsets, local_waypoint_offsets, coordinate_frame
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def dynamic_block_plan_stats_path(
    dataset_root: str | Path,
    block_anchor_offsets: Sequence[int],
    local_waypoint_offsets: Sequence[int],
    coordinate_frame: str = BLOCK_PLAN_COORDINATE_FRAME,
) -> Path:
    digest = block_plan_spec_hash(
        block_anchor_offsets, local_waypoint_offsets, coordinate_frame
    )
    return Path(dataset_root) / "meta/dynamic_plan_stats" / f"block_plan_{digest}.json"
