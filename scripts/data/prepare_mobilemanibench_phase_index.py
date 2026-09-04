#!/usr/bin/env python3
"""Build split-safe MobileManiBench task/phase sidecar indices.

The converted parquet already contains the source object trajectory and success
flag, so this tool never rewrites the dataset and does not expose privileged
signals to the policy input.  They are used only to choose training roots.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PHASES = ("navigation", "approach", "grasp", "manipulation")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def episode_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    pattern = info["data_path"]
    chunk_size = int(info.get("chunks_size", info.get("chunk_size", 1000)))
    return root / pattern.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )


def first_persistent(mask: np.ndarray, count: int) -> int | None:
    count = max(1, int(count))
    if len(mask) < count:
        return None
    hits = np.convolve(mask.astype(np.int32), np.ones(count, dtype=np.int32), "valid")
    indices = np.flatnonzero(hits == count)
    return int(indices[0]) if len(indices) else None


def object_motion_progress(values: np.ndarray) -> np.ndarray:
    """Scale pose displacement by the episode's demonstrated pose range."""
    pose = np.asarray(values, dtype=np.float64)[:, :6].copy()
    if pose.shape[1] >= 6:
        pose[:, 3:6] = np.unwrap(pose[:, 3:6], axis=0)
    displacement = pose - pose[:1]
    scale = np.linalg.norm(np.ptp(pose, axis=0))
    if not np.isfinite(scale) or scale < 1e-6:
        return np.zeros(len(pose), dtype=np.float64)
    return np.linalg.norm(displacement, axis=1) / scale


def phase_ranges(
    length: int,
    object_move: int,
    approach_lead: int,
    grasp_lead: int,
) -> list[tuple[str, int, int]]:
    approach_start = max(0, object_move - approach_lead)
    grasp_start = max(approach_start, object_move - grasp_lead)
    ranges = (
        ("navigation", 0, approach_start - 1),
        ("approach", approach_start, grasp_start - 1),
        ("grasp", grasp_start, object_move - 1),
        ("manipulation", object_move, length - 1),
    )
    return [item for item in ranges if item[1] <= item[2]]


def build(args: argparse.Namespace) -> None:
    root = Path(args.dataset_root)
    if args.plan_config:
        config = yaml.safe_load(Path(args.plan_config).read_text(encoding="utf-8"))
        phase = config["phase_index"]
        args.control_fps = float(config["control_fps"])
        args.progress_threshold = float(phase["progress_threshold"])
        args.persistent_frames = int(phase["persistent_frames"])
        args.approach_lead_seconds = float(phase["approach_lead_seconds"])
        args.grasp_lead_seconds = float(phase["grasp_lead_seconds"])
    output = Path(args.output or root / "meta/phase_index.jsonl")
    summary_path = output.with_name(f"{output.stem}_summary.json")
    if output.is_file() and summary_path.is_file() and args.reuse_existing:
        existing = read_json(summary_path)
        expected = {
            "split": args.split,
            "control_fps": args.control_fps,
            "progress_threshold": args.progress_threshold,
            "persistent_frames": args.persistent_frames,
            "approach_lead_seconds": args.approach_lead_seconds,
            "grasp_lead_seconds": args.grasp_lead_seconds,
        }
        if all(existing.get(key) == value for key, value in expected.items()):
            print(f"Reusing phase index: {output}")
            return
        print(f"Phase index configuration changed; rebuilding {output}")

    info = read_json(root / "meta/info.json")
    split_manifest = read_json(
        Path(args.split_manifest or root / "meta/plan_splits.json")
    )
    episode_ids = {
        int(value)
        for value in split_manifest["splits"][args.split]["episode_indices"]
    }
    episodes = [
        json.loads(line)
        for line in (root / "meta/episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    no_object_motion = 0
    parse_failed = 0
    approach_lead = int(round(args.approach_lead_seconds * args.control_fps))
    grasp_lead = int(round(args.grasp_lead_seconds * args.control_fps))

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        if episode_index not in episode_ids:
            continue
        task = str(episode["tasks"][0])
        try:
            frame = pd.read_parquet(
                episode_path(root, info, episode_index),
                columns=["supervision.object", "success"],
            )
            objects = np.stack(frame["supervision.object"].to_numpy())
            success = np.asarray(frame["success"], dtype=np.float64) > 0.5
            progress = object_motion_progress(objects)
            object_move = first_persistent(
                progress >= args.progress_threshold, args.persistent_frames
            )
            if object_move is None:
                no_object_motion += 1
                continue
            success_hits = np.flatnonzero(success)
            success_frame = int(success_hits[0]) if len(success_hits) else None
            for phase, start, end in phase_ranges(
                len(frame), object_move, approach_lead, grasp_lead
            ):
                record = {
                    "episode_index": episode_index,
                    "task": task,
                    "phase": phase,
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "object_move_frame": int(object_move),
                    "grasp_frame": int(max(0, object_move - grasp_lead)),
                    "success_frame": success_frame,
                    "has_object_motion": True,
                    "is_success_episode": success_frame is not None,
                    "success_hold": bool(
                        success_frame is not None and start >= success_frame
                    ),
                }
                rows.append(record)
                counts[phase] += end - start + 1
                task_counts[task][phase] += end - start + 1
        except Exception as error:
            parse_failed += 1
            if parse_failed <= 10:
                print(f"WARNING: episode {episode_index} phase parse failed: {error}")

    if not rows:
        raise RuntimeError("No phase records were generated")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "version": 1,
        "split": args.split,
        "control_fps": args.control_fps,
        "progress_threshold": args.progress_threshold,
        "persistent_frames": args.persistent_frames,
        "approach_lead_seconds": args.approach_lead_seconds,
        "grasp_lead_seconds": args.grasp_lead_seconds,
        "phase_frame_counts": dict(counts),
        "task_phase_frame_counts": {
            task: dict(value) for task, value in task_counts.items()
        },
        "indexed_episode_count": len({row["episode_index"] for row in rows}),
        "no_object_motion_episode_count": no_object_motion,
        "parse_failed_episode_count": parse_failed,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(rows)} phase ranges to {output}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--plan-config")
    parser.add_argument("--split-manifest")
    parser.add_argument("--output")
    parser.add_argument("--control-fps", type=float, default=30.0)
    parser.add_argument("--progress-threshold", type=float, default=0.02)
    parser.add_argument("--persistent-frames", type=int, default=4)
    parser.add_argument("--approach-lead-seconds", type=float, default=2.0)
    parser.add_argument("--grasp-lead-seconds", type=float, default=0.5)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
