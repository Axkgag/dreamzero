#!/usr/bin/env python3
"""Create a one-anchor LeRobot dataset view without duplicating media.

The destination receives an independent copy of ``meta/`` plus symlinks to the
source payload directories. DreamZero's native ``meta/step_filter.jsonl`` keeps
exactly one (episode, frame) pair in ``all_steps`` while the complete source
episode remains available for video delta sampling.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument(
        "--required-video-offset",
        type=int,
        default=32,
        help="Largest future RGB delta required by training.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_episode_path(source: Path, info: dict[str, Any], episode: int) -> Path:
    chunk_size = int(info["chunks_size"])
    relative = str(info["data_path"]).format(
        episode_chunk=episode // chunk_size,
        episode_index=episode,
    )
    return source / relative


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if source == destination:
        raise ValueError("Source and destination must differ")
    if not (source / "meta").is_dir():
        raise FileNotFoundError(source / "meta")
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists; refusing to modify it: {destination}"
        )

    info = read_json(source / "meta/info.json")
    episodes = read_jsonl(source / "meta/episodes.jsonl")
    episode_by_id = {int(row["episode_index"]): row for row in episodes}
    if args.episode_index not in episode_by_id:
        raise ValueError(f"Unknown episode index: {args.episode_index}")
    episode_length = int(episode_by_id[args.episode_index]["length"])
    if not 0 <= args.frame_index < episode_length:
        raise ValueError(
            f"frame_index={args.frame_index} is outside episode length "
            f"{episode_length}"
        )
    final_video_frame = args.frame_index + args.required_video_offset
    if final_video_frame >= episode_length:
        raise ValueError(
            f"Anchor does not provide the required video window: "
            f"{args.frame_index}+{args.required_video_offset}="
            f"{final_video_frame} >= {episode_length}"
        )

    parquet_path = resolve_episode_path(source, info, args.episode_index)
    frame = pd.read_parquet(parquet_path)
    if len(frame) != episode_length:
        raise ValueError(
            f"Episode metadata/parquet length mismatch: {episode_length} vs "
            f"{len(frame)}"
        )
    selected = frame.iloc[args.frame_index]
    required_columns = (
        "action.plan.base_waypoints",
        "action.plan.manipulator",
        "action.plan.valid",
        "annotation.task",
    )
    missing = [name for name in required_columns if name not in frame.columns]
    if missing:
        raise KeyError(f"Selected parquet is missing columns: {missing}")
    plan_valid = np.asarray(selected["action.plan.valid"], dtype=bool)
    if not np.all(plan_valid):
        raise ValueError(
            f"Selected anchor does not have a complete Plan: "
            f"{plan_valid.tolist()}"
        )

    temporary = destination.parent / (
        f".{destination.name}.building-{os.getpid()}"
    )
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    try:
        shutil.copytree(source / "meta", temporary / "meta")
        for child in source.iterdir():
            if child.name == "meta":
                continue
            (temporary / child.name).symlink_to(child.resolve(), target_is_directory=child.is_dir())

        step_filter_rows: list[dict[str, Any]] = []
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            length = int(episode["length"])
            retained = (
                args.frame_index if episode_index == args.episode_index else None
            )
            filtered = [
                index for index in range(length) if index != retained
            ]
            step_filter_rows.append(
                {
                    "episode_index": episode_index,
                    "step_indices": filtered,
                }
            )
        write_jsonl(temporary / "meta/step_filter.jsonl", step_filter_rows)
        write_json(
            temporary / "meta/single_anchor.json",
            {
                "type": "lerobot_step_filter_view",
                "source_dataset": str(source),
                "episode_index": args.episode_index,
                "frame_index": args.frame_index,
                "episode_length": episode_length,
                "required_video_offsets": [
                    0,
                    args.required_video_offset,
                ],
                "video_frame_range": [
                    args.frame_index,
                    final_video_frame,
                ],
                "plan_valid": plan_valid.tolist(),
                "task": str(selected["annotation.task"]),
                "effective_num_samples": 1,
                "normalization_stats": "copied from source training dataset",
                "payload_storage": "absolute symlinks to source data/media",
            },
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary)
        raise

    print(f"Created single-anchor dataset view: {destination}")
    print(f"  source={source}")
    print(
        f"  anchor=(episode={args.episode_index}, frame={args.frame_index})"
    )
    print(
        f"  video_window=[{args.frame_index}, {final_video_frame}]"
    )
    print(f"  task={selected['annotation.task']}")


if __name__ == "__main__":
    main()
