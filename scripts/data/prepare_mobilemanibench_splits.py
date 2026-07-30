#!/usr/bin/env python3
"""Create deterministic, task-stratified MobileManiBench train/val splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_key(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()


def group_name(source_relative_path: str, group_by: str) -> str:
    if group_by == "episode":
        return source_relative_path.rsplit("/state_infos.pkl", 1)[0]
    episode_marker = source_relative_path.rfind("/episode_")
    if episode_marker < 0:
        raise ValueError(f"Cannot derive trajectory group from {source_relative_path}")
    return source_relative_path[:episode_marker]


def split_summary(
    episode_ids: list[int],
    episode_by_id: dict[int, dict[str, Any]],
    group_by_episode: dict[int, str],
) -> dict[str, Any]:
    task_counts: Counter[str] = Counter()
    frames = 0
    groups: set[str] = set()
    for episode_id in episode_ids:
        episode = episode_by_id[episode_id]
        frames += int(episode["length"])
        groups.add(group_by_episode[episode_id])
        task_counts.update(episode["tasks"])
    return {
        "episode_indices": episode_ids,
        "num_episodes": len(episode_ids),
        "num_frames": frames,
        "num_samples": frames,
        "num_groups": len(groups),
        "task_episode_counts": dict(sorted(task_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-by",
        choices=("trajectory", "episode"),
        default="trajectory",
        help="Use episode only for tiny smoke datasets.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in (0, 1)")

    root = args.dataset_root.resolve()
    output = root / "meta/plan_splits.json"
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} already exists; pass --force to replace it")

    episodes = read_jsonl(root / "meta/episodes.jsonl")
    sources = read_jsonl(root / "meta/source_episodes.jsonl")
    if len(episodes) != len(sources):
        raise ValueError(
            f"episodes/source metadata length mismatch: {len(episodes)} != {len(sources)}"
        )

    episode_by_id: dict[int, dict[str, Any]] = {}
    group_by_episode: dict[int, str] = {}
    group_episodes: dict[str, list[int]] = defaultdict(list)
    group_task: dict[str, str] = {}
    official_scene_splits: Counter[str] = Counter()
    source_path_splits: Counter[str] = Counter()

    for episode, source in zip(episodes, sources):
        episode_id = int(episode["episode_index"])
        if episode_id != int(source["episode_index"]):
            raise ValueError(f"episode index mismatch at {episode_id}")
        tasks = episode.get("tasks", [])
        if len(tasks) != 1:
            raise ValueError(f"episode {episode_id} must have exactly one task")
        taxonomy = source["source_taxonomy"]
        group = group_name(taxonomy["source_relative_path"], args.group_by)
        task = str(tasks[0])
        previous_task = group_task.setdefault(group, task)
        if previous_task != task:
            raise ValueError(f"group {group} contains multiple tasks")
        episode_by_id[episode_id] = episode
        group_by_episode[episode_id] = group
        group_episodes[group].append(episode_id)
        official_scene_splits[
            str(source["scene"]["room_infos"].get("split", "missing"))
        ] += 1
        source_path_splits[str(taxonomy.get("train_split", "missing"))] += 1

    groups_by_task: dict[str, list[str]] = defaultdict(list)
    for group, task in group_task.items():
        groups_by_task[task].append(group)

    validation_groups: set[str] = set()
    unsplittable_tasks: list[str] = []
    for task, groups in sorted(groups_by_task.items()):
        ordered = sorted(groups, key=lambda value: stable_key(args.seed, value))
        if len(ordered) < 2:
            unsplittable_tasks.append(task)
            continue
        num_validation = max(
            1,
            min(
                len(ordered) - 1,
                round(len(ordered) * args.validation_fraction),
            ),
        )
        validation_groups.update(ordered[:num_validation])

    if not validation_groups:
        raise ValueError(
            "No validation groups were selected. For a two-episode smoke dataset, "
            "rerun with --group-by episode."
        )

    train_ids: list[int] = []
    validation_ids: list[int] = []
    for episode_id in sorted(episode_by_id):
        if group_by_episode[episode_id] in validation_groups:
            validation_ids.append(episode_id)
        else:
            train_ids.append(episode_id)
    if not train_ids or not validation_ids:
        raise ValueError("Both train and validation splits must be non-empty")

    result = {
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "method": "task_stratified_stable_hash_group_holdout",
        "group_by": args.group_by,
        "group_definition": (
            "source path through trajectories/traj_*; all episode_* siblings stay together"
            if args.group_by == "trajectory"
            else "individual source episode"
        ),
        "validation_fraction_target": args.validation_fraction,
        "seed": args.seed,
        "official_source_split_counts": dict(sorted(official_scene_splits.items())),
        "source_path_split_counts": dict(sorted(source_path_splits.items())),
        "num_tasks": len(groups_by_task),
        "num_groups": len(group_episodes),
        "unsplittable_tasks_kept_in_train": unsplittable_tasks,
        "splits": {
            "train": split_summary(train_ids, episode_by_id, group_by_episode),
            "val": split_summary(
                validation_ids, episode_by_id, group_by_episode
            ),
        },
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Wrote {output}")
    for name, split in result["splits"].items():
        print(
            f"{name}: {split['num_groups']} groups, "
            f"{split['num_episodes']} episodes, {split['num_frames']} frames"
        )


if __name__ == "__main__":
    main()
