#!/usr/bin/env python3
"""Create a lightweight task-balanced view of a converted MobileManiBench root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TASKS = (
    "close box",
    "pull cart",
    "open faucet",
    "open window",
    "open dishwasher",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_key(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()


def trajectory_group(source_relative_path: str) -> str:
    marker = source_relative_path.rfind("/episode_")
    if marker < 0:
        raise ValueError(f"Cannot derive trajectory group from {source_relative_path}")
    return source_relative_path[:marker]


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
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--train-groups-per-task", type=int, default=120)
    parser.add_argument("--val-groups-per-task", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.train_groups_per_task <= 0 or args.val_groups_per_task <= 0:
        parser.error("group limits must be positive")

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    if not (source_root / "meta/plan_splits.json").is_file():
        raise FileNotFoundError(source_root / "meta/plan_splits.json")

    episodes = read_jsonl(source_root / "meta/episodes.jsonl")
    sources = read_jsonl(source_root / "meta/source_episodes.jsonl")
    if len(episodes) != len(sources):
        raise ValueError("episodes.jsonl and source_episodes.jsonl disagree")
    episode_by_id = {int(row["episode_index"]): row for row in episodes}
    source_by_id = {int(row["episode_index"]): row for row in sources}
    if episode_by_id.keys() != source_by_id.keys():
        raise ValueError("episode IDs and source episode IDs disagree")

    group_by_episode: dict[int, str] = {}
    groups_by_split_task: dict[str, dict[str, dict[str, list[int]]]] = {
        "train": defaultdict(lambda: defaultdict(list)),
        "val": defaultdict(lambda: defaultdict(list)),
    }
    original_splits = read_json(source_root / "meta/plan_splits.json")["splits"]
    requested_tasks = tuple(dict.fromkeys(str(task) for task in args.tasks))
    requested_task_set = set(requested_tasks)
    for split in ("train", "val"):
        for episode_id in original_splits[split]["episode_indices"]:
            episode_id = int(episode_id)
            episode = episode_by_id[episode_id]
            task = str(episode["tasks"][0])
            if task not in requested_task_set:
                continue
            group = trajectory_group(
                source_by_id[episode_id]["source_taxonomy"]["source_relative_path"]
            )
            group_by_episode[episode_id] = group
            groups_by_split_task[split][task][group].append(episode_id)

    selected: dict[str, list[int]] = {"train": [], "val": []}
    selection_report: dict[str, dict[str, dict[str, int]]] = {
        "train": {},
        "val": {},
    }
    limits = {
        "train": args.train_groups_per_task,
        "val": args.val_groups_per_task,
    }
    for split in ("train", "val"):
        for task in requested_tasks:
            groups = groups_by_split_task[split].get(task, {})
            if len(groups) < limits[split]:
                raise ValueError(
                    f"{split}/{task} has {len(groups)} groups, "
                    f"fewer than requested {limits[split]}"
                )
            chosen_groups = sorted(
                groups, key=lambda value: stable_key(args.seed, value)
            )[: limits[split]]
            ids = sorted(
                episode_id
                for group in chosen_groups
                for episode_id in groups[group]
            )
            selected[split].extend(ids)
            selection_report[split][task] = {
                "num_groups": len(chosen_groups),
                "num_episodes": len(ids),
                "num_frames": sum(
                    int(episode_by_id[episode_id]["length"]) for episode_id in ids
                ),
            }
        selected[split].sort()

    train_set = set(selected["train"])
    val_set = set(selected["val"])
    if train_set & val_set:
        raise ValueError("Train and validation episode IDs overlap")
    selected_ids = train_set | val_set

    output_root.mkdir(parents=True)
    (output_root / "meta").mkdir()
    os.symlink(source_root / "data", output_root / "data", target_is_directory=True)
    os.symlink(source_root / "videos", output_root / "videos", target_is_directory=True)

    passthrough_meta = (
        "modality.json",
        "embodiment.json",
        "relative_stats_dreamzero.json",
        "tasks.jsonl",
        "robot_schema.json",
        "calibration.json",
        "extensions.json",
    )
    for name in passthrough_meta:
        shutil.copy2(source_root / "meta" / name, output_root / "meta" / name)

    selected_episodes = [
        episode_by_id[episode_id] for episode_id in sorted(selected_ids)
    ]
    selected_sources = [
        source_by_id[episode_id] for episode_id in sorted(selected_ids)
    ]
    write_jsonl(output_root / "meta/episodes.jsonl", selected_episodes)
    write_jsonl(output_root / "meta/source_episodes.jsonl", selected_sources)

    source_manifest_path = source_root / "meta/source_manifest.jsonl"
    if source_manifest_path.is_file():
        with source_manifest_path.open("r", encoding="utf-8") as source_handle:
            with (output_root / "meta/source_manifest.jsonl").open(
                "w", encoding="utf-8"
            ) as output_handle:
                for line in source_handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if int(row["episode_index"]) in selected_ids:
                        output_handle.write(
                            json.dumps(row, ensure_ascii=False) + "\n"
                        )

    original_info = read_json(source_root / "meta/info.json")
    subset_info = dict(original_info)
    subset_info["total_episodes"] = len(selected_ids)
    subset_info["total_frames"] = sum(
        int(episode_by_id[episode_id]["length"]) for episode_id in selected_ids
    )
    subset_info["splits"] = {
        "train": selected["train"],
        "val": selected["val"],
    }
    subset_info["subset"] = {
        "source_root": str(source_root),
        "tasks": list(requested_tasks),
        "selection_seed": args.seed,
        "train_groups_per_task": args.train_groups_per_task,
        "val_groups_per_task": args.val_groups_per_task,
        "data_and_videos": "absolute directory symlinks to source_root",
    }
    write_json(output_root / "meta/info.json", subset_info)

    plan_splits = {
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(output_root),
        "source_dataset_root": str(source_root),
        "method": "task_balanced_stable_hash_trajectory_subset",
        "group_by": "trajectory",
        "seed": args.seed,
        "tasks": list(requested_tasks),
        "selection": selection_report,
        "splits": {
            split: split_summary(
                selected[split], episode_by_id, group_by_episode
            )
            for split in ("train", "val")
        },
    }
    write_json(output_root / "meta/plan_splits.json", plan_splits)

    readme = f"""# MobileManiBench G1 balanced five-task view

- Source: `{source_root}`
- Tasks: {", ".join(f"`{task}`" for task in requested_tasks)}
- Train groups/task: {args.train_groups_per_task}
- Validation groups/task: {args.val_groups_per_task}
- Seed: {args.seed}
- `data/` and `videos/` are directory symlinks; source data is not duplicated.
- Run `prepare_mobilemanibench_plan_metadata.py --split train --write-core-stats`
  before training.
"""
    (output_root / "SUBSET_README.md").write_text(readme, encoding="utf-8")

    print(f"Wrote balanced task subset to {output_root}")
    for split in ("train", "val"):
        summary = plan_splits["splits"][split]
        print(
            f"{split}: {summary['num_groups']} groups, "
            f"{summary['num_episodes']} episodes, "
            f"{summary['num_frames']} frames"
        )
        for task in requested_tasks:
            item = selection_report[split][task]
            print(
                f"  {task}: {item['num_groups']} groups, "
                f"{item['num_episodes']} episodes, "
                f"{item['num_frames']} frames"
            )


if __name__ == "__main__":
    main()
