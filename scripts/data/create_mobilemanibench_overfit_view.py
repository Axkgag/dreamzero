#!/usr/bin/env python3
"""Create a small full-episode MobileManiBench view for overfit tests.

The view owns its metadata but symlinks the converted parquet and video payloads.
Train and validation intentionally contain the same episodes so that validation
measures memorization rather than generalization.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_payload_path(
    root: Path,
    template: str,
    episode_index: int,
    chunk_size: int,
    **values: Any,
) -> Path:
    return root / template.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
        **values,
    )


def split_summary(
    episode_ids: list[int], episode_by_id: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    frames = sum(int(episode_by_id[index]["length"]) for index in episode_ids)
    task_counts: dict[str, int] = {}
    for index in episode_ids:
        task = str(episode_by_id[index]["tasks"][0])
        task_counts[task] = task_counts.get(task, 0) + 1
    return {
        "episode_indices": episode_ids,
        "num_episodes": len(episode_ids),
        "num_frames": frames,
        "num_samples": frames,
        "num_groups": len(episode_ids),
        "task_episode_counts": dict(sorted(task_counts.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--episode-indices", type=int, nargs="+", required=True
    )
    parser.add_argument(
        "--required-final-offset",
        type=int,
        default=192,
        help="Largest global tick required by one complete training sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    episode_ids = list(dict.fromkeys(int(index) for index in args.episode_indices))
    if len(episode_ids) < 2:
        raise ValueError("An overfit view should contain at least two episodes")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    info = read_json(source / "meta/info.json")
    episodes = read_jsonl(source / "meta/episodes.jsonl")
    episode_by_id = {int(row["episode_index"]): row for row in episodes}
    missing = [index for index in episode_ids if index not in episode_by_id]
    if missing:
        raise ValueError(f"Unknown episode indices: {missing}")

    chunk_size = int(info["chunks_size"])
    required_offset = int(args.required_final_offset)
    reports: list[dict[str, Any]] = []
    for episode_index in episode_ids:
        row = episode_by_id[episode_index]
        length = int(row["length"])
        if length <= required_offset:
            raise ValueError(
                f"Episode {episode_index} has {length} frames and cannot cover "
                f"global offset {required_offset}"
            )
        parquet = resolve_payload_path(
            source, str(info["data_path"]), episode_index, chunk_size
        )
        frame = pd.read_parquet(
            parquet, columns=["frame_index", "success", "annotation.task"]
        )
        if len(frame) != length:
            raise ValueError(
                f"Episode {episode_index} metadata/parquet length mismatch: "
                f"{length} vs {len(frame)}"
            )
        video_paths = {}
        for key in ("observation.images.head", "observation.images.wrist"):
            path = resolve_payload_path(
                source,
                str(info["video_path"]),
                episode_index,
                chunk_size,
                video_key=key,
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            video_paths[key] = str(path)
        success_frames = frame.loc[frame["success"] > 0.5, "frame_index"]
        reports.append(
            {
                "episode_index": episode_index,
                "task": str(frame["annotation.task"].iloc[0]),
                "length": length,
                "num_complete_roots": length - required_offset,
                "first_success_frame": (
                    int(success_frames.iloc[0]) if len(success_frames) else None
                ),
                "last_success_frame": (
                    int(success_frames.iloc[-1]) if len(success_frames) else None
                ),
                "core_videos": video_paths,
            }
        )

    temporary = output.parent / f".{output.name}.building-{os.getpid()}"
    temporary.mkdir(parents=True)
    try:
        shutil.copytree(
            source / "meta",
            temporary / "meta",
            ignore=shutil.ignore_patterns(
                "phase_index*", "dynamic_plan_stats"
            ),
        )
        for child in source.iterdir():
            if child.name == "meta":
                continue
            (temporary / child.name).symlink_to(
                child.resolve(), target_is_directory=child.is_dir()
            )

        selected_episodes = [episode_by_id[index] for index in episode_ids]
        write_jsonl(temporary / "meta/episodes.jsonl", selected_episodes)

        source_episodes_path = source / "meta/source_episodes.jsonl"
        if source_episodes_path.is_file():
            selected = {
                int(row["episode_index"]): row
                for row in read_jsonl(source_episodes_path)
                if int(row["episode_index"]) in set(episode_ids)
            }
            write_jsonl(
                temporary / "meta/source_episodes.jsonl",
                [selected[index] for index in episode_ids],
            )

        source_manifest_path = source / "meta/source_manifest.jsonl"
        if source_manifest_path.is_file():
            selected_set = set(episode_ids)
            write_jsonl(
                temporary / "meta/source_manifest.jsonl",
                (
                    row
                    for row in read_jsonl(source_manifest_path)
                    if int(row["episode_index"]) in selected_set
                ),
            )

        subset_info = dict(info)
        subset_info["total_episodes"] = len(episode_ids)
        subset_info["total_frames"] = sum(item["length"] for item in reports)
        subset_info["total_tasks"] = len({item["task"] for item in reports})
        subset_info["splits"] = {"train": episode_ids, "val": episode_ids}
        subset_info["overfit_view"] = {
            "source_root": str(source),
            "episode_indices": episode_ids,
            "intentional_train_val_overlap": True,
            "payload_storage": "absolute symlinks to source data/media",
        }
        write_json(temporary / "meta/info.json", subset_info)

        plan_splits = {
            "version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_root": str(output),
            "source_dataset_root": str(source),
            "method": "full_episode_overfit_view",
            "group_by": "episode",
            "intentional_train_val_overlap": True,
            "required_final_offset": required_offset,
            "episodes": reports,
            "splits": {
                "train": split_summary(episode_ids, episode_by_id),
                "val": split_summary(episode_ids, episode_by_id),
            },
        }
        write_json(temporary / "meta/plan_splits.json", plan_splits)
        write_json(
            temporary / "meta/overfit_view.json",
            {
                "type": "full_episode_overfit_view",
                "source_dataset": str(source),
                "required_final_offset": required_offset,
                "episodes": reports,
                "normalization_stats": "copied from source training dataset",
                "payload_storage": "absolute symlinks to source data/media",
            },
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary)
        raise

    print(f"Created overfit dataset view: {output}")
    print(f"  source={source}")
    for report in reports:
        print(
            f"  episode={report['episode_index']} task={report['task']!r} "
            f"frames={report['length']} "
            f"complete_roots={report['num_complete_roots']} "
            f"success=[{report['first_success_frame']},"
            f"{report['last_success_frame']}]"
        )


if __name__ == "__main__":
    main()
