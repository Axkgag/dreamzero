#!/usr/bin/env python3
"""Offline evaluation for MobileManiBench dual Base/Manipulator plans.

The evaluator deliberately uses the deployment-time flow sampler:

* only the current RGB observation is provided to the model;
* future video latents and action latents start from deterministic noise;
* predictions are inverse-normalized to physical units before metrics;
* episode selection is controlled by ``--split``.

For the converted smoke dataset, ``--split train`` evaluates the same two
episodes used for micro-overfit training. Future conversions can provide
``meta/split_manifest.jsonl`` without changing this script.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

# UniPC's compiled update specializes on the Python ``step_index`` argument.
# A 16-step sampling run therefore exceeds PyTorch 2.5's default per-function
# cache limit of 8 before inference finishes.
torch._dynamo.config.cache_size_limit = max(
    torch._dynamo.config.cache_size_limit,
    64,
)

from groot.vla.data.dataset import MobileManiBenchPlanDataset
from groot.vla.data.transform import MobilePlanTransform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        default="train",
        help=(
            "Episode split to evaluate. Resolution order is "
            "meta/split_manifest.jsonl, source room split, then meta/info.json."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-samples", type=int, default=0, help="0 evaluates every selected anchor")
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1140)
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Validate split/sample construction without loading the model",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def all_episode_ids(dataset_root: Path) -> list[int]:
    rows = read_jsonl(dataset_root / "meta/episodes.jsonl")
    if not rows:
        raise FileNotFoundError(dataset_root / "meta/episodes.jsonl")
    return sorted(int(row["episode_index"]) for row in rows)


def split_from_manifest(dataset_root: Path, split: str) -> set[int] | None:
    rows = read_jsonl(dataset_root / "meta/split_manifest.jsonl")
    if not rows:
        return None
    known = {str(row.get("split")) for row in rows}
    if split not in known:
        return None
    return {
        int(row["episode_index"])
        for row in rows
        if str(row.get("split")) == split
    }


def split_from_source_rooms(dataset_root: Path, split: str) -> set[int] | None:
    rows = read_jsonl(dataset_root / "meta/source_episodes.jsonl")
    matches: set[int] = set()
    known: set[str] = set()
    for row in rows:
        room_split = (
            row.get("scene", {}).get("room_infos", {}).get("split")
        )
        if room_split is None:
            continue
        room_split = str(room_split)
        known.add(room_split)
        if room_split == split:
            matches.add(int(row["episode_index"]))
    if split in known:
        return matches
    return None


def split_from_info(dataset_root: Path, split: str) -> set[int] | None:
    info = read_json(dataset_root / "meta/info.json")
    spec = info.get("splits", {}).get(split)
    if spec is None:
        return None
    episode_ids = all_episode_ids(dataset_root)
    if isinstance(spec, list):
        return {int(value) for value in spec}
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(f"Unsupported split specification for {split!r}: {spec!r}")
    start_text, end_text = spec.split(":", 1)
    start_percent = float(start_text)
    end_percent = float(end_text)
    if not (0 <= start_percent <= end_percent <= 100):
        raise ValueError(f"Invalid percentage split {split!r}: {spec!r}")
    count = len(episode_ids)
    start = math.floor(count * start_percent / 100.0)
    end = count if end_percent == 100 else math.floor(count * end_percent / 100.0)
    return set(episode_ids[start:end])


def resolve_episode_split(dataset_root: Path, split: str) -> tuple[set[int], str]:
    resolvers = (
        ("meta/split_manifest.jsonl", split_from_manifest),
        ("meta/source_episodes.jsonl:scene.room_infos.split", split_from_source_rooms),
        ("meta/info.json:splits", split_from_info),
    )
    for source, resolver in resolvers:
        selected = resolver(dataset_root, split)
        if selected is not None:
            if not selected:
                raise ValueError(f"Split {split!r} exists in {source} but contains no episodes")
            return selected, source
    available: dict[str, Any] = {}
    manifest = read_jsonl(dataset_root / "meta/split_manifest.jsonl")
    if manifest:
        available["split_manifest"] = sorted({str(row.get("split")) for row in manifest})
    source_rows = read_jsonl(dataset_root / "meta/source_episodes.jsonl")
    available["source_room_splits"] = sorted(
        {
            str(value)
            for row in source_rows
            if (
                value := row.get("scene", {})
                .get("room_infos", {})
                .get("split")
            )
            is not None
        }
    )
    available["info_splits"] = sorted(
        read_json(dataset_root / "meta/info.json").get("splits", {})
    )
    raise ValueError(f"Unknown split {split!r}; available split metadata: {available}")


def select_dataset_indices(
    dataset: MobileManiBenchPlanDataset,
    episode_ids: set[int],
    stride: int,
    max_samples: int,
) -> list[int]:
    if stride < 1:
        raise ValueError("--sample-stride must be >= 1")
    indices = [
        index
        for index, (episode_id, _) in enumerate(dataset.all_steps)
        if int(episode_id) in episode_ids
    ]
    indices = indices[::stride]
    if max_samples > 0:
        indices = indices[:max_samples]
    if not indices:
        raise ValueError("No dataset anchors remain after split/stride/max-samples filtering")
    return indices


def iter_batches(values: list[int], batch_size: int) -> Iterable[list[int]]:
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def initialize_distributed() -> tuple[torch.device, DeviceMesh | None, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Model evaluation requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    mesh = None
    if world_size > 1:
        if world_size != 2:
            raise ValueError("DreamZero inference parallelism currently supports 1 or 2 GPUs")
        dist.init_process_group("nccl")
        mesh = init_device_mesh(
            "cuda",
            mesh_shape=(world_size,),
            mesh_dim_names=("ip",),
        )
    return device, mesh, local_rank, world_size


def load_weights_into_model(
    model: torch.nn.Module,
    checkpoint_dir: Path,
    label: str,
    *,
    allow_unexpected: bool = False,
) -> None:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    single_path = checkpoint_dir / "model.safetensors"
    if index_path.is_file():
        index = read_json(index_path)
        files = sorted(set(index["weight_map"].values()))
    elif single_path.is_file():
        files = [single_path.name]
    else:
        raise FileNotFoundError(
            f"{label}: expected model.safetensors or model.safetensors.index.json in {checkpoint_dir}"
        )

    unexpected: set[str] = set()
    for filename in files:
        path = checkpoint_dir / filename
        print(f"[weights] {label}: {path}", flush=True)
        state = load_file(str(path), device="cpu")
        incompatible = model.load_state_dict(state, strict=False)
        unexpected.update(incompatible.unexpected_keys)
        del state
        gc.collect()
    if unexpected and not allow_unexpected:
        preview = sorted(unexpected)[:20]
        raise RuntimeError(
            f"{label}: {len(unexpected)} unexpected checkpoint keys; first keys={preview}"
        )
    if unexpected:
        print(
            f"[weights] {label}: ignored {len(unexpected)} architecture-specific "
            "keys from the source checkpoint",
            flush=True,
        )


def load_model(
    checkpoint: Path,
    device: torch.device,
    mesh: DeviceMesh | None,
    num_inference_steps: int,
):
    config_path = checkpoint / "experiment_cfg/conf.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cfg = OmegaConf.load(config_path)
    model = instantiate(cfg.model)

    base_checkpoint = Path(str(cfg.pretrained_model_path))
    # The source DreamZero checkpoint contains its legacy step-action
    # encoder/decoder. Those keys are intentionally absent from the dual-plan
    # architecture; shared Wan weights still load by name.
    load_weights_into_model(
        model,
        base_checkpoint,
        "pretrained base",
        allow_unexpected=True,
    )
    action_head = model.action_head
    if (
        hasattr(action_head, "inject_lora_after_loading")
        and action_head.config.defer_lora_injection
    ):
        action_head.inject_lora_after_loading()
    load_weights_into_model(model, checkpoint, "trained overlay")

    model.eval()
    model.requires_grad_(False)
    if action_head.train_architecture == "lora":
        action_head.model = action_head.model.merge_and_unload()
    action_head.num_inference_steps = int(num_inference_steps)
    if not 1 <= action_head.num_inference_steps <= len(action_head.dit_step_mask):
        raise ValueError(
            f"--num-inference-steps must be in [1,{len(action_head.dit_step_mask)}]"
        )

    model.to(device=device, dtype=torch.bfloat16)
    model.post_initialize()
    if mesh is not None:
        model.parallelize(device_mesh=mesh)
    return model, cfg


def build_dataset_and_collator(dataset_root: Path, cfg: Any):
    # Evaluation supplies exactly the current frame. The future 32 RGB frames
    # used as training targets must never be passed as clean inference context.
    model_transform = instantiate(cfg.train_dataset.plan_transform)
    model_transform.eval()
    dataset = MobileManiBenchPlanDataset(
        dataset_path=dataset_root,
        video_delta_indices=[0],
        load_videos=True,
        video_backend="decord",
        max_manipulator_dim=int(cfg.max_manipulator_action_dim),
        # Keep physical GT and episode/frame identity in the raw sample. The
        # training transform intentionally drops those non-model fields.
        plan_transform=None,
    )
    collator = instantiate(cfg.data_collator)
    return dataset, model_transform, collator


def reset_sampler_state(action_head: Any, seed: int) -> None:
    action_head.seed = int(seed)
    action_head.language = None
    action_head.current_start_frame = 0
    action_head.clip_feas = None
    action_head.ys = None
    action_head.kv_cache1 = None
    action_head.kv_cache_neg = None
    action_head.crossattn_cache = None
    action_head.crossattn_cache_neg = None


def sample_seed(global_seed: int, sample_ids: list[tuple[int, int]]) -> int:
    payload = json.dumps([global_seed, sample_ids], separators=(",", ":")).encode()
    # Torch's manual_seed accepts 64-bit values, but keep this in the signed
    # 31-bit range for identical behavior across CUDA/PyTorch versions.
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7FFFFFFF


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def normalize_rows(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    fallback = np.zeros_like(vector)
    fallback[..., 0] = 1.0
    return np.where(norm > eps, vector / np.maximum(norm, eps), fallback)


def rotation6d_rows_to_matrix(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64).reshape(*value.shape[:-1], 2, 3)
    first = normalize_rows(rows[..., 0, :])
    second_raw = rows[..., 1, :] - np.sum(
        rows[..., 1, :] * first, axis=-1, keepdims=True
    ) * first
    second = normalize_rows(second_raw)
    third = normalize_rows(np.cross(first, second))
    # Recompute the second row so the result is right-handed even for a nearly
    # degenerate network prediction.
    second = normalize_rows(np.cross(third, first))
    return np.stack([first, second, third], axis=-2)


def rotation_geodesic_deg(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = prediction @ np.swapaxes(target, -1, -2)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    cosine = np.where(cosine > 1.0 - 1e-10, 1.0, cosine)
    return np.degrees(np.arccos(cosine))


def wrap_angle(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def rz(yaw: np.ndarray) -> np.ndarray:
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    result = np.zeros(yaw.shape + (3, 3), dtype=np.float64)
    result[..., 0, 0] = cosine
    result[..., 0, 1] = -sine
    result[..., 1, 0] = sine
    result[..., 1, 1] = cosine
    result[..., 2, 2] = 1.0
    return result


class MetricAccumulator:
    def __init__(self, offsets: list[int]):
        self.offsets = offsets
        self.values: dict[str, list[float]] = defaultdict(list)
        self.per_horizon: list[dict[str, list[float]]] = [
            defaultdict(list) for _ in offsets
        ]
        self.per_sample: list[dict[str, Any]] = []
        self.predictions: dict[str, list[np.ndarray]] = defaultdict(list)

    @staticmethod
    def _finite(values: np.ndarray) -> list[float]:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        return [float(value) for value in array[np.isfinite(array)]]

    def add(
        self,
        episode_index: int,
        frame_index: int,
        base_pred: np.ndarray,
        base_gt: np.ndarray,
        manip_pred: np.ndarray,
        manip_gt: np.ndarray,
        valid: np.ndarray,
        hand_dim: int,
    ) -> None:
        valid = np.asarray(valid, dtype=bool)
        base_position_error = np.linalg.norm(base_pred[:, :2] - base_gt[:, :2], axis=-1)
        base_yaw_pred = np.arctan2(base_pred[:, 2], base_pred[:, 3])
        base_yaw_gt = np.arctan2(base_gt[:, 2], base_gt[:, 3])
        base_yaw_error = np.degrees(np.abs(wrap_angle(base_yaw_pred - base_yaw_gt)))

        eef_position_error = np.linalg.norm(manip_pred[:, :3] - manip_gt[:, :3], axis=-1)
        eef_rotation_pred = rotation6d_rows_to_matrix(manip_pred[:, 3:9])
        eef_rotation_gt = rotation6d_rows_to_matrix(manip_gt[:, 3:9])
        eef_orientation_error = rotation_geodesic_deg(
            eef_rotation_pred, eef_rotation_gt
        )

        if hand_dim:
            hand_abs_error = np.abs(
                manip_pred[:, 9 : 9 + hand_dim] - manip_gt[:, 9 : 9 + hand_dim]
            )
            hand_waypoint_mae = hand_abs_error.mean(axis=-1)
        else:
            hand_abs_error = np.empty((len(valid), 0))
            hand_waypoint_mae = np.full(len(valid), np.nan)

        base_rotation_pred = rz(base_yaw_pred)
        base_rotation_gt = rz(base_yaw_gt)
        base_translation_pred = np.pad(base_pred[:, :2], ((0, 0), (0, 1)))
        base_translation_gt = np.pad(base_gt[:, :2], ((0, 0), (0, 1)))
        rel_eef_position_pred = np.einsum(
            "hji,hj->hi",
            base_rotation_pred,
            manip_pred[:, :3] - base_translation_pred,
        )
        rel_eef_position_gt = np.einsum(
            "hji,hj->hi",
            base_rotation_gt,
            manip_gt[:, :3] - base_translation_gt,
        )
        relative_position_error = np.linalg.norm(
            rel_eef_position_pred - rel_eef_position_gt, axis=-1
        )
        rel_eef_rotation_pred = np.einsum(
            "hji,hjk->hik", base_rotation_pred, eef_rotation_pred
        )
        rel_eef_rotation_gt = np.einsum(
            "hji,hjk->hik", base_rotation_gt, eef_rotation_gt
        )
        relative_orientation_error = rotation_geodesic_deg(
            rel_eef_rotation_pred, rel_eef_rotation_gt
        )

        metric_arrays = {
            "base_position_error_m": base_position_error,
            "base_yaw_error_deg": base_yaw_error,
            "eef_position_error_m": eef_position_error,
            "eef_orientation_error_deg": eef_orientation_error,
            "hand_joint_mae": hand_waypoint_mae,
            "relative_eef_position_error_m": relative_position_error,
            "relative_eef_orientation_error_deg": relative_orientation_error,
        }
        for name, values in metric_arrays.items():
            self.values[name].extend(self._finite(values[valid]))
            for horizon_index, is_valid in enumerate(valid):
                if is_valid and np.isfinite(values[horizon_index]):
                    self.per_horizon[horizon_index][name].append(
                        float(values[horizon_index])
                    )
        if hand_dim:
            self.values["hand_joint_abs_error"].extend(
                self._finite(hand_abs_error[valid])
            )

        final_index = len(self.offsets) - 1
        if valid[final_index]:
            self.values["base_fde_m"].append(float(base_position_error[final_index]))
            self.values["base_final_yaw_error_deg"].append(
                float(base_yaw_error[final_index])
            )
            self.values["eef_fde_m"].append(float(eef_position_error[final_index]))
            self.values["eef_final_orientation_error_deg"].append(
                float(eef_orientation_error[final_index])
            )
            if hand_dim:
                self.values["hand_final_mae"].append(
                    float(hand_waypoint_mae[final_index])
                )

        row: dict[str, Any] = {
            "episode_index": episode_index,
            "frame_index": frame_index,
            "valid_waypoints": int(valid.sum()),
        }
        for name, values in metric_arrays.items():
            selected = np.asarray(values)[valid]
            row[name.replace("_error", "_mean_error")] = (
                float(np.nanmean(selected)) if selected.size else None
            )
        self.per_sample.append(row)
        self.predictions["episode_index"].append(np.asarray(episode_index))
        self.predictions["frame_index"].append(np.asarray(frame_index))
        self.predictions["base_pred"].append(base_pred.astype(np.float32))
        self.predictions["base_gt"].append(base_gt.astype(np.float32))
        self.predictions["manipulator_pred"].append(manip_pred.astype(np.float32))
        self.predictions["manipulator_gt"].append(manip_gt.astype(np.float32))
        self.predictions["plan_valid"].append(valid)

    @staticmethod
    def summarize_values(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "mean": None, "median": None, "p90": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "p90": float(np.quantile(array, 0.90)),
        }

    def summary(self) -> dict[str, Any]:
        def mean(name: str) -> float | None:
            values = self.values.get(name, [])
            return float(np.mean(values)) if values else None

        return {
            "num_samples": len(self.per_sample),
            "primary_metrics": {
                "base_ade_m": mean("base_position_error_m"),
                "base_fde_m": mean("base_fde_m"),
                "base_yaw_mae_deg": mean("base_yaw_error_deg"),
                "eef_position_ade_m": mean("eef_position_error_m"),
                "eef_position_fde_m": mean("eef_fde_m"),
                "eef_orientation_mae_deg": mean("eef_orientation_error_deg"),
                "hand_joint_mae": mean("hand_joint_abs_error"),
                "relative_eef_position_mae_m": mean(
                    "relative_eef_position_error_m"
                ),
                "relative_eef_orientation_mae_deg": mean(
                    "relative_eef_orientation_error_deg"
                ),
            },
            "metrics": {
                name: self.summarize_values(values)
                for name, values in sorted(self.values.items())
            },
            "per_horizon": [
                {
                    "offset": self.offsets[index],
                    "metrics": {
                        name: self.summarize_values(values)
                        for name, values in sorted(bucket.items())
                    },
                }
                for index, bucket in enumerate(self.per_horizon)
            ],
        }

    def save(self, output_dir: Path, metadata: dict[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = self.summary()
        summary["evaluation"] = metadata
        write_json(output_dir / "summary.json", summary)
        with (output_dir / "per_sample_metrics.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in self.per_sample:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        metric_names = sorted(
            {
                name
                for bucket in self.per_horizon
                for name in bucket
            }
        )
        with (output_dir / "per_horizon_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["horizon_index", "offset", "time_seconds"]
                + [f"{name}_mean" for name in metric_names]
                + [f"{name}_count" for name in metric_names],
            )
            writer.writeheader()
            control_fps = float(metadata["control_fps"])
            for index, bucket in enumerate(self.per_horizon):
                row: dict[str, Any] = {
                    "horizon_index": index,
                    "offset": self.offsets[index],
                    "time_seconds": self.offsets[index] / control_fps,
                }
                for name in metric_names:
                    stats = self.summarize_values(bucket.get(name, []))
                    row[f"{name}_mean"] = stats["mean"]
                    row[f"{name}_count"] = stats["count"]
                writer.writerow(row)
        np.savez_compressed(
            output_dir / "predictions.npz",
            **{
                key: np.stack(values)
                for key, values in self.predictions.items()
            },
        )


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    episode_ids, split_source = resolve_episode_split(dataset_root, args.split)

    if args.inspect_only:
        # The resolved training config is not needed to verify episode routing.
        episodes = read_jsonl(dataset_root / "meta/episodes.jsonl")
        selected = [row for row in episodes if int(row["episode_index"]) in episode_ids]
        print(
            json.dumps(
                {
                    "dataset_root": str(dataset_root),
                    "split": args.split,
                    "split_source": split_source,
                    "episode_ids": sorted(episode_ids),
                    "num_episodes": len(selected),
                    "num_frames": sum(int(row["length"]) for row in selected),
                },
                indent=2,
            )
        )
        return 0

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --inspect-only is used")
    checkpoint = args.checkpoint.resolve()
    device, mesh, rank, world_size = initialize_distributed()
    model, cfg = load_model(
        checkpoint=checkpoint,
        device=device,
        mesh=mesh,
        num_inference_steps=args.num_inference_steps,
    )
    dataset, model_transform, collator = build_dataset_and_collator(dataset_root, cfg)
    indices = select_dataset_indices(
        dataset,
        episode_ids=episode_ids,
        stride=args.sample_stride,
        max_samples=args.max_samples,
    )
    if args.batch_size != 1:
        raise ValueError(
            "Independent offline anchors currently require --batch-size 1 so "
            "each sample receives its own deterministic noise and fresh KV cache"
        )

    plan_transform = MobilePlanTransform(
        stats_path=dataset_root / "meta/plan_stats.json"
    )
    offsets = [int(value) for value in dataset.plan_offsets]
    accumulator = MetricAccumulator(offsets)
    if rank == 0:
        print(
            f"Evaluating {len(indices)} anchors from {len(episode_ids)} episodes "
            f"(split={args.split!r}, source={split_source})",
            flush=True,
        )

    for ordinal, batch_indices in enumerate(iter_batches(indices, args.batch_size), start=1):
        raw_samples = [dataset[index] for index in batch_indices]
        sample_ids = [
            (int(sample["episode_index"]), int(sample["frame_index"]))
            for sample in raw_samples
        ]
        current_seed = sample_seed(args.seed, sample_ids)
        reset_sampler_state(model.action_head, current_seed)
        model_samples = [model_transform(dict(sample)) for sample in raw_samples]
        batch = collator(model_samples)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            output = model.get_action(batch)

        if rank == 0:
            normalized_base = output["base_plan_pred"].detach().float().cpu()
            normalized_manipulator = (
                output["manipulator_plan_pred"].detach().float().cpu()
            )
            physical = plan_transform.unapply(
                {
                    "base_action": normalized_base,
                    "manipulator_action": normalized_manipulator,
                }
            )
            base_predictions = to_numpy(physical["base_plan"])
            manipulator_predictions = to_numpy(physical["manipulator_plan"])
            for local_index, sample in enumerate(raw_samples):
                accumulator.add(
                    episode_index=int(sample["episode_index"]),
                    frame_index=int(sample["frame_index"]),
                    base_pred=base_predictions[local_index],
                    base_gt=to_numpy(sample["base_plan"]),
                    manip_pred=manipulator_predictions[local_index],
                    manip_gt=to_numpy(sample["manipulator_plan"]),
                    valid=to_numpy(sample["plan_valid"]),
                    hand_dim=int(sample["hand_dim"]),
                )
            if ordinal == 1 or ordinal % 10 == 0 or ordinal == len(indices):
                print(f"[eval] {ordinal}/{len(indices)}", flush=True)
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else checkpoint / f"mobile_plan_eval_{args.split}"
        )
        accumulator.save(
            output_dir,
            metadata={
                "checkpoint": str(checkpoint),
                "dataset_root": str(dataset_root),
                "split": args.split,
                "split_source": split_source,
                "episode_ids": sorted(episode_ids),
                "sample_stride": args.sample_stride,
                "max_samples": args.max_samples,
                "seed": args.seed,
                "num_inference_steps": args.num_inference_steps,
                "world_size": world_size,
                "control_fps": float(dataset.control_fps),
                "plan_offsets": offsets,
                "observation_video_delta_indices": [0],
            },
        )
        print(f"Wrote evaluation to {output_dir}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
