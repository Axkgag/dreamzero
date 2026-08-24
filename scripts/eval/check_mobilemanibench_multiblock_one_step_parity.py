#!/usr/bin/env python3
"""Capture or compare one deterministic multiblock training DiT prediction.

Run ``capture`` once per execution mode (merged/unmerged LoRA, one/two GPUs),
then use ``compare`` on rank-0 artifacts.  This checks the raw one-step action
velocity before an ODE solver can amplify numerical differences.
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from evaluate_mobilemanibench_multiblock_plan import (
    ORACLE_MODE,
    _resolve_plan_spec,
    _resolve_stats_path,
    build_datasets,
    build_transform_and_collator,
)
from evaluate_mobilemanibench_plan import initialize_distributed, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--checkpoint", type=Path, required=True)
    capture.add_argument("--dataset-root", type=Path, required=True)
    capture.add_argument("--dataset-index", type=int, default=0)
    capture.add_argument("--seed", type=int, default=1140)
    capture.add_argument("--lora-mode", choices=("merged", "unmerged"), required=True)
    capture.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--atol", type=float, default=1e-2)
    compare.add_argument("--rtol", type=float, default=1e-2)
    return parser.parse_args()


def _capture(args: argparse.Namespace) -> int:
    checkpoint = args.checkpoint.resolve()
    dataset_root = args.dataset_root.resolve()
    cfg = OmegaConf.load(checkpoint / "experiment_cfg/conf.yaml")
    anchors, local_offsets, block_stride = _resolve_plan_spec(cfg)
    stats_path = _resolve_stats_path(dataset_root, cfg, anchors, local_offsets)
    dataset = build_datasets(
        dataset_root,
        cfg,
        anchors,
        local_offsets,
        block_stride,
        ORACLE_MODE,
    )["root"]
    if not 0 <= args.dataset_index < len(dataset):
        raise IndexError(
            f"dataset-index {args.dataset_index} is outside [0,{len(dataset)})"
        )

    device, mesh, rank, world_size = initialize_distributed()
    model, cfg = load_model(
        checkpoint,
        device,
        mesh,
        num_inference_steps=16,
        merge_lora=args.lora_mode == "merged",
    )
    transform, collator = build_transform_and_collator(cfg, stats_path)
    raw = dataset[args.dataset_index]
    batch = collator([transform(dict(raw))])

    captured: dict[str, torch.Tensor] = {}
    action_head = model.action_head
    original = action_head.compute_action_losses

    def capture_losses(this: Any, *positional: Any, **keyword: Any):
        del this
        names = (
            "action_noise_pred",
            "training_target_action",
            "action_mask",
            "has_real_action",
            "timestep_action",
        )
        for index, name in enumerate(names):
            value = keyword.get(name, positional[index] if index < len(positional) else None)
            if torch.is_tensor(value):
                captured[name] = value.detach().float().cpu()
        for name in ("noisy_actions", "clean_actions"):
            value = keyword.get(name)
            if torch.is_tensor(value):
                captured[name] = value.detach().float().cpu()
        return original(*positional, **keyword)

    action_head.compute_action_losses = types.MethodType(capture_losses, action_head)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        output = model(batch)
    action_head.compute_action_losses = original

    if rank == 0:
        if "action_noise_pred" not in captured:
            raise RuntimeError("The training forward did not expose action_noise_pred")
        metadata = {
            "checkpoint": str(checkpoint),
            "dataset_root": str(dataset_root),
            "dataset_index": args.dataset_index,
            "episode_index": int(raw["episode_index"]),
            "frame_index": int(raw["frame_index"]),
            "seed": args.seed,
            "lora_mode": args.lora_mode,
            "world_size": world_size,
        }
        arrays = {name: value.numpy() for name, value in captured.items()}
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
        arrays["loss"] = np.asarray(float(output["loss"].detach().float().cpu()))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **arrays)
        print(json.dumps(metadata, indent=2), flush=True)
        print(f"Wrote {args.output}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


def _compare(args: argparse.Namespace) -> int:
    with np.load(args.reference, allow_pickle=False) as reference, np.load(
        args.candidate, allow_pickle=False
    ) as candidate:
        names = sorted(
            set(reference.files) & set(candidate.files) - {"metadata_json"}
        )
        report: dict[str, Any] = {}
        passed = True
        for name in names:
            left = np.asarray(reference[name])
            right = np.asarray(candidate[name])
            if left.shape != right.shape:
                report[name] = {
                    "shape_reference": list(left.shape),
                    "shape_candidate": list(right.shape),
                    "allclose": False,
                }
                passed = False
                continue
            difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
            close = bool(np.allclose(left, right, atol=args.atol, rtol=args.rtol))
            report[name] = {
                "shape": list(left.shape),
                "max_abs": float(difference.max(initial=0.0)),
                "mean_abs": float(difference.mean()) if difference.size else 0.0,
                "allclose": close,
            }
            passed = passed and close
    print(
        json.dumps(
            {
                "reference": str(args.reference),
                "candidate": str(args.candidate),
                "atol": args.atol,
                "rtol": args.rtol,
                "passed": passed,
                "tensors": report,
            },
            indent=2,
        )
    )
    return 0 if passed else 2


def main() -> int:
    args = parse_args()
    return _capture(args) if args.command == "capture" else _compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
