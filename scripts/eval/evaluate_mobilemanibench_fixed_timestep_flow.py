#!/usr/bin/env python3
"""Evaluate MobileManiBench action-flow errors at fixed diffusion times.

This diagnostic follows the training-time teacher-forcing path rather than the
deployment sampler.  It loads the model and sample once, caches the expensive
text/image/video encodings, and then evaluates every requested diffusion
timestep and noise seed without reloading the checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate

from groot.vla.data.dataset import MobileManiBenchPlanDataset

from evaluate_mobilemanibench_plan import initialize_distributed, load_model


DEFAULT_TIMESTEPS = (50.0, 100.0, 250.0, 500.0, 750.0, 900.0)
DEFAULT_SEEDS = (1140, 1141, 1142, 1143, 1144)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument(
        "--timesteps",
        type=float,
        nargs="+",
        default=list(DEFAULT_TIMESTEPS),
        help=(
            "Requested scheduler timestep values, not scheduler array indices. "
            "The nearest value in the 1000-step training schedule is used."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Noise seeds evaluated at every requested timestep.",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def locate_anchor(
    dataset: MobileManiBenchPlanDataset,
    episode_index: int,
    frame_index: int,
) -> int:
    matches = [
        index
        for index, (episode_id, frame_id) in enumerate(dataset.all_steps)
        if int(episode_id) == episode_index and int(frame_id) == frame_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one anchor for episode={episode_index}, frame={frame_index}; "
            f"found {len(matches)}"
        )
    return matches[0]


def build_training_sample(dataset_root: Path, cfg: Any, episode: int, frame: int):
    video_delta_indices = [
        int(value) for value in cfg.train_dataset.video_delta_indices
    ]
    dataset = MobileManiBenchPlanDataset(
        dataset_path=dataset_root,
        video_delta_indices=video_delta_indices,
        load_videos=True,
        video_backend=str(cfg.train_dataset.video_backend),
        max_manipulator_dim=int(cfg.train_dataset.max_manipulator_dim),
        plan_transform=None,
    )
    index = locate_anchor(dataset, episode, frame)
    raw_sample = dataset[index]
    transform = instantiate(cfg.train_dataset.plan_transform)
    # Unlike deployment evaluation, this diagnostic needs the normalized GT
    # action/action_mask fields emitted only by the transform's training mode.
    # The MobileManiBench config uses zero language dropout, so this does not
    # introduce stochastic language conditioning.
    transform.train()
    collator = instantiate(cfg.data_collator)
    batch = collator([transform(dict(raw_sample))])
    return batch, raw_sample, video_delta_indices


def normalize_training_video(action_head: Any, images: torch.Tensor) -> torch.Tensor:
    videos = images.permute(0, 4, 1, 2, 3)
    if videos.dtype == torch.uint8:
        videos = videos.float() / 255.0
        batch, channels, frames, height, width = videos.shape
        videos = videos.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        videos = action_head.normalize_video(videos)
        videos = videos.reshape(
            batch, frames, channels, height, width
        ).permute(0, 2, 1, 3, 4)
    videos = videos.to(device=action_head._device, dtype=action_head.dtype)

    target_height = getattr(action_head.config, "target_video_height", None)
    target_width = getattr(action_head.config, "target_video_width", None)
    if target_height is None or target_width is None:
        if getattr(action_head.model, "frame_seqlen", None) in (50, 55):
            target_height, target_width = 176, 320
    if target_height is not None and target_width is not None:
        _, _, _, height, width = videos.shape
        if (height, width) != (target_height, target_width):
            batch, channels, frames, _, _ = videos.shape
            videos = torch.nn.functional.interpolate(
                videos.reshape(batch * frames, channels, height, width),
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch, channels, frames, target_height, target_width)
    return videos


def prepare_cached_inputs(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    _, action_input = model.prepare_input(batch)
    action_head = model.action_head
    action_head.set_frozen_modules_to_eval_mode()
    action_model_kwargs = action_head.prepare_action_model_kwargs(action_input)

    videos = normalize_training_video(action_head, action_input.images)
    prompt_embeddings = action_head.encode_prompt(
        action_input.text,
        action_input.text_attention_mask,
    ).to(action_head._device)
    latents = action_head.encode_video(
        videos,
        action_head.tiled,
        (action_head.tile_size_height, action_head.tile_size_width),
        (action_head.tile_stride_height, action_head.tile_stride_width),
    ).to(action_head._device)

    _, _, num_frames, height, width = videos.shape
    image = videos[:, :, :1].transpose(1, 2)
    clip_features, image_condition, _ = action_head.encode_image(
        image, num_frames, height, width
    )

    # The training forward transposes VAE latents from BCFHW to BFCHW before
    # adding noise and passes BFCHW back to the DiT as BCFHW.
    clean_latents = latents.transpose(1, 2)
    tokens_per_frame = (clean_latents.shape[3] // 2) * (
        clean_latents.shape[4] // 2
    )
    sequence_length = clean_latents.shape[1] * tokens_per_frame

    action_head.validate_action_video_layout(
        action_input.action,
        clean_latents,
        action_input.state,
        videos,
        clean_latents,
    )
    return {
        "action_input": action_input,
        "action_model_kwargs": action_model_kwargs,
        "videos": videos,
        "clean_latents": clean_latents,
        "prompt_embeddings": prompt_embeddings,
        "clip_features": clip_features.to(action_head._device),
        "image_condition": image_condition.to(action_head._device),
        "sequence_length": sequence_length,
    }


def resolve_timestep(scheduler: Any, requested: float) -> dict[str, float | int]:
    timesteps = scheduler.timesteps.detach().float().cpu()
    index = int(torch.argmin(torch.abs(timesteps - float(requested))).item())
    actual = float(timesteps[index].item())
    sigma = float(scheduler.sigmas[index].item())
    return {
        "requested_timestep": float(requested),
        "timestep_id": index,
        "actual_timestep": actual,
        "sigma": sigma,
    }


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float | None:
    active = mask.bool()
    if not bool(active.any()):
        return None
    error = torch.square(prediction.float() - target.float())
    return float(error[active].mean().item())


def slice_metrics(
    action_head: Any,
    prediction: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor,
    training_weight: float,
) -> dict[str, float | None]:
    horizon = int(action_head.plan_horizon)
    slices = {
        "base_xy_mse": (
            prediction[:, :horizon, 0:2],
            target[:, :horizon, 0:2],
            action_mask[:, :horizon, 0:2],
        ),
        "base_yaw_sincos_mse": (
            prediction[:, :horizon, 2:4],
            target[:, :horizon, 2:4],
            action_mask[:, :horizon, 2:4],
        ),
        "base_total_mse": (
            prediction[:, :horizon, : action_head.base_action_dim],
            target[:, :horizon, : action_head.base_action_dim],
            action_mask[:, :horizon, : action_head.base_action_dim],
        ),
        "eef_position_mse": (
            prediction[:, horizon:, 0:3],
            target[:, horizon:, 0:3],
            action_mask[:, horizon:, 0:3],
        ),
        "eef_rotation6d_mse": (
            prediction[:, horizon:, 3:9],
            target[:, horizon:, 3:9],
            action_mask[:, horizon:, 3:9],
        ),
        "hand_configuration_mse": (
            prediction[:, horizon:, 9:],
            target[:, horizon:, 9:],
            action_mask[:, horizon:, 9:],
        ),
        "manipulator_total_mse": (
            prediction[:, horizon:, : action_head.manipulator_action_dim],
            target[:, horizon:, : action_head.manipulator_action_dim],
            action_mask[:, horizon:, : action_head.manipulator_action_dim],
        ),
    }
    result = {
        name: masked_mse(pred, truth, mask)
        for name, (pred, truth, mask) in slices.items()
    }
    result["base_weighted_loss"] = (
        None
        if result["base_total_mse"] is None
        else result["base_total_mse"] * training_weight
    )
    result["manipulator_weighted_loss"] = (
        None
        if result["manipulator_total_mse"] is None
        else result["manipulator_total_mse"] * training_weight
    )
    return result


def evaluate_one(
    action_head: Any,
    cached: dict[str, Any],
    timestep_info: dict[str, float | int],
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    action_input = cached["action_input"]
    clean_latents = cached["clean_latents"]
    actions = action_input.action
    device = actions.device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    video_noise = torch.randn(
        clean_latents.shape,
        generator=generator,
        device=device,
        dtype=clean_latents.dtype,
    )
    action_noise = torch.randn(
        actions.shape,
        generator=generator,
        device=device,
        dtype=actions.dtype,
    )

    actual_timestep = float(timestep_info["actual_timestep"])
    video_timestep = torch.full(
        clean_latents.shape[:2],
        actual_timestep,
        device=device,
        dtype=torch.float32,
    )
    action_timestep = torch.full(
        actions.shape[:2],
        actual_timestep,
        device=device,
        dtype=torch.float32,
    )
    noisy_latents = action_head.scheduler.add_noise(
        clean_latents.flatten(0, 1),
        video_noise.flatten(0, 1),
        video_timestep.flatten(0, 1),
    ).unflatten(0, clean_latents.shape[:2])
    noisy_actions = action_head.scheduler.add_noise(
        actions.flatten(0, 1),
        action_noise.flatten(0, 1),
        action_timestep.flatten(0, 1),
    ).unflatten(0, actions.shape[:2])
    target_action_flow = action_head.scheduler.training_target(
        actions,
        action_noise,
        action_timestep,
    )

    # _forward_train relies on its gradient-checkpointing wrapper to strip the
    # per-block KV-cache return value. Enable that control-flow path exactly as
    # in training. load_model() has already set requires_grad_(False), so this
    # does not accumulate parameter gradients or build a trainable graph.
    with torch.enable_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        _, predicted_action_flow = action_head.model(
            noisy_latents.transpose(1, 2),
            timestep=video_timestep,
            clip_feature=cached["clip_features"],
            y=cached["image_condition"],
            context=cached["prompt_embeddings"],
            seq_len=cached["sequence_length"],
            state=action_input.state,
            embodiment_id=action_input.embodiment_id,
            action=noisy_actions,
            timestep_action=action_timestep,
            clean_x=clean_latents.transpose(1, 2),
            **cached["action_model_kwargs"],
        )

    weight = float(
        action_head.scheduler.training_weight(
            action_timestep[:, :1].flatten()
        )[0].item()
    )
    row: dict[str, Any] = {
        **timestep_info,
        "noise_seed": int(seed),
        "training_weight": weight,
    }
    row.update(
        slice_metrics(
            action_head,
            predicted_action_flow,
            target_action_flow,
            action_input.action_mask,
            weight,
        )
    )
    arrays = {
        "predicted_action_flow": predicted_action_flow.detach().float().cpu().numpy(),
        "target_action_flow": target_action_flow.detach().float().cpu().numpy(),
        "noisy_action": noisy_actions.detach().float().cpu().numpy(),
    }
    return row, arrays


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = [
        "base_xy_mse",
        "base_yaw_sincos_mse",
        "base_total_mse",
        "eef_position_mse",
        "eef_rotation6d_mse",
        "hand_configuration_mse",
        "manipulator_total_mse",
        "base_weighted_loss",
        "manipulator_weighted_loss",
    ]
    buckets: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[float(row["requested_timestep"])].append(row)

    summaries: list[dict[str, Any]] = []
    for requested in sorted(buckets):
        bucket = buckets[requested]
        summary: dict[str, Any] = {
            "requested_timestep": requested,
            "actual_timestep": bucket[0]["actual_timestep"],
            "timestep_id": bucket[0]["timestep_id"],
            "sigma": bucket[0]["sigma"],
            "training_weight": bucket[0]["training_weight"],
            "num_seeds": len(bucket),
        }
        for name in metric_names:
            values = np.asarray(
                [row[name] for row in bucket if row[name] is not None],
                dtype=np.float64,
            )
            summary[f"{name}_mean"] = (
                float(values.mean()) if values.size else None
            )
            summary[f"{name}_std"] = (
                float(values.std()) if values.size else None
            )
        summaries.append(summary)
    return summaries


def write_outputs(
    output_dir: Path,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    arrays: dict[str, list[np.ndarray]],
    action: torch.Tensor,
    action_mask: torch.Tensor,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "summary.json",
        {
            "evaluation": metadata,
            "per_timestep": summaries,
        },
    )
    if rows:
        with (output_dir / "per_seed_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if summaries:
        with (output_dir / "per_timestep_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    np.savez_compressed(
        output_dir / "flow_predictions.npz",
        **{key: np.concatenate(values, axis=0) for key, values in arrays.items()},
        action=action.detach().float().cpu().numpy(),
        action_mask=action_mask.detach().bool().cpu().numpy(),
        requested_timestep=np.asarray(
            [row["requested_timestep"] for row in rows], dtype=np.float32
        ),
        actual_timestep=np.asarray(
            [row["actual_timestep"] for row in rows], dtype=np.float32
        ),
        timestep_id=np.asarray(
            [row["timestep_id"] for row in rows], dtype=np.int64
        ),
        sigma=np.asarray([row["sigma"] for row in rows], dtype=np.float32),
        noise_seed=np.asarray([row["noise_seed"] for row in rows], dtype=np.int64),
    )


def main() -> int:
    args = parse_args()
    if not args.timesteps:
        raise ValueError("--timesteps must contain at least one value")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one value")

    checkpoint = args.checkpoint.resolve()
    dataset_root = args.dataset_root.resolve()
    device, mesh, rank, world_size = initialize_distributed()
    model, cfg = load_model(
        checkpoint=checkpoint,
        device=device,
        mesh=mesh,
        num_inference_steps=16,
    )
    batch, raw_sample, video_delta_indices = build_training_sample(
        dataset_root,
        cfg,
        args.episode_index,
        args.frame_index,
    )

    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        cached = prepare_cached_inputs(model, batch)

    timestep_infos = [
        resolve_timestep(model.action_head.scheduler, value)
        for value in args.timesteps
    ]
    rows: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    total = len(timestep_infos) * len(args.seeds)
    completed = 0

    if rank == 0:
        print(
            f"Loaded checkpoint and cached sample once; evaluating "
            f"{len(timestep_infos)} timesteps x {len(args.seeds)} seeds "
            f"= {total} forwards",
            flush=True,
        )

    for timestep_info in timestep_infos:
        for seed in args.seeds:
            row, current_arrays = evaluate_one(
                model.action_head,
                cached,
                timestep_info,
                seed,
            )
            completed += 1
            if rank == 0:
                rows.append(row)
                for name, value in current_arrays.items():
                    arrays[name].append(value)
                print(
                    f"[flow] {completed}/{total} "
                    f"requested_t={timestep_info['requested_timestep']:.1f} "
                    f"actual_t={timestep_info['actual_timestep']:.3f} "
                    f"sigma={timestep_info['sigma']:.5f} seed={seed}",
                    flush=True,
                )
            if world_size > 1:
                dist.barrier()

    if rank == 0:
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else checkpoint
            / (
                "mobile_plan_fixed_timestep_flow_"
                f"ep{args.episode_index:03d}_frame{args.frame_index:04d}"
            )
        )
        summaries = summarize_rows(rows)
        metadata = {
            "checkpoint": str(checkpoint),
            "dataset_root": str(dataset_root),
            "episode_index": args.episode_index,
            "frame_index": args.frame_index,
            "requested_timesteps": [float(value) for value in args.timesteps],
            "noise_seeds": [int(value) for value in args.seeds],
            "world_size": world_size,
            "video_delta_indices": video_delta_indices,
            "model_loaded_once": True,
            "sample_encoded_once": True,
            "teacher_forcing_clean_x": True,
            "video_action_timestep_coupled": True,
            "raw_hand_dim": int(raw_sample["hand_dim"]),
        }
        write_outputs(
            output_dir,
            metadata,
            rows,
            summaries,
            arrays,
            cached["action_input"].action,
            cached["action_input"].action_mask,
        )
        print(f"Wrote fixed-timestep flow diagnostics to {output_dir}", flush=True)

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
