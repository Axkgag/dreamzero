"""Serve MobileManiBench WAM with real-history cache rebuilding.

The client may send either one 30 Hz observation or a batch of observations on
each call.  Only timestamped real RGB/state samples are retained.  Every plan
request resets the model cache, prefills up to three complete 1.6 s real blocks,
and returns a 48-tick plan whose first 16 ticks are marked executable.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

EVAL_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/eval"
if str(EVAL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_SCRIPT_DIR))

import numpy as np
import torch
import tyro
from openpi_client.base_policy import BasePolicy

from eval_utils.policy_server import PolicyServerConfig, WebsocketPolicyServer
from groot.vla.data.transform.mobile_plan import MobilePlanTransform
from groot.vla.utils.mobilemanibench_receding_horizon import (
    RealObservationBuffer,
    TimedMobileObservation,
    executable_waypoint_indices,
    reset_wam_cache,
)
from scripts.eval.evaluate_mobilemanibench_multiblock_plan import (
    _physical_predictions,
    _resolve_plan_spec,
    _resolve_stats_path,
    build_transform_and_collator,
    prepare_inference_batch,
)
from scripts.eval.evaluate_mobilemanibench_plan import (
    initialize_distributed,
    load_model,
)


LOGGER = logging.getLogger(__name__)


def _first(obs: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in obs:
            return obs[key]
    raise KeyError(f"Observation is missing all supported keys: {keys}")


def _as_time_batch(value: Any, count: int, width: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim == (1 if width is not None else 3):
        result = result[None]
    if len(result) == 1 and count > 1:
        result = np.repeat(result, count, axis=0)
    if len(result) != count:
        raise ValueError(f"Expected {count} observations, got shape {result.shape}")
    return result


class MobileManiBenchWAMRuntime(BasePolicy):
    def __init__(
        self,
        *,
        model: Any,
        cfg: Any,
        model_transform: Any,
        collator: Any,
        plan_transform: MobilePlanTransform,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.model_transform = model_transform
        self.collator = collator
        self.plan_transform = plan_transform
        self.seed = int(seed)
        self.local_offsets = [int(value) for value in cfg.plan_local_offsets]
        self.anchors = [int(value) for value in cfg.block_anchor_offsets]
        self.blocks = len(self.anchors)
        self.waypoints = len(self.local_offsets)
        self.flow_tokens_per_block = 2 * self.waypoints
        inference = cfg.inference
        self.prediction_horizon_ticks = int(inference.prediction_horizon_ticks)
        self.execution_horizon_ticks = int(inference.execution_horizon_ticks)
        if not bool(inference.rebuild_cache_from_real_history):
            raise ValueError("This runtime requires rebuild_cache_from_real_history=true")
        self.buffer = RealObservationBuffer(
            video_fps=float(cfg.video_fps),
            block_duration_seconds=self.prediction_horizon_ticks
            / float(cfg.control_fps),
            max_history_blocks=self.blocks - 1,
        )
        self.prompt = ""

    def _append_observations(self, obs: dict[str, Any]) -> None:
        head = np.asarray(
            _first(obs, "video.head", "observation/exterior_image_0_left")
        )
        wrist = np.asarray(
            _first(obs, "video.wrist", "observation/wrist_image_left")
        )
        if head.ndim == 3:
            head = head[None]
        if wrist.ndim == 3:
            wrist = wrist[None]
        count = len(head)
        if len(wrist) != count:
            raise ValueError("Head and wrist history lengths differ")
        timestamps = np.asarray(
            obs.get("timestamps", obs.get("timestamp", time.monotonic())),
            dtype=np.float64,
        ).reshape(-1)
        if len(timestamps) == 1 and count > 1:
            raise ValueError("Batched RGB requires one timestamp per observation")
        eef_position = _as_time_batch(
            _first(obs, "state.eef_position", "observation/eef_position"),
            count,
            3,
        )
        eef_rotation = _as_time_batch(
            _first(obs, "state.eef_rotation_rpy", "observation/eef_rotation_rpy"),
            count,
            3,
        )
        latest = self.buffer.latest_timestamp
        latest = -np.inf if latest is None else latest
        for index, timestamp in enumerate(timestamps):
            if float(timestamp) <= latest:
                continue
            self.buffer.append(
                TimedMobileObservation(
                    timestamp=float(timestamp),
                    head_rgb=np.asarray(head[index]),
                    wrist_rgb=np.asarray(wrist[index]),
                    eef_position=np.asarray(eef_position[index], dtype=np.float32),
                    eef_rotation_rpy=np.asarray(
                        eef_rotation[index], dtype=np.float32
                    ),
                )
            )
            latest = float(timestamp)

    def _raw_sample(
        self, observations: list[TimedMobileObservation]
    ) -> dict[str, Any]:
        current = observations[-1]
        physical_state = np.concatenate(
            [current.eef_position, current.eef_rotation_rpy]
        ).astype(np.float32)
        video_count = len(observations)
        latent_count = 1 + max(0, video_count - 1) // 4
        return {
            "video.head": np.stack([item.head_rgb for item in observations]),
            "video.wrist": np.stack([item.wrist_rgb for item in observations]),
            "annotation.task": [self.prompt],
            "base_plan": np.zeros((self.blocks, self.waypoints, 4), np.float32),
            "manipulator_plan": np.zeros(
                (self.blocks, self.waypoints, int(self.cfg.max_manipulator_action_dim)),
                np.float32,
            ),
            "plan_valid": np.zeros((self.blocks, self.waypoints), np.bool_),
            "base_dim_mask": np.ones((self.blocks, self.waypoints, 4), np.bool_),
            "manipulator_dim_mask": np.ones(
                (self.blocks, self.waypoints, int(self.cfg.max_manipulator_action_dim)),
                np.bool_,
            ),
            "plan_local_offsets": np.asarray(self.local_offsets, np.int64),
            "plan_time_seconds": np.asarray(self.local_offsets, np.float32)
            / float(self.cfg.control_fps),
            "block_anchor_offsets": np.asarray(self.anchors, np.int64),
            "global_plan_offsets": np.asarray(
                [anchor + offset for anchor in self.anchors for offset in self.local_offsets],
                np.int64,
            ),
            "block_state_valid": np.ones(self.blocks, np.bool_),
            "block_valid": np.ones(self.blocks, np.bool_),
            "video_valid": np.ones(video_count, np.bool_),
            "video_latent_valid": np.ones(latent_count, np.bool_),
            "physical_block_state": np.repeat(
                physical_state[None], self.blocks, axis=0
            ),
            "eef_rotation_representation": str(
                self.cfg.eef_rotation_representation
            ),
            "sample_phase_id": np.int64(-1),
            "sample_task_id": np.int64(-1),
            "sampled_block_slot": np.int64(-1),
            "sampled_horizon": np.int64(-1),
            "full_window": np.bool_(True),
            "success_hold": np.bool_(False),
        }

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._append_observations(obs)
        if obs.get("prompt"):
            self.prompt = str(obs["prompt"])
        sequence = self.buffer.rebuild_sequence()
        action_head = self.model.action_head
        reset_wam_cache(action_head)
        output = None
        for observations in sequence:
            action_head.seed = self.seed
            batch = prepare_inference_batch(
                self._raw_sample(observations),
                self.model_transform,
                self.collator,
                self.flow_tokens_per_block,
            )
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                output = self.model.get_action(batch)
        assert output is not None
        base, manipulator, base_prior, eef_prior = _physical_predictions(
            output,
            self.plan_transform,
            int(action_head.manipulator_action_dim),
        )
        result: dict[str, Any] = {
            "base_plan": base.astype(np.float32),
            "manipulator_plan": manipulator.astype(np.float32),
            "plan_local_offsets": np.asarray(self.local_offsets, np.int64),
            "executable_waypoint_indices": np.asarray(
                executable_waypoint_indices(
                    self.local_offsets, self.execution_horizon_ticks
                ),
                np.int64,
            ),
            "prediction_horizon_ticks": self.prediction_horizon_ticks,
            "execution_horizon_ticks": self.execution_horizon_ticks,
            "real_history_blocks": len(sequence) - 1,
        }
        if base_prior is not None:
            result["base_prior"] = np.asarray(base_prior, np.float32)
        if eef_prior is not None:
            result["eef_prior"] = np.asarray(eef_prior, np.float32)
        return result

    def reset(self, reset_info: dict[str, Any]) -> None:
        del reset_info
        self.buffer.clear()
        self.prompt = ""
        reset_wam_cache(self.model.action_head)


def main(
    checkpoint: str,
    dataset_root: str = "/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_g1_5tasks/g1",
    port: int = 8000,
    host: str = "0.0.0.0",
    num_inference_steps: int = 16,
    seed: int = 42,
) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    checkpoint_path = Path(checkpoint).resolve()
    root = Path(dataset_root).resolve()
    device, mesh, rank, world_size = initialize_distributed()
    if world_size != 1 or rank != 0:
        raise ValueError("The websocket runtime currently supports one GPU")
    model, cfg = load_model(
        checkpoint_path, device, mesh, int(num_inference_steps)
    )
    anchors, offsets, _ = _resolve_plan_spec(cfg)
    stats_path = _resolve_stats_path(root, cfg, anchors, offsets)
    model_transform, collator = build_transform_and_collator(cfg, stats_path)
    plan_transform = MobilePlanTransform(
        stats_path=stats_path,
        eef_rotation_representation=str(cfg.eef_rotation_representation),
    )
    runtime = MobileManiBenchWAMRuntime(
        model=model,
        cfg=cfg,
        model_transform=model_transform,
        collator=collator,
        plan_transform=plan_transform,
        seed=seed,
    )
    server = WebsocketPolicyServer(
        policy=runtime,
        server_config=PolicyServerConfig(
            image_resolution=(160, 320),
            needs_wrist_camera=True,
            n_external_cameras=1,
            needs_session_id=True,
            action_space="cartesian_position",
        ),
        host=host,
        port=port,
    )
    LOGGER.info("Serving real-history MobileManiBench WAM on %s:%d", host, port)
    server.serve_forever()


if __name__ == "__main__":
    tyro.cli(main)
