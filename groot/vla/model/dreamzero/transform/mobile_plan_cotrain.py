"""DreamZero model transform and collator for dual MobileManiBench plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import PrivateAttr

from groot.vla.data.schema import EmbodimentTag

from .dreamzero_cotrain import DefaultDataCollator, DreamTransform


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class MobilePlanDataCollator(DefaultDataCollator):
    """Stack a research batch while preserving both semantic plan branches."""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().__call__(features)
        batch_size = len(features)
        expected = {
            "base_action": (batch_size, 6, 4),
            "manipulator_action": (batch_size, 6, 21),
            "base_action_mask": (batch_size, 6, 4),
            "manipulator_action_mask": (batch_size, 6, 21),
            "plan_time_offsets": (batch_size, 6),
        }
        for key, shape in expected.items():
            if tuple(batch[key].shape) != shape:
                raise ValueError(f"{key}: expected {shape}, got {tuple(batch[key].shape)}")
        return batch


class MobilePlanCotrainTransform(DreamTransform):
    """Adapt Phase-1 samples to DreamZero without collapsing plan semantics."""

    plan_horizon: int = 6
    base_action_dim: int = 4
    manipulator_action_dim: int = 21
    state_stats_path: str | Path
    control_fps: float = 30.0
    image_resolution_height: int = 176
    image_resolution_width: int = 320
    _mobile_state_stats: dict[str, Any] = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with Path(self.state_stats_path).open("r", encoding="utf-8") as handle:
            self._mobile_state_stats = json.load(handle)["observation.state"]
        if self.action_horizon != 2 * self.plan_horizon:
            raise ValueError(
                "DreamZero packed action_horizon must equal "
                f"2 * plan_horizon ({2 * self.plan_horizon})"
            )
        # This dedicated transform is only used by the xdof MobileManiBench root.
        self.embodiment_tag = EmbodimentTag.XDOF

    def _resize_video(self, video: np.ndarray) -> np.ndarray:
        """Resize each camera view before building the 2x2 Wan input grid."""
        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError(f"Expected THWC RGB video, got {video.shape}")
        target = (self.image_resolution_height, self.image_resolution_width)
        if video.shape[1:3] == target:
            return video
        source_dtype = video.dtype
        tensor = torch.as_tensor(video).permute(0, 3, 1, 2).float()
        resized = F.interpolate(
            tensor,
            size=target,
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        if np.issubdtype(source_dtype, np.integer):
            resized = resized.round().clamp(0, 255)
        return resized.to(torch.uint8).cpu().numpy()

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        q01 = np.asarray(self._mobile_state_stats["q01"], dtype=np.float32)
        q99 = np.asarray(self._mobile_state_stats["q99"], dtype=np.float32)
        varying = q99 != q01
        normalized = np.zeros_like(state, dtype=np.float32)
        normalized[..., varying] = (
            2.0
            * (state[..., varying] - q01[varying])
            / (q99[varying] - q01[varying])
            - 1.0
        )
        return np.clip(normalized, -1.0, 1.0)

    def _canonicalize(self, data: dict[str, Any]) -> dict[str, Any]:
        result = dict(data)
        if "video" not in result:
            head = self._resize_video(_numpy(result["video.head"]))
            wrist = self._resize_video(_numpy(result["video.wrist"]))
            # DreamTransform's generic grid places view 0 at top-left and view
            # 2 at top-right. Keep the unused lower-left slot black.
            black = np.zeros_like(head)
            result["video"] = np.stack([head, black, wrist], axis=1)
        if "state" not in result:
            state = (
                _numpy(result["physical_block_state"]).astype(np.float32)
                if "physical_block_state" in result
                else np.concatenate(
                    [
                        _numpy(result["state.eef_position"]),
                        _numpy(result["state.eef_rotation_rpy"]),
                    ],
                    axis=-1,
                ).astype(np.float32)
            )
            result["physical_block_state"] = state.copy()
            result["state"] = self._normalize_state(state)
        return result

    def _prepare_action(self, data: dict):
        base = _numpy(data["base_action"]).astype(np.float32)
        manipulator = _numpy(data["manipulator_action"]).astype(np.float32)
        base_mask = _numpy(data["base_action_mask"]).astype(bool)
        manipulator_mask = _numpy(data["manipulator_action_mask"]).astype(bool)
        if base.shape != (self.plan_horizon, self.base_action_dim):
            raise ValueError(f"Unexpected base action shape: {base.shape}")
        if manipulator.shape != (
            self.plan_horizon,
            self.manipulator_action_dim,
        ):
            raise ValueError(f"Unexpected manipulator action shape: {manipulator.shape}")

        packed_base = np.zeros(
            (self.plan_horizon, self.manipulator_action_dim), dtype=np.float32
        )
        packed_base[:, : self.base_action_dim] = base
        packed_base_mask = np.zeros_like(packed_base, dtype=bool)
        packed_base_mask[:, : self.base_action_dim] = base_mask
        action = np.concatenate([packed_base, manipulator], axis=0)
        action_mask = np.concatenate([packed_base_mask, manipulator_mask], axis=0)
        return action, action_mask, action.shape[0]

    def apply_single(self, data: dict) -> dict:
        data = self._canonicalize(data)
        transformed = super().apply_single(data)
        for key in (
            "base_action",
            "manipulator_action",
            "base_action_mask",
            "manipulator_action_mask",
            "plan_valid",
            "plan_time_offsets",
            "plan_time_seconds",
        ):
            transformed[key] = _numpy(data[key])
        return transformed

    def apply(self, data: dict) -> dict:
        # The torch DataLoader calls transforms per sample; batching belongs to
        # MobilePlanDataCollator, which retains the semantic branch keys.
        return self.apply_single(data)


class MobileBlockPlanDataCollator(DefaultDataCollator):
    """Stack multiblock samples while retaining their semantic axes."""

    def __init__(
        self,
        tokenizer_path: str = "google/umt5-xxl",
        max_length: int = 512,
        num_views: int = 1,
        embodiment_tag_mapping=None,
        num_plan_blocks: int = 4,
        plan_waypoints_per_block: int = 2,
        base_action_dim: int = 4,
        manipulator_action_dim: int = 21,
        max_state_dim: int = 64,
    ):
        super().__init__(
            tokenizer_path=tokenizer_path,
            max_length=max_length,
            num_views=num_views,
            embodiment_tag_mapping=embodiment_tag_mapping,
        )
        self.num_plan_blocks = int(num_plan_blocks)
        self.plan_waypoints_per_block = int(plan_waypoints_per_block)
        self.base_action_dim = int(base_action_dim)
        self.manipulator_action_dim = int(manipulator_action_dim)
        self.max_state_dim = int(max_state_dim)
        if self.num_plan_blocks <= 0 or self.plan_waypoints_per_block <= 0:
            raise ValueError("Plan block and waypoint counts must be positive")

    def _validate_batch_shapes(
        self, batch: Dict[str, Any], batch_size: int
    ) -> None:
        blocks = self.num_plan_blocks
        waypoints = self.plan_waypoints_per_block
        packed_width = 2 * waypoints

        expected = {
            "base_action": (
                batch_size,
                blocks,
                waypoints,
                self.base_action_dim,
            ),
            "manipulator_action": (
                batch_size,
                blocks,
                waypoints,
                self.manipulator_action_dim,
            ),
            "base_action_mask": (
                batch_size,
                blocks,
                waypoints,
                self.base_action_dim,
            ),
            "manipulator_action_mask": (
                batch_size,
                blocks,
                waypoints,
                self.manipulator_action_dim,
            ),
            "plan_local_offsets": (batch_size, waypoints),
            "block_anchor_offsets": (batch_size, blocks),
            "global_plan_offsets": (batch_size, blocks * waypoints),
        }
        for key, shape in expected.items():
            if tuple(batch[key].shape) != shape:
                raise ValueError(
                    f"{key}: expected {shape}, got {tuple(batch[key].shape)}"
                )
        expected_action = (blocks * packed_width, self.manipulator_action_dim)
        if tuple(batch["action"].shape[1:]) != expected_action:
            raise ValueError(
                f"Expected packed action [B,{expected_action[0]},"
                f"{expected_action[1]}], got {batch['action'].shape}"
            )
        expected_state = (blocks, self.max_state_dim)
        if tuple(batch["state"].shape[1:]) != expected_state:
            raise ValueError(
                f"Expected state [B,{blocks},{self.max_state_dim}], "
                f"got {batch['state'].shape}"
            )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().__call__(features)
        self._validate_batch_shapes(batch, len(features))
        return batch


class MobileBlockPlanCotrainTransform(MobilePlanCotrainTransform):
    """Pack configurable local dual-plan chunks for DreamZero block routing."""

    num_plan_blocks: int = 4
    plan_waypoints_per_block: int = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.plan_horizon != self.plan_waypoints_per_block:
            raise ValueError("plan_horizon must equal waypoints per block")
        if self.action_horizon != 2 * self.plan_waypoints_per_block:
            raise ValueError("action_horizon must be the single-block flow width")

    def _prepare_action(self, data: dict):
        base = _numpy(data["base_action"]).astype(np.float32)
        manipulator = _numpy(data["manipulator_action"]).astype(np.float32)
        base_mask = _numpy(data["base_action_mask"]).astype(bool)
        manipulator_mask = _numpy(data["manipulator_action_mask"]).astype(bool)
        expected_base = (
            self.num_plan_blocks,
            self.plan_waypoints_per_block,
            self.base_action_dim,
        )
        expected_manipulator = (
            self.num_plan_blocks,
            self.plan_waypoints_per_block,
            self.manipulator_action_dim,
        )
        if base.shape != expected_base:
            raise ValueError(f"Unexpected block Base action shape: {base.shape}")
        if manipulator.shape != expected_manipulator:
            raise ValueError(
                f"Unexpected block Manipulator action shape: {manipulator.shape}"
            )
        padded_base = np.zeros(
            (*base.shape[:-1], self.manipulator_action_dim), dtype=np.float32
        )
        padded_base[..., : self.base_action_dim] = base
        padded_base_mask = np.zeros_like(padded_base, dtype=bool)
        padded_base_mask[..., : self.base_action_dim] = base_mask
        block_action = np.concatenate([padded_base, manipulator], axis=1)
        block_mask = np.concatenate([padded_base_mask, manipulator_mask], axis=1)
        action = block_action.reshape(
            self.num_plan_blocks * self.action_horizon,
            self.manipulator_action_dim,
        )
        action_mask = block_mask.reshape(action.shape)
        return action, action_mask, action.shape[0]

    def apply_single(self, data: dict) -> dict:
        data = self._canonicalize(data)
        transformed = DreamTransform.apply_single(self, data)
        for key in (
            "base_action",
            "manipulator_action",
            "base_action_mask",
            "manipulator_action_mask",
            "plan_valid",
            "plan_local_offsets",
            "plan_time_seconds",
            "block_anchor_offsets",
            "global_plan_offsets",
            "block_state_valid",
            "physical_block_state",
        ):
            transformed[key] = _numpy(data[key])
        return transformed
