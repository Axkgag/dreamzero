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
            state = np.concatenate(
                [
                    _numpy(result["state.eef_position"]),
                    _numpy(result["state.eef_rotation_rpy"]),
                ],
                axis=-1,
            ).astype(np.float32)
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
