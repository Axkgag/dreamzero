"""Slice-aware normalization for MobileManiBench two-branch action plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import Field, PrivateAttr, model_validator

from .base import InvertibleModalityTransform


class MobilePlanTransform(InvertibleModalityTransform):
    """Normalize only translational and hand slices of a realized plan."""

    apply_to: list[str] = Field(
        default_factory=lambda: ["base_plan", "manipulator_plan"]
    )
    stats_path: str | Path | None = None
    statistics: dict[str, Any] | None = None
    validate_geometry: bool = True
    geometry_atol: float = 5e-3
    _stats: dict[str, Any] = PrivateAttr()

    @model_validator(mode="after")
    def _load_statistics(self) -> "MobilePlanTransform":
        if (self.stats_path is None) == (self.statistics is None):
            raise ValueError("Provide exactly one of stats_path or statistics")
        if self.statistics is not None:
            self._stats = self.statistics
        else:
            path = Path(self.stats_path)  # type: ignore[arg-type]
            with path.open("r", encoding="utf-8") as handle:
                self._stats = json.load(handle)
        for key in ("base_xy", "eef_xyz", "hand"):
            if key not in self._stats["statistics"]:
                raise ValueError(f"plan stats are missing statistics.{key}")
        return self

    @staticmethod
    def _tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype) if dtype is not None else value
        return torch.as_tensor(value, dtype=dtype)

    def _q99(self, value: torch.Tensor, name: str, inverse: bool = False) -> torch.Tensor:
        stats = self._stats["statistics"][name]
        q01 = torch.as_tensor(stats["q01"], dtype=value.dtype, device=value.device)
        q99 = torch.as_tensor(stats["q99"], dtype=value.dtype, device=value.device)
        varying = q99 != q01
        if inverse:
            result = (value + 1.0) * 0.5 * (q99 - q01) + q01
            return torch.where(varying, result, q01)
        safe_width = torch.where(varying, q99 - q01, torch.ones_like(q99))
        result = 2.0 * (value - q01) / safe_width - 1.0
        result = torch.where(varying, result, torch.zeros_like(result))
        return torch.clamp(result, -1.0, 1.0)

    def _check_geometry(
        self, base: torch.Tensor, manipulator: torch.Tensor, valid: torch.Tensor
    ) -> None:
        if not torch.isfinite(base).all() or not torch.isfinite(manipulator).all():
            raise ValueError("Non-finite value found in MobileManiBench plan")
        active = valid.bool()
        if not torch.any(active):
            return
        base_active = base[active]
        manip_active = manipulator[active]
        yaw_norm = torch.linalg.vector_norm(base_active[..., 2:4], dim=-1)
        rot = manip_active[..., 3:9].reshape(-1, 2, 3)
        row_norm = torch.linalg.vector_norm(rot, dim=-1)
        row_dot = torch.sum(rot[:, 0] * rot[:, 1], dim=-1)
        if not torch.allclose(
            yaw_norm, torch.ones_like(yaw_norm), atol=self.geometry_atol, rtol=0
        ):
            raise ValueError("Base yaw sin/cos does not have unit norm")
        if not torch.allclose(
            row_norm, torch.ones_like(row_norm), atol=self.geometry_atol, rtol=0
        ) or not torch.allclose(
            row_dot, torch.zeros_like(row_dot), atol=self.geometry_atol, rtol=0
        ):
            raise ValueError("EEF rotation6d rows are not orthonormal")

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        base = self._tensor(data["base_plan"], dtype=torch.float32)
        manipulator = self._tensor(data["manipulator_plan"], dtype=torch.float32)
        valid = self._tensor(data["plan_valid"]).bool()
        base_dim = self._tensor(data["base_dim_mask"]).bool()
        manipulator_dim = self._tensor(data["manipulator_dim_mask"]).bool()

        if base.shape[-2:] != (len(self._stats["plan_time_offsets"]), 4):
            raise ValueError(f"Expected base plan [6,4], got {tuple(base.shape)}")
        if manipulator.shape[-2] != base.shape[-2] or manipulator.shape[-1] < 9:
            raise ValueError(f"Invalid manipulator plan shape: {tuple(manipulator.shape)}")
        if self.validate_geometry:
            self._check_geometry(base, manipulator, valid)

        base_action = base.clone()
        manipulator_action = manipulator.clone()
        base_action[..., 0:2] = self._q99(base[..., 0:2], "base_xy")
        manipulator_action[..., 0:3] = self._q99(
            manipulator[..., 0:3], "eef_xyz"
        )
        hand_dim = len(self._stats["statistics"]["hand"]["q01"])
        if hand_dim:
            manipulator_action[..., 9 : 9 + hand_dim] = self._q99(
                manipulator[..., 9 : 9 + hand_dim], "hand"
            )

        data["base_action"] = base_action
        data["manipulator_action"] = manipulator_action
        data["base_action_mask"] = base_dim & valid.unsqueeze(-1)
        data["manipulator_action_mask"] = manipulator_dim & valid.unsqueeze(-1)
        data["plan_valid"] = valid
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        base_action = self._tensor(data["base_action"], dtype=torch.float32)
        manipulator_action = self._tensor(
            data["manipulator_action"], dtype=torch.float32
        )
        base_plan = base_action.clone()
        manipulator_plan = manipulator_action.clone()
        base_plan[..., 0:2] = self._q99(
            base_action[..., 0:2], "base_xy", inverse=True
        )
        manipulator_plan[..., 0:3] = self._q99(
            manipulator_action[..., 0:3], "eef_xyz", inverse=True
        )
        hand_dim = len(self._stats["statistics"]["hand"]["q01"])
        if hand_dim:
            manipulator_plan[..., 9 : 9 + hand_dim] = self._q99(
                manipulator_action[..., 9 : 9 + hand_dim], "hand", inverse=True
            )
        data["base_plan"] = base_plan
        data["manipulator_plan"] = manipulator_plan
        return data

