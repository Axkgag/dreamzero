"""Differentiable, slice-aware losses for MobileManiBench plans."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....utils.mobile_plan_spec import (
    EEF_ROTATION_ANCHOR_BASE_6D,
    EEF_ROTATION_CURRENT_EEF_DELTA_ROTVEC,
    eef_rotation_dim,
)


def _safe_normalize(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    fallback = torch.zeros_like(value)
    fallback[..., 0] = 1.0
    return torch.where(norm > eps, value / norm.clamp_min(eps), fallback)


def rotation6d_rows_to_matrix(value: torch.Tensor) -> torch.Tensor:
    """Convert the repository's first-two-rows rotation6d convention to SO(3)."""
    rows = value.float().reshape(*value.shape[:-1], 2, 3)
    first = _safe_normalize(rows[..., 0, :])
    second_raw = rows[..., 1, :] - (
        rows[..., 1, :] * first
    ).sum(dim=-1, keepdim=True) * first
    second = _safe_normalize(second_raw)
    third = _safe_normalize(torch.linalg.cross(first, second, dim=-1))
    second = _safe_normalize(torch.linalg.cross(third, first, dim=-1))
    return torch.stack([first, second, third], dim=-2)


def rotation_vector_to_matrix(value: torch.Tensor) -> torch.Tensor:
    """Differentiable SO(3) exponential map for axis-angle vectors."""
    value = value.float()
    x, y, z = value.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(*value.shape[:-1], 3, 3)
    theta_sq = value.square().sum(-1)
    theta = torch.sqrt(theta_sq.clamp_min(1e-12))
    small = theta_sq < 1e-8
    a = torch.where(
        small,
        1.0 - theta_sq / 6.0 + theta_sq.square() / 120.0,
        torch.sin(theta) / theta,
    )
    b = torch.where(
        small,
        0.5 - theta_sq / 24.0 + theta_sq.square() / 720.0,
        (1.0 - torch.cos(theta)) / theta_sq.clamp_min(1e-12),
    )
    identity = torch.eye(3, device=value.device, dtype=value.dtype).expand(
        *value.shape[:-1], 3, 3
    )
    return identity + a[..., None, None] * skew + b[..., None, None] * (skew @ skew)


def euler_rpy_to_matrix(value: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = value.float().unbind(-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    return torch.stack(
        [
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
            -sp,
            cp * sr,
            cp * cr,
        ],
        dim=-1,
    ).reshape(*value.shape[:-1], 3, 3)


def rotation_geodesic(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    relative = prediction @ target.transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5)
    skew = torch.stack(
        [
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ],
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    return torch.atan2(sine, cosine.clamp(-1.0, 1.0))


def yaw_matrix(sincos: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Build a yaw rotation without the undefined gradient of atan2(0, 0)."""
    norm = torch.linalg.vector_norm(sincos, dim=-1, keepdim=True)
    normalized = sincos / norm.clamp_min(eps)
    fallback = torch.zeros_like(sincos)
    fallback[..., 1] = 1.0
    normalized = torch.where(norm > eps, normalized, fallback)
    sine = normalized[..., 0]
    cosine = normalized[..., 1]
    result = torch.zeros(
        *sine.shape, 3, 3, device=sine.device, dtype=sine.dtype
    )
    result[..., 0, 0] = cosine
    result[..., 0, 1] = -sine
    result[..., 1, 0] = sine
    result[..., 1, 1] = cosine
    result[..., 2, 2] = 1.0
    return result


def _safe_base_pose_for_geometry(
    value: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Replace masked Base poses by identity yaw before geometry operations."""
    fallback = torch.zeros_like(value)
    fallback[..., 3] = 1.0
    return torch.where(valid.unsqueeze(-1), value, fallback)


def _safe_eef_pose_for_geometry(
    value: torch.Tensor,
    valid: torch.Tensor,
    rotation_representation: str = EEF_ROTATION_ANCHOR_BASE_6D,
) -> torch.Tensor:
    """Replace masked EEF poses by an identity rotation before geometry."""
    rotation_dim = eef_rotation_dim(rotation_representation)
    pose = value[..., : 3 + rotation_dim]
    fallback = torch.zeros_like(pose)
    if rotation_representation == EEF_ROTATION_ANCHOR_BASE_6D:
        fallback[..., 3] = 1.0
        fallback[..., 7] = 1.0
    return torch.where(valid.unsqueeze(-1), pose, fallback)


def matrix_to_rotation6d_rows(value: torch.Tensor) -> torch.Tensor:
    """Convert SO(3) matrices to the repository's first-two-rows convention."""
    return value[..., :2, :].reshape(*value.shape[:-2], 6)


def eef_current_to_future_base(
    base: torch.Tensor, eef_current_base: torch.Tensor
) -> torch.Tensor:
    """Express an EEF pose from B(t) in the predicted future Base frame."""
    base_rotation = yaw_matrix(base[..., 2:4])
    base_translation = F.pad(base[..., :2], (0, 1))
    eef_rotation = rotation6d_rows_to_matrix(eef_current_base[..., 3:9])
    relative_position = torch.einsum(
        "...ji,...j->...i",
        base_rotation,
        eef_current_base[..., :3] - base_translation,
    )
    relative_rotation = base_rotation.transpose(-1, -2) @ eef_rotation
    return torch.cat(
        [relative_position, matrix_to_rotation6d_rows(relative_rotation)],
        dim=-1,
    )


def eef_future_to_current_base(
    base: torch.Tensor, eef_future_base: torch.Tensor
) -> torch.Tensor:
    """Compose a future-Base-relative EEF pose back into the B(t) frame."""
    base_rotation = yaw_matrix(base[..., 2:4])
    base_translation = F.pad(base[..., :2], (0, 1))
    eef_rotation = rotation6d_rows_to_matrix(eef_future_base[..., 3:9])
    current_position = (
        torch.einsum(
            "...ij,...j->...i",
            base_rotation,
            eef_future_base[..., :3],
        )
        + base_translation
    )
    current_rotation = base_rotation @ eef_rotation
    return torch.cat(
        [current_position, matrix_to_rotation6d_rows(current_rotation)],
        dim=-1,
    )


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1)


def _masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    value = F.smooth_l1_loss(
        prediction.float(), target.float(), reduction="none", beta=beta
    )
    return _masked_mean(value, mask)


class MobilePlanPhysicalConsistencyLosses(nn.Module):
    """Compute plan-component and Base/EEF consistency losses."""

    def __init__(
        self,
        stats_path: str | Path,
        *,
        plan_horizon: int = 6,
        base_action_dim: int = 4,
        manipulator_action_dim: int = 21,
        huber_beta: float = 0.1,
        eef_rotation_representation: str = EEF_ROTATION_ANCHOR_BASE_6D,
        eef_rotation_sigma_weight_base: float = 1.0,
        eef_rotation_sigma_weight_scale: float = 0.0,
    ):
        super().__init__()
        with Path(stats_path).open("r", encoding="utf-8") as handle:
            metadata: dict[str, Any] = json.load(handle)
        if metadata.get("fit_split") != "train":
            raise ValueError(
                "Physical loss statistics must be fit on the train split"
            )
        self.plan_horizon = plan_horizon
        self.base_action_dim = base_action_dim
        self.manipulator_action_dim = manipulator_action_dim
        self.hand_dim = int(metadata["hand_dim"])
        self.huber_beta = float(huber_beta)
        self.eef_rotation_representation = eef_rotation_representation
        self.eef_rotation_dim = eef_rotation_dim(eef_rotation_representation)
        self.eef_rotation_slice = slice(3, 3 + self.eef_rotation_dim)
        self.hand_start = 3 + self.eef_rotation_dim
        self.eef_rotation_sigma_weight_base = float(
            eef_rotation_sigma_weight_base
        )
        self.eef_rotation_sigma_weight_scale = float(
            eef_rotation_sigma_weight_scale
        )
        stats_representation = metadata.get(
            "eef_rotation_representation", EEF_ROTATION_ANCHOR_BASE_6D
        )
        if stats_representation != eef_rotation_representation:
            raise ValueError(
                f"Physical-loss stats use {stats_representation}, but the model "
                f"uses {eef_rotation_representation}"
            )

        statistics = metadata["statistics"]
        for name in ("base_xy", "eef_xyz", "hand"):
            q01 = torch.as_tensor(statistics[name]["q01"], dtype=torch.float32)
            q99 = torch.as_tensor(statistics[name]["q99"], dtype=torch.float32)
            self.register_buffer(f"{name}_q01", q01, persistent=True)
            self.register_buffer(f"{name}_q99", q99, persistent=True)

    @staticmethod
    def _denormalize(
        value: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor
    ) -> torch.Tensor:
        return (value + 1.0) * 0.5 * (q99 - q01) + q01

    @staticmethod
    def _scale(q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
        return (0.5 * (q99 - q01)).clamp_min(1e-6)

    def physical_plans(
        self, packed_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        horizon = self.plan_horizon
        base = packed_action[:, :horizon, : self.base_action_dim].float().clone()
        manipulator = (
            packed_action[:, horizon:, : self.manipulator_action_dim]
            .float()
            .clone()
        )
        base[..., :2] = self._denormalize(
            base[..., :2], self.base_xy_q01, self.base_xy_q99
        )
        manipulator[..., :3] = self._denormalize(
            manipulator[..., :3], self.eef_xyz_q01, self.eef_xyz_q99
        )
        if self.hand_dim:
            hand_slice = slice(self.hand_start, self.hand_start + self.hand_dim)
            manipulator[..., hand_slice] = self._denormalize(
                manipulator[..., hand_slice], self.hand_q01, self.hand_q99
            )
        return base, manipulator

    def physical_base_prior(self, prediction: torch.Tensor) -> torch.Tensor:
        base = prediction[..., : self.base_action_dim].float().clone()
        base[..., :2] = self._denormalize(
            base[..., :2], self.base_xy_q01, self.base_xy_q99
        )
        return base

    def physical_eef_prior(self, prediction: torch.Tensor) -> torch.Tensor:
        eef = prediction[..., : 3 + self.eef_rotation_dim].float().clone()
        eef[..., :3] = self._denormalize(
            eef[..., :3], self.eef_xyz_q01, self.eef_xyz_q99
        )
        return eef

    def _rotation_matrix(
        self,
        eef: torch.Tensor,
        anchor_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rotation_value = eef[..., self.eef_rotation_slice]
        if self.eef_rotation_representation == EEF_ROTATION_ANCHOR_BASE_6D:
            return rotation6d_rows_to_matrix(rotation_value)
        if self.eef_rotation_representation == EEF_ROTATION_CURRENT_EEF_DELTA_ROTVEC:
            delta = rotation_vector_to_matrix(rotation_value)
            if anchor_state is None:
                return delta
            anchor = anchor_state.float()
            while anchor.ndim < eef.ndim:
                anchor = anchor.unsqueeze(-2)
            anchor = anchor.expand(*eef.shape[:-1], anchor.shape[-1])
            return euler_rpy_to_matrix(anchor[..., 3:6]) @ delta
        raise AssertionError(self.eef_rotation_representation)

    @staticmethod
    def _rotation_bucket_metrics(
        angle: torch.Tensor,
        sigma: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if sigma is None:
            return {}
        result: dict[str, torch.Tensor] = {}
        sigma = sigma.float()
        bins = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01))
        for lower, upper in bins:
            bucket_mask = mask & (sigma >= lower) & (sigma < upper)
            suffix = f"sigma_{int(lower * 100):03d}_{int(min(upper, 1.0) * 100):03d}"
            key = f"eef_rotation_error_deg_{suffix}_metric"
            result[key] = _masked_mean(torch.rad2deg(angle), bucket_mask)
            result[f"eef_rotation_count_{suffix}_metric"] = (
                bucket_mask.sum().float()
            )
        high_mask = mask & (sigma >= 0.8)
        result["eef_rotation_error_deg_sigma_ge_080_metric"] = _masked_mean(
            torch.rad2deg(angle), high_mask
        )
        result["eef_rotation_count_sigma_ge_080_metric"] = high_mask.sum().float()
        return result

    def prior_terms(
        self,
        *,
        base_prediction: torch.Tensor | None,
        eef_prediction: torch.Tensor | None,
        clean_target: torch.Tensor,
        action_mask: torch.Tensor,
        has_real_action: torch.Tensor,
        eef_frame: str,
        anchor_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute direct Base/EEF Prior terms and their composition loss."""
        valid_eef_frames = {"current_base", "future_base"}
        if self.eef_rotation_representation == EEF_ROTATION_CURRENT_EEF_DELTA_ROTVEC:
            valid_eef_frames = {"current_eef_delta"}
        if eef_frame not in valid_eef_frames:
            raise ValueError(f"Unknown Prior EEF frame: {eef_frame}")
        horizon = self.plan_horizon
        base_gt, manip_gt = self.physical_plans(clean_target)
        base_mask = action_mask[:, :horizon, : self.base_action_dim].bool()
        eef_pose_dim = 3 + self.eef_rotation_dim
        eef_mask = action_mask[:, horizon:, :eef_pose_dim].bool()
        sample_mask = has_real_action.bool().view(-1, 1, 1)
        base_mask = base_mask & sample_mask
        eef_mask = eef_mask & sample_mask
        base_geometry_valid = base_mask[..., :4].all(-1)
        eef_geometry_valid = eef_mask[..., :eef_pose_dim].all(-1)
        base_gt_geometry = _safe_base_pose_for_geometry(
            base_gt, base_geometry_valid
        )
        manip_gt_geometry = _safe_eef_pose_for_geometry(
            manip_gt,
            eef_geometry_valid,
            self.eef_rotation_representation,
        )
        zero = clean_target.sum() * 0.0
        base_scale = self._scale(self.base_xy_q01, self.base_xy_q99)
        eef_scale = self._scale(self.eef_xyz_q01, self.eef_xyz_q99)

        if base_prediction is None:
            base_pred = None
            base_xy_loss = zero
            base_yaw_loss = zero
            base_yaw_unit_loss = zero
            base_position_error = zero
        else:
            base_pred = self.physical_base_prior(base_prediction)
            base_xy_loss = _masked_smooth_l1(
                base_pred[..., :2] / base_scale,
                base_gt[..., :2] / base_scale,
                base_mask[..., :2],
                self.huber_beta,
            )
            base_yaw_loss = _masked_smooth_l1(
                base_pred[..., 2:4],
                base_gt[..., 2:4],
                base_mask[..., 2:4],
                self.huber_beta,
            )
            base_yaw_mask = base_mask[..., 2:4].all(-1)
            base_yaw_unit_loss = _masked_mean(
                (
                    torch.linalg.vector_norm(base_pred[..., 2:4], dim=-1)
                    - 1.0
                ).square(),
                base_yaw_mask,
            )
            base_position_error = _masked_mean(
                torch.linalg.vector_norm(
                    base_pred[..., :2] - base_gt[..., :2], dim=-1
                ),
                base_mask[..., :2].all(-1),
            )

        if eef_prediction is None:
            eef_pred = None
            eef_position_loss = zero
            eef_rotation_loss = zero
            eef_position_error = zero
            eef_rotation_error = zero
        else:
            eef_pred = self.physical_eef_prior(eef_prediction)
            if eef_frame == "future_base":
                eef_target = eef_current_to_future_base(
                    base_gt_geometry, manip_gt_geometry
                )
                direct_position_mask = (
                    eef_mask[..., :3].all(-1)
                    & base_mask[..., :2].all(-1)
                )
                direct_rotation_mask = (
                    eef_mask[..., self.eef_rotation_slice].all(-1)
                    & base_mask[..., 2:4].all(-1)
                )
            else:
                eef_target = manip_gt_geometry
                direct_position_mask = eef_mask[..., :3].all(-1)
                direct_rotation_mask = eef_mask[..., self.eef_rotation_slice].all(-1)
            eef_position_loss = _masked_smooth_l1(
                eef_pred[..., :3] / eef_scale,
                eef_target[..., :3] / eef_scale,
                direct_position_mask.unsqueeze(-1).expand_as(
                    eef_pred[..., :3]
                ),
                self.huber_beta,
            )
            eef_rotation_angle = rotation_geodesic(
                self._rotation_matrix(eef_pred),
                self._rotation_matrix(eef_target),
            )
            eef_rotation_loss = _masked_mean(
                eef_rotation_angle / math.pi, direct_rotation_mask
            )
            eef_position_error = _masked_mean(
                torch.linalg.vector_norm(
                    eef_pred[..., :3] - eef_target[..., :3], dim=-1
                ),
                direct_position_mask,
            )
            eef_rotation_error = _masked_mean(
                torch.rad2deg(eef_rotation_angle), direct_rotation_mask
            )

        if base_pred is None or eef_pred is None:
            joint_position_loss = zero
            joint_rotation_loss = zero
        else:
            base_pred_geometry = _safe_base_pose_for_geometry(
                base_pred, base_geometry_valid
            )
            eef_pred_geometry = _safe_eef_pose_for_geometry(
                eef_pred,
                eef_geometry_valid,
                self.eef_rotation_representation,
            )
            joint_position_mask = (
                base_mask[..., :2].all(-1)
                & eef_mask[..., :3].all(-1)
            )
            joint_rotation_mask = (
                base_mask[..., 2:4].all(-1)
                & eef_mask[..., self.eef_rotation_slice].all(-1)
            )
            if eef_frame == "future_base":
                joint_prediction = eef_future_to_current_base(
                    base_pred_geometry, eef_pred_geometry
                )
                joint_target = manip_gt_geometry
                joint_prediction_rotation = rotation6d_rows_to_matrix(
                    joint_prediction[..., 3:9]
                )
                joint_target_rotation = rotation6d_rows_to_matrix(
                    joint_target[..., 3:9]
                )
            elif eef_frame == "current_eef_delta":
                if anchor_state is None:
                    raise ValueError(
                        "current_eef_delta Prior consistency requires anchor_state"
                    )
                base_prediction_rotation = yaw_matrix(
                    base_pred_geometry[..., 2:4]
                )
                base_target_rotation = yaw_matrix(base_gt_geometry[..., 2:4])
                base_prediction_translation = F.pad(
                    base_pred_geometry[..., :2], (0, 1)
                )
                base_target_translation = F.pad(
                    base_gt_geometry[..., :2], (0, 1)
                )
                joint_prediction_position = torch.einsum(
                    "...ji,...j->...i",
                    base_prediction_rotation,
                    eef_pred_geometry[..., :3] - base_prediction_translation,
                )
                joint_target_position = torch.einsum(
                    "...ji,...j->...i",
                    base_target_rotation,
                    manip_gt_geometry[..., :3] - base_target_translation,
                )
                joint_prediction_rotation = (
                    base_prediction_rotation.transpose(-1, -2)
                    @ self._rotation_matrix(eef_pred_geometry, anchor_state)
                )
                joint_target_rotation = (
                    base_target_rotation.transpose(-1, -2)
                    @ self._rotation_matrix(manip_gt_geometry, anchor_state)
                )
                joint_prediction = torch.cat(
                    [
                        joint_prediction_position,
                        matrix_to_rotation6d_rows(joint_prediction_rotation),
                    ],
                    dim=-1,
                )
                joint_target = torch.cat(
                    [
                        joint_target_position,
                        matrix_to_rotation6d_rows(joint_target_rotation),
                    ],
                    dim=-1,
                )
            else:
                joint_prediction = eef_current_to_future_base(
                    base_pred_geometry, eef_pred_geometry
                )
                joint_target = eef_current_to_future_base(
                    base_gt_geometry, manip_gt_geometry
                )
                joint_prediction_rotation = rotation6d_rows_to_matrix(
                    joint_prediction[..., 3:9]
                )
                joint_target_rotation = rotation6d_rows_to_matrix(
                    joint_target[..., 3:9]
                )
            joint_position_loss = _masked_smooth_l1(
                joint_prediction[..., :3] / eef_scale,
                joint_target[..., :3] / eef_scale,
                joint_position_mask.unsqueeze(-1).expand_as(
                    joint_prediction[..., :3]
                ),
                self.huber_beta,
            )
            joint_rotation_angle = rotation_geodesic(
                joint_prediction_rotation,
                joint_target_rotation,
            )
            joint_rotation_loss = _masked_mean(
                joint_rotation_angle / math.pi, joint_rotation_mask
            )

        return {
            "base_prior_xy_loss": base_xy_loss,
            "base_prior_yaw_loss": base_yaw_loss,
            "base_prior_unit_loss": base_yaw_unit_loss,
            "eef_prior_position_loss": eef_position_loss,
            "eef_prior_rotation_loss": eef_rotation_loss,
            "joint_prior_consistency_position_loss": joint_position_loss,
            "joint_prior_consistency_rotation_loss": joint_rotation_loss,
            "base_prior_position_error_m": base_position_error,
            "eef_prior_position_error_m": eef_position_error,
            "eef_prior_rotation_error_deg": eef_rotation_error,
        }

    def forward(
        self,
        clean_prediction: torch.Tensor,
        clean_target: torch.Tensor,
        action_mask: torch.Tensor,
        has_real_action: torch.Tensor,
        action_sigma: torch.Tensor | None = None,
        anchor_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        horizon = self.plan_horizon
        base_pred, manip_pred = self.physical_plans(clean_prediction)
        base_gt, manip_gt = self.physical_plans(clean_target)
        base_mask = action_mask[:, :horizon, : self.base_action_dim].bool()
        manip_mask = action_mask[:, horizon:, : self.manipulator_action_dim].bool()
        sample_mask = has_real_action.bool().view(-1, 1, 1)
        base_mask = base_mask & sample_mask
        manip_mask = manip_mask & sample_mask
        base_geometry_valid = base_mask[..., :4].all(-1)
        eef_pose_dim = 3 + self.eef_rotation_dim
        manip_geometry_valid = manip_mask[..., :eef_pose_dim].all(-1)
        base_pred_geometry = _safe_base_pose_for_geometry(
            base_pred, base_geometry_valid
        )
        base_gt_geometry = _safe_base_pose_for_geometry(
            base_gt, base_geometry_valid
        )
        manip_pred_geometry = _safe_eef_pose_for_geometry(
            manip_pred,
            manip_geometry_valid,
            self.eef_rotation_representation,
        )
        manip_gt_geometry = _safe_eef_pose_for_geometry(
            manip_gt,
            manip_geometry_valid,
            self.eef_rotation_representation,
        )

        base_xy_scale = self._scale(self.base_xy_q01, self.base_xy_q99)
        eef_scale = self._scale(self.eef_xyz_q01, self.eef_xyz_q99)
        base_xy_loss = _masked_smooth_l1(
            base_pred[..., :2] / base_xy_scale,
            base_gt[..., :2] / base_xy_scale,
            base_mask[..., :2],
            self.huber_beta,
        )
        base_yaw_loss = _masked_smooth_l1(
            base_pred[..., 2:4],
            base_gt[..., 2:4],
            base_mask[..., 2:4],
            self.huber_beta,
        )
        yaw_token_mask = base_mask[..., 2:4].all(-1)
        yaw_unit_loss = _masked_mean(
            (
                torch.linalg.vector_norm(base_pred[..., 2:4], dim=-1) - 1.0
            ).square(),
            yaw_token_mask,
        )

        eef_position_loss = _masked_smooth_l1(
            manip_pred[..., :3] / eef_scale,
            manip_gt[..., :3] / eef_scale,
            manip_mask[..., :3],
            self.huber_beta,
        )
        rotation_pred = self._rotation_matrix(manip_pred_geometry, anchor_state)
        rotation_gt = self._rotation_matrix(manip_gt_geometry, anchor_state)
        rotation_token_mask = manip_mask[..., self.eef_rotation_slice].all(-1)
        rotation_angle = rotation_geodesic(rotation_pred, rotation_gt)
        manipulator_sigma = (
            action_sigma[:, horizon:].float() if action_sigma is not None else None
        )
        rotation_weight = (
            self.eef_rotation_sigma_weight_base
            + self.eef_rotation_sigma_weight_scale * manipulator_sigma
            if manipulator_sigma is not None
            else self.eef_rotation_sigma_weight_base
        )
        eef_rotation_loss = _masked_mean(
            rotation_angle / math.pi * rotation_weight, rotation_token_mask
        )

        if self.hand_dim:
            hand_slice = slice(self.hand_start, self.hand_start + self.hand_dim)
            hand_scale = self._scale(self.hand_q01, self.hand_q99)
            hand_loss = _masked_smooth_l1(
                manip_pred[..., hand_slice] / hand_scale,
                manip_gt[..., hand_slice] / hand_scale,
                manip_mask[..., hand_slice],
                self.huber_beta,
            )
        else:
            hand_loss = clean_prediction.sum() * 0.0

        base_rotation_pred = yaw_matrix(base_pred_geometry[..., 2:4])
        base_rotation_gt = yaw_matrix(base_gt_geometry[..., 2:4])
        base_translation_pred = F.pad(base_pred_geometry[..., :2], (0, 1))
        base_translation_gt = F.pad(base_gt_geometry[..., :2], (0, 1))
        relative_position_pred = torch.einsum(
            "...ji,...j->...i",
            base_rotation_pred,
            manip_pred_geometry[..., :3] - base_translation_pred,
        )
        relative_position_gt = torch.einsum(
            "...ji,...j->...i",
            base_rotation_gt,
            manip_gt_geometry[..., :3] - base_translation_gt,
        )
        relative_rotation_pred = base_rotation_pred.transpose(-1, -2) @ rotation_pred
        relative_rotation_gt = base_rotation_gt.transpose(-1, -2) @ rotation_gt
        consistency_position_mask = (
            base_mask[..., :2].all(-1)
            & manip_mask[..., :3].all(-1)
        )
        consistency_rotation_mask = (
            base_mask[..., 2:4].all(-1)
            & rotation_token_mask
        )
        relative_position_loss = _masked_smooth_l1(
            relative_position_pred / eef_scale,
            relative_position_gt / eef_scale,
            consistency_position_mask.unsqueeze(-1).expand_as(
                relative_position_pred
            ),
            self.huber_beta,
        )
        relative_rotation_angle = rotation_geodesic(
            relative_rotation_pred, relative_rotation_gt
        )
        relative_rotation_loss = _masked_mean(
            relative_rotation_angle / math.pi, consistency_rotation_mask
        )

        base_position_error_m = _masked_mean(
            torch.linalg.vector_norm(
                base_pred[..., :2] - base_gt[..., :2], dim=-1
            ),
            base_mask[..., :2].all(-1),
        )
        eef_position_error_m = _masked_mean(
            torch.linalg.vector_norm(
                manip_pred[..., :3] - manip_gt[..., :3], dim=-1
            ),
            manip_mask[..., :3].all(-1),
        )
        eef_rotation_error_deg = _masked_mean(
            torch.rad2deg(rotation_angle), rotation_token_mask
        )
        determinant_error = _masked_mean(
            # CUDA linalg.det has no BF16 kernel.  This is a diagnostic metric;
            # evaluate it in FP32 so BF16 autocast training remains supported.
            (torch.linalg.det(rotation_pred.float()) - 1.0).abs(),
            rotation_token_mask,
        )

        result = {
            "base_xy_loss": base_xy_loss,
            "base_yaw_loss": base_yaw_loss,
            "base_yaw_unit_loss": yaw_unit_loss,
            "eef_position_loss": eef_position_loss,
            "eef_rotation_loss": eef_rotation_loss,
            "hand_loss": hand_loss,
            "base_eef_consistency_position_loss": relative_position_loss,
            "base_eef_consistency_rotation_loss": relative_rotation_loss,
            "base_position_error_m": base_position_error_m,
            "eef_position_error_m": eef_position_error_m,
            "eef_rotation_error_deg": eef_rotation_error_deg,
            "eef_rotation_determinant_error": determinant_error,
        }
        result.update(
            self._rotation_bucket_metrics(
                rotation_angle,
                manipulator_sigma,
                rotation_token_mask,
            )
        )
        if self.eef_rotation_representation == EEF_ROTATION_ANCHOR_BASE_6D:
            raw = manip_pred[..., self.eef_rotation_slice].float().reshape(
                -1, 2, 3
            )
            raw_mask = rotation_token_mask.reshape(-1)
            row_norm = torch.linalg.vector_norm(raw, dim=-1)
            row_dot = (raw[:, 0] * raw[:, 1]).sum(-1).abs()
            result["eef_rotation6d_row_norm_error_metric"] = _masked_mean(
                (row_norm - 1.0).abs().mean(-1), raw_mask
            )
            result["eef_rotation6d_abs_row_dot_metric"] = _masked_mean(
                row_dot, raw_mask
            )
        else:
            raw_angle = torch.linalg.vector_norm(
                manip_pred[..., self.eef_rotation_slice].float(), dim=-1
            )
            result["eef_rotvec_pred_norm_rad_metric"] = _masked_mean(
                raw_angle, rotation_token_mask
            )
        return result
