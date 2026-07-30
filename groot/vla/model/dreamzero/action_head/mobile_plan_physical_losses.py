"""Differentiable, slice-aware losses for MobileManiBench plans."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def yaw_matrix(sincos: torch.Tensor) -> torch.Tensor:
    yaw = torch.atan2(sincos[..., 0], sincos[..., 1])
    sine = torch.sin(yaw)
    cosine = torch.cos(yaw)
    result = torch.zeros(
        *yaw.shape, 3, 3, device=yaw.device, dtype=yaw.dtype
    )
    result[..., 0, 0] = cosine
    result[..., 0, 1] = -sine
    result[..., 1, 0] = sine
    result[..., 1, 1] = cosine
    result[..., 2, 2] = 1.0
    return result


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
            hand_slice = slice(9, 9 + self.hand_dim)
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
        eef = prediction[..., :9].float().clone()
        eef[..., :3] = self._denormalize(
            eef[..., :3], self.eef_xyz_q01, self.eef_xyz_q99
        )
        return eef

    def prior_terms(
        self,
        *,
        base_prediction: torch.Tensor | None,
        eef_prediction: torch.Tensor | None,
        clean_target: torch.Tensor,
        action_mask: torch.Tensor,
        has_real_action: torch.Tensor,
        eef_frame: str,
    ) -> dict[str, torch.Tensor]:
        """Compute direct Base/EEF Prior terms and their composition loss."""
        if eef_frame not in {"current_base", "future_base"}:
            raise ValueError(f"Unknown Prior EEF frame: {eef_frame}")
        horizon = self.plan_horizon
        base_gt, manip_gt = self.physical_plans(clean_target)
        base_mask = action_mask[:, :horizon, : self.base_action_dim].bool()
        eef_mask = action_mask[:, horizon:, :9].bool()
        sample_mask = has_real_action.bool().view(-1, 1, 1)
        base_mask = base_mask & sample_mask
        eef_mask = eef_mask & sample_mask
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
                eef_target = eef_current_to_future_base(base_gt, manip_gt)
                direct_position_mask = (
                    eef_mask[..., :3].all(-1)
                    & base_mask[..., :2].all(-1)
                )
                direct_rotation_mask = (
                    eef_mask[..., 3:9].all(-1)
                    & base_mask[..., 2:4].all(-1)
                )
            else:
                eef_target = manip_gt[..., :9]
                direct_position_mask = eef_mask[..., :3].all(-1)
                direct_rotation_mask = eef_mask[..., 3:9].all(-1)
            eef_position_loss = _masked_smooth_l1(
                eef_pred[..., :3] / eef_scale,
                eef_target[..., :3] / eef_scale,
                direct_position_mask.unsqueeze(-1).expand_as(
                    eef_pred[..., :3]
                ),
                self.huber_beta,
            )
            eef_rotation_angle = rotation_geodesic(
                rotation6d_rows_to_matrix(eef_pred[..., 3:9]),
                rotation6d_rows_to_matrix(eef_target[..., 3:9]),
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
            joint_position_mask = (
                base_mask[..., :2].all(-1)
                & eef_mask[..., :3].all(-1)
            )
            joint_rotation_mask = (
                base_mask[..., 2:4].all(-1)
                & eef_mask[..., 3:9].all(-1)
            )
            if eef_frame == "future_base":
                joint_prediction = eef_future_to_current_base(
                    base_pred, eef_pred
                )
                joint_target = manip_gt[..., :9]
            else:
                joint_prediction = eef_current_to_future_base(
                    base_pred, eef_pred
                )
                joint_target = eef_current_to_future_base(base_gt, manip_gt)
            joint_position_loss = _masked_smooth_l1(
                joint_prediction[..., :3] / eef_scale,
                joint_target[..., :3] / eef_scale,
                joint_position_mask.unsqueeze(-1).expand_as(
                    joint_prediction[..., :3]
                ),
                self.huber_beta,
            )
            joint_rotation_angle = rotation_geodesic(
                rotation6d_rows_to_matrix(joint_prediction[..., 3:9]),
                rotation6d_rows_to_matrix(joint_target[..., 3:9]),
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
    ) -> dict[str, torch.Tensor]:
        horizon = self.plan_horizon
        base_pred, manip_pred = self.physical_plans(clean_prediction)
        base_gt, manip_gt = self.physical_plans(clean_target)
        base_mask = action_mask[:, :horizon, : self.base_action_dim].bool()
        manip_mask = action_mask[:, horizon:, : self.manipulator_action_dim].bool()
        sample_mask = has_real_action.bool().view(-1, 1, 1)
        base_mask = base_mask & sample_mask
        manip_mask = manip_mask & sample_mask

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
        rotation_pred = rotation6d_rows_to_matrix(manip_pred[..., 3:9])
        rotation_gt = rotation6d_rows_to_matrix(manip_gt[..., 3:9])
        rotation_token_mask = manip_mask[..., 3:9].all(-1)
        rotation_angle = rotation_geodesic(rotation_pred, rotation_gt)
        eef_rotation_loss = _masked_mean(
            rotation_angle / math.pi, rotation_token_mask
        )

        if self.hand_dim:
            hand_slice = slice(9, 9 + self.hand_dim)
            hand_scale = self._scale(self.hand_q01, self.hand_q99)
            hand_loss = _masked_smooth_l1(
                manip_pred[..., hand_slice] / hand_scale,
                manip_gt[..., hand_slice] / hand_scale,
                manip_mask[..., hand_slice],
                self.huber_beta,
            )
        else:
            hand_loss = clean_prediction.sum() * 0.0

        base_rotation_pred = yaw_matrix(base_pred[..., 2:4])
        base_rotation_gt = yaw_matrix(base_gt[..., 2:4])
        base_translation_pred = F.pad(base_pred[..., :2], (0, 1))
        base_translation_gt = F.pad(base_gt[..., :2], (0, 1))
        relative_position_pred = torch.einsum(
            "...ji,...j->...i",
            base_rotation_pred,
            manip_pred[..., :3] - base_translation_pred,
        )
        relative_position_gt = torch.einsum(
            "...ji,...j->...i",
            base_rotation_gt,
            manip_gt[..., :3] - base_translation_gt,
        )
        relative_rotation_pred = (
            base_rotation_pred.transpose(-1, -2) @ rotation_pred
        )
        relative_rotation_gt = (
            base_rotation_gt.transpose(-1, -2) @ rotation_gt
        )
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
            (torch.linalg.det(rotation_pred) - 1.0).abs(), rotation_token_mask
        )

        return {
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
