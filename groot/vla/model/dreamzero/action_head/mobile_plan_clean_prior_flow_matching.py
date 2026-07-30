"""Flow head for configurable sparse clean Base/EEF Prior targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from transformers.feature_extraction_utils import BatchFeature

from ..modules.wan_video_dit_dual_plan_prior import (
    MobilePlanPriorConfig,
    coerce_mobile_plan_prior_config,
    resolve_prior_flow_indices,
)
from .mobile_plan_flow_matching import (
    MobilePlanFlowMatchingActionHead,
    MobilePlanPolicyHeadConfig,
)
from .mobile_plan_physical_losses import MobilePlanPhysicalConsistencyLosses


@dataclass(init=False)
class MobilePlanCleanPriorPolicyHeadConfig(MobilePlanPolicyHeadConfig):
    prior_condition_mode: str = field(default="normal")
    prior: dict[str, Any] = field(
        default_factory=lambda: asdict(MobilePlanPriorConfig())
    )
    base_prior_loss_weight: float = field(default=0.0)
    base_prior_xy_loss_weight: float = field(default=1.0)
    base_prior_yaw_loss_weight: float = field(default=1.0)
    base_prior_unit_loss_weight: float = field(default=0.01)
    base_prior_loss_start_step: int = field(default=0)
    base_prior_loss_ramp_steps: int = field(default=0)
    base_prior_target_gradient_ratio: float = field(default=0.1)
    eef_prior_loss_weight: float = field(default=0.1)
    eef_prior_position_loss_weight: float = field(default=1.0)
    eef_prior_rotation_loss_weight: float = field(default=1.0)
    eef_prior_loss_start_step: int = field(default=200)
    eef_prior_loss_ramp_steps: int = field(default=500)
    eef_prior_target_gradient_ratio: float = field(default=0.1)
    joint_prior_consistency_loss_weight: float = field(default=0.05)
    joint_prior_consistency_position_loss_weight: float = field(default=1.0)
    joint_prior_consistency_rotation_loss_weight: float = field(default=1.0)
    joint_prior_consistency_loss_start_step: int = field(default=700)
    joint_prior_consistency_loss_ramp_steps: int = field(default=500)
    joint_prior_consistency_target_gradient_ratio: float = field(default=0.05)

    def __init__(self, **kwargs):
        prior = kwargs.pop("prior", None)
        legacy_time_offsets = kwargs.pop("prior_time_offsets", None)
        super().__init__(**kwargs)
        self.prior = asdict(
            coerce_mobile_plan_prior_config(
                prior, legacy_time_offsets=legacy_time_offsets
            )
        )


class MobilePlanCleanPriorFlowMatchingActionHead(MobilePlanFlowMatchingActionHead):
    """Train flow outputs plus direct clean coarse Base/EEF predictions."""

    config_class = MobilePlanCleanPriorPolicyHeadConfig

    def __init__(self, config: MobilePlanCleanPriorPolicyHeadConfig):
        if not config.plan_stats_path:
            raise ValueError("Clean Prior training requires plan_stats_path")
        super().__init__(config)
        self.prior_config = coerce_mobile_plan_prior_config(config.prior)
        model_prior_config = getattr(self.model, "prior_config", None)
        if (
            model_prior_config is not None
            and model_prior_config != self.prior_config
        ):
            raise ValueError(
                "Action-head and diffusion-model Prior configurations differ"
            )
        self.prior_flow_indices = resolve_prior_flow_indices(
            config.plan_time_offsets, self.prior_config.time_offsets
        )
        self.prior_horizon = len(self.prior_flow_indices)
        if self.physical_losses is None:
            self.physical_losses = MobilePlanPhysicalConsistencyLosses(
                config.plan_stats_path,
                plan_horizon=config.plan_horizon,
                base_action_dim=config.base_action_dim,
                manipulator_action_dim=config.manipulator_action_dim,
                huber_beta=config.physical_loss_huber_beta,
            )
        self.prior_physical_losses = MobilePlanPhysicalConsistencyLosses(
            config.plan_stats_path,
            plan_horizon=self.prior_horizon,
            base_action_dim=config.base_action_dim,
            manipulator_action_dim=config.manipulator_action_dim,
            huber_beta=config.physical_loss_huber_beta,
        )
        self._latest_base_prior: torch.Tensor | None = None
        self._latest_eef_prior: torch.Tensor | None = None

    def prepare_action_model_kwargs(self, action_input: BatchFeature) -> dict:
        kwargs = super().prepare_action_model_kwargs(action_input)
        kwargs["prior_condition_mode"] = self.config.prior_condition_mode
        kwargs["prior_time_offsets"] = self.prior_config.time_offsets
        return kwargs

    def compute_action_losses(
        self,
        action_noise_pred: torch.Tensor,
        training_target_action: torch.Tensor,
        action_mask: torch.Tensor,
        has_real_action: torch.Tensor,
        timestep_action: torch.Tensor,
        noisy_actions: torch.Tensor | None = None,
        clean_actions: torch.Tensor | None = None,
        action_model_aux: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        losses = super().compute_action_losses(
            action_noise_pred,
            training_target_action,
            action_mask,
            has_real_action,
            timestep_action,
            noisy_actions=noisy_actions,
            clean_actions=clean_actions,
            action_model_aux=action_model_aux,
        )
        if clean_actions is None:
            raise ValueError("Clean Base Prior loss requires clean_actions")
        horizon = self.plan_horizon
        base_prior = (
            action_noise_pred[
                :,
                self.prior_flow_indices,
                self.base_action_dim : 2 * self.base_action_dim,
            ]
            if self.prior_config.predict_base
            else None
        )
        eef_prior = (
            action_noise_pred[
                :,
                self.prior_flow_indices,
                2 * self.base_action_dim : 2 * self.base_action_dim + 9,
            ]
            if self.prior_config.predict_eef
            else None
        )
        prior_clean_actions = torch.cat(
            [
                clean_actions[:, self.prior_flow_indices],
                clean_actions[
                    :,
                    tuple(horizon + index for index in self.prior_flow_indices),
                ],
            ],
            dim=1,
        )
        prior_action_mask = torch.cat(
            [
                action_mask[:, self.prior_flow_indices],
                action_mask[
                    :,
                    tuple(horizon + index for index in self.prior_flow_indices),
                ],
            ],
            dim=1,
        )
        prior_terms = self.prior_physical_losses.prior_terms(
            base_prediction=base_prior,
            eef_prediction=eef_prior,
            clean_target=prior_clean_actions,
            action_mask=prior_action_mask,
            has_real_action=has_real_action,
            eef_frame=self.prior_config.eef_frame,
        )
        base_prior_loss = (
            self.config.base_prior_xy_loss_weight
            * prior_terms["base_prior_xy_loss"]
            + self.config.base_prior_yaw_loss_weight
            * prior_terms["base_prior_yaw_loss"]
            + self.config.base_prior_unit_loss_weight
            * prior_terms["base_prior_unit_loss"]
        )
        eef_prior_loss = (
            self.config.eef_prior_position_loss_weight
            * prior_terms["eef_prior_position_loss"]
            + self.config.eef_prior_rotation_loss_weight
            * prior_terms["eef_prior_rotation_loss"]
        )
        joint_prior_consistency_loss = (
            self.config.joint_prior_consistency_position_loss_weight
            * prior_terms["joint_prior_consistency_position_loss"]
            + self.config.joint_prior_consistency_rotation_loss_weight
            * prior_terms["joint_prior_consistency_rotation_loss"]
        )
        step = int(self.global_step)
        effective_base_weight = (
            self._ramped_weight(
                self.config.base_prior_loss_weight,
                step,
                self.config.base_prior_loss_start_step,
                self.config.base_prior_loss_ramp_steps,
            )
            if self.prior_config.predict_base
            else 0.0
        )
        effective_eef_weight = (
            self._ramped_weight(
                self.config.eef_prior_loss_weight,
                step,
                self.config.eef_prior_loss_start_step,
                self.config.eef_prior_loss_ramp_steps,
            )
            if self.prior_config.predict_eef
            else 0.0
        )
        effective_joint_weight = (
            self._ramped_weight(
                self.config.joint_prior_consistency_loss_weight,
                step,
                self.config.joint_prior_consistency_loss_start_step,
                self.config.joint_prior_consistency_loss_ramp_steps,
            )
            if self.prior_config.predict_base
            and self.prior_config.predict_eef
            else 0.0
        )
        weighted_base_prior_loss = effective_base_weight * base_prior_loss
        weighted_eef_prior_loss = effective_eef_weight * eef_prior_loss
        weighted_joint_prior_loss = (
            effective_joint_weight * joint_prior_consistency_loss
        )
        losses["action_loss"] = (
            losses["action_loss"]
            + weighted_base_prior_loss
            + weighted_eef_prior_loss
            + weighted_joint_prior_loss
        )
        losses.update(prior_terms)
        losses.update(
            {
                "base_prior_loss": base_prior_loss,
                "eef_prior_loss": eef_prior_loss,
                "joint_prior_consistency_loss": (
                    joint_prior_consistency_loss
                ),
                "weighted_base_prior_loss": weighted_base_prior_loss,
                "weighted_eef_prior_loss": weighted_eef_prior_loss,
                "weighted_joint_prior_consistency_loss": (
                    weighted_joint_prior_loss
                ),
                "effective_base_prior_loss_weight": torch.as_tensor(
                    effective_base_weight, device=losses["action_loss"].device
                ),
                "effective_eef_prior_loss_weight": torch.as_tensor(
                    effective_eef_weight, device=losses["action_loss"].device
                ),
                "effective_joint_prior_consistency_loss_weight": torch.as_tensor(
                    effective_joint_weight,
                    device=losses["action_loss"].device,
                ),
            }
        )
        interval = max(
            int(getattr(self.config, "loss_gradient_log_interval", 50)),
            1,
        )
        if (
            bool(getattr(self.config, "log_loss_gradient_metrics", False))
            and torch.is_grad_enabled()
            and int(self.global_step) % interval == 0
        ):
            flow_gradient = losses.get("plan_flow_gradient_norm_metric")
            if flow_gradient is None:
                flow_gradient = self._gradient_norm(
                    losses["base_flow_loss"] + losses["manipulator_flow_loss"],
                    action_noise_pred,
                )
            epsilon = torch.finfo(torch.float32).eps
            if self.prior_config.predict_base:
                base_prior_gradient = self._gradient_norm(
                    base_prior_loss, action_noise_pred
                )
                losses["base_prior_gradient_norm_metric"] = (
                    base_prior_gradient
                )
                losses["recommended_base_prior_loss_weight_metric"] = (
                    float(self.config.base_prior_target_gradient_ratio)
                    * flow_gradient
                    / base_prior_gradient.clamp_min(epsilon)
                )
            if self.prior_config.predict_eef:
                eef_prior_gradient = self._gradient_norm(
                    eef_prior_loss, action_noise_pred
                )
                losses["eef_prior_gradient_norm_metric"] = eef_prior_gradient
                losses["recommended_eef_prior_loss_weight_metric"] = (
                    float(self.config.eef_prior_target_gradient_ratio)
                    * flow_gradient
                    / eef_prior_gradient.clamp_min(epsilon)
                )
            if (
                self.prior_config.predict_base
                and self.prior_config.predict_eef
            ):
                joint_prior_gradient = self._gradient_norm(
                    joint_prior_consistency_loss, action_noise_pred
                )
                losses["joint_prior_gradient_norm_metric"] = (
                    joint_prior_gradient
                )
                losses[
                    "recommended_joint_prior_consistency_loss_weight_metric"
                ] = (
                    float(
                        self.config.joint_prior_consistency_target_gradient_ratio
                    )
                    * flow_gradient
                    / joint_prior_gradient.clamp_min(epsilon)
                )
        return losses

    def capture_action_model_aux(
        self, context_index: int, action_model_prediction: torch.Tensor
    ) -> None:
        if context_index == 0:
            if self.prior_config.predict_base:
                self._latest_base_prior = action_model_prediction[
                    :,
                    self.prior_flow_indices,
                    self.base_action_dim : 2 * self.base_action_dim,
                ].detach()
            if self.prior_config.predict_eef:
                self._latest_eef_prior = action_model_prediction[
                    :,
                    self.prior_flow_indices,
                    2 * self.base_action_dim : 2 * self.base_action_dim + 9,
                ].detach()

    def get_action(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        self._latest_base_prior = None
        self._latest_eef_prior = None
        output = super().get_action(backbone_output, action_input)
        if self.prior_config.predict_base and self._latest_base_prior is None:
            raise RuntimeError(
                "Clean-prior model did not expose a coarse Base prediction"
            )
        if self.prior_config.predict_eef and self._latest_eef_prior is None:
            raise RuntimeError(
                "Clean-prior model did not expose a coarse EEF prediction"
            )
        if self._latest_base_prior is not None:
            output["base_prior_pred"] = self._latest_base_prior
        if self._latest_eef_prior is not None:
            output["eef_prior_pred"] = self._latest_eef_prior
        return output
