"""Block-major MobileManiBench flow heads using DreamZero teacher forcing."""

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
from .mobile_plan_clean_prior_flow_matching import (
    MobilePlanCleanPriorPolicyHeadConfig,
)
from .mobile_plan_flow_matching import (
    MobilePlanFlowMatchingActionHead,
    MobilePlanPolicyHeadConfig,
)
from .mobile_plan_physical_losses import MobilePlanPhysicalConsistencyLosses


@dataclass(init=False)
class MobilePlanMultiBlockPolicyHeadConfig(MobilePlanPolicyHeadConfig):
    num_plan_blocks: int = field(default=4)
    plan_waypoints_per_block: int = field(default=2)
    plan_local_offsets: tuple[int, ...] = field(default=(4, 8))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MobilePlanMultiBlockFlowMatchingActionHead(MobilePlanFlowMatchingActionHead):
    """Apply flow and physical losses after unpacking block-major actions."""

    config_class = MobilePlanMultiBlockPolicyHeadConfig

    def __init__(self, config: MobilePlanMultiBlockPolicyHeadConfig):
        if config.plan_horizon != config.plan_waypoints_per_block:
            raise ValueError("plan_horizon must mean waypoints per block")
        if len(config.plan_local_offsets) != config.plan_waypoints_per_block:
            raise ValueError("plan_local_offsets length must match local horizon")
        super().__init__(config)
        self.num_plan_blocks = int(config.num_plan_blocks)
        self.waypoints_per_block = int(config.plan_waypoints_per_block)
        self.flow_tokens_per_block = 2 * self.waypoints_per_block
        if config.action_horizon != self.flow_tokens_per_block:
            raise ValueError(
                "action_horizon is the single-block flow width and must equal "
                f"{self.flow_tokens_per_block}"
            )

    def prepare_action_model_kwargs(self, action_input: BatchFeature) -> dict:
        offsets = action_input.plan_local_offsets
        expected = torch.as_tensor(
            self.config.plan_local_offsets,
            dtype=offsets.dtype,
            device=offsets.device,
        ).unsqueeze(0)
        if not torch.equal(offsets, expected.expand_as(offsets)):
            raise ValueError(
                f"Unexpected plan_local_offsets: {offsets.detach().cpu().tolist()}"
            )
        return {"plan_local_offsets": offsets}

    def _num_action_blocks(self, action: torch.Tensor) -> int:
        if action.shape[1] % self.flow_tokens_per_block:
            raise ValueError(
                f"Action length {action.shape[1]} is not divisible by "
                f"{self.flow_tokens_per_block}"
            )
        return action.shape[1] // self.flow_tokens_per_block

    def align_action_timestep_ids(
        self, timestep_action_id: torch.Tensor
    ) -> torch.Tensor:
        num_blocks = self._num_action_blocks(timestep_action_id.unsqueeze(-1))
        block = timestep_action_id.reshape(
            timestep_action_id.shape[0], num_blocks, self.flow_tokens_per_block
        )
        return block[:, :, :1].expand_as(block).reshape_as(timestep_action_id)

    def validate_action_video_layout(
        self,
        actions: torch.Tensor,
        noise: torch.Tensor,
        state_features: torch.Tensor,
        videos: torch.Tensor,
        latents: torch.Tensor,
    ) -> None:
        latent_future_frames = noise.shape[1] - 1
        if latent_future_frames <= 0 or latent_future_frames % self.num_frame_per_block:
            raise ValueError(
                "Future latent count must be a positive multiple of "
                f"num_frame_per_block={self.num_frame_per_block}; got "
                f"{latent_future_frames} from video={tuple(videos.shape)}, "
                f"latents={tuple(latents.shape)}"
            )
        video_blocks = latent_future_frames // self.num_frame_per_block
        action_blocks = self._num_action_blocks(actions)
        state_blocks = state_features.shape[1] // self.model.num_state_per_block
        if not (video_blocks == action_blocks == state_blocks):
            raise ValueError(
                "Video/action/state block mismatch: "
                f"video={video_blocks}, action={action_blocks}, state={state_blocks}"
            )
        if video_blocks != self.num_plan_blocks:
            raise ValueError(
                f"Expected {self.num_plan_blocks} complete training blocks, "
                f"got {video_blocks}"
            )

    def build_coupled_action_timestep_ids(
        self,
        timestep_id_block: torch.Tensor,
        actions: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        del noise
        num_blocks = self._num_action_blocks(actions)
        if timestep_id_block.shape[1] != num_blocks:
            raise ValueError(
                f"Expected {num_blocks} video timestep blocks, got "
                f"{timestep_id_block.shape[1]}"
            )
        block_timestep = timestep_id_block[:, :, 0]
        return block_timestep.unsqueeze(-1).expand(
            -1, -1, self.flow_tokens_per_block
        ).reshape(actions.shape[0], actions.shape[1])

    def _branch_major_per_block(self, value: torch.Tensor) -> torch.Tensor:
        num_blocks = self._num_action_blocks(value)
        block = value.reshape(
            value.shape[0],
            num_blocks,
            self.flow_tokens_per_block,
            *value.shape[2:],
        )
        return block.reshape(
            value.shape[0] * num_blocks,
            self.flow_tokens_per_block,
            *value.shape[2:],
        )

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
        num_blocks = self._num_action_blocks(action_noise_pred)
        repeated_real = has_real_action.repeat_interleave(num_blocks)
        converted_aux = dict(action_model_aux or {})
        physical_state = converted_aux.get("physical_block_state")
        if physical_state is not None:
            if physical_state.shape[:2] != (
                action_noise_pred.shape[0],
                num_blocks,
            ):
                raise ValueError(
                    "physical_block_state must be [batch,num_blocks,state_dim], "
                    f"got {tuple(physical_state.shape)}"
                )
            converted_aux["physical_block_state"] = physical_state.reshape(
                action_noise_pred.shape[0] * num_blocks,
                physical_state.shape[-1],
            )
        converted = {
            "action_noise_pred": self._branch_major_per_block(action_noise_pred),
            "training_target_action": self._branch_major_per_block(
                training_target_action
            ),
            "action_mask": self._branch_major_per_block(action_mask),
            "timestep_action": self._branch_major_per_block(
                timestep_action.unsqueeze(-1)
            ).squeeze(-1),
            "noisy_actions": (
                self._branch_major_per_block(noisy_actions)
                if noisy_actions is not None
                else None
            ),
            "clean_actions": (
                self._branch_major_per_block(clean_actions)
                if clean_actions is not None
                else None
            ),
        }
        losses = super().compute_action_losses(
            converted["action_noise_pred"],
            converted["training_target_action"],
            converted["action_mask"],
            repeated_real,
            converted["timestep_action"],
            noisy_actions=converted["noisy_actions"],
            clean_actions=converted["clean_actions"],
            action_model_aux=converted_aux,
        )

        prediction_block = action_noise_pred.reshape(
            action_noise_pred.shape[0],
            num_blocks,
            self.flow_tokens_per_block,
            action_noise_pred.shape[-1],
        )
        target_block = training_target_action.reshape_as(prediction_block)
        mask_block = action_mask.reshape_as(prediction_block)
        timestep_block = timestep_action.reshape(
            timestep_action.shape[0], num_blocks, self.flow_tokens_per_block
        )
        for block_index in range(num_blocks):
            base = slice(0, self.waypoints_per_block)
            manipulator = slice(
                self.waypoints_per_block, self.flow_tokens_per_block
            )
            losses[f"base_flow_loss/block_{block_index}"] = self._masked_branch_loss(
                prediction_block[:, block_index, base, : self.base_action_dim],
                target_block[:, block_index, base, : self.base_action_dim],
                mask_block[:, block_index, base, : self.base_action_dim],
                has_real_action,
                timestep_block[:, block_index, base],
            )
            losses[
                f"manipulator_flow_loss/block_{block_index}"
            ] = self._masked_branch_loss(
                prediction_block[
                    :, block_index, manipulator, : self.manipulator_action_dim
                ],
                target_block[
                    :, block_index, manipulator, : self.manipulator_action_dim
                ],
                mask_block[
                    :, block_index, manipulator, : self.manipulator_action_dim
                ],
                has_real_action,
                timestep_block[:, block_index, manipulator],
            )
        return losses

    def get_action(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        output = super().get_action(backbone_output, action_input)
        packed = output["action_pred"]
        if packed.shape[1] != self.flow_tokens_per_block:
            raise ValueError(
                "Cached multiblock inference must return exactly one action block; "
                f"got {packed.shape[1]} tokens"
            )
        output["base_plan_pred"] = packed[
            :, : self.waypoints_per_block, : self.base_action_dim
        ]
        output["manipulator_plan_pred"] = packed[
            :, self.waypoints_per_block :, : self.manipulator_action_dim
        ]
        return output


@dataclass(init=False)
class MobilePlanMultiBlockCleanPriorPolicyHeadConfig(
    MobilePlanCleanPriorPolicyHeadConfig
):
    num_plan_blocks: int = field(default=4)
    plan_waypoints_per_block: int = field(default=2)
    plan_local_offsets: tuple[int, ...] = field(default=(4, 8))

    def __init__(self, **kwargs):
        prior = kwargs.pop("prior", None)
        if prior is None:
            prior = asdict(MobilePlanPriorConfig(time_offsets=(8,)))
        kwargs["prior"] = prior
        super().__init__(**kwargs)


class MobilePlanMultiBlockCleanPriorFlowMatchingActionHead(
    MobilePlanMultiBlockFlowMatchingActionHead
):
    """Multiblock flow head with one clean endpoint Prior per block."""

    config_class = MobilePlanMultiBlockCleanPriorPolicyHeadConfig

    def __init__(self, config: MobilePlanMultiBlockCleanPriorPolicyHeadConfig):
        if not config.plan_stats_path:
            raise ValueError("Multiblock clean Prior requires plan_stats_path")
        super().__init__(config)
        self.prior_config = coerce_mobile_plan_prior_config(config.prior)
        prior_indices = resolve_prior_flow_indices(
            config.plan_local_offsets, self.prior_config.time_offsets
        )
        if len(prior_indices) != 1:
            raise ValueError("Multiblock WAM supports one endpoint Prior per block")
        self.prior_flow_index = int(prior_indices[0])
        self.prior_physical_losses = MobilePlanPhysicalConsistencyLosses(
            config.plan_stats_path,
            plan_horizon=1,
            base_action_dim=config.base_action_dim,
            manipulator_action_dim=config.manipulator_action_dim,
            huber_beta=config.physical_loss_huber_beta,
            eef_rotation_representation=config.eef_rotation_representation,
            eef_rotation_sigma_weight_base=config.eef_rotation_sigma_weight_base,
            eef_rotation_sigma_weight_scale=config.eef_rotation_sigma_weight_scale,
        )
        self._latest_base_prior: torch.Tensor | None = None
        self._latest_eef_prior: torch.Tensor | None = None

    def prepare_action_model_kwargs(self, action_input: BatchFeature) -> dict:
        kwargs = super().prepare_action_model_kwargs(action_input)
        offsets = torch.as_tensor(
            self.prior_config.time_offsets,
            dtype=action_input.plan_local_offsets.dtype,
            device=action_input.plan_local_offsets.device,
        ).unsqueeze(0).expand(action_input.plan_local_offsets.shape[0], -1)
        kwargs.update(
            {
                "prior_condition_mode": self.config.prior_condition_mode,
                "prior_time_offsets": offsets,
            }
        )
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
        if clean_actions is None:
            raise ValueError("Multiblock clean Prior requires clean_actions")
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
        num_blocks = self._num_action_blocks(action_noise_pred)
        batch = action_noise_pred.shape[0]
        prediction_block = action_noise_pred.reshape(
            batch,
            num_blocks,
            self.flow_tokens_per_block,
            action_noise_pred.shape[-1],
        )
        clean_block = clean_actions.reshape_as(prediction_block)
        mask_block = action_mask.reshape_as(prediction_block)
        endpoint = self.prior_flow_index
        base_prior = (
            prediction_block[
                :,
                :,
                endpoint,
                self.base_action_dim : 2 * self.base_action_dim,
            ].reshape(batch * num_blocks, 1, self.base_action_dim)
            if self.prior_config.predict_base
            else None
        )
        eef_prior = (
            prediction_block[
                :,
                :,
                endpoint,
                2 * self.base_action_dim : (
                    2 * self.base_action_dim
                    + 3
                    + self.prior_physical_losses.eef_rotation_dim
                ),
            ].reshape(
                batch * num_blocks,
                1,
                3 + self.prior_physical_losses.eef_rotation_dim,
            )
            if self.prior_config.predict_eef
            else None
        )
        base_target = clean_block[:, :, endpoint : endpoint + 1]
        manipulator_index = self.waypoints_per_block + endpoint
        manipulator_target = clean_block[
            :, :, manipulator_index : manipulator_index + 1
        ]
        prior_target = torch.cat([base_target, manipulator_target], dim=2).reshape(
            batch * num_blocks, 2, clean_actions.shape[-1]
        )
        prior_mask = torch.cat(
            [
                mask_block[:, :, endpoint : endpoint + 1],
                mask_block[:, :, manipulator_index : manipulator_index + 1],
            ],
            dim=2,
        ).reshape(batch * num_blocks, 2, action_mask.shape[-1])
        repeated_real = has_real_action.repeat_interleave(num_blocks)
        terms = self.prior_physical_losses.prior_terms(
            base_prediction=base_prior,
            eef_prediction=eef_prior,
            clean_target=prior_target,
            action_mask=prior_mask,
            has_real_action=repeated_real,
            eef_frame=self.prior_config.eef_frame,
            anchor_state=(
                action_model_aux["physical_block_state"].reshape(
                    batch * num_blocks, -1
                )
                if action_model_aux is not None
                and action_model_aux.get("physical_block_state") is not None
                else None
            ),
        )
        base_loss = (
            self.config.base_prior_xy_loss_weight * terms["base_prior_xy_loss"]
            + self.config.base_prior_yaw_loss_weight * terms["base_prior_yaw_loss"]
            + self.config.base_prior_unit_loss_weight
            * terms["base_prior_unit_loss"]
        )
        eef_loss = (
            self.config.eef_prior_position_loss_weight
            * terms["eef_prior_position_loss"]
            + self.config.eef_prior_rotation_loss_weight
            * terms["eef_prior_rotation_loss"]
        )
        joint_loss = (
            self.config.joint_prior_consistency_position_loss_weight
            * terms["joint_prior_consistency_position_loss"]
            + self.config.joint_prior_consistency_rotation_loss_weight
            * terms["joint_prior_consistency_rotation_loss"]
        )
        step = int(self.global_step)
        base_weight = (
            self._ramped_weight(
                self.config.base_prior_loss_weight,
                step,
                self.config.base_prior_loss_start_step,
                self.config.base_prior_loss_ramp_steps,
            )
            if self.prior_config.predict_base
            else 0.0
        )
        eef_weight = (
            self._ramped_weight(
                self.config.eef_prior_loss_weight,
                step,
                self.config.eef_prior_loss_start_step,
                self.config.eef_prior_loss_ramp_steps,
            )
            if self.prior_config.predict_eef
            else 0.0
        )
        joint_weight = (
            self._ramped_weight(
                self.config.joint_prior_consistency_loss_weight,
                step,
                self.config.joint_prior_consistency_loss_start_step,
                self.config.joint_prior_consistency_loss_ramp_steps,
            )
            if self.prior_config.predict_base and self.prior_config.predict_eef
            else 0.0
        )
        weighted_base = base_weight * base_loss
        weighted_eef = eef_weight * eef_loss
        weighted_joint = joint_weight * joint_loss
        losses["action_loss"] = (
            losses["action_loss"] + weighted_base + weighted_eef + weighted_joint
        )
        losses.update(terms)
        losses.update(
            {
                "base_prior_loss": base_loss,
                "eef_prior_loss": eef_loss,
                "joint_prior_consistency_loss": joint_loss,
                "weighted_base_prior_loss": weighted_base,
                "weighted_eef_prior_loss": weighted_eef,
                "weighted_joint_prior_consistency_loss": weighted_joint,
                "effective_base_prior_loss_weight": torch.as_tensor(
                    base_weight, device=losses["action_loss"].device
                ),
                "effective_eef_prior_loss_weight": torch.as_tensor(
                    eef_weight, device=losses["action_loss"].device
                ),
                "effective_joint_prior_consistency_loss_weight": torch.as_tensor(
                    joint_weight, device=losses["action_loss"].device
                ),
            }
        )
        return losses

    def capture_action_model_aux(
        self, context_index: int, action_model_prediction: torch.Tensor
    ) -> None:
        if context_index != 0:
            return
        block = action_model_prediction.reshape(
            action_model_prediction.shape[0],
            -1,
            self.flow_tokens_per_block,
            action_model_prediction.shape[-1],
        )
        endpoint = self.prior_flow_index
        if self.prior_config.predict_base:
            self._latest_base_prior = block[
                :,
                :,
                endpoint,
                self.base_action_dim : 2 * self.base_action_dim,
            ].detach()
        if self.prior_config.predict_eef:
            self._latest_eef_prior = block[
                :,
                :,
                endpoint,
                2 * self.base_action_dim : (
                    2 * self.base_action_dim
                    + 3
                    + self.prior_physical_losses.eef_rotation_dim
                ),
            ].detach()

    def get_action(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        self._latest_base_prior = None
        self._latest_eef_prior = None
        output = super().get_action(backbone_output, action_input)
        if self._latest_base_prior is not None:
            output["base_prior_pred"] = self._latest_base_prior
        if self._latest_eef_prior is not None:
            output["eef_prior_pred"] = self._latest_eef_prior
        return output
