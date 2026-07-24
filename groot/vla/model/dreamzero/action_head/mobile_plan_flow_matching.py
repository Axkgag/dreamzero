"""Flow-matching policy head with decoupled Base and Manipulator plan losses."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers.feature_extraction_utils import BatchFeature

from .wan_flow_matching_action_tf import WANPolicyHead, WANPolicyHeadConfig


@dataclass(init=False)
class MobilePlanPolicyHeadConfig(WANPolicyHeadConfig):
    plan_horizon: int = field(default=6)
    base_action_dim: int = field(default=4)
    manipulator_action_dim: int = field(default=21)
    plan_time_offsets: tuple[int, ...] = field(
        default=(1, 4, 8, 12, 16, 24)
    )
    control_fps: float = field(default=30.0)
    base_flow_loss_weight: float = field(default=1.0)
    manipulator_flow_loss_weight: float = field(default=1.0)

    def __init__(self, **kwargs):
        # WAN configs intentionally accept additional research fields supplied
        # by the shared Hydra hierarchy.
        super().__init__(**kwargs)


class MobilePlanFlowMatchingActionHead(WANPolicyHead):
    """Reuse Wan video dynamics while treating the two plans as typed tokens."""

    config_class = MobilePlanPolicyHeadConfig

    def __init__(self, config: MobilePlanPolicyHeadConfig):
        if config.action_horizon != 2 * config.plan_horizon:
            raise ValueError("Packed action_horizon must be 2 * plan_horizon")
        if config.action_dim != config.manipulator_action_dim:
            raise ValueError("Packed action_dim must equal manipulator_action_dim")
        super().__init__(config)
        self.plan_horizon = config.plan_horizon
        self.base_action_dim = config.base_action_dim
        self.manipulator_action_dim = config.manipulator_action_dim

    def prepare_action_model_kwargs(self, action_input: BatchFeature) -> dict:
        offsets = action_input.plan_time_offsets
        expected = torch.as_tensor(
            self.config.plan_time_offsets,
            dtype=offsets.dtype,
            device=offsets.device,
        ).unsqueeze(0)
        if not torch.equal(offsets, expected.expand_as(offsets)):
            raise ValueError(
                f"Unexpected plan_time_offsets: {offsets.detach().cpu().tolist()}"
            )
        return {"plan_time_offsets": offsets}

    def align_action_timestep_ids(
        self, timestep_action_id: torch.Tensor
    ) -> torch.Tensor:
        if timestep_action_id.shape[1] != 2 * self.plan_horizon:
            raise ValueError(
                f"Expected {2 * self.plan_horizon} action timesteps, "
                f"got {timestep_action_id.shape[1]}"
            )
        base_timestep = timestep_action_id[:, : self.plan_horizon]
        return torch.cat([base_timestep, base_timestep], dim=1)

    def validate_action_video_layout(
        self,
        actions: torch.Tensor,
        noise: torch.Tensor,
        state_features: torch.Tensor,
        videos: torch.Tensor,
        latents: torch.Tensor,
    ) -> None:
        """A dual plan is one register block for one complete plan window."""
        latent_future_frames = noise.shape[1] - 1
        if latent_future_frames != self.num_frame_per_block:
            raise ValueError(
                "Mobile dual-plan training requires one video block per plan "
                f"window: got {latent_future_frames} future latent frames and "
                f"num_frame_per_block={self.num_frame_per_block}. "
                f"video={tuple(videos.shape)}, latents={tuple(latents.shape)}"
            )
        expected_actions = self.model.num_action_per_block
        expected_states = self.model.num_state_per_block
        if actions.shape[1] != expected_actions:
            raise ValueError(
                f"Expected {expected_actions} dual-plan tokens, "
                f"got {actions.shape[1]}"
            )
        if state_features.shape[1] != expected_states:
            raise ValueError(
                f"Expected {expected_states} state tokens, "
                f"got {state_features.shape[1]}"
            )

    def build_coupled_action_timestep_ids(
        self,
        timestep_id_block: torch.Tensor,
        actions: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Give every token in the plan window the block's diffusion time."""
        if timestep_id_block.shape[1] != 1:
            raise ValueError(
                "Mobile dual-plan timestep coupling expects exactly one "
                f"video block, got {timestep_id_block.shape[1]}"
            )
        block_timestep = timestep_id_block[:, :, 0]
        return block_timestep.expand(-1, actions.shape[1])

    def _masked_branch_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        has_real_action: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask.bool()
        squared_error = torch.nn.functional.mse_loss(
            prediction.float(), target.float(), reduction="none"
        )
        active_dims = mask.sum(dim=-1)
        token_loss = (squared_error * mask).sum(dim=-1) / active_dims.clamp_min(1)
        token_valid = active_dims > 0
        sample_valid = has_real_action.bool().unsqueeze(-1)
        valid = token_valid & sample_valid
        timestep_weight = self.scheduler.training_weight(
            timestep.flatten(0, 1)
        ).unflatten(0, timestep.shape).to(self._device)
        weighted = token_loss * timestep_weight * valid
        return weighted.sum() / valid.sum().clamp_min(1)

    def compute_action_losses(
        self,
        action_noise_pred: torch.Tensor,
        training_target_action: torch.Tensor,
        action_mask: torch.Tensor,
        has_real_action: torch.Tensor,
        timestep_action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        horizon = self.plan_horizon
        base_loss = self._masked_branch_loss(
            action_noise_pred[:, :horizon, : self.base_action_dim],
            training_target_action[:, :horizon, : self.base_action_dim],
            action_mask[:, :horizon, : self.base_action_dim],
            has_real_action,
            timestep_action[:, :horizon],
        )
        manipulator_loss = self._masked_branch_loss(
            action_noise_pred[:, horizon:, : self.manipulator_action_dim],
            training_target_action[:, horizon:, : self.manipulator_action_dim],
            action_mask[:, horizon:, : self.manipulator_action_dim],
            has_real_action,
            timestep_action[:, horizon:],
        )
        action_loss = (
            self.config.base_flow_loss_weight * base_loss
            + self.config.manipulator_flow_loss_weight * manipulator_loss
        )
        return {
            "action_loss": action_loss,
            "base_flow_loss": base_loss,
            "manipulator_flow_loss": manipulator_loss,
        }

    def get_action(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        # WANPolicyHead inherits ActionHead.get_action(), whose default behavior
        # calls the training forward pass. Dual-plan inference must instead run
        # the joint video/action flow sampler from a current observation.
        output = super().lazy_joint_video_action(backbone_output, action_input)
        packed = output["action_pred"]
        output["base_plan_pred"] = packed[
            :, : self.plan_horizon, : self.base_action_dim
        ]
        output["manipulator_plan_pred"] = packed[:, self.plan_horizon :]
        return output
