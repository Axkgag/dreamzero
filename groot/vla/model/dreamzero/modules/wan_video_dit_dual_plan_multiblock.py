"""Block-major Base/Manipulator plan adapters for DreamZero teacher forcing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wan_video_dit_action_casual_chunk import CategorySpecificMLP, CausalWanModel
from .wan_video_dit_dual_plan import PlanOffsetEmbedding
from .wan_video_dit_dual_plan_prior import (
    PRIOR_CONDITION_MODES,
    CleanPriorDirectedCausalWanSelfAttention,
    MobilePlanPriorConfig,
    coerce_mobile_plan_prior_config,
    resolve_prior_flow_indices,
)
from .wan_video_dit_action_casual_chunk import MultiEmbodimentActionEncoder
from ....utils.mobile_plan_spec import (
    EEF_ROTATION_ANCHOR_BASE_6D,
    eef_rotation_dim,
)


class MultiBlockDualPlanActionEncoder(nn.Module):
    """Encode ``[Base(W), Manipulator(W)]`` independently inside every block."""

    def __init__(
        self,
        base_action_dim: int,
        manipulator_action_dim: int,
        hidden_size: int,
        num_embodiments: int,
        plan_local_offsets: Sequence[int],
        control_fps: float,
    ) -> None:
        super().__init__()
        self.waypoints_per_block = len(plan_local_offsets)
        self.flow_tokens_per_block = 2 * self.waypoints_per_block
        self.base_action_dim = int(base_action_dim)
        self.manipulator_action_dim = int(manipulator_action_dim)
        self.base_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.base_action_dim,
            hidden_size=hidden_size,
            num_embodiments=num_embodiments,
        )
        self.manipulator_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.manipulator_action_dim,
            hidden_size=hidden_size,
            num_embodiments=num_embodiments,
        )
        self.type_embedding = nn.Parameter(torch.empty(2, hidden_size))
        nn.init.normal_(self.type_embedding, std=0.02)
        self.offset_embedding = PlanOffsetEmbedding(hidden_size)
        self.register_buffer(
            "offset_seconds",
            torch.as_tensor(plan_local_offsets, dtype=torch.float32) / control_fps,
            persistent=True,
        )

    def forward(
        self,
        packed_action: torch.Tensor,
        timesteps: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> torch.Tensor:
        if packed_action.ndim != 3:
            raise ValueError(f"Expected [B,T,D] action, got {packed_action.shape}")
        batch, token_count, action_dim = packed_action.shape
        if action_dim != self.manipulator_action_dim:
            raise ValueError(
                f"Expected action dim {self.manipulator_action_dim}, got {action_dim}"
            )
        if token_count % self.flow_tokens_per_block:
            raise ValueError(
                f"Action tokens {token_count} are not divisible by the per-block "
                f"flow width {self.flow_tokens_per_block}"
            )
        if timesteps.shape != packed_action.shape[:2]:
            raise ValueError(
                f"Action timestep shape {timesteps.shape} does not match "
                f"{packed_action.shape[:2]}"
            )
        num_blocks = token_count // self.flow_tokens_per_block
        block = packed_action.reshape(
            batch, num_blocks, self.flow_tokens_per_block, action_dim
        )
        block_timestep = timesteps.reshape(
            batch, num_blocks, self.flow_tokens_per_block
        )
        waypoint_count = self.waypoints_per_block
        base = block[:, :, :waypoint_count, : self.base_action_dim].reshape(
            batch, num_blocks * waypoint_count, self.base_action_dim
        )
        manipulator = block[:, :, waypoint_count:].reshape(
            batch, num_blocks * waypoint_count, self.manipulator_action_dim
        )
        base_timestep = block_timestep[:, :, :waypoint_count].reshape(
            batch, num_blocks * waypoint_count
        )
        manipulator_timestep = block_timestep[:, :, waypoint_count:].reshape(
            batch, num_blocks * waypoint_count
        )
        base_token = self.base_encoder(base, base_timestep, category_ids)
        manipulator_token = self.manipulator_encoder(
            manipulator, manipulator_timestep, category_ids
        )
        seconds = self.offset_seconds.to(device=packed_action.device)
        seconds = seconds.repeat(num_blocks).unsqueeze(0).expand(batch, -1)
        offset = self.offset_embedding(seconds, base_token.dtype)
        base_token = base_token + offset + self.type_embedding[0].to(base_token.dtype)
        manipulator_token = (
            manipulator_token
            + offset
            + self.type_embedding[1].to(manipulator_token.dtype)
        )
        base_token = base_token.reshape(batch, num_blocks, waypoint_count, -1)
        manipulator_token = manipulator_token.reshape(
            batch, num_blocks, waypoint_count, -1
        )
        return torch.cat([base_token, manipulator_token], dim=2).reshape(
            batch, num_blocks * self.flow_tokens_per_block, -1
        )


class MultiBlockDualPlanActionDecoder(nn.Module):
    """Decode block-major hidden registers back to padded flow actions."""

    def __init__(
        self,
        base_action_dim: int,
        manipulator_action_dim: int,
        hidden_size: int,
        model_dim: int,
        num_embodiments: int,
        waypoints_per_block: int,
    ) -> None:
        super().__init__()
        self.base_action_dim = int(base_action_dim)
        self.manipulator_action_dim = int(manipulator_action_dim)
        self.waypoints_per_block = int(waypoints_per_block)
        self.flow_tokens_per_block = 2 * self.waypoints_per_block
        self.base_decoder = CategorySpecificMLP(
            num_categories=num_embodiments,
            input_dim=model_dim,
            hidden_dim=hidden_size,
            output_dim=self.base_action_dim,
        )
        self.manipulator_decoder = CategorySpecificMLP(
            num_categories=num_embodiments,
            input_dim=model_dim,
            hidden_dim=hidden_size,
            output_dim=self.manipulator_action_dim,
        )

    def forward(
        self, hidden: torch.Tensor, category_ids: torch.Tensor
    ) -> torch.Tensor:
        if hidden.shape[1] % self.flow_tokens_per_block:
            raise ValueError(
                f"Hidden tokens {hidden.shape[1]} are not block-major width "
                f"{self.flow_tokens_per_block}"
            )
        batch = hidden.shape[0]
        num_blocks = hidden.shape[1] // self.flow_tokens_per_block
        block = hidden.reshape(batch, num_blocks, self.flow_tokens_per_block, -1)
        base_hidden = block[:, :, : self.waypoints_per_block].reshape(
            batch, num_blocks * self.waypoints_per_block, -1
        )
        manipulator_hidden = block[:, :, self.waypoints_per_block :].reshape(
            batch, num_blocks * self.waypoints_per_block, -1
        )
        base = self.base_decoder(base_hidden, category_ids).reshape(
            batch, num_blocks, self.waypoints_per_block, self.base_action_dim
        )
        manipulator = self.manipulator_decoder(
            manipulator_hidden, category_ids
        ).reshape(
            batch,
            num_blocks,
            self.waypoints_per_block,
            self.manipulator_action_dim,
        )
        padded_base = F.pad(
            base, (0, self.manipulator_action_dim - self.base_action_dim)
        )
        return torch.cat([padded_base, manipulator], dim=2).reshape(
            batch, num_blocks * self.flow_tokens_per_block, self.manipulator_action_dim
        )


class WanVideoDiTMultiBlockDualPlan(CausalWanModel):
    """DreamZero causal Wan model with four block-major plan registers per block."""

    def __init__(
        self,
        plan_waypoints_per_block: int = 2,
        base_action_dim: int = 4,
        manipulator_action_dim: int = 21,
        plan_local_offsets: Sequence[int] = (4, 8),
        control_fps: float = 30.0,
        eef_rotation_representation: str = EEF_ROTATION_ANCHOR_BASE_6D,
        **kwargs,
    ) -> None:
        if len(plan_local_offsets) != plan_waypoints_per_block:
            raise ValueError(
                "plan_local_offsets length must equal plan_waypoints_per_block"
            )
        flow_tokens_per_block = 2 * plan_waypoints_per_block
        kwargs["action_dim"] = manipulator_action_dim
        kwargs["num_action_per_block"] = flow_tokens_per_block
        super().__init__(**kwargs)
        self.plan_waypoints_per_block = int(plan_waypoints_per_block)
        self.flow_tokens_per_block = int(flow_tokens_per_block)
        self.base_action_dim = int(base_action_dim)
        self.manipulator_action_dim = int(manipulator_action_dim)
        self.eef_rotation_representation = eef_rotation_representation
        self.register_buffer(
            "expected_plan_local_offsets",
            torch.as_tensor(plan_local_offsets, dtype=torch.long),
            persistent=True,
        )
        self.action_encoder = MultiBlockDualPlanActionEncoder(
            base_action_dim=base_action_dim,
            manipulator_action_dim=manipulator_action_dim,
            hidden_size=self.dim,
            num_embodiments=1,
            plan_local_offsets=plan_local_offsets,
            control_fps=control_fps,
        )
        self.action_decoder = MultiBlockDualPlanActionDecoder(
            base_action_dim=base_action_dim,
            manipulator_action_dim=manipulator_action_dim,
            hidden_size=self.hidden_size,
            model_dim=self.dim,
            num_embodiments=1,
            waypoints_per_block=plan_waypoints_per_block,
        )

    def _validate_offsets(self, plan_local_offsets: torch.Tensor | Sequence[int]) -> None:
        offsets = torch.as_tensor(
            plan_local_offsets,
            device=self.expected_plan_local_offsets.device,
            dtype=torch.long,
        )
        if offsets.ndim == 1:
            offsets = offsets.unsqueeze(0)
        expected = self.expected_plan_local_offsets.unsqueeze(0).expand_as(offsets)
        if not torch.equal(offsets, expected):
            raise ValueError(
                "Expected local plan offsets "
                f"{self.expected_plan_local_offsets.tolist()}, got {offsets.tolist()}"
            )

    def forward(self, *args, plan_local_offsets=None, **kwargs):
        if plan_local_offsets is None:
            raise ValueError("Multiblock dual plan requires plan_local_offsets")
        self._validate_offsets(plan_local_offsets)
        return super().forward(*args, **kwargs)


class MultiBlockCleanPriorActionEncoder(MultiBlockDualPlanActionEncoder):
    """Insert clean Prior queries at the start of every action block."""

    def __init__(self, *args, prior_flow_indices: Sequence[int], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if len(prior_flow_indices) != 1:
            raise ValueError("Multiblock endpoint prior requires one local offset")
        self.prior_flow_index = int(prior_flow_indices[0])
        hidden_size = self.type_embedding.shape[-1]
        self.prior_query = nn.Parameter(torch.empty(1, hidden_size))
        self.prior_type_embedding = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.prior_query, std=0.02)
        nn.init.normal_(self.prior_type_embedding, std=0.02)

    def forward(self, packed_action, timesteps, category_ids):
        flow = super().forward(packed_action, timesteps, category_ids)
        batch = flow.shape[0]
        num_blocks = flow.shape[1] // self.flow_tokens_per_block
        flow = flow.reshape(batch, num_blocks, self.flow_tokens_per_block, -1)
        seconds = self.offset_seconds[self.prior_flow_index].to(
            device=packed_action.device
        )
        seconds = seconds.reshape(1, 1).expand(batch * num_blocks, 1)
        offset = self.offset_embedding(seconds, flow.dtype).reshape(
            batch, num_blocks, 1, -1
        )
        prior = self.prior_query.to(flow.dtype).reshape(1, 1, 1, -1) + offset
        prior = prior + self.prior_type_embedding.to(flow.dtype)
        return torch.cat([prior.expand(batch, num_blocks, -1, -1), flow], dim=2).reshape(
            batch, num_blocks * (self.flow_tokens_per_block + 1), -1
        )


class MultiBlockCleanPriorActionDecoder(MultiBlockDualPlanActionDecoder):
    """Decode one clean Prior plus four noisy flow registers per block."""

    def __init__(
        self,
        *args,
        prior_flow_index: int,
        eef_prior_dim: int = 9,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.eef_prior_dim = int(eef_prior_dim)
        if 2 * self.base_action_dim + self.eef_prior_dim > self.manipulator_action_dim:
            raise ValueError("Packed Base channels cannot hold Base and EEF Prior outputs")
        self.prior_flow_index = int(prior_flow_index)
        model_dim = self.base_decoder.layer1.W.shape[1]
        hidden_size = self.base_decoder.layer1.W.shape[2]
        num_embodiments = self.base_decoder.layer1.W.shape[0]
        self.base_prior_head = CategorySpecificMLP(
            num_categories=num_embodiments,
            input_dim=model_dim,
            hidden_dim=hidden_size,
            output_dim=self.base_action_dim,
        )
        self.eef_prior_head = CategorySpecificMLP(
            num_categories=num_embodiments,
            input_dim=model_dim,
            hidden_dim=hidden_size,
            output_dim=self.eef_prior_dim,
        )

    def forward(self, hidden, category_ids):
        internal_width = self.flow_tokens_per_block + 1
        if hidden.shape[1] % internal_width:
            raise ValueError("Prior hidden registers do not align with action blocks")
        batch = hidden.shape[0]
        num_blocks = hidden.shape[1] // internal_width
        block = hidden.reshape(batch, num_blocks, internal_width, -1)
        prior_hidden = block[:, :, 0].reshape(batch, num_blocks, -1)
        flow_hidden = block[:, :, 1:].reshape(
            batch, num_blocks * self.flow_tokens_per_block, -1
        )
        flow = super().forward(flow_hidden, category_ids).reshape(
            batch,
            num_blocks,
            self.flow_tokens_per_block,
            self.manipulator_action_dim,
        ).clone()
        base_prior = self.base_prior_head(prior_hidden, category_ids)
        eef_prior = self.eef_prior_head(prior_hidden, category_ids)
        slot = self.prior_flow_index
        flow[:, :, slot, self.base_action_dim : 2 * self.base_action_dim] = base_prior
        flow[
            :,
            :,
            slot,
            2 * self.base_action_dim : 2 * self.base_action_dim + self.eef_prior_dim,
        ] = eef_prior
        return flow.reshape(
            batch, num_blocks * self.flow_tokens_per_block, self.manipulator_action_dim
        )


class WanVideoDiTMultiBlockDualPlanPrior(WanVideoDiTMultiBlockDualPlan):
    """Multiblock dual plan with one directed clean endpoint Prior per block."""

    def __init__(
        self,
        prior: MobilePlanPriorConfig | Mapping[str, object] | None = None,
        prior_time_offsets: Sequence[int] | None = None,
        prior_condition_mode: str = "normal",
        **kwargs,
    ) -> None:
        plan_local_offsets = tuple(kwargs.get("plan_local_offsets", (4, 8)))
        if prior is None and prior_time_offsets is None:
            prior = MobilePlanPriorConfig(time_offsets=(8,))
        prior_config = coerce_mobile_plan_prior_config(
            prior, legacy_time_offsets=prior_time_offsets
        )
        prior_indices = resolve_prior_flow_indices(
            plan_local_offsets, prior_config.time_offsets
        )
        if len(prior_indices) != 1:
            raise ValueError("Multiblock WAM supports one endpoint Prior per block")
        if prior_condition_mode not in PRIOR_CONDITION_MODES:
            raise ValueError(f"Unknown prior_condition_mode: {prior_condition_mode}")
        super().__init__(**kwargs)
        self.prior_config = prior_config
        self.prior_condition_mode = prior_condition_mode
        self.prior_flow_index = int(prior_indices[0])
        internal_width = self.flow_tokens_per_block + 1
        self.num_action_per_block = internal_width
        for block in self.blocks:
            previous = block.self_attn
            replacement = CleanPriorDirectedCausalWanSelfAttention(
                dim=previous.dim,
                num_heads=previous.num_heads,
                frame_seqlen=previous.frame_seqlen,
                local_attn_size=previous.local_attn_size,
                sink_size=previous.sink_size,
                num_frame_per_block=previous.num_frame_per_block,
                qk_norm=previous.qk_norm,
                eps=previous.eps,
                num_action_per_block=internal_width,
                num_state_per_block=previous.num_state_per_block,
                num_base_prior_tokens=1,
            )
            replacement.load_state_dict(previous.state_dict())
            block.self_attn = replacement
        self.action_encoder = MultiBlockCleanPriorActionEncoder(
            base_action_dim=self.base_action_dim,
            manipulator_action_dim=self.manipulator_action_dim,
            hidden_size=self.dim,
            num_embodiments=1,
            plan_local_offsets=plan_local_offsets,
            prior_flow_indices=prior_indices,
            control_fps=float(kwargs.get("control_fps", 30.0)),
        )
        self.action_decoder = MultiBlockCleanPriorActionDecoder(
            base_action_dim=self.base_action_dim,
            manipulator_action_dim=self.manipulator_action_dim,
            hidden_size=self.hidden_size,
            model_dim=self.dim,
            num_embodiments=1,
            waypoints_per_block=self.plan_waypoints_per_block,
            prior_flow_index=self.prior_flow_index,
            eef_prior_dim=3 + eef_rotation_dim(
                self.eef_rotation_representation
            ),
        )

    def _action_register_timesteps(
        self, timestep_action, action_features, state_features
    ):
        if timestep_action.shape[1] % self.flow_tokens_per_block:
            raise ValueError("Flow timesteps do not align with multiblock actions")
        batch = timestep_action.shape[0]
        num_blocks = timestep_action.shape[1] // self.flow_tokens_per_block
        expected_registers = num_blocks * (self.flow_tokens_per_block + 1)
        if action_features.shape[1] != expected_registers:
            raise ValueError(
                f"Expected {expected_registers} internal action registers, got "
                f"{action_features.shape[1]}"
            )
        if state_features.shape[1] != num_blocks:
            raise ValueError("Prior action/state block counts differ")
        flow_timestep = timestep_action.reshape(
            batch, num_blocks, self.flow_tokens_per_block
        )
        prior_timestep = torch.zeros_like(flow_timestep[:, :, :1])
        internal = torch.cat([prior_timestep, flow_timestep], dim=2).reshape(
            batch, expected_registers
        )
        return internal, flow_timestep[:, :, 0]

    def forward(
        self,
        *args,
        prior_time_offsets=None,
        prior_condition_mode=None,
        **kwargs,
    ):
        if prior_time_offsets is None:
            raise ValueError("Multiblock Prior requires prior_time_offsets")
        offsets = torch.as_tensor(
            prior_time_offsets,
            device=self.expected_plan_local_offsets.device,
            dtype=torch.long,
        )
        if offsets.ndim == 1:
            offsets = offsets.unsqueeze(0)
        expected = torch.as_tensor(
            self.prior_config.time_offsets,
            device=offsets.device,
            dtype=torch.long,
        ).unsqueeze(0).expand_as(offsets)
        if not torch.equal(offsets, expected):
            raise ValueError(
                f"Expected Prior offsets {list(self.prior_config.time_offsets)}, "
                f"got {offsets.tolist()}"
            )
        mode = prior_condition_mode or self.prior_condition_mode
        if mode not in PRIOR_CONDITION_MODES:
            raise ValueError(f"Unknown prior_condition_mode: {mode}")
        for block in self.blocks:
            block.self_attn.prior_condition_mode = mode
        return super().forward(*args, **kwargs)
