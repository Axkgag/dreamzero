"""Clean Base/EEF-Prior extension of the MobileManiBench dual-plan Wan DiT."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn

from .wan_video_dit_action_casual_chunk import (
    CausalWanModel,
    CausalWanSelfAttention,
    CategorySpecificMLP,
)
from .wan_video_dit_dual_plan import (
    DualPlanActionDecoder,
    DualPlanActionEncoder,
)


PRIOR_CONDITION_MODES = frozenset({"normal", "masked", "shuffled"})


@dataclass(frozen=True)
class MobilePlanPriorConfig:
    """Semantic targets and temporal grid for clean mobile-plan Prior tokens."""

    time_offsets: tuple[int, ...] = (8, 16, 24)
    predict_base: bool = True
    predict_eef: bool = False
    eef_frame: str = "future_base"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "time_offsets",
            tuple(int(offset) for offset in self.time_offsets),
        )
        if not self.predict_base and not self.predict_eef:
            raise ValueError("At least one Prior target must be enabled")
        if self.eef_frame not in {"current_base", "future_base"}:
            raise ValueError(
                "prior.eef_frame must be 'current_base' or 'future_base'"
            )


def coerce_mobile_plan_prior_config(
    value: MobilePlanPriorConfig | Mapping[str, Any] | None,
    *,
    legacy_time_offsets: Sequence[int] | None = None,
) -> MobilePlanPriorConfig:
    """Normalize Hydra mappings while retaining flat-config compatibility."""
    if value is None:
        kwargs: dict[str, Any] = {}
        if legacy_time_offsets is not None:
            kwargs["time_offsets"] = tuple(legacy_time_offsets)
        return MobilePlanPriorConfig(**kwargs)
    if isinstance(value, MobilePlanPriorConfig):
        result = value
    elif isinstance(value, Mapping):
        result = MobilePlanPriorConfig(**dict(value))
    else:
        raise TypeError(
            "prior must be a MobilePlanPriorConfig or mapping, got "
            f"{type(value).__name__}"
        )
    if legacy_time_offsets is not None and tuple(legacy_time_offsets) != tuple(
        result.time_offsets
    ):
        raise ValueError(
            "Conflicting prior.time_offsets and legacy prior_time_offsets"
        )
    return result


def resolve_prior_flow_indices(
    plan_time_offsets: Sequence[int],
    prior_time_offsets: Sequence[int],
) -> tuple[int, ...]:
    """Map configurable Prior offsets onto the sparse Flow waypoint grid."""
    plan_offsets = tuple(int(offset) for offset in plan_time_offsets)
    prior_offsets = tuple(int(offset) for offset in prior_time_offsets)
    if not prior_offsets:
        raise ValueError("prior_time_offsets must contain at least one offset")
    if tuple(sorted(set(prior_offsets))) != prior_offsets:
        raise ValueError(
            "prior_time_offsets must be unique and strictly increasing"
        )
    plan_index = {offset: index for index, offset in enumerate(plan_offsets)}
    missing = [offset for offset in prior_offsets if offset not in plan_index]
    if missing:
        raise ValueError(
            "prior_time_offsets must be a subset of plan_time_offsets; "
            f"missing offsets: {missing}"
        )
    return tuple(plan_index[offset] for offset in prior_offsets)


def _condition_prior(value: torch.Tensor, mode: str) -> torch.Tensor | None:
    if mode == "masked":
        return None
    if mode == "shuffled":
        if value.shape[0] < 2:
            raise ValueError("shuffled prior_condition_mode requires batch_size >= 2")
        return torch.roll(value, shifts=1, dims=0)
    if mode != "normal":
        raise ValueError(f"Unknown prior_condition_mode: {mode}")
    return value


class CleanPriorDirectedCausalWanSelfAttention(CausalWanSelfAttention):
    """Give clean Prior queries a one-way path into downstream tokens."""

    def __init__(self, *args, num_base_prior_tokens: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_base_prior_tokens = num_base_prior_tokens
        self.prior_condition_mode = "normal"

    def _downstream_context(self, prior_key, prior_value):
        key = _condition_prior(prior_key, self.prior_condition_mode)
        value = _condition_prior(prior_value, self.prior_condition_mode)
        if (key is None) != (value is None):
            raise RuntimeError("Prior key/value conditioning mismatch")
        return key, value

    def _process_noisy_image_blocks(
        self,
        noisy_image_q,
        noisy_image_k,
        noisy_image_v,
        clean_image_k,
        clean_image_v,
        noisy_action_k,
        noisy_action_v,
        noisy_state_k,
        noisy_state_v,
        half_frames,
        action_horizon,
        state_horizon,
    ):
        del action_horizon, state_horizon
        block_size = self.frame_seqlen * self.num_frame_per_block
        num_blocks = (half_frames - 1) // self.num_frame_per_block
        output = torch.empty_like(noisy_image_q)
        output[:, : self.frame_seqlen] = self.attn(
            noisy_image_q[:, : self.frame_seqlen],
            noisy_image_k[:, : self.frame_seqlen],
            noisy_image_v[:, : self.frame_seqlen],
        )
        for block_index in range(num_blocks):
            noisy_start = self.frame_seqlen + block_index * block_size
            noisy_end = min(noisy_start + block_size, noisy_image_q.shape[1])
            clean_end = self.frame_seqlen + block_index * block_size
            action_start = block_index * self.num_action_per_block
            prior_start = action_start
            prior_end = prior_start + self.num_base_prior_tokens
            flow_end = action_start + self.num_action_per_block
            state_start = block_index * self.num_state_per_block
            state_end = state_start + self.num_state_per_block
            prior_key, prior_value = self._downstream_context(
                noisy_action_k[:, prior_start:prior_end],
                noisy_action_v[:, prior_start:prior_end],
            )
            keys = [
                clean_image_k[:, :clean_end],
                noisy_image_k[:, noisy_start:noisy_end],
            ]
            values = [
                clean_image_v[:, :clean_end],
                noisy_image_v[:, noisy_start:noisy_end],
            ]
            if prior_key is not None:
                keys.append(prior_key)
                values.append(prior_value)
            keys.extend(
                [
                    noisy_action_k[:, prior_end:flow_end],
                    noisy_state_k[:, state_start:state_end],
                ]
            )
            values.extend(
                [
                    noisy_action_v[:, prior_end:flow_end],
                    noisy_state_v[:, state_start:state_end],
                ]
            )
            output[:, noisy_start:noisy_end] = self.attn(
                noisy_image_q[:, noisy_start:noisy_end],
                torch.cat(keys, dim=1),
                torch.cat(values, dim=1),
            )
        return output

    def _process_noisy_action_blocks(
        self,
        noisy_action_q,
        noisy_action_k,
        noisy_action_v,
        clean_image_k,
        clean_image_v,
        noisy_image_k,
        noisy_image_v,
        noisy_state_k,
        noisy_state_v,
        half_frames,
        action_horizon,
        state_horizon,
    ):
        del action_horizon, state_horizon
        num_blocks = (half_frames - 1) // self.num_frame_per_block
        output = torch.empty_like(noisy_action_q)
        block_size = self.frame_seqlen * self.num_frame_per_block
        for block_index in range(num_blocks):
            action_start = block_index * self.num_action_per_block
            prior_start = action_start
            prior_end = prior_start + self.num_base_prior_tokens
            flow_end = action_start + self.num_action_per_block
            clean_end = self.frame_seqlen + block_index * block_size
            noisy_start = self.frame_seqlen + block_index * block_size
            noisy_end = noisy_start + block_size
            state_start = block_index * self.num_state_per_block
            state_end = state_start + self.num_state_per_block

            # The Prior sees only legal clean context, current state and Prior peers.
            prior_keys = torch.cat(
                [
                    clean_image_k[:, :clean_end],
                    noisy_action_k[:, prior_start:prior_end],
                    noisy_state_k[:, state_start:state_end],
                ],
                dim=1,
            )
            prior_values = torch.cat(
                [
                    clean_image_v[:, :clean_end],
                    noisy_action_v[:, prior_start:prior_end],
                    noisy_state_v[:, state_start:state_end],
                ],
                dim=1,
            )
            output[:, prior_start:prior_end] = self.attn(
                noisy_action_q[:, prior_start:prior_end],
                prior_keys,
                prior_values,
            )

            prior_key, prior_value = self._downstream_context(
                noisy_action_k[:, prior_start:prior_end],
                noisy_action_v[:, prior_start:prior_end],
            )
            flow_keys = [
                clean_image_k[:, :clean_end],
                noisy_image_k[:, noisy_start:noisy_end],
            ]
            flow_values = [
                clean_image_v[:, :clean_end],
                noisy_image_v[:, noisy_start:noisy_end],
            ]
            if prior_key is not None:
                flow_keys.append(prior_key)
                flow_values.append(prior_value)
            flow_keys.extend(
                [
                    noisy_action_k[:, prior_end:flow_end],
                    noisy_state_k[:, state_start:state_end],
                ]
            )
            flow_values.extend(
                [
                    noisy_action_v[:, prior_end:flow_end],
                    noisy_state_v[:, state_start:state_end],
                ]
            )
            output[:, prior_end:flow_end] = self.attn(
                noisy_action_q[:, prior_end:flow_end],
                torch.cat(flow_keys, dim=1),
                torch.cat(flow_values, dim=1),
            )
        return output

    def _cached_multimodal_attention(
        self,
        video_query,
        video_key,
        video_value,
        cached_key,
        cached_value,
        action_query,
        action_key,
        action_value,
    ):
        prior_count = self.num_base_prior_tokens
        state_count = self.num_state_per_block
        prior_query = action_query[:, :prior_count]
        flow_query = action_query[:, prior_count:-state_count]
        state_query = action_query[:, -state_count:]
        prior_key_raw = action_key[:, :prior_count]
        prior_value_raw = action_value[:, :prior_count]
        flow_key = action_key[:, prior_count:-state_count]
        flow_value = action_value[:, prior_count:-state_count]
        state_key = action_key[:, -state_count:]
        state_value = action_value[:, -state_count:]

        prior_output = self.attn(
            prior_query,
            torch.cat([cached_key, prior_key_raw, state_key], dim=1),
            torch.cat([cached_value, prior_value_raw, state_value], dim=1),
        )
        prior_key, prior_value = self._downstream_context(
            prior_key_raw, prior_value_raw
        )
        downstream_keys = [cached_key, video_key]
        downstream_values = [cached_value, video_value]
        if prior_key is not None:
            downstream_keys.append(prior_key)
            downstream_values.append(prior_value)
        downstream_keys.extend([flow_key, state_key])
        downstream_values.extend([flow_value, state_value])
        combined_key = torch.cat(downstream_keys, dim=1)
        combined_value = torch.cat(downstream_values, dim=1)
        video_output = self.attn(video_query, combined_key, combined_value)
        flow_output = self.attn(flow_query, combined_key, combined_value)
        state_output = self.attn(state_query, state_key, state_value)
        return torch.cat(
            [video_output, prior_output, flow_output, state_output], dim=1
        )


class CleanPriorDualPlanActionEncoder(DualPlanActionEncoder):
    def __init__(self, *args, prior_flow_indices: Sequence[int], **kwargs):
        super().__init__(*args, **kwargs)
        hidden_size = self.type_embedding.shape[-1]
        self.prior_horizon = len(prior_flow_indices)
        self.register_buffer(
            "prior_flow_indices",
            torch.as_tensor(prior_flow_indices, dtype=torch.long),
            persistent=False,
        )
        self.prior_query = nn.Parameter(
            torch.empty(self.prior_horizon, hidden_size)
        )
        self.prior_type_embedding = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.prior_query, std=0.02)
        nn.init.normal_(self.prior_type_embedding, std=0.02)

    def forward(self, packed_action, timesteps, category_ids):
        flow_tokens = super().forward(packed_action, timesteps, category_ids)
        seconds = self.offset_seconds.to(device=packed_action.device).index_select(
            0, self.prior_flow_indices
        )
        seconds = seconds.unsqueeze(0).expand(packed_action.shape[0], -1)
        offset = self.offset_embedding(seconds, flow_tokens.dtype)
        prior = self.prior_query.to(flow_tokens.dtype).unsqueeze(0) + offset
        prior = prior + self.prior_type_embedding.to(flow_tokens.dtype)
        return torch.cat(
            [prior.expand(packed_action.shape[0], -1, -1), flow_tokens],
            dim=1,
        )


class CleanPriorDualPlanActionDecoder(DualPlanActionDecoder):
    eef_prior_dim = 9

    def __init__(self, *args, prior_flow_indices: Sequence[int], **kwargs):
        super().__init__(*args, **kwargs)
        if (
            2 * self.base_action_dim + self.eef_prior_dim
            > self.manipulator_action_dim
        ):
            raise ValueError(
                "Packed flow output does not have enough unused Base channels "
                "for Base and EEF Prior predictions"
            )
        self.prior_horizon = len(prior_flow_indices)
        self.register_buffer(
            "prior_flow_indices",
            torch.as_tensor(prior_flow_indices, dtype=torch.long),
            persistent=False,
        )
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
        prior_hidden = hidden[:, : self.prior_horizon]
        flow_hidden = hidden[:, self.prior_horizon :]
        flow = super().forward(flow_hidden, category_ids)
        base_prior = self.base_prior_head(prior_hidden, category_ids)
        eef_prior = self.eef_prior_head(prior_hidden, category_ids)
        base = flow[:, : self.plan_horizon].clone()
        base[
            :,
            self.prior_flow_indices,
            self.base_action_dim : 2 * self.base_action_dim,
        ] = base_prior
        base[
            :,
            self.prior_flow_indices,
            2 * self.base_action_dim : (
                2 * self.base_action_dim + self.eef_prior_dim
            ),
        ] = eef_prior
        return torch.cat([base, flow[:, self.plan_horizon :]], dim=1)


class WanVideoDiTDualPlanPrior(CausalWanModel):
    """Wan DiT with configurable clean Prior and 12 noisy plan registers."""

    def __init__(
        self,
        plan_horizon: int = 6,
        base_action_dim: int = 4,
        manipulator_action_dim: int = 21,
        plan_time_offsets: Sequence[int] = (1, 4, 8, 12, 16, 24),
        prior: MobilePlanPriorConfig | Mapping[str, Any] | None = None,
        prior_time_offsets: Sequence[int] | None = None,
        control_fps: float = 30.0,
        prior_condition_mode: str = "normal",
        **kwargs,
    ):
        if len(plan_time_offsets) != plan_horizon:
            raise ValueError("plan_time_offsets length must equal plan_horizon")
        if prior_condition_mode not in PRIOR_CONDITION_MODES:
            raise ValueError(f"Unknown prior_condition_mode: {prior_condition_mode}")
        prior_config = coerce_mobile_plan_prior_config(
            prior, legacy_time_offsets=prior_time_offsets
        )
        prior_flow_indices = resolve_prior_flow_indices(
            plan_time_offsets, prior_config.time_offsets
        )
        prior_horizon = len(prior_flow_indices)
        kwargs["action_dim"] = manipulator_action_dim
        kwargs["num_action_per_block"] = 2 * plan_horizon + prior_horizon
        super().__init__(**kwargs)
        self.plan_horizon = plan_horizon
        self.prior_horizon = prior_horizon
        self.base_action_dim = base_action_dim
        self.manipulator_action_dim = manipulator_action_dim
        self.prior_condition_mode = prior_condition_mode
        self.prior_config = prior_config
        self.register_buffer(
            "expected_plan_time_offsets",
            torch.as_tensor(plan_time_offsets, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "expected_prior_time_offsets",
            torch.as_tensor(prior_config.time_offsets, dtype=torch.long),
            persistent=True,
        )
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
                num_action_per_block=2 * plan_horizon + prior_horizon,
                num_state_per_block=previous.num_state_per_block,
                num_base_prior_tokens=prior_horizon,
            )
            replacement.load_state_dict(previous.state_dict())
            block.self_attn = replacement
        self.action_encoder = CleanPriorDualPlanActionEncoder(
            base_action_dim=base_action_dim,
            manipulator_action_dim=manipulator_action_dim,
            hidden_size=self.dim,
            num_embodiments=1,
            plan_time_offsets=plan_time_offsets,
            prior_flow_indices=prior_flow_indices,
            control_fps=control_fps,
        )
        self.action_decoder = CleanPriorDualPlanActionDecoder(
            base_action_dim=base_action_dim,
            manipulator_action_dim=manipulator_action_dim,
            hidden_size=self.hidden_size,
            model_dim=self.dim,
            num_embodiments=1,
            plan_horizon=plan_horizon,
            prior_flow_indices=prior_flow_indices,
        )

    def _action_register_timesteps(
        self, timestep_action, action_features, state_features
    ):
        expected_flow_tokens = 2 * self.plan_horizon
        if timestep_action.shape[1] != expected_flow_tokens:
            raise ValueError(
                f"Expected {expected_flow_tokens} flow timesteps, got "
                f"{timestep_action.shape[1]}"
            )
        expected_registers = self.prior_horizon + expected_flow_tokens
        if action_features.shape[1] != expected_registers:
            raise ValueError(
                "Prior action encoder must return "
                f"{expected_registers} registers"
            )
        prior_timestep = torch.zeros_like(timestep_action[:, : self.prior_horizon])
        stride = timestep_action.shape[1] // state_features.shape[1]
        return (
            torch.cat([prior_timestep, timestep_action], dim=1),
            timestep_action[:, ::stride],
        )

    def forward(
        self,
        *args,
        plan_time_offsets=None,
        prior_time_offsets=None,
        prior_condition_mode=None,
        **kwargs,
    ):
        if plan_time_offsets is None:
            raise ValueError(
                "Prior dual-plan DiT requires explicit plan_time_offsets"
            )
        offsets = torch.as_tensor(
            plan_time_offsets,
            device=self.expected_plan_time_offsets.device,
            dtype=torch.long,
        )
        if offsets.ndim == 1:
            offsets = offsets.unsqueeze(0)
        expected = self.expected_plan_time_offsets.unsqueeze(0).expand_as(offsets)
        if not torch.equal(offsets, expected):
            raise ValueError(
                "Expected plan offsets "
                f"{self.expected_plan_time_offsets.tolist()}, got "
                f"{offsets.tolist()}"
            )
        if prior_time_offsets is None:
            raise ValueError(
                "Prior dual-plan DiT requires explicit prior_time_offsets"
            )
        prior_offsets = torch.as_tensor(
            prior_time_offsets,
            device=self.expected_prior_time_offsets.device,
            dtype=torch.long,
        )
        if prior_offsets.ndim == 1:
            prior_offsets = prior_offsets.unsqueeze(0)
        expected_prior = self.expected_prior_time_offsets.unsqueeze(0).expand_as(
            prior_offsets
        )
        if not torch.equal(prior_offsets, expected_prior):
            raise ValueError(
                "Expected Prior offsets "
                f"{self.expected_prior_time_offsets.tolist()}, got "
                f"{prior_offsets.tolist()}"
            )
        mode = prior_condition_mode or self.prior_condition_mode
        if mode not in PRIOR_CONDITION_MODES:
            raise ValueError(f"Unknown prior_condition_mode: {mode}")
        for block in self.blocks:
            block.self_attn.prior_condition_mode = mode
        return super().forward(*args, **kwargs)
