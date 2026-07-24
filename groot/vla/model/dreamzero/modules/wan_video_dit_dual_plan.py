"""Dual Base/Manipulator plan-token adapter for the existing Wan video DiT."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from groot.vla.model.n1_5.modules.action_encoder import (
    SinusoidalPositionalEncoding,
)

from .wan_video_dit_action_casual_chunk import (
    CategorySpecificMLP,
    CausalWanModel,
    MultiEmbodimentActionEncoder,
)


class PlanOffsetEmbedding(nn.Module):
    """Embed physical future seconds rather than the ordinal token index."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.positional = SinusoidalPositionalEncoding(hidden_size)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, seconds: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.projection(self.positional(seconds).to(dtype=dtype))


class DualPlanActionEncoder(nn.Module):
    """Use independent projections and type embeddings for the two token streams."""

    def __init__(
        self,
        base_action_dim: int,
        manipulator_action_dim: int,
        hidden_size: int,
        num_embodiments: int,
        plan_time_offsets: Sequence[int],
        control_fps: float,
    ):
        super().__init__()
        self.plan_horizon = len(plan_time_offsets)
        self.base_action_dim = base_action_dim
        self.manipulator_action_dim = manipulator_action_dim
        self.base_encoder = MultiEmbodimentActionEncoder(
            action_dim=base_action_dim,
            hidden_size=hidden_size,
            num_embodiments=num_embodiments,
        )
        self.manipulator_encoder = MultiEmbodimentActionEncoder(
            action_dim=manipulator_action_dim,
            hidden_size=hidden_size,
            num_embodiments=num_embodiments,
        )
        self.type_embedding = nn.Parameter(torch.empty(2, hidden_size))
        nn.init.normal_(self.type_embedding, std=0.02)
        self.offset_embedding = PlanOffsetEmbedding(hidden_size)
        self.register_buffer(
            "offset_seconds",
            torch.as_tensor(plan_time_offsets, dtype=torch.float32) / control_fps,
            persistent=True,
        )

    def forward(
        self,
        packed_action: torch.Tensor,
        timesteps: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> torch.Tensor:
        expected_tokens = 2 * self.plan_horizon
        if packed_action.shape[1:] != (
            expected_tokens,
            self.manipulator_action_dim,
        ):
            raise ValueError(
                "Packed dual plan must be "
                f"[B,{expected_tokens},{self.manipulator_action_dim}], got "
                f"{tuple(packed_action.shape)}"
            )
        base = packed_action[:, : self.plan_horizon, : self.base_action_dim]
        manipulator = packed_action[:, self.plan_horizon :]
        base_timestep = timesteps[:, : self.plan_horizon]
        manipulator_timestep = timesteps[:, self.plan_horizon :]
        base_token = self.base_encoder(base, base_timestep, category_ids)
        manipulator_token = self.manipulator_encoder(
            manipulator, manipulator_timestep, category_ids
        )
        seconds = self.offset_seconds.to(device=packed_action.device)
        seconds = seconds.unsqueeze(0).expand(packed_action.shape[0], -1)
        offset = self.offset_embedding(seconds, base_token.dtype)
        base_token = base_token + offset + self.type_embedding[0].to(base_token.dtype)
        manipulator_token = (
            manipulator_token
            + offset
            + self.type_embedding[1].to(manipulator_token.dtype)
        )
        return torch.cat([base_token, manipulator_token], dim=1)


class DualPlanActionDecoder(nn.Module):
    """Decode each semantic branch with its own output projection."""

    def __init__(
        self,
        base_action_dim: int,
        manipulator_action_dim: int,
        hidden_size: int,
        model_dim: int,
        num_embodiments: int,
        plan_horizon: int,
    ):
        super().__init__()
        self.base_action_dim = base_action_dim
        self.manipulator_action_dim = manipulator_action_dim
        self.plan_horizon = plan_horizon
        self.base_decoder = CategorySpecificMLP(
            num_categories=num_embodiments,
            input_dim=model_dim,
            hidden_dim=hidden_size,
            output_dim=base_action_dim,
        )
        self.manipulator_decoder = CategorySpecificMLP(
            num_categories=num_embodiments,
            input_dim=model_dim,
            hidden_dim=hidden_size,
            output_dim=manipulator_action_dim,
        )

    def forward(
        self, hidden: torch.Tensor, category_ids: torch.Tensor
    ) -> torch.Tensor:
        base_hidden = hidden[:, : self.plan_horizon]
        manipulator_hidden = hidden[:, self.plan_horizon :]
        base = self.base_decoder(base_hidden, category_ids)
        manipulator = self.manipulator_decoder(
            manipulator_hidden, category_ids
        )
        packed_base = F.pad(
            base,
            (0, self.manipulator_action_dim - self.base_action_dim),
        )
        return torch.cat([packed_base, manipulator], dim=1)


class WanVideoDiTDualPlan(CausalWanModel):
    """CausalWanModel with 6 Base tokens followed by 6 Manipulator tokens."""

    def __init__(
        self,
        plan_horizon: int = 6,
        base_action_dim: int = 4,
        manipulator_action_dim: int = 21,
        plan_time_offsets: Sequence[int] = (1, 4, 8, 12, 16, 24),
        control_fps: float = 30.0,
        **kwargs,
    ):
        if len(plan_time_offsets) != plan_horizon:
            raise ValueError("plan_time_offsets length must equal plan_horizon")
        kwargs["action_dim"] = manipulator_action_dim
        kwargs["num_action_per_block"] = 2 * plan_horizon
        super().__init__(**kwargs)
        self.plan_horizon = plan_horizon
        self.base_action_dim = base_action_dim
        self.manipulator_action_dim = manipulator_action_dim
        self.register_buffer(
            "expected_plan_time_offsets",
            torch.as_tensor(plan_time_offsets, dtype=torch.long),
            persistent=True,
        )
        num_embodiments = 1
        self.action_encoder = DualPlanActionEncoder(
            base_action_dim=base_action_dim,
            manipulator_action_dim=manipulator_action_dim,
            hidden_size=self.dim,
            num_embodiments=num_embodiments,
            plan_time_offsets=plan_time_offsets,
            control_fps=control_fps,
        )
        self.action_decoder = DualPlanActionDecoder(
            base_action_dim=base_action_dim,
            manipulator_action_dim=manipulator_action_dim,
            hidden_size=self.hidden_size,
            model_dim=self.dim,
            num_embodiments=num_embodiments,
            plan_horizon=plan_horizon,
        )

    def forward(self, *args, plan_time_offsets=None, **kwargs):
        if plan_time_offsets is None:
            raise ValueError("Dual plan DiT requires explicit plan_time_offsets")
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
                f"Expected plan offsets {self.expected_plan_time_offsets.tolist()}, "
                f"got {offsets.tolist()}"
            )
        return super().forward(*args, **kwargs)

