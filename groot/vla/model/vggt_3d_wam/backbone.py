"""A local alternating-attention VGGT-style image feature aggregator."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

LOGGER = logging.getLogger(__name__)


def _sinusoidal_1d(
    length: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    half = max(1, dim // 2)
    denominator = max(1, half - 1)
    frequency = torch.exp(
        -math.log(10000)
        * torch.arange(half, device=device, dtype=torch.float32)
        / denominator
    )
    position = torch.arange(length, device=device, dtype=torch.float32)
    encoding = torch.cat(
        ((position[:, None] * frequency).sin(), (position[:, None] * frequency).cos()),
        dim=-1,
    )
    return F.pad(encoding, (0, max(0, dim - encoding.shape[-1])))[:, :dim].to(dtype)


def _sinusoidal_2d(
    height: int,
    width: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y_dim = dim // 2
    x_dim = dim - y_dim
    y = _sinusoidal_1d(height, y_dim, device=device, dtype=dtype)
    x = _sinusoidal_1d(width, x_dim, device=device, dtype=dtype)
    return torch.cat(
        (
            y[:, None].expand(-1, width, -1),
            x[None].expand(height, -1, -1),
        ),
        dim=-1,
    ).reshape(height * width, dim)


class LoRALinear(nn.Module):
    """Low-rank residual update around a frozen linear projection."""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.down = nn.Linear(linear.in_features, rank, bias=False)
        self.up = nn.Linear(rank, linear.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs) + self.up(self.dropout(self.down(inputs))) * self.scale


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, length, dim = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        output = F.scaled_dot_product_attention(query, key, value)
        output = output.transpose(1, 2).reshape(batch, length, dim)
        return self.proj(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(inputs), approximate="tanh"))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_ratio)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attn(self.norm1(inputs))
        return inputs + self.mlp(self.norm2(inputs))


class ConvPatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(images)


def _replace_with_lora(
    module: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in {"qkv", "proj", "fc1", "fc2"}:
            setattr(module, name, LoRALinear(child, rank, alpha, dropout))
        else:
            _replace_with_lora(child, rank, alpha, dropout)


def _extract_state_dict(value: object) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict):
        raise TypeError(f"Checkpoint must contain a state dict, got {type(value)!r}")
    for key in ("state_dict", "model", "module"):
        nested = value.get(key)
        if isinstance(nested, dict) and nested:
            value = nested
            break
    result: dict[str, torch.Tensor] = {}
    for raw_key, tensor in value.items():
        if not torch.is_tensor(tensor):
            continue
        key = str(raw_key)
        for prefix in ("module.", "model.", "vggt_model.", "backbone.", "aggregator."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        result[key] = tensor
    return result


def _adapt_checkpoint_state(
    state: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map official VGGT/DINO names and positional layout to local modules."""
    adapted: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        key = raw_key
        if key == "patch_embed.register_tokens":
            key = "patch_embed.reg_token"
        if (
            key == "patch_embed.pos_embed"
            and key in expected
            and value.ndim == 3
            and value.shape[1] == expected[key].shape[1] + 1
        ):
            # Official DINO stores the class position followed by the 37x37
            # patch grid. timm's no_embed_class layout stores only the grid.
            value = value[:, 1:]
        adapted[key] = value
    return adapted


class VGGTBackbone(nn.Module):
    """Alternate per-frame and global attention over patch tokens."""

    def __init__(
        self,
        patch_size: int,
        patch_embed_type: str,
        pretrain_image_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        checkpoint_path: str | None,
        init_random: bool,
        min_checkpoint_match_fraction: float,
        freeze: bool,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        global_temporal_window: int,
        freeze_dino: bool = True,
        dino_image_chunk_size: int = 4,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed_type = patch_embed_type
        self.freeze_dino = bool(
            freeze_dino and patch_embed_type == "dinov2_vitl14_reg"
        )
        self.dino_image_chunk_size = int(dino_image_chunk_size)
        if self.dino_image_chunk_size < 1:
            raise ValueError("dino_image_chunk_size must be positive")
        self.global_temporal_window = int(global_temporal_window)
        if self.global_temporal_window < 1:
            raise ValueError("global_temporal_window must be positive")
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if patch_embed_type == "conv":
            self.patch_embed = ConvPatchEmbed(patch_size, embed_dim)
        elif patch_embed_type == "dinov2_vitl14_reg":
            if patch_size != 14 or embed_dim != 1024:
                raise ValueError(
                    "dinov2_vitl14_reg requires patch_size=14 and embed_dim=1024"
                )
            import timm

            self.patch_embed = timm.create_model(
                "vit_large_patch14_reg4_dinov2",
                pretrained=False,
                img_size=pretrain_image_size,
                dynamic_img_size=True,
                num_classes=0,
            )
        else:
            raise ValueError(f"Unknown patch_embed_type: {patch_embed_type}")
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.empty(1, 2, 4, embed_dim))
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)
        self.patch_start_idx = 5
        self.frame_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.global_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.checkpoint_load_report: dict[str, object] | None = None

        if checkpoint_path:
            self.load_vggt_checkpoint(checkpoint_path, min_checkpoint_match_fraction)
        elif not init_random:
            raise ValueError(
                "vggt_checkpoint_path is required unless init_random=true is set explicitly"
            )

        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad = False
        elif self.freeze_dino:
            for parameter in self.patch_embed.parameters():
                parameter.requires_grad = False
        if self.freeze_dino:
            self.patch_embed.eval()
        if lora_rank > 0:
            # DINOv2 remains a completely frozen 2D feature extractor.
            # Adapt only VGGT's cross-view/frame aggregation blocks.
            _replace_with_lora(
                self.frame_blocks,
                lora_rank,
                lora_alpha,
                lora_dropout,
            )
            _replace_with_lora(
                self.global_blocks,
                lora_rank,
                lora_alpha,
                lora_dropout,
            )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_dino:
            self.patch_embed.eval()
        return self

    def load_vggt_checkpoint(
        self, checkpoint_path: str, min_match_fraction: float
    ) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"VGGT checkpoint does not exist: {path}")
        # mmap keeps unused VGGT heads and patch-encoder tensors off the host
        # heap. Only compatible aggregator pages are materialized during copy.
        state = _extract_state_dict(
            torch.load(
                path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        )
        own_state = self.state_dict()
        state = _adapt_checkpoint_state(state, own_state)
        compatible = {
            key: value
            for key, value in state.items()
            if key in own_state and own_state[key].shape == value.shape
        }
        fraction = len(compatible) / max(1, len(own_state))
        if fraction < min_match_fraction:
            raise ValueError(
                f"Only {len(compatible)}/{len(own_state)} backbone tensors match "
                f"{path}; required fraction is {min_match_fraction:.3f}. "
                "Use an adapted checkpoint or set init_random=true for a scratch run."
            )
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        self.checkpoint_load_report = {
            "path": str(path),
            "checkpoint_tensors": len(state),
            "model_tensors": len(own_state),
            "matched_tensors": len(compatible),
            "matched_tensor_fraction": fraction,
            "matched_numel": sum(value.numel() for value in compatible.values()),
            "model_numel": sum(value.numel() for value in own_state.values()),
            "missing_tensors": len(missing),
        }
        LOGGER.info(
            "Loaded %d VGGT backbone tensors from %s (%d missing, %d ignored)",
            len(compatible),
            path,
            len(missing),
            len(state) - len(compatible) + len(unexpected),
        )
        print(
            "VGGT checkpoint load: "
            f"{len(compatible)}/{len(own_state)} model tensors matched "
            f"from {path}"
        )

    def patch_aligned_image_size(
        self,
        image_size: tuple[int, int],
    ) -> tuple[int, int]:
        """Pad only the bottom/right edge to the patch-size lattice."""
        height, width = image_size
        aligned_height = math.ceil(height / self.patch_size) * self.patch_size
        aligned_width = math.ceil(width / self.patch_size) * self.patch_size
        return aligned_height, aligned_width

    def _extract_patch_tokens(
        self,
        images: torch.Tensor,
        patch_h: int,
        patch_w: int,
    ) -> torch.Tensor:
        if self.patch_embed_type == "conv":
            feature_map = self.patch_embed(images)
            return feature_map.flatten(2).transpose(1, 2)
        patch_tokens = self.patch_embed.forward_features(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        else:
            patch_tokens = patch_tokens[:, self.patch_embed.num_prefix_tokens :]
        if patch_tokens.shape[1] != patch_h * patch_w:
            raise ValueError(
                "DINO patch token count does not match the input grid: "
                f"{patch_tokens.shape[1]} versus {patch_h}x{patch_w}"
            )
        return patch_tokens

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Return patch features as ``[B, T, V, C, Hp, Wp]``."""
        batch, time, views, channels, height, width = video.shape
        if channels != 3:
            raise ValueError(f"Expected RGB video, got {video.shape}")
        images = video.reshape(batch * time * views, channels, height, width)
        aligned_height, aligned_width = self.patch_aligned_image_size((height, width))
        if (aligned_height, aligned_width) != (height, width):
            images = F.pad(
                images,
                (0, aligned_width - width, 0, aligned_height - height),
                value=0,
            )
            height, width = aligned_height, aligned_width
        images = (images - self.image_mean) / self.image_std
        patch_h, patch_w = height // self.patch_size, width // self.patch_size
        if self.freeze_dino:
            patch_chunks = []
            with torch.no_grad():
                for start in range(0, len(images), self.dino_image_chunk_size):
                    patch_chunks.append(
                        self._extract_patch_tokens(
                            images[start : start + self.dino_image_chunk_size],
                            patch_h,
                            patch_w,
                        )
                    )
            patch_tokens = torch.cat(patch_chunks, dim=0).detach()
        else:
            patch_tokens = self._extract_patch_tokens(
                images,
                patch_h,
                patch_w,
            )
        sequence = time * views
        patches = patch_h * patch_w
        tokens = patch_tokens.reshape(
            batch, time, views, patches, self.embed_dim
        )
        spatial_position = _sinusoidal_2d(
            patch_h,
            patch_w,
            self.embed_dim,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        time_position = _sinusoidal_1d(
            time,
            self.embed_dim,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        view_position = _sinusoidal_1d(
            views,
            self.embed_dim,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        patch_tokens = (
            tokens
            + spatial_position[None, None, None]
            + time_position[None, :, None, None]
            + view_position[None, None, :, None]
        )
        special_index = torch.ones(sequence, dtype=torch.long, device=tokens.device)
        special_index[0] = 0
        camera_token = self.camera_token[:, special_index].expand(batch, -1, -1, -1)
        register_token = self.register_token[:, special_index].expand(
            batch, -1, -1, -1
        )
        special_tokens = torch.cat((camera_token, register_token), dim=2).reshape(
            batch, time, views, self.patch_start_idx, self.embed_dim
        )
        tokens = torch.cat((special_tokens, patch_tokens), dim=3).reshape(
            batch * sequence, patches + self.patch_start_idx, self.embed_dim
        )
        tokens_per_frame = patches + self.patch_start_idx
        for frame_block, global_block in zip(self.frame_blocks, self.global_blocks):
            if self.training and self.gradient_checkpointing:
                tokens = checkpoint(
                    frame_block,
                    tokens,
                    use_reentrant=False,
                )
            else:
                tokens = frame_block(tokens)
            tokens_by_time = tokens.reshape(
                batch,
                time,
                views,
                tokens_per_frame,
                self.embed_dim,
            )
            global_windows = []
            for start in range(0, time, self.global_temporal_window):
                window = tokens_by_time[
                    :, start : start + self.global_temporal_window
                ]
                window_time = window.shape[1]
                window = window.reshape(
                    batch,
                    window_time * views * tokens_per_frame,
                    self.embed_dim,
                )
                if self.training and self.gradient_checkpointing:
                    window = checkpoint(
                        global_block,
                        window,
                        use_reentrant=False,
                    )
                else:
                    window = global_block(window)
                global_windows.append(
                    window.reshape(
                        batch,
                        window_time,
                        views,
                        tokens_per_frame,
                        self.embed_dim,
                    )
                )
            tokens = torch.cat(global_windows, dim=1).reshape(
                batch * sequence,
                tokens_per_frame,
                self.embed_dim,
            )
        tokens = tokens[:, self.patch_start_idx :]
        return (
            tokens.transpose(1, 2)
            .reshape(batch, time, views, self.embed_dim, patch_h, patch_w)
            .contiguous()
        )
