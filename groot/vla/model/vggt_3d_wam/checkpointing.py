"""Safe partial initialization for changed VGGT tokenizer architectures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


def resolve_matching_checkpoint(
    checkpoint: str | Path | None,
) -> Path | None:
    """Resolve a non-empty safetensors checkpoint, otherwise return ``None``."""
    if checkpoint is None:
        return None
    checkpoint_text = str(checkpoint).strip()
    if not checkpoint_text:
        return None
    checkpoint_path = Path(checkpoint_text).expanduser()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "model.safetensors"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Matching-only initialization checkpoint does not exist: "
            f"{checkpoint_path}"
        )
    if checkpoint_path.suffix != ".safetensors":
        raise ValueError(
            "Matching-only initialization currently requires a "
            f"model.safetensors file, got: {checkpoint_path}"
        )
    return checkpoint_path


def load_matching_trainable_parameters(
    model: torch.nn.Module,
    checkpoint: str | Path | None,
) -> dict[str, Any] | None:
    """Load only name/shape-compatible trainable parameters.

    Frozen VGGT/DINO weights are deliberately left at their normal ReconDrive
    initialization. Loading tensors one by one avoids materializing the full
    multi-gigabyte checkpoint in host memory.
    """
    checkpoint_path = resolve_matching_checkpoint(checkpoint)
    if checkpoint_path is None:
        return None

    trainable_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    matched_names: list[str] = []
    mismatched_shapes: dict[str, dict[str, list[int]]] = {}
    missing_names: list[str] = []
    matched_numel = 0
    total_numel = sum(
        parameter.numel() for parameter in trainable_parameters.values()
    )

    with safe_open(
        checkpoint_path,
        framework="pt",
        device="cpu",
    ) as source:
        source_names = set(source.keys())
        with torch.no_grad():
            for name, parameter in trainable_parameters.items():
                if name not in source_names:
                    missing_names.append(name)
                    continue
                source_shape = tuple(source.get_slice(name).get_shape())
                target_shape = tuple(parameter.shape)
                if source_shape != target_shape:
                    mismatched_shapes[name] = {
                        "checkpoint": list(source_shape),
                        "model": list(target_shape),
                    }
                    continue
                source_tensor = source.get_tensor(name)
                parameter.copy_(
                    source_tensor.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
                matched_names.append(name)
                matched_numel += parameter.numel()

    if not matched_names:
        raise RuntimeError(
            "Matching-only initialization found no compatible trainable "
            f"parameters in {checkpoint_path}"
        )
    return {
        "checkpoint": str(checkpoint_path),
        "matched_tensors": len(matched_names),
        "mismatched_tensors": len(mismatched_shapes),
        "missing_tensors": len(missing_names),
        "matched_numel": matched_numel,
        "total_trainable_numel": total_numel,
        "matched_trainable_fraction": (
            matched_numel / total_numel if total_numel else 0.0
        ),
        "mismatched_shapes": mismatched_shapes,
    }
