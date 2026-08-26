"""Helpers for making parameter-efficient checkpoints self-contained."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch


CHECKPOINT_MANIFEST_NAME = "dreamzero_checkpoint_manifest.json"
CHECKPOINT_FORMAT_VERSION = 2


def is_reconstructible_checkpoint_state_key(name: str) -> bool:
    """Return whether a persistent state entry is derived entirely from config."""
    return name == "offset_seconds" or name.endswith(".offset_seconds")


def collect_required_checkpoint_state_keys(
    model: torch.nn.Module,
) -> set[str]:
    """Collect module-declared state that cannot be rebuilt from base weights."""
    model_state_keys = set(model.state_dict())
    required: set[str] = set()
    for module_name, module in model.named_modules():
        provider = getattr(module, "checkpoint_required_state_keys", None)
        if provider is None:
            continue
        prefix = f"{module_name}." if module_name else ""
        for local_name in provider():
            full_name = prefix + str(local_name)
            if full_name not in model_state_keys:
                raise RuntimeError(
                    f"{type(module).__name__} declared missing checkpoint "
                    f"state key {full_name!r}"
                )
            required.add(full_name)
    return required


def select_parameter_efficient_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], set[str]]:
    """Keep trainable state plus frozen tensors absent from the base checkpoint."""
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    required = collect_required_checkpoint_state_keys(model)
    selected_names = trainable | required
    missing = selected_names - set(state_dict)
    if missing:
        preview = sorted(missing)[:20]
        raise RuntimeError(
            "Checkpoint state dict is missing trainable/required tensors; "
            f"first keys={preview}, total={len(missing)}"
        )
    return {
        name: value for name, value in state_dict.items()
        if name in selected_names
    }, required


def write_checkpoint_manifest(
    output_dir: str | Path,
    *,
    parameter_efficient: bool,
    required_state_keys: Iterable[str],
) -> None:
    """Record the exact non-base state contract for strict future loading."""
    path = Path(output_dir) / CHECKPOINT_MANIFEST_NAME
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "parameter_efficient": bool(parameter_efficient),
        "required_state_keys": sorted(set(required_state_keys)),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_checkpoint_required_keys(
    checkpoint_dir: str | Path,
    loaded_state_keys: Iterable[str],
    model: torch.nn.Module,
) -> tuple[set[str], bool]:
    """Validate new manifests; report legacy missing state without hiding it."""
    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / CHECKPOINT_MANIFEST_NAME
    loaded = set(loaded_state_keys)
    runtime_required = collect_required_checkpoint_state_keys(model)
    has_manifest = manifest_path.is_file()
    if has_manifest:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        version = int(manifest.get("format_version", 0))
        if version != CHECKPOINT_FORMAT_VERSION:
            raise RuntimeError(
                f"Unsupported DreamZero checkpoint format version {version} "
                f"in {manifest_path}"
            )
        manifest_required = set(manifest.get("required_state_keys", []))
        undeclared = runtime_required - manifest_required
        if undeclared:
            preview = sorted(undeclared)[:20]
            raise RuntimeError(
                "Checkpoint manifest does not cover current required state; "
                f"first keys={preview}, total={len(undeclared)}"
            )
        required = runtime_required | manifest_required
    else:
        required = runtime_required
    missing = required - loaded
    if missing and has_manifest:
        preview = sorted(missing)[:20]
        raise RuntimeError(
            "Self-contained checkpoint is missing required tensors; "
            f"first keys={preview}, total={len(missing)}"
        )
    return missing, has_manifest
