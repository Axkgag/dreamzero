"""Hydra and Hugging Face Trainer entry point for VGGT tokenizer training."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import warnings

import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import Subset
from transformers import Trainer, TrainerCallback, set_seed
from transformers.trainer_utils import SaveStrategy

from groot.vla.model.vggt_3d_wam.checkpointing import (
    load_matching_trainable_parameters,
)
from groot.vla.model.vggt_3d_wam.visualization import save_vggt_visualization


def allow_trusted_numpy_rng_state_types() -> None:
    """Allow PyTorch to read NumPy RNG state from our own Trainer checkpoint.

    Transformers restores ``rng_state*.pth`` with ``weights_only=True``.
    Checkpoints written by NumPy 1.x contain an ndarray reconstruction helper
    and parameterized dtype classes that PyTorch does not allow by default.
    Restrict the allowlist to the primitive NumPy types used by RNG state
    rather than falling back to unrestricted pickle loading.
    """
    from numpy.core.multiarray import _reconstruct

    torch.serialization.add_safe_globals(
        [
            _reconstruct,
            np.ndarray,
            np.dtype,
            type(np.dtype(np.uint32)),
            type(np.dtype(np.float64)),
        ]
    )


def get_last_complete_checkpoint(output_dir: str | Path) -> str | None:
    """Return the newest fully resumable checkpoint, ignoring partial saves."""
    output_path = Path(output_dir)
    candidates: list[tuple[int, Path]] = []
    for path in output_path.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if path.is_dir() and match:
            candidates.append((int(match.group(1)), path))
    model_files = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    required_files = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
    for _, path in sorted(candidates, reverse=True):
        has_model = any((path / filename).is_file() for filename in model_files)
        missing = [
            filename
            for filename in required_files
            if not (path / filename).is_file()
        ]
        if has_model and not missing:
            return str(path)
        details = []
        if not has_model:
            details.append("model weights")
        details.extend(missing)
        warnings.warn(
            f"Ignoring incomplete checkpoint {path}: missing "
            + ", ".join(details),
            stacklevel=2,
        )
    return None


class VGGTJSONLLossLoggerCallback(TrainerCallback):
    """Append scalar VGGT loss diagnostics to ``loss_log.jsonl``."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _scalar(value):
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            return value.detach().item()
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                return None
        if isinstance(value, (bool, int, float)):
            return value
        return None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return
        entry = {"step": int(state.global_step)}
        for key, value in logs.items():
            keep = (
                key in {"loss", "eval_loss", "learning_rate", "grad_norm", "epoch"}
                or key.endswith("_learning_rate")
                or key.endswith("_avg")
                or key.endswith("_ratio")
            )
            if not keep:
                continue
            scalar = self._scalar(value)
            if scalar is not None:
                entry[key] = scalar
        if len(entry) == 1:
            return
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


class VGGTTrainer(Trainer):
    """Trainer adapter for DreamZero's dictionary-in/dictionary-out convention."""

    def __init__(self, *args, **kwargs) -> None:
        self.visualization_config = kwargs.pop("visualization_config", {})
        self.backbone_learning_rate = float(
            kwargs.pop("backbone_learning_rate")
        )
        super().__init__(*args, **kwargs)
        self.loss_windows: dict[str, list[float]] = {}
        self._last_train_visualization_step = -1
        self._val_visualizations_saved = 0

    def create_optimizer(self):
        """Use a conservative LR for pretrained LoRA and a faster LR for heads."""
        if self.optimizer is not None:
            return self.optimizer

        model = self.model_wrapped
        decay_parameters = self.get_decay_parameter_names(model)
        groups: dict[tuple[str, bool], list[torch.nn.Parameter]] = {
            ("backbone", True): [],
            ("backbone", False): [],
            ("heads", True): [],
            ("heads", False): [],
        }
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            normalized_name = name.removeprefix("module.")
            new_tap_projection = normalized_name.startswith(
                (
                    "backbone.frame_tap_projections.",
                    "backbone.global_tap_projections.",
                )
            )
            family = (
                "backbone"
                if normalized_name.startswith("backbone.")
                and not new_tap_projection
                else "heads"
            )
            groups[(family, name in decay_parameters)].append(parameter)

        optimizer_groups = []
        for (family, use_decay), parameters in groups.items():
            if not parameters:
                continue
            optimizer_groups.append(
                {
                    "params": parameters,
                    "lr": (
                        self.backbone_learning_rate
                        if family == "backbone"
                        else self.args.learning_rate
                    ),
                    "weight_decay": (
                        self.args.weight_decay if use_decay else 0.0
                    ),
                    "group_name": family,
                }
            )

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args, model
        )
        unsupported = {
            key
            for key in ("params", "model", "optimizer_dict")
            if key in optimizer_kwargs
        }
        if unsupported:
            raise ValueError(
                "VGGT grouped learning rates do not support optimizer kwargs "
                f"{sorted(unsupported)}; use optim=adamw_torch."
            )
        self.optimizer = optimizer_cls(optimizer_groups, **optimizer_kwargs)
        return self.optimizer

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
        learning_rate=None,
    ):
        """Save a scheduled checkpoint before validation at the same step."""
        save_before_evaluate = (
            self.control.should_save
            and self.control.should_evaluate
            and self.args.save_strategy != SaveStrategy.BEST
        )
        if save_before_evaluate:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )
            self.control.should_save = False
        return super()._maybe_log_save_evaluate(
            tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
            learning_rate,
        )

    @staticmethod
    def _distributed_diagnostics(
        outputs: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        tracked = {
            key: value.detach().float().reshape(())
            for key, value in outputs.items()
            if key != "loss"
            and torch.is_tensor(value)
            and value.numel() == 1
            and key.endswith(("_loss", "_count", "_weight"))
        }
        if not tracked:
            return {}
        keys = sorted(tracked)
        values = torch.stack([tracked[key] for key in keys])
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                values, op=torch.distributed.ReduceOp.SUM
            )
            world_size = torch.distributed.get_world_size()
        return {
            key: float(
                values[index] / world_size
                if key.endswith("_loss") or key == "geometry_loss_weight"
                else values[index]
            )
            for index, key in enumerate(keys)
        }

    def _maybe_save_visualization(self, inputs, outputs, training: bool) -> None:
        config = self.visualization_config
        if not config.get("enabled", False) or not self.is_world_process_zero():
            return
        step = int(self.state.global_step)
        if training:
            interval = int(config.get("train_interval", 0))
            if (
                interval <= 0
                or step % interval
                or self._last_train_visualization_step == step
            ):
                return
            split = "train"
            sample_index = 0
        else:
            interval = int(config.get("val_interval", 0))
            maximum = int(config.get("val_max_samples", 0))
            if (
                interval <= 0
                or step % interval
                or self._val_visualizations_saved >= maximum
            ):
                return
            split = "val"
            sample_index = self._val_visualizations_saved

        try:
            save_vggt_visualization(
                inputs,
                outputs,
                self.args.output_dir,
                split=split,
                step=step,
                sample_index=sample_index,
                max_time_steps=int(config.get("max_time_steps", 4)),
                max_views=int(config.get("max_views", 2)),
                scatter_max_points=int(
                    config.get("scatter_max_points", 4096)
                ),
                confidence_threshold=float(
                    config.get("confidence_threshold", 0.01)
                ),
                log_to_wandb=bool(config.get("log_to_wandb", False)),
            )
            if training:
                self._last_train_visualization_step = step
            else:
                self._val_visualizations_saved += 1
        except Exception as error:
            message = f"VGGT {split} visualization failed at step {step}: {error}"
            if config.get("fail_on_error", False):
                raise RuntimeError(message) from error
            warnings.warn(message, stacklevel=2)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        outputs = model(inputs)
        for key, value in self._distributed_diagnostics(outputs).items():
            window = self.loss_windows.setdefault(key, [])
            window.append(value)
            if len(window) > 10:
                window.pop(0)
        self._maybe_save_visualization(inputs, outputs, training=model.training)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """Validate the positional batch-dict interface without gathering logits."""
        del prediction_loss_only, ignore_keys
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(inputs)
        self._maybe_save_visualization(inputs, outputs, training=False)
        return outputs["loss"].mean().detach(), None, None

    def training_step(self, model, inputs, *args, **kwargs):
        unwrapped = self.accelerator.unwrap_model(model)
        unwrapped.global_step = self.state.global_step
        return super().training_step(model, inputs, *args, **kwargs)

    def log(self, logs, *args, **kwargs):
        if self.optimizer is not None:
            family_lrs: dict[str, float] = {}
            for group in self.optimizer.param_groups:
                family = group.get("group_name")
                if family is not None:
                    family_lrs[str(family)] = float(group["lr"])
            for family, learning_rate in family_lrs.items():
                logs[f"{family}_learning_rate"] = learning_rate
        if self.state.global_step % 10 == 0:
            for key, values in self.loss_windows.items():
                if values:
                    logs[f"{key}_avg"] = sum(values) / len(values)
            ratio_pairs = {
                "pointmap_inside_grid_ratio": (
                    "pointmap_inside_grid_count",
                    "pointmap_raw_valid_count",
                ),
                "pointmap_inside_grid_weight_ratio": (
                    "pointmap_inside_grid_weight",
                    "pointmap_raw_valid_weight",
                ),
                "ray_valid_ratio": (
                    "ray_valid_sample_count",
                    "ray_total_sample_count",
                ),
                "ray_supervised_pixel_ratio": (
                    "ray_supervised_pixel_count",
                    "ray_inside_grid_count",
                ),
                "free_space_sample_ratio": (
                    "free_space_sample_count",
                    "ray_valid_sample_count",
                ),
                "surface_sample_ratio": (
                    "surface_sample_count",
                    "ray_valid_sample_count",
                ),
                "multiview_correspondence_ratio": (
                    "multiview_correspondence_count",
                    "multiview_candidate_count",
                ),
            }
            for ratio_name, (numerator_key, denominator_key) in ratio_pairs.items():
                numerator = self.loss_windows.get(numerator_key, [])
                denominator = self.loss_windows.get(denominator_key, [])
                denominator_sum = sum(denominator)
                if numerator and denominator_sum > 0:
                    logs[ratio_name] = sum(numerator) / denominator_sum
        return super().log(logs, *args, **kwargs)

    def evaluate(self, *args, **kwargs):
        self._val_visualizations_saved = 0
        return super().evaluate(*args, **kwargs)


@hydra.main(config_path="../configs", config_name="vggt_3d_wam", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = get_last_complete_checkpoint(output_dir)
    cfg.training_args.run_name = output_dir.name
    os.environ.setdefault("WANDB_PROJECT", str(cfg.wandb_project))
    os.environ.setdefault("WANDB_DIR", str(output_dir))
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)
    with (output_dir / "geometry_quality.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "quality": "lossy_h264_pseudo_robot_centric_pointmap",
                "calibration_requirement": "verified calibration required for metric claims",
            },
            handle,
            indent=2,
        )

    set_seed(int(cfg.seed))
    model = instantiate(cfg.model)
    init_checkpoint = cfg.get("init_checkpoint")
    if checkpoint is not None and init_checkpoint:
        print(
            "VGGT matching-only initialization skipped because automatic "
            f"resume will load {checkpoint}"
        )
        init_checkpoint = None
    initialization = load_matching_trainable_parameters(
        model,
        init_checkpoint,
    )
    if initialization is not None:
        print(
            "VGGT matching-only initialization: "
            f"{initialization['matched_tensors']} tensors, "
            f"{initialization['matched_trainable_fraction']:.2%} of "
            "trainable parameters loaded from "
            f"{initialization['checkpoint']}"
        )
        if initialization["mismatched_shapes"]:
            print(
                "VGGT matching-only skipped shape mismatches: "
                f"{initialization['mismatched_shapes']}"
            )
    train_dataset = instantiate(cfg.train_dataset)
    eval_dataset = instantiate(cfg.val_dataset) if cfg.do_eval else None
    if eval_dataset is not None and int(cfg.max_eval_samples) > 0:
        maximum = min(int(cfg.max_eval_samples), len(eval_dataset))
        if maximum < len(eval_dataset):
            # Even spacing is deterministic and covers the complete held-out
            # episode list better than taking one contiguous prefix.
            indices = (
                torch.linspace(0, len(eval_dataset) - 1, maximum)
                .round()
                .to(torch.long)
                .tolist()
            )
            eval_dataset = Subset(eval_dataset, indices)
    data_collator = instantiate(cfg.data_collator)
    training_args = instantiate(cfg.training_args)
    trainer = VGGTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        backbone_learning_rate=cfg.backbone_learning_rate,
        visualization_config=OmegaConf.to_container(
            cfg.visualization, resolve=True
        ),
    )
    trainer.add_callback(
        VGGTJSONLLossLoggerCallback(output_dir / "loss_log.jsonl")
    )

    if checkpoint is not None:
        allow_trusted_numpy_rng_state_types()
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(output_dir))
    trainer.save_state()


if __name__ == "__main__":
    main()
