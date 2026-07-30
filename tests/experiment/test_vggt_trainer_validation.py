from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("DS_ACCELERATOR", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch.utils.data import Dataset
from transformers import PretrainedConfig, TrainingArguments

from groot.vla.experiment.vggt_3d_wam import (
    VGGTTrainer,
    get_last_complete_checkpoint,
)


class _VideoDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"video": torch.tensor([float(index + 1)])}


class _BatchDictOnlyVGGT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.config = PretrainedConfig()
        self.received_batch_dict = False

    def forward(self, inputs: dict[str, torch.Tensor]):
        self.received_batch_dict = isinstance(inputs, dict)
        return {"loss": inputs["video"].float().mean() * self.scale}


def _collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {"video": torch.stack([item["video"] for item in items])}


class _SaveOrderVGGTTrainer(VGGTTrainer):
    def __init__(self, *args, **kwargs) -> None:
        self.events: list[str] = []
        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, model, trial) -> None:
        self.events.append("save")

    def _evaluate(self, trial, ignore_keys_for_eval):
        self.events.append("evaluate")
        return {}


class VGGTTrainerValidationTest(unittest.TestCase):
    def test_incomplete_latest_checkpoint_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir)
            complete = output_path / "checkpoint-4000"
            complete.mkdir()
            for filename in (
                "model.safetensors",
                "trainer_state.json",
                "optimizer.pt",
                "scheduler.pt",
            ):
                (complete / filename).touch()
            incomplete = output_path / "checkpoint-5000"
            incomplete.mkdir()
            (incomplete / "config.json").touch()

            with self.assertWarnsRegex(
                UserWarning,
                "Ignoring incomplete checkpoint",
            ):
                checkpoint = get_last_complete_checkpoint(output_path)

        self.assertEqual(checkpoint, str(complete))

    def test_evaluate_passes_one_batch_dictionary_to_model(self) -> None:
        model = _BatchDictOnlyVGGT()
        with tempfile.TemporaryDirectory() as output_dir:
            args = TrainingArguments(
                output_dir=output_dir,
                per_device_eval_batch_size=2,
                prediction_loss_only=True,
                report_to=[],
                remove_unused_columns=False,
            )
            trainer = VGGTTrainer(
                model=model,
                args=args,
                eval_dataset=_VideoDataset(),
                data_collator=_collate,
                backbone_learning_rate=2e-5,
                visualization_config={"enabled": False},
            )
            metrics = trainer.evaluate()

        self.assertTrue(model.received_batch_dict)
        self.assertIn("eval_loss", metrics)
        self.assertAlmostEqual(metrics["eval_loss"], 2.5)

    def test_same_step_checkpoint_is_saved_before_evaluation(self) -> None:
        model = _BatchDictOnlyVGGT()
        with tempfile.TemporaryDirectory() as output_dir:
            args = TrainingArguments(
                output_dir=output_dir,
                prediction_loss_only=True,
                report_to=[],
                remove_unused_columns=False,
                save_strategy="steps",
            )
            trainer = _SaveOrderVGGTTrainer(
                model=model,
                args=args,
                backbone_learning_rate=2e-5,
                visualization_config={"enabled": False},
            )
            trainer.control.should_log = False
            trainer.control.should_save = True
            trainer.control.should_evaluate = True
            trainer._maybe_log_save_evaluate(
                tr_loss=torch.tensor(0.0),
                grad_norm=None,
                model=model,
                trial=None,
                epoch=0,
                ignore_keys_for_eval=None,
                start_time=0.0,
            )

        self.assertEqual(trainer.events, ["save", "evaluate"])


if __name__ == "__main__":
    unittest.main()
