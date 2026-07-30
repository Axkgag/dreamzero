from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("DS_ACCELERATOR", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch.utils.data import Dataset
from transformers import BatchFeature, PretrainedConfig, TrainingArguments

from groot.vla.experiment.base import BaseTrainer


class _StateDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"state": torch.tensor([float(index + 1)])}


class _BatchDictOnlyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.config = PretrainedConfig()
        self.received_batch_dict = False

    def forward(self, inputs: dict[str, torch.Tensor]):
        self.received_batch_dict = isinstance(inputs, dict)
        return {"loss": inputs["state"].float().mean() * self.scale}


class _BatchFeatureModel(_BatchDictOnlyModel):
    def forward(self, inputs: dict[str, torch.Tensor]):
        self.received_batch_dict = isinstance(inputs, dict)
        return BatchFeature(
            {"loss": inputs["state"].float().mean() * self.scale}
        )


def _collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {"state": torch.stack([item["state"] for item in items])}


class _SaveOrderTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        self.events: list[str] = []
        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, model, trial) -> None:
        self.events.append("save")

    def _evaluate(self, trial, ignore_keys_for_eval):
        self.events.append("evaluate")
        return {}


class BaseTrainerPredictionStepTest(unittest.TestCase):
    def test_evaluate_passes_one_batch_dictionary_to_model(self) -> None:
        model = _BatchDictOnlyModel()
        with tempfile.TemporaryDirectory() as output_dir:
            args = TrainingArguments(
                output_dir=output_dir,
                per_device_eval_batch_size=2,
                report_to=[],
                remove_unused_columns=False,
            )
            trainer = BaseTrainer(
                model=model,
                args=args,
                eval_dataset=_StateDataset(),
                data_collator=_collate,
                compute_dtype=torch.float32,
                output_dir=output_dir,
            )
            metrics = trainer.evaluate()

        self.assertTrue(model.received_batch_dict)
        self.assertIn("eval_loss", metrics)
        self.assertAlmostEqual(metrics["eval_loss"], 2.5)

    def test_evaluate_reads_loss_from_batch_feature_mapping(self) -> None:
        model = _BatchFeatureModel()
        with tempfile.TemporaryDirectory() as output_dir:
            args = TrainingArguments(
                output_dir=output_dir,
                per_device_eval_batch_size=2,
                report_to=[],
                remove_unused_columns=False,
            )
            trainer = BaseTrainer(
                model=model,
                args=args,
                eval_dataset=_StateDataset(),
                data_collator=_collate,
                compute_dtype=torch.float32,
                output_dir=output_dir,
            )
            metrics = trainer.evaluate()

        self.assertTrue(model.received_batch_dict)
        self.assertIn("eval_loss", metrics)
        self.assertAlmostEqual(metrics["eval_loss"], 2.5)

    def test_same_step_checkpoint_is_saved_before_evaluation(self) -> None:
        model = _BatchDictOnlyModel()
        with tempfile.TemporaryDirectory() as output_dir:
            args = TrainingArguments(
                output_dir=output_dir,
                report_to=[],
                remove_unused_columns=False,
                save_strategy="steps",
            )
            trainer = _SaveOrderTrainer(
                model=model,
                args=args,
                compute_dtype=torch.float32,
                output_dir=output_dir,
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
