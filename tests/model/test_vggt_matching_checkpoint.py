from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from groot.vla.model.vggt_3d_wam.checkpointing import (
    load_matching_trainable_parameters,
    resolve_matching_checkpoint,
)


class TinyChangedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matching = torch.nn.Linear(3, 2, bias=False)
        self.changed = torch.nn.Linear(4, 2, bias=False)
        self.new = torch.nn.Linear(2, 1, bias=False)
        self.frozen = torch.nn.Parameter(
            torch.zeros(2),
            requires_grad=False,
        )


class VGGTMatchingCheckpointTest(unittest.TestCase):
    def test_empty_checkpoint_is_a_strict_no_op(self) -> None:
        model = TinyChangedModel()
        original = model.matching.weight.detach().clone()

        self.assertIsNone(resolve_matching_checkpoint(None))
        self.assertIsNone(resolve_matching_checkpoint(""))
        self.assertIsNone(resolve_matching_checkpoint("   "))
        self.assertIsNone(load_matching_trainable_parameters(model, ""))
        torch.testing.assert_close(model.matching.weight, original)

    def test_only_matching_trainable_parameters_are_loaded(self) -> None:
        model = TinyChangedModel()
        changed_before = model.changed.weight.detach().clone()
        new_before = model.new.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "model.safetensors"
            save_file(
                {
                    "matching.weight": torch.full((2, 3), 7.0),
                    "changed.weight": torch.full((2, 3), 8.0),
                    "frozen": torch.full((2,), 9.0),
                },
                checkpoint_path,
            )

            result = load_matching_trainable_parameters(model, directory)

        self.assertIsNotNone(result)
        self.assertEqual(result["matched_tensors"], 1)
        self.assertEqual(result["mismatched_tensors"], 1)
        self.assertEqual(result["missing_tensors"], 1)
        torch.testing.assert_close(
            model.matching.weight,
            torch.full((2, 3), 7.0),
        )
        torch.testing.assert_close(model.changed.weight, changed_before)
        torch.testing.assert_close(model.new.weight, new_before)
        torch.testing.assert_close(model.frozen, torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
