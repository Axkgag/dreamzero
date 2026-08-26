from __future__ import annotations

import tempfile
import unittest

import torch

from groot.vla.utils.checkpoint_state import (
    is_reconstructible_checkpoint_state_key,
    select_parameter_efficient_state_dict,
    validate_checkpoint_required_keys,
    write_checkpoint_manifest,
)


class RequiredFrozenModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(2, 2, bias=False)
        self.adapter = torch.nn.Linear(2, 1, bias=False)
        self.extension = torch.nn.Linear(2, 2, bias=False)
        self.base.requires_grad_(False)
        self.extension.requires_grad_(False)

    def checkpoint_required_state_keys(self) -> list[str]:
        return ["extension.weight"]


class CheckpointStateTest(unittest.TestCase):
    def test_wan_required_state_excludes_reconstructible_offset_seconds(self) -> None:
        self.assertTrue(is_reconstructible_checkpoint_state_key("offset_seconds"))
        self.assertTrue(
            is_reconstructible_checkpoint_state_key(
                "model.action_encoder.offset_seconds"
            )
        )
        self.assertFalse(
            is_reconstructible_checkpoint_state_key(
                "model.blocks.0.cross_attn.k_img.weight"
            )
        )

    def test_parameter_efficient_state_keeps_required_frozen_tensor(self) -> None:
        model = RequiredFrozenModule()
        selected, required = select_parameter_efficient_state_dict(
            model,
            model.state_dict(),
        )
        self.assertEqual(required, {"extension.weight"})
        self.assertEqual(
            set(selected),
            {"adapter.weight", "extension.weight"},
        )

    def test_manifest_rejects_missing_required_tensor(self) -> None:
        model = RequiredFrozenModule()
        with tempfile.TemporaryDirectory() as directory:
            write_checkpoint_manifest(
                directory,
                parameter_efficient=True,
                required_state_keys={"extension.weight"},
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "missing required tensors",
            ):
                validate_checkpoint_required_keys(
                    directory,
                    {"adapter.weight"},
                    model,
                )

    def test_legacy_checkpoint_reports_missing_without_claiming_complete(self) -> None:
        model = RequiredFrozenModule()
        with tempfile.TemporaryDirectory() as directory:
            missing, has_manifest = validate_checkpoint_required_keys(
                directory,
                {"adapter.weight"},
                model,
            )
        self.assertFalse(has_manifest)
        self.assertEqual(missing, {"extension.weight"})


if __name__ == "__main__":
    unittest.main()
