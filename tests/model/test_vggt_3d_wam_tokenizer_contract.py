import json
from types import SimpleNamespace

import pytest
import torch

from groot.vla.experiment.vggt_3d_wam import (
    VGGTJSONLLossLoggerCallback,
    VGGTTrainer,
)
from groot.vla.model.vggt_3d_wam.configuration import VGGT3DWAMConfig
from groot.vla.model.vggt_3d_wam.geometry import rays_in_frame, scale_intrinsics
from groot.vla.model.vggt_3d_wam.metric_tokens import (
    MetricTokenEncoder,
    MultiViewDeformableCrossAttention,
)
from groot.vla.model.vggt_3d_wam.model import VGGT3DWAMModel
from groot.vla.model.vggt_3d_wam.temporal_codec import (
    WanTemporalDecoder,
    WanTemporalEncoder,
    wan_latent_time,
    wan_video_time,
)


def _tiny_config() -> VGGT3DWAMConfig:
    return VGGT3DWAMConfig(
        image_size=(32, 64),
        patch_size=16,
        patch_embed_type="conv",
        vggt_pretrain_image_size=32,
        backbone_dim=16,
        backbone_depth=1,
        backbone_heads=4,
        backbone_mlp_ratio=2,
        vggt_checkpoint_path=None,
        init_random=True,
        freeze_backbone=False,
        lora_rank=0,
        global_temporal_window=1,
        latent_dim=8,
        latent_spatial_stride=16,
        latent_temporal_stride=4,
        video_temporal_layers=1,
        video_temporal_heads=2,
        video_decoder_dim=32,
        geometry_dim=8,
        geometry_heads=2,
        geometry_temporal_layers=1,
        geometry_deformable_layers=2,
        geometry_deformable_levels=2,
        deformable_points=1,
        deformable_offset_scale=1.0,
        grid_size=(1, 2, 2),
        grid_x_range=(0.1, 1.0),
        grid_y_range=(-0.5, 0.5),
        grid_z_range=(0.1, 1.0),
        pointmap_size=(4, 8),
        pointmap_depth_bins=4,
        pointmap_min_range=0.1,
        pointmap_max_range=1.0,
        pointmap_ray_chunk_size=16,
        pointmap_loss_warmup_steps=0,
    )


def _camera_tensors(time: int, views: int) -> tuple[torch.Tensor, torch.Tensor]:
    intrinsics = torch.tensor(
        [[32.0, 0.0, 31.5], [0.0, 32.0, 15.5], [0.0, 0.0, 1.0]]
    )
    camera_k = intrinsics.view(1, 1, 1, 3, 3).expand(
        1, time, views, -1, -1
    ).clone()
    transforms = torch.eye(4).view(1, 1, 1, 4, 4).expand(
        1, time, views, -1, -1
    ).clone()
    return camera_k, transforms


def test_wan_temporal_contract_and_gradients():
    assert wan_latent_time(33) == 9
    assert wan_video_time(9) == 33
    with pytest.raises(ValueError, match="4k\\+1"):
        wan_latent_time(32)

    encoder = WanTemporalEncoder(8)
    decoder = WanTemporalDecoder(8)
    frames = torch.randn(2, 8, 33, 2, 3, requires_grad=True)
    latent = encoder(frames)
    reconstructed = decoder(latent)

    assert latent.shape == (2, 8, 9, 2, 3)
    assert reconstructed.shape == frames.shape
    reconstructed.square().mean().backward()
    assert frames.grad is not None
    assert encoder.chunk.weight.grad is not None
    assert decoder.chunk.weight.grad is not None


def test_vggt_jsonl_loss_logger_appends_rank_zero_metrics(tmp_path):
    output_path = tmp_path / "loss_log.jsonl"
    callback = VGGTJSONLLossLoggerCallback(output_path)
    state = SimpleNamespace(is_world_process_zero=True, global_step=10)
    callback.on_log(
        None,
        state,
        None,
        logs={
            "loss": 0.5,
            "learning_rate": 1e-5,
            "grad_norm": 2.0,
            "video_recon_loss_avg": 0.4,
            "free_space_sample_count_avg": 12.0,
            "multiview_correspondence_ratio": 0.25,
            "ignored_runtime": 12.0,
        },
    )
    state.global_step = 20
    callback.on_log(
        None,
        state,
        None,
        logs={"loss": 0.25, "pointmap_loss_avg": torch.tensor(0.1)},
    )
    state.is_world_process_zero = False
    callback.on_log(None, state, None, logs={"loss": 999.0})

    entries = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert entries == [
        {
            "step": 10,
            "loss": 0.5,
            "learning_rate": 1e-5,
            "grad_norm": 2.0,
            "video_recon_loss_avg": 0.4,
            "free_space_sample_count_avg": 12.0,
            "multiview_correspondence_ratio": 0.25,
        },
        {
            "step": 20,
            "loss": 0.25,
            "pointmap_loss_avg": pytest.approx(0.1),
        },
    ]


def test_trainer_collects_scalar_losses_counts_and_weights():
    diagnostics = VGGTTrainer._distributed_diagnostics(
        {
            "loss": torch.tensor(10.0),
            "pointmap_loss": torch.tensor(0.2),
            "free_space_sample_count": torch.tensor(12),
            "pointmap_raw_valid_weight": torch.tensor(3.5),
            "non_scalar_count": torch.ones(2),
            "ignored": torch.tensor(99.0),
        }
    )
    assert diagnostics == {
        "free_space_sample_count": 12.0,
        "pointmap_loss": pytest.approx(0.2),
        "pointmap_raw_valid_weight": 3.5,
    }


def test_temporal_encoder_preserves_causal_chunk_boundaries():
    encoder = WanTemporalEncoder(8).eval()
    frames = torch.randn(1, 8, 9, 2, 3)
    changed_future = frames.clone()
    changed_future[:, :, 5:] += 100

    with torch.no_grad():
        original = encoder(frames)
        changed = encoder(changed_future)

    torch.testing.assert_close(original[:, :, :2], changed[:, :, :2])
    assert not torch.allclose(original[:, :, 2:], changed[:, :, 2:])


def test_tokenizer_native_multiview_contract():
    model = VGGT3DWAMModel(_tiny_config()).eval()
    video = torch.randint(0, 256, (1, 9, 2, 3, 32, 64), dtype=torch.uint8)
    camera_k, transforms = _camera_tensors(time=9, views=2)

    with torch.no_grad():
        encoded = model.encode(
            video,
            camera_k,
            transforms,
            sample_posterior=False,
        )
        reconstructed = model.decode_2d(
            encoded["z_2d_video"],
            output_time=9,
            output_size=(32, 64),
        )
        geometry = model.decode_3d(
            encoded["z_3d_video"],
            camera_k,
            transforms,
            output_time=9,
        )

    assert encoded["z_2d_video"].shape == (1, 2, 8, 3, 2, 4)
    assert encoded["z_3d_video"].shape == (1, 3, 4, 8)
    assert reconstructed.shape == (1, 9, 2, 3, 32, 64)
    assert geometry["frame_tokens"].shape == (1, 9, 4, 8)
    assert geometry["pointmap"].shape == (1, 9, 2, 3, 4, 8)
    assert geometry["occupancy_logits"].shape == (1, 9, 2, 4, 4, 8)
    assert geometry["ray_sample_valid"].shape == (1, 9, 2, 4, 4, 8)


def test_tokenizer_wan_2d_drop_in_layout():
    model = VGGT3DWAMModel(_tiny_config()).eval()
    video = torch.rand(1, 3, 9, 32, 64) * 2 - 1

    with torch.no_grad():
        latent = model.encode_2d(
            video,
            sample_posterior=False,
            input_range="minus_one_one",
        )
        reconstructed = model.decode_2d(latent)

    assert latent.shape == (1, 8, 3, 2, 4)
    assert reconstructed.shape == video.shape
    assert reconstructed.min() >= -1
    assert reconstructed.max() <= 1


def test_tokenizer_training_forward_is_finite():
    model = VGGT3DWAMModel(_tiny_config()).train()
    video = torch.randint(0, 256, (1, 9, 2, 3, 32, 64), dtype=torch.uint8)
    camera_k, transforms = _camera_tensors(time=9, views=2)
    inputs = {
        "video": video,
        "camera_K": camera_k,
        "T_b0_camera": transforms,
        "pseudo_pointmap_b0": torch.zeros(1, 9, 2, 3, 4, 8),
        "pointmap_valid": torch.ones(1, 9, 2, 4, 8),
    }

    outputs = model(inputs)

    assert torch.isfinite(outputs["loss"])
    assert outputs["z_2d_video"].shape[3] == 3
    assert outputs["z_3d_video"].shape[1] == 3
    assert outputs["reconstructed_video"].shape[1] == 9
    assert outputs["decoded_z_3d_video"].shape[1] == 9
    assert torch.isfinite(outputs["free_space_loss"])
    assert torch.isfinite(outputs["surface_occupancy_loss"])
    assert torch.isfinite(outputs["multiview_consistency_loss"])


def _synthetic_pointmap(
    camera_k: torch.Tensor,
    transforms: torch.Tensor,
    output_size: tuple[int, int],
    surface_range: float = 0.8,
) -> torch.Tensor:
    render_k = scale_intrinsics(camera_k, (32, 64), output_size)
    origins, directions = rays_in_frame(render_k, transforms, *output_size)
    points = origins + surface_range * directions
    return points.permute(0, 1, 2, 5, 3, 4).contiguous()


def test_free_space_supervision_backpropagates_to_voxel_grid():
    model = VGGT3DWAMModel(_tiny_config()).eval()
    camera_k, transforms = _camera_tensors(time=1, views=1)
    token_grid = torch.randn(1, 1, 8, 1, 2, 2, requires_grad=True)
    pointmap, depth_logits, occupancy_logits, sample_valid = (
        model.pointmap_decoder(token_grid, camera_k, transforms)
    )
    target = _synthetic_pointmap(camera_k, transforms, (4, 8))
    confidence = torch.ones(1, 1, 1, 4, 8)
    _, _, free_loss, surface_loss, diagnostics = model._pointmap_losses(
        pointmap,
        depth_logits,
        occupancy_logits,
        sample_valid,
        target,
        confidence,
        transforms,
    )

    assert torch.isfinite(free_loss)
    assert torch.isfinite(surface_loss)
    assert free_loss > 0
    assert surface_loss > 0
    assert diagnostics["free_space_sample_count"] > 0
    assert diagnostics["surface_sample_count"] > 0
    (free_loss + surface_loss).backward()
    assert token_grid.grad is not None
    assert (
        model.pointmap_decoder.occupancy_head.output_projection.weight.grad
        is not None
    )


def test_pointmap_decoder_uses_learned_two_x_refinement():
    config = _tiny_config()
    config.pointmap_size = [8, 16]
    config.pointmap_ray_size = [4, 8]
    model = VGGT3DWAMModel(config).eval()
    camera_k, transforms = _camera_tensors(time=1, views=1)
    token_grid = torch.randn(1, 1, 8, 1, 2, 2)

    with torch.no_grad():
        pointmap, depth_logits, occupancy_logits, sample_valid = (
            model.pointmap_decoder(token_grid, camera_k, transforms)
        )

    assert pointmap.shape == (1, 1, 1, 3, 8, 16)
    assert depth_logits.shape[-2:] == (4, 8)
    assert occupancy_logits.shape[-2:] == (4, 8)
    assert sample_valid.shape[-2:] == (4, 8)


def test_multiview_consistency_uses_geometric_correspondence():
    model = VGGT3DWAMModel(_tiny_config()).eval()
    camera_k, transforms = _camera_tensors(time=1, views=2)
    target = _synthetic_pointmap(camera_k, transforms, (4, 8))
    confidence = torch.ones(1, 1, 2, 4, 8)

    aligned_loss, aligned_diagnostics = model._multiview_consistency_loss(
        target,
        target,
        confidence,
        camera_k,
        transforms,
    )
    perturbed = target.clone()
    perturbed[:, :, 1, 0] += 0.2
    perturbed_loss, perturbed_diagnostics = model._multiview_consistency_loss(
        perturbed,
        target,
        confidence,
        camera_k,
        transforms,
    )

    assert aligned_loss < 1e-6
    assert perturbed_loss > aligned_loss
    assert aligned_diagnostics["multiview_candidate_count"] > 0
    assert (
        aligned_diagnostics["multiview_correspondence_count"]
        == aligned_diagnostics["multiview_candidate_count"]
    )
    assert (
        perturbed_diagnostics["multiview_candidate_count"]
        == aligned_diagnostics["multiview_candidate_count"]
    )


def test_deformable_metric_encoder_gradients_and_all_invisible_fallback():
    encoder = MetricTokenEncoder(
        input_dim=8,
        token_dim=8,
        num_heads=2,
        temporal_layers=1,
        deformable_layers=2,
        deformable_levels=2,
        deformable_points=2,
        deformable_offset_scale=1.0,
        temporal_stride=4,
        grid_size=(1, 2, 2),
        x_range=(0.0, 1.0),
        y_range=(-0.5, 0.5),
        z_range=(-1.0, -0.1),
    )
    features = torch.randn(1, 9, 2, 8, 2, 4, requires_grad=True)
    camera_k, transforms = _camera_tensors(time=9, views=2)

    tokens, grid, visible = encoder(
        features,
        camera_k,
        transforms,
        image_size=(32, 64),
    )

    assert tokens.shape == (1, 3, 4, 8)
    assert grid.shape == (1, 3, 8, 1, 2, 2)
    assert not visible.any()
    assert torch.isfinite(tokens).all()

    tokens.square().mean().backward()
    offset_head = (
        encoder.deformable_encoder[1]
        .cross_attention.sampling_offsets
    )
    assert offset_head.weight.grad is not None
    assert torch.isfinite(offset_head.weight.grad).all()


def test_deformable_samples_cannot_enter_bottom_right_padding():
    attention = MultiViewDeformableCrossAttention(
        input_dim=4,
        token_dim=4,
        num_heads=1,
        num_levels=1,
        num_points=1,
        offset_scale=1.0,
    ).eval()
    with torch.no_grad():
        attention.sampling_offsets.weight.zero_()
        attention.sampling_offsets.bias.copy_(
            torch.tensor([0.1, 0.0])
        )
    query = torch.zeros(1, 1, 1, 4)
    feature = torch.ones(1, 1, 1, 4, 4, 4)
    reference = torch.tensor([[[[[0.92, 0.0]]]]])
    visible = torch.ones(1, 1, 1, 1, dtype=torch.bool)

    unmasked = attention(
        query,
        [feature],
        reference,
        visible,
    )
    padding_masked = attention(
        query,
        [feature],
        reference,
        visible,
        valid_grid_max=torch.tensor([0.95, 1.0]),
    )

    assert not torch.allclose(unmasked, query)
    torch.testing.assert_close(padding_masked, query)


def test_backbone_lora_scope_and_checkpoint_backward():
    config = _tiny_config()
    config.freeze_backbone = True
    config.lora_rank = 2
    config.backbone_gradient_checkpointing = True
    model = VGGT3DWAMModel(config).train()

    trainable_backbone = {
        name
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad
    }
    assert trainable_backbone
    assert not any(name.startswith("patch_embed.") for name in trainable_backbone)
    assert all(
        name.startswith(("frame_blocks.", "global_blocks."))
        and (".down." in name or ".up." in name)
        for name in trainable_backbone
    )

    video = torch.rand(1, 9, 1, 3, 32, 64)
    features = model.backbone(video)
    features.square().mean().backward()

    assert any(
        parameter.grad is not None
        for name, parameter in model.backbone.named_parameters()
        if ".up." in name
    )
