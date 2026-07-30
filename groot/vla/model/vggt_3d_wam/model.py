"""Shared-backbone 2D video and metric-3D VGGT tokenizer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from .backbone import VGGTBackbone
from .configuration import VGGT3DWAMConfig
from .geometry import invert_transform, points_in_metric_grid, scale_intrinsics
from .losses import (
    ChunkedLPIPSLoss,
    charbonnier_loss,
    spatial_gradient_loss,
    ssim_loss,
    temporal_difference_loss,
)
from .metric_tokens import MetricTokenDecoder, MetricTokenEncoder
from .pointmap_decoder import PointMapDecoder
from .video_latent import VideoDecoder, VideoLatentBranch


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1)


class VGGT3DWAMModel(PreTrainedModel):
    config_class = VGGT3DWAMConfig
    base_model_prefix = "vggt_3d_wam"
    _keys_to_ignore_on_load_missing = [r"lpips_loss\..*"]

    def __init__(self, config: VGGT3DWAMConfig) -> None:
        super().__init__(config)
        image_size = tuple(config.image_size)
        grid_size = tuple(config.grid_size)
        x_range = tuple(config.grid_x_range)
        y_range = tuple(config.grid_y_range)
        z_range = tuple(config.grid_z_range)
        self.backbone = VGGTBackbone(
            patch_size=config.patch_size,
            patch_embed_type=config.patch_embed_type,
            pretrain_image_size=config.vggt_pretrain_image_size,
            embed_dim=config.backbone_dim,
            depth=config.backbone_depth,
            num_heads=config.backbone_heads,
            mlp_ratio=config.backbone_mlp_ratio,
            checkpoint_path=config.vggt_checkpoint_path,
            init_random=config.init_random,
            min_checkpoint_match_fraction=config.min_checkpoint_match_fraction,
            freeze=config.freeze_backbone,
            freeze_dino=config.freeze_dino,
            dino_image_chunk_size=config.dino_image_chunk_size,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            global_temporal_window=config.global_temporal_window,
            gradient_checkpointing=config.backbone_gradient_checkpointing,
        )
        self.video_encoder = VideoLatentBranch(
            input_dim=config.backbone_dim,
            latent_dim=config.latent_dim,
            spatial_stride=config.latent_spatial_stride,
            temporal_stride=config.latent_temporal_stride,
            temporal_layers=config.video_temporal_layers,
            temporal_heads=config.video_temporal_heads,
        )
        self.video_decoder = VideoDecoder(
            config.latent_dim,
            config.video_decoder_dim,
            config.latent_spatial_stride,
        )
        self.metric_encoder = MetricTokenEncoder(
            input_dim=config.backbone_dim,
            token_dim=config.geometry_dim,
            num_heads=config.geometry_heads,
            temporal_layers=config.geometry_temporal_layers,
            deformable_layers=config.geometry_deformable_layers,
            deformable_levels=config.geometry_deformable_levels,
            deformable_points=config.deformable_points,
            deformable_offset_scale=config.deformable_offset_scale,
            temporal_stride=config.latent_temporal_stride,
            grid_size=grid_size,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
        )
        self.metric_decoder = MetricTokenDecoder(
            token_dim=config.geometry_dim,
            num_heads=config.geometry_heads,
            temporal_layers=config.geometry_temporal_layers,
            grid_size=grid_size,
        )
        self.pointmap_decoder = PointMapDecoder(
            token_dim=config.geometry_dim,
            image_size=image_size,
            output_size=tuple(config.pointmap_size),
            ray_size=(
                None
                if config.pointmap_ray_size is None
                else tuple(config.pointmap_ray_size)
            ),
            depth_bins=config.pointmap_depth_bins,
            min_range=config.pointmap_min_range,
            max_range=config.pointmap_max_range,
            ray_chunk_size=config.pointmap_ray_chunk_size,
            hidden_dim=config.pointmap_decoder_dim,
            residual_layers=config.pointmap_decoder_residual_layers,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
        )
        self.lpips_loss = (
            ChunkedLPIPSLoss(
                net=config.lpips_net,
                chunk_size=config.lpips_chunk_size,
            )
            if config.lpips_loss_weight > 0
            else None
        )
        self.global_step = 0

    @staticmethod
    def _canonical_video(
        video: torch.Tensor,
        input_range: str = "auto",
    ) -> tuple[torch.Tensor, bool]:
        """Return ``[B,T,V,3,H,W]`` in ``[-1,1]`` and a Wan-layout flag."""
        wan_layout = video.ndim == 5
        if wan_layout:
            if video.shape[1] != 3:
                raise ValueError(
                    "A 5D video must use Wan layout [B,3,T,H,W], "
                    f"got {tuple(video.shape)}"
                )
            video = video.permute(0, 2, 1, 3, 4)[:, :, None]
        elif video.ndim != 6 or video.shape[3] != 3:
            raise ValueError(
                "Video must be [B,3,T,H,W] or [B,T,V,3,H,W], "
                f"got {tuple(video.shape)}"
            )
        if video.dtype == torch.uint8:
            normalized = video.float() * (2.0 / 255.0) - 1.0
        else:
            normalized = video.float()
            if input_range == "zero_one":
                normalized = normalized * 2.0 - 1.0
            elif input_range == "minus_one_one":
                pass
            elif input_range == "auto":
                if float(normalized.detach().amin()) >= 0:
                    normalized = normalized * 2.0 - 1.0
            else:
                raise ValueError(
                    "input_range must be auto, zero_one, or minus_one_one"
                )
        return normalized, wan_layout

    def extract_features(
        self,
        video: torch.Tensor,
        input_range: str = "auto",
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Extract shared VGGT features once for both tokenizer branches."""
        canonical, wan_layout = self._canonical_video(video, input_range)
        backbone_video = (canonical + 1.0) * 0.5
        return self.backbone(backbone_video), canonical, wan_layout

    def _encode_2d_features(
        self,
        features: torch.Tensor,
        video_size: tuple[int, int],
        *,
        sample_posterior: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.video_encoder(
            features,
            video_size,
            sample_posterior=sample_posterior,
        )

    def encode_2d(
        self,
        video: torch.Tensor,
        *,
        sample_posterior: bool | None = None,
        return_stats: bool = False,
        input_range: str = "auto",
    ):
        """Encode RGB video using a Wan-compatible tensor contract.

        ``[B,3,T,H,W]`` returns ``[B,48,T',H/16,W/16]``. The native
        multi-view layout ``[B,T,V,3,H,W]`` returns
        ``[B,V,48,T',H/16,W/16]``.
        """
        features, canonical, wan_layout = self.extract_features(video, input_range)
        if sample_posterior is None:
            sample_posterior = self.training
        latent, mu, logvar = self._encode_2d_features(
            features,
            tuple(canonical.shape[-2:]),
            sample_posterior=sample_posterior,
        )
        if wan_layout:
            latent, mu, logvar = latent[:, 0], mu[:, 0], logvar[:, 0]
        if return_stats:
            return {"latent": latent, "mu": mu, "logvar": logvar}
        return latent

    def encode_3d(
        self,
        video: torch.Tensor,
        camera_k: torch.Tensor,
        base_from_camera: torch.Tensor,
        *,
        input_range: str = "auto",
    ) -> torch.Tensor:
        """Encode metric 3D tokens on the same ``T→T'`` lattice as 2D."""
        features, canonical, wan_layout = self.extract_features(video, input_range)
        if wan_layout:
            if camera_k.ndim == 4:
                camera_k = camera_k[:, :, None]
                base_from_camera = base_from_camera[:, :, None]
            elif camera_k.shape[2] != 1:
                raise ValueError("Wan-layout video contains exactly one view")
        tokens, _, _ = self.metric_encoder(
            features,
            camera_k,
            base_from_camera,
            self.backbone.patch_aligned_image_size(
                tuple(canonical.shape[-2:])
            ),
            valid_image_size=tuple(canonical.shape[-2:]),
        )
        return tokens

    def encode(
        self,
        video: torch.Tensor,
        camera_k: torch.Tensor,
        base_from_camera: torch.Tensor,
        *,
        sample_posterior: bool | None = None,
        input_range: str = "auto",
    ) -> dict[str, torch.Tensor]:
        """Jointly encode 2D and 3D latents from one shared backbone pass."""
        features, canonical, wan_layout = self.extract_features(video, input_range)
        if sample_posterior is None:
            sample_posterior = self.training
        latent_2d, mu_2d, logvar_2d = self._encode_2d_features(
            features,
            tuple(canonical.shape[-2:]),
            sample_posterior=sample_posterior,
        )
        latent_3d, _, voxel_visible = self.metric_encoder(
            features,
            camera_k,
            base_from_camera,
            self.backbone.patch_aligned_image_size(
                tuple(canonical.shape[-2:])
            ),
            valid_image_size=tuple(canonical.shape[-2:]),
        )
        if wan_layout:
            latent_2d = latent_2d[:, 0]
            mu_2d = mu_2d[:, 0]
            logvar_2d = logvar_2d[:, 0]
        return {
            "z_2d_video": latent_2d,
            "mu_2d": mu_2d,
            "logvar_2d": logvar_2d,
            "z_3d_video": latent_3d,
            "voxel_visible": voxel_visible,
        }

    def decode_2d(
        self,
        latent: torch.Tensor,
        *,
        output_time: int | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Decode either Wan layout or native multi-view 2D latents."""
        wan_layout = latent.ndim == 5
        if wan_layout:
            latent = latent[:, None]
        elif latent.ndim != 6:
            raise ValueError(
                "2D latent must be [B,C,T,H,W] or [B,V,C,T,H,W]"
            )
        decoded = self.video_decoder(latent, output_time, output_size)
        if wan_layout:
            return decoded[:, :, 0].permute(0, 2, 1, 3, 4).contiguous()
        return decoded

    def decode_3d(
        self,
        latent: torch.Tensor,
        camera_k: torch.Tensor,
        base_from_camera: torch.Tensor,
        *,
        output_time: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode compressed 3D latents into full-time grids and PointMaps."""
        frame_tokens, token_grid = self.metric_decoder(latent, output_time)
        (
            pointmap,
            depth_logits,
            occupancy_logits,
            ray_sample_valid,
        ) = self.pointmap_decoder(
            token_grid,
            camera_k,
            base_from_camera,
        )
        return {
            "frame_tokens": frame_tokens,
            "token_grid": token_grid,
            "pointmap": pointmap,
            "depth_logits": depth_logits,
            "occupancy_logits": occupancy_logits,
            "ray_sample_valid": ray_sample_valid,
        }

    def save_pretrained(self, save_directory, *args, **kwargs):
        """Save trainable tokenizer state without the frozen LPIPS network."""
        checkpoint_path = self.config.vggt_checkpoint_path
        init_random = self.config.init_random
        self.config.vggt_checkpoint_path = None
        self.config.init_random = True
        state_dict = kwargs.get("state_dict")
        if state_dict is None:
            state_dict = self.state_dict()
        kwargs["state_dict"] = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("lpips_loss.")
        }
        try:
            return super().save_pretrained(save_directory, *args, **kwargs)
        finally:
            self.config.vggt_checkpoint_path = checkpoint_path
            self.config.init_random = init_random

    def _geometry_weight(self) -> float:
        warmup = self.config.pointmap_loss_warmup_steps
        quality_weight = self.config.geometry_quality_weight
        if warmup <= 0:
            return self.config.pointmap_loss_weight * quality_weight
        progress = min(1.0, float(self.global_step + 1) / warmup)
        return (
            self.config.pointmap_loss_weight
            * quality_weight
            * progress
        )

    def _pointmap_losses(
        self,
        prediction: torch.Tensor,
        depth_logits: torch.Tensor,
        occupancy_logits: torch.Tensor,
        ray_sample_valid: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        base_from_camera: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        coordinate_target_xyz = target.permute(0, 1, 2, 4, 5, 3)
        coordinate_target_inside = points_in_metric_grid(
            coordinate_target_xyz,
            self.config.grid_x_range,
            self.config.grid_y_range,
            self.config.grid_z_range,
        )
        coordinate_error = F.smooth_l1_loss(
            prediction, target, reduction="none", beta=0.05
        )
        coordinate_loss = _weighted_mean(
            coordinate_error,
            (valid * coordinate_target_inside)[:, :, :, None],
        )

        ray_height, ray_width = depth_logits.shape[-2:]
        if target.shape[-2:] == (ray_height, ray_width):
            ray_target = target
            ray_valid = valid
        else:
            batch, time, views = target.shape[:3]
            ray_target = F.interpolate(
                target.reshape(
                    batch * time * views, 3, *target.shape[-2:]
                ),
                size=(ray_height, ray_width),
                mode="nearest",
            ).reshape(batch, time, views, 3, ray_height, ray_width)
            ray_valid = F.interpolate(
                valid.reshape(
                    batch * time * views, 1, *valid.shape[-2:]
                ),
                size=(ray_height, ray_width),
                mode="nearest",
            ).reshape(batch, time, views, ray_height, ray_width)
        target_xyz = ray_target.permute(0, 1, 2, 4, 5, 3)
        target_inside = points_in_metric_grid(
            target_xyz,
            self.config.grid_x_range,
            self.config.grid_y_range,
            self.config.grid_z_range,
        )
        origin = base_from_camera[..., :3, 3]
        target_range = torch.linalg.vector_norm(
            target_xyz - origin[..., None, None, :], dim=-1
        )
        depths = self.pointmap_decoder.depth_values.to(target_range.dtype)
        depth_grid = depths.view(1, 1, 1, -1, 1, 1)
        target_range_grid = target_range[:, :, :, None]
        sample_valid = ray_sample_valid.to(dtype=torch.bool)
        target_bin_distance = (
            target_range_grid - depth_grid
        ).abs().masked_fill(~sample_valid, torch.inf)
        target_bin = target_bin_distance.argmin(dim=3)
        target_bin_valid = sample_valid.any(dim=3)
        ray_error = F.cross_entropy(
            depth_logits.reshape(-1, depth_logits.shape[3], *depth_logits.shape[-2:]),
            target_bin.reshape(-1, *target_bin.shape[-2:]),
            reduction="none",
        ).reshape_as(ray_valid)
        ray_error = torch.where(
            target_bin_valid, ray_error, torch.zeros_like(ray_error)
        )
        ray_loss = _weighted_mean(
            ray_error, ray_valid * target_inside * target_bin_valid
        )

        margin = float(self.config.free_space_surface_margin)
        free_mask = depth_grid < target_range_grid - margin
        surface_mask = (
            (depth_grid - target_range_grid).abs() <= margin
        ) & target_inside[:, :, :, None]
        confidence = ray_valid[:, :, :, None]
        free_weights = confidence * (free_mask & sample_valid)
        surface_weights = confidence * (surface_mask & sample_valid)

        free_error = F.binary_cross_entropy_with_logits(
            occupancy_logits,
            torch.zeros_like(occupancy_logits),
            reduction="none",
        )
        surface_error = F.binary_cross_entropy_with_logits(
            occupancy_logits,
            torch.ones_like(occupancy_logits),
            reduction="none",
        )
        free_space_loss = _weighted_mean(free_error, free_weights)
        surface_occupancy_loss = _weighted_mean(
            surface_error, surface_weights
        )
        coordinate_positive_confidence = valid > 0
        positive_confidence = ray_valid > 0
        eligible_samples = positive_confidence[:, :, :, None].expand_as(
            sample_valid
        )
        valid_samples = eligible_samples & sample_valid
        diagnostics = {
            "pointmap_raw_valid_count": coordinate_positive_confidence.sum(),
            "pointmap_inside_grid_count": (
                coordinate_positive_confidence & coordinate_target_inside
            ).sum(),
            "pointmap_raw_valid_weight": valid.sum(),
            "pointmap_inside_grid_weight": (
                valid * coordinate_target_inside
            ).sum(),
            "ray_inside_grid_count": (
                positive_confidence & target_inside
            ).sum(),
            "ray_total_sample_count": eligible_samples.sum(),
            "ray_valid_sample_count": valid_samples.sum(),
            "ray_supervised_pixel_count": (
                positive_confidence & target_inside & target_bin_valid
            ).sum(),
            "free_space_sample_count": (
                free_mask & valid_samples
            ).sum(),
            "surface_sample_count": (
                surface_mask & valid_samples
            ).sum(),
        }
        return (
            coordinate_loss,
            ray_loss,
            free_space_loss,
            surface_occupancy_loss,
            diagnostics,
        )

    @staticmethod
    def _surface_normal_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        prediction_xyz = prediction.permute(0, 1, 2, 4, 5, 3)
        target_xyz = target.permute(0, 1, 2, 4, 5, 3)
        prediction_dx = (
            prediction_xyz[..., :-1, 1:, :]
            - prediction_xyz[..., :-1, :-1, :]
        )
        prediction_dy = (
            prediction_xyz[..., 1:, :-1, :]
            - prediction_xyz[..., :-1, :-1, :]
        )
        target_dx = (
            target_xyz[..., :-1, 1:, :]
            - target_xyz[..., :-1, :-1, :]
        )
        target_dy = (
            target_xyz[..., 1:, :-1, :]
            - target_xyz[..., :-1, :-1, :]
        )
        prediction_normal = F.normalize(
            torch.linalg.cross(prediction_dx, prediction_dy, dim=-1),
            dim=-1,
            eps=1e-6,
        )
        target_normal = F.normalize(
            torch.linalg.cross(target_dx, target_dy, dim=-1),
            dim=-1,
            eps=1e-6,
        )
        normal_error = 1.0 - (
            prediction_normal * target_normal
        ).sum(dim=-1).clamp(-1, 1)
        normal_valid = torch.minimum(
            torch.minimum(valid[..., :-1, :-1], valid[..., :-1, 1:]),
            valid[..., 1:, :-1],
        )
        return _weighted_mean(normal_error, normal_valid)

    @staticmethod
    def _depth_gradient_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        base_from_camera: torch.Tensor,
    ) -> torch.Tensor:
        origin = base_from_camera[..., :3, 3, None, None]
        prediction_range = torch.linalg.vector_norm(
            prediction - origin, dim=3
        )
        target_range = torch.linalg.vector_norm(target - origin, dim=3)
        prediction_dx = prediction_range[..., :, 1:] - prediction_range[..., :, :-1]
        target_dx = target_range[..., :, 1:] - target_range[..., :, :-1]
        prediction_dy = prediction_range[..., 1:, :] - prediction_range[..., :-1, :]
        target_dy = target_range[..., 1:, :] - target_range[..., :-1, :]
        valid_x = torch.minimum(valid[..., :, 1:], valid[..., :, :-1])
        valid_y = torch.minimum(valid[..., 1:, :], valid[..., :-1, :])
        x_loss = _weighted_mean(
            F.smooth_l1_loss(
                prediction_dx, target_dx, reduction="none", beta=0.02
            ),
            valid_x,
        )
        y_loss = _weighted_mean(
            F.smooth_l1_loss(
                prediction_dy, target_dy, reduction="none", beta=0.02
            ),
            valid_y,
        )
        return 0.5 * (x_loss + y_loss)

    @staticmethod
    def _project_b0_points(
        points_b0: torch.Tensor,
        camera_k: torch.Tensor,
        base_from_camera: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        camera_from_base = invert_transform(base_from_camera)
        camera_points = torch.einsum(
            "...ij,...hwj->...hwi",
            camera_from_base[..., :3, :3],
            points_b0,
        ) + camera_from_base[..., None, None, :3, 3]
        depth = camera_points[..., 2]
        pixels_h = torch.einsum(
            "...ij,...hwj->...hwi", camera_k, camera_points
        )
        pixels = pixels_h[..., :2] / pixels_h[..., 2:].clamp_min(1e-4)
        grid = torch.empty_like(pixels)
        grid[..., 0] = 2 * (pixels[..., 0] + 0.5) / width - 1
        grid[..., 1] = 2 * (pixels[..., 1] + 0.5) / height - 1
        visible = (depth > 1e-4) & (grid.abs() <= 1).all(dim=-1)
        return grid, visible

    @staticmethod
    def _sample_view(
        values: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        batch, time, height, width = values.shape[:4]
        channels = 1 if values.ndim == 4 else values.shape[-1]
        if values.ndim == 4:
            feature = values.reshape(batch * time, 1, height, width)
        else:
            feature = values.reshape(
                batch * time, height, width, channels
            ).permute(0, 3, 1, 2)
        sampled = F.grid_sample(
            feature,
            grid.reshape(batch * time, height, width, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.permute(0, 2, 3, 1).reshape(
            batch, time, height, width, channels
        )
        return sampled[..., 0] if values.ndim == 4 else sampled

    def _multiview_consistency_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        camera_k: torch.Tensor,
        base_from_camera: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        views = prediction.shape[2]
        if views < 2:
            zero = prediction.new_zeros(())
            return zero, {
                "multiview_candidate_count": zero,
                "multiview_correspondence_count": zero,
                "multiview_correspondence_weight": zero,
            }
        height, width = prediction.shape[-2:]
        render_k = scale_intrinsics(
            camera_k,
            self.pointmap_decoder.image_size,
            (height, width),
        )
        predicted_points = prediction.permute(0, 1, 2, 4, 5, 3)
        target_points = target.permute(0, 1, 2, 4, 5, 3)
        numerator = prediction.new_zeros(())
        denominator = prediction.new_zeros(())
        candidate_count = prediction.new_zeros(())
        correspondence_count = prediction.new_zeros(())
        threshold = float(self.config.multiview_occlusion_threshold)

        for source_view in range(views):
            source_gt = target_points[:, :, source_view]
            source_prediction = predicted_points[:, :, source_view]
            source_confidence = valid[:, :, source_view]
            source_inside = points_in_metric_grid(
                source_gt,
                self.config.grid_x_range,
                self.config.grid_y_range,
                self.config.grid_z_range,
            )
            for target_view in range(views):
                if source_view == target_view:
                    continue
                grid, projection_visible = self._project_b0_points(
                    source_gt,
                    render_k[:, :, target_view],
                    base_from_camera[:, :, target_view],
                    height,
                    width,
                )
                sampled_prediction = self._sample_view(
                    predicted_points[:, :, target_view], grid
                )
                sampled_gt = self._sample_view(
                    target_points[:, :, target_view], grid
                )
                sampled_confidence = self._sample_view(
                    valid[:, :, target_view], grid
                )
                # A source surface is mutually visible only if the target
                # pseudo PointMap contains the same B0 point after projection.
                overlap_error = torch.linalg.vector_norm(
                    sampled_gt - source_gt, dim=-1
                )
                overlap = overlap_error <= threshold
                candidate = (
                    projection_visible
                    & source_inside
                    & (source_confidence > 0)
                    & (sampled_confidence > 0)
                )
                correspondence = candidate & overlap
                weights = (
                    source_confidence
                    * sampled_confidence
                    * correspondence
                )
                error = F.smooth_l1_loss(
                    source_prediction,
                    sampled_prediction,
                    reduction="none",
                    beta=0.05,
                ).mean(dim=-1)
                numerator = numerator + (error * weights).sum()
                denominator = denominator + weights.sum()
                candidate_count = candidate_count + candidate.sum()
                correspondence_count = (
                    correspondence_count + correspondence.sum()
                )
        return numerator / denominator.clamp_min(1), {
            "multiview_candidate_count": candidate_count,
            "multiview_correspondence_count": correspondence_count,
            "multiview_correspondence_weight": denominator,
        }

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        video, wan_layout = self._canonical_video(inputs["video"])
        if wan_layout:
            raise ValueError("Training batches must use [B,T,V,3,H,W]")
        camera_k = inputs["camera_K"].float()
        base_from_camera = inputs["T_b0_camera"].float()
        target_pointmap = inputs["pseudo_pointmap_b0"].float()
        pointmap_valid = inputs["pointmap_valid"].float()
        masked_video = video
        if self.training and self.config.masked_view_probability > 0:
            view_mask = (
                torch.rand(
                    video.shape[0],
                    1,
                    video.shape[2],
                    1,
                    1,
                    1,
                    device=video.device,
                )
                < self.config.masked_view_probability
            )
            masked_video = video.masked_fill(view_mask, -1)

        features = self.backbone((masked_video + 1.0) * 0.5)
        latent, mu, logvar = self._encode_2d_features(
            features,
            tuple(video.shape[-2:]),
            sample_posterior=self.training,
        )
        geometry_tokens, token_grid, voxel_visible = self.metric_encoder(
            features,
            camera_k,
            base_from_camera,
            self.backbone.patch_aligned_image_size(tuple(video.shape[-2:])),
            valid_image_size=tuple(video.shape[-2:]),
        )
        reconstructed = self.video_decoder(
            latent,
            video.shape[1],
            tuple(video.shape[-2:]),
        )
        decoded_geometry_tokens, decoded_token_grid = self.metric_decoder(
            geometry_tokens,
            video.shape[1],
        )
        (
            predicted_pointmap,
            depth_logits,
            occupancy_logits,
            ray_sample_valid,
        ) = self.pointmap_decoder(
            decoded_token_grid,
            camera_k,
            base_from_camera,
        )

        video_recon_loss = charbonnier_loss(reconstructed, video)
        video_ssim_loss = (
            ssim_loss(reconstructed, video)
            if self.config.ssim_loss_weight > 0
            else video.new_zeros(())
        )
        video_spatial_gradient_loss = (
            spatial_gradient_loss(reconstructed, video)
            if self.config.spatial_gradient_loss_weight > 0
            else video.new_zeros(())
        )
        video_temporal_difference_loss = (
            temporal_difference_loss(reconstructed, video)
            if self.config.temporal_difference_loss_weight > 0
            else video.new_zeros(())
        )
        video_lpips_loss = (
            self.lpips_loss(reconstructed, video)
            if self.lpips_loss is not None
            else video.new_zeros(())
        )
        video_quality_loss = (
            self.config.video_reconstruction_loss_weight
            * video_recon_loss
            + self.config.lpips_loss_weight * video_lpips_loss
            + self.config.ssim_loss_weight * video_ssim_loss
            + self.config.spatial_gradient_loss_weight
            * video_spatial_gradient_loss
            + self.config.temporal_difference_loss_weight
            * video_temporal_difference_loss
        )
        kl_2d_loss = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
        (
            pointmap_loss,
            ray_surface_loss,
            free_space_loss,
            surface_occupancy_loss,
            pointmap_diagnostics,
        ) = self._pointmap_losses(
            predicted_pointmap,
            depth_logits,
            occupancy_logits,
            ray_sample_valid,
            target_pointmap,
            pointmap_valid,
            base_from_camera,
        )
        (
            multiview_consistency_loss,
            multiview_diagnostics,
        ) = self._multiview_consistency_loss(
            predicted_pointmap,
            target_pointmap,
            pointmap_valid,
            camera_k,
            base_from_camera,
        )
        temporal_geometry_loss = video.new_zeros(())
        if (
            self.config.temporal_geometry_loss_weight > 0
            and predicted_pointmap.shape[1] > 1
        ):
            predicted_delta = predicted_pointmap[:, 1:] - predicted_pointmap[:, :-1]
            target_delta = target_pointmap[:, 1:] - target_pointmap[:, :-1]
            temporal_valid = torch.minimum(
                pointmap_valid[:, 1:], pointmap_valid[:, :-1]
            )
            temporal_geometry_loss = _weighted_mean(
                F.smooth_l1_loss(
                    predicted_delta, target_delta, reduction="none", beta=0.05
                ),
                temporal_valid[:, :, :, None],
            )
        surface_normal_loss = (
            self._surface_normal_loss(
                predicted_pointmap,
                target_pointmap,
                pointmap_valid,
            )
            if self.config.surface_normal_loss_weight > 0
            else video.new_zeros(())
        )
        depth_gradient_loss = (
            self._depth_gradient_loss(
                predicted_pointmap,
                target_pointmap,
                pointmap_valid,
                base_from_camera,
            )
            if self.config.depth_gradient_loss_weight > 0
            else video.new_zeros(())
        )

        geometry_weight = self._geometry_weight()
        geometry_objective_loss = (
            pointmap_loss
            + self.config.ray_loss_weight * ray_surface_loss
            + self.config.free_space_loss_weight
            * (free_space_loss + surface_occupancy_loss)
            + self.config.multiview_consistency_loss_weight
            * multiview_consistency_loss
            + self.config.temporal_geometry_loss_weight
            * temporal_geometry_loss
            + self.config.surface_normal_loss_weight * surface_normal_loss
            + self.config.depth_gradient_loss_weight * depth_gradient_loss
        )
        weighted_geometry_loss = geometry_weight * geometry_objective_loss
        total_loss = (
            video_quality_loss
            + self.config.beta_2d * kl_2d_loss
            + weighted_geometry_loss
        )
        outputs = {
            "loss": total_loss,
            "video_recon_loss": video_recon_loss,
            "video_lpips_loss": video_lpips_loss,
            "video_ssim_loss": video_ssim_loss,
            "video_spatial_gradient_loss": video_spatial_gradient_loss,
            "video_temporal_difference_loss": video_temporal_difference_loss,
            "video_quality_loss": video_quality_loss,
            "kl_2d_loss": kl_2d_loss,
            "pointmap_loss": pointmap_loss,
            "ray_surface_loss": ray_surface_loss,
            "free_space_loss": free_space_loss,
            "surface_occupancy_loss": surface_occupancy_loss,
            "multiview_consistency_loss": multiview_consistency_loss,
            "temporal_geometry_loss": temporal_geometry_loss,
            "surface_normal_loss": surface_normal_loss,
            "depth_gradient_loss": depth_gradient_loss,
            "geometry_objective_loss": geometry_objective_loss,
            "weighted_geometry_loss": weighted_geometry_loss,
            "geometry_loss_weight": total_loss.new_tensor(geometry_weight),
            "z_2d_video": latent,
            "z_3d_video": geometry_tokens,
            "decoded_z_3d_video": decoded_geometry_tokens,
            "voxel_visible": voxel_visible,
            "reconstructed_video": reconstructed,
            "predicted_pointmap_b0": predicted_pointmap,
            "occupancy_logits": occupancy_logits,
            "ray_sample_valid": ray_sample_valid,
        }
        outputs.update(pointmap_diagnostics)
        outputs.update(multiview_diagnostics)
        return outputs
