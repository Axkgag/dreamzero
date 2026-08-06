"""Configuration for the local VGGT 2D/3D tokenizer."""

from __future__ import annotations

from transformers import PretrainedConfig


class VGGT3DWAMConfig(PretrainedConfig):
    model_type = "vggt_3d_wam"

    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        patch_size: int = 16,
        patch_embed_type: str = "conv",
        vggt_pretrain_image_size: int = 518,
        backbone_dim: int = 384,
        backbone_depth: int = 8,
        backbone_heads: int = 6,
        backbone_mlp_ratio: float = 4.0,
        vggt_checkpoint_path: str | None = None,
        init_random: bool = False,
        min_checkpoint_match_fraction: float = 0.05,
        freeze_backbone: bool = True,
        freeze_dino: bool = True,
        dino_image_chunk_size: int = 4,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        global_temporal_window: int = 4,
        align_global_windows_to_codec: bool = True,
        global_attention_causal: bool = False,
        feature_tap_layers: tuple[int, int, int, int] = (4, 11, 17, 23),
        feature_tap_dim: int = 128,
        backbone_gradient_checkpointing: bool = True,
        latent_dim: int = 32,
        latent_spatial_stride: int = 16,
        latent_temporal_stride: int = 4,
        video_temporal_layers: int = 2,
        video_temporal_heads: int = 4,
        video_fusion_dim: int = 256,
        video_query_heads: int = 8,
        video_decoder_dim: int = 128,
        temporal_codec_num_downsample_stages: int = 2,
        temporal_decoder_num_upsample_stages: int = 2,
        temporal_codec_use_layer_cache: bool = True,
        video_reconstruction_loss_weight: float = 1.0,
        lpips_loss_weight: float = 0.0,
        lpips_net: str = "alex",
        lpips_chunk_size: int = 8,
        ssim_loss_weight: float = 0.0,
        spatial_gradient_loss_weight: float = 0.0,
        temporal_difference_loss_weight: float = 0.0,
        geometry_dim: int = 192,
        geometry_heads: int = 6,
        geometry_temporal_layers: int = 2,
        geometry_deformable_layers: int = 2,
        geometry_deformable_levels: int = 2,
        geometry_feature_layers: tuple[int, int] = (11, 23),
        deformable_points: int = 4,
        deformable_offset_scale: float = 2.0,
        grid_size: tuple[int, int, int] = (8, 12, 8),
        grid_x_range: tuple[float, float] = (0.0, 3.0),
        grid_y_range: tuple[float, float] = (-2.0, 2.0),
        grid_z_range: tuple[float, float] = (-0.5, 2.0),
        pointmap_size: tuple[int, int] = (32, 32),
        pointmap_ray_size: tuple[int, int] | None = None,
        pointmap_depth_bins: int = 48,
        pointmap_min_range: float = 0.05,
        pointmap_max_range: float = 5.0,
        pointmap_ray_chunk_size: int = 1024,
        pointmap_decoder_dim: int = 128,
        pointmap_decoder_residual_layers: int = 2,
        beta_2d: float = 1e-6,
        pointmap_loss_weight: float = 0.1,
        geometry_quality_weight: float = 1.0,
        pointmap_loss_warmup_steps: int = 1000,
        ray_loss_weight: float = 0.1,
        free_space_loss_weight: float = 0.1,
        free_space_surface_margin: float = 0.1,
        multiview_consistency_loss_weight: float = 0.05,
        multiview_occlusion_threshold: float = 0.15,
        temporal_geometry_loss_weight: float = 0.0,
        surface_normal_loss_weight: float = 0.0,
        depth_gradient_loss_weight: float = 0.0,
        mask_auxiliary_losses_to_grid: bool = True,
        masked_view_probability: float = 0.0,
        model_dtype: str = "float32",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.image_size = list(image_size)
        self.patch_size = patch_size
        self.patch_embed_type = patch_embed_type
        self.vggt_pretrain_image_size = vggt_pretrain_image_size
        self.backbone_dim = backbone_dim
        self.backbone_depth = backbone_depth
        self.backbone_heads = backbone_heads
        self.backbone_mlp_ratio = backbone_mlp_ratio
        self.vggt_checkpoint_path = vggt_checkpoint_path
        self.init_random = init_random
        self.min_checkpoint_match_fraction = min_checkpoint_match_fraction
        self.freeze_backbone = freeze_backbone
        self.freeze_dino = freeze_dino
        self.dino_image_chunk_size = dino_image_chunk_size
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.global_temporal_window = global_temporal_window
        self.align_global_windows_to_codec = align_global_windows_to_codec
        self.global_attention_causal = global_attention_causal
        self.feature_tap_layers = list(feature_tap_layers)
        self.feature_tap_dim = feature_tap_dim
        self.backbone_gradient_checkpointing = backbone_gradient_checkpointing
        self.latent_dim = latent_dim
        self.latent_spatial_stride = latent_spatial_stride
        self.latent_temporal_stride = latent_temporal_stride
        self.video_temporal_layers = video_temporal_layers
        self.video_temporal_heads = video_temporal_heads
        self.video_fusion_dim = video_fusion_dim
        self.video_query_heads = video_query_heads
        self.video_decoder_dim = video_decoder_dim
        self.temporal_codec_num_downsample_stages = (
            temporal_codec_num_downsample_stages
        )
        self.temporal_decoder_num_upsample_stages = (
            temporal_decoder_num_upsample_stages
        )
        self.temporal_codec_use_layer_cache = temporal_codec_use_layer_cache
        self.video_reconstruction_loss_weight = (
            video_reconstruction_loss_weight
        )
        self.lpips_loss_weight = lpips_loss_weight
        self.lpips_net = lpips_net
        self.lpips_chunk_size = lpips_chunk_size
        self.ssim_loss_weight = ssim_loss_weight
        self.spatial_gradient_loss_weight = spatial_gradient_loss_weight
        self.temporal_difference_loss_weight = (
            temporal_difference_loss_weight
        )
        self.geometry_dim = geometry_dim
        self.geometry_heads = geometry_heads
        self.geometry_temporal_layers = geometry_temporal_layers
        self.geometry_deformable_layers = geometry_deformable_layers
        self.geometry_deformable_levels = geometry_deformable_levels
        self.geometry_feature_layers = list(geometry_feature_layers)
        self.deformable_points = deformable_points
        self.deformable_offset_scale = deformable_offset_scale
        self.grid_size = list(grid_size)
        self.grid_x_range = list(grid_x_range)
        self.grid_y_range = list(grid_y_range)
        self.grid_z_range = list(grid_z_range)
        self.pointmap_size = list(pointmap_size)
        self.pointmap_ray_size = (
            None if pointmap_ray_size is None else list(pointmap_ray_size)
        )
        self.pointmap_depth_bins = pointmap_depth_bins
        self.pointmap_min_range = pointmap_min_range
        self.pointmap_max_range = pointmap_max_range
        self.pointmap_ray_chunk_size = pointmap_ray_chunk_size
        self.pointmap_decoder_dim = pointmap_decoder_dim
        self.pointmap_decoder_residual_layers = (
            pointmap_decoder_residual_layers
        )
        self.beta_2d = beta_2d
        self.pointmap_loss_weight = pointmap_loss_weight
        self.geometry_quality_weight = geometry_quality_weight
        self.pointmap_loss_warmup_steps = pointmap_loss_warmup_steps
        self.ray_loss_weight = ray_loss_weight
        self.free_space_loss_weight = free_space_loss_weight
        self.free_space_surface_margin = free_space_surface_margin
        self.multiview_consistency_loss_weight = (
            multiview_consistency_loss_weight
        )
        self.multiview_occlusion_threshold = multiview_occlusion_threshold
        self.temporal_geometry_loss_weight = temporal_geometry_loss_weight
        self.surface_normal_loss_weight = surface_normal_loss_weight
        self.depth_gradient_loss_weight = depth_gradient_loss_weight
        self.mask_auxiliary_losses_to_grid = mask_auxiliary_losses_to_grid
        self.masked_view_probability = masked_view_probability
        self.model_dtype = model_dtype
