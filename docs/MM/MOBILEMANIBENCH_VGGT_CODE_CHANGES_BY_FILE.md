# MobileManiBench VGGT Encoder-Decoder：代码改动逐文件说明

## 1. 文档目的与统计范围

> 当前配置校对：2026-07-30
> 本文前 9 节描述当前实现；第 10 节以后是按时间追加的开发与修复记录，旧数值仅表示
> 当时状态。当前常量始终以第 3、7 节和实际 YAML 为准。

本文用于逐文件 review MobileManiBench 多视角 VGGT encoder-decoder 实现。实现对应
`docs/vggt_3d_wam_proposal.md` 第 4 节，代码位于 DreamZero 主仓库，不直接复制
ReconDrive。ReconDrive `hnr_v2` 仅作为训练流程、alternating attention、LoRA 和
voxel query 聚合的设计参考。

在初版四帧 tokenizer 原型基础上，本次进一步完成 Wan2.2-compatible 时空接口改造。
文件名统一使用 `vggt` 或 `vggt_3d_wam`，不使用 `stage1`。

## 2. 已完成链路与边界

当前链路为：

```text
RGB head/wrist + depth head/wrist + camera pose + base pose
  -> synchronized 33-frame multi-view clip
  -> shared VGGT-style alternating-attention backbone
  -> causal 2D temporal codec: 33 -> 9
  -> 2D variational video latent -> learned 9 -> 33 RGB reconstruction
  -> metric voxel queries -> projected multi-view sampling
  -> causal geometry-space temporal transformer
  -> causal 3D temporal codec: 33 -> 9
  -> learned 3D temporal decoder: 9 -> 33
  -> ray depth-bin + occupancy decoder -> 33-frame robot-centric PointMap video
  -> Charbonnier + LPIPS + SSIM + spatial/temporal visual losses
     + KL + PointMap + ray-bin + free-space/surface occupancy
     + multi-view/temporal/normal/depth-gradient geometry losses
  -> rank-zero train/val reconstruction and PointMap visualization
```

当前深度来自有损 H.264 pseudo-range，标定状态为 `nominal_unverified`。因此代码会降低
PointMap 权重并打印 warning；该输出只能用于 coarse geometry supervision，不能用于
毫米级重建、硬 occupancy、碰撞或接触标签。

## 3. 核心张量协议

```text
video                 [B, T, V, 3, H, W]       uint8
camera_K              [B, T, V, 3, 3]          float32
T_b0_camera           [B, T, V, 4, 4]          float32
pseudo_pointmap_b0    [B, T, V, 3, Hq, Wq]     float32
pointmap_valid        [B, T, V, Hq, Wq]         float32 confidence
z_2d_video            [B, V, 48, 9, H/16, W/16]
z_3d_video            [B, 9, N_voxel, C3]
decoded_z_3d_video    [B, 33, N_voxel, C3]
```

`B0` 是 clip 当前帧的底盘坐标系。相机外参按
`inverse(T_world_base0) @ T_world_camera @ T_camera_optical` 构造。

生产配置的完整协议为：

```text
video                 [B, 33, V, 3, 160, 320]
z_2d_video            [B, V, 48, 9, 10, 20]
z_3d_video            [B, 9, 768, 256]
decoded 3D grid       [B, 33, 256, 8, 12, 8]
reconstructed_video   [B, 33, V, 3, 160, 320]
predicted_pointmap    [B, 33, V, 3, 80, 160]
depth_logits          [B, 33, V, 64, 40, 80]
occupancy_logits      [B, 33, V, 64, 40, 80]
ray_sample_valid      [B, 33, V, 64, 40, 80]
```

2D 和 3D latent 使用完全相同的时间 lattice：

```text
latent[0] <- frame[0]
latent[1] <- frame[1:5]
...
latent[8] <- frame[29:33]
```

## 4. 文件总览

| 文件 | 核心职责 |
|---|---|
| `groot/vla/data/dataset/mobilemanibench_vggt.py` | RGB/depth/camera/PointMap 数据链路 |
| `groot/vla/model/vggt_3d_wam/configuration.py` | 可保存的模型配置 |
| `groot/vla/model/vggt_3d_wam/geometry.py` | 坐标变换、ray、投影和 metric grid |
| `groot/vla/model/vggt_3d_wam/backbone.py` | alternating-attention backbone 与 LoRA |
| `groot/vla/model/vggt_3d_wam/temporal_codec.py` | 2D/3D 共用的 Wan-compatible 因果时间编解码 |
| `groot/vla/model/vggt_3d_wam/video_latent.py` | 2D video VAE bottleneck 与 RGB decoder |
| `groot/vla/model/vggt_3d_wam/metric_tokens.py` | metric queries 与 deformable 多视角聚合 |
| `groot/vla/model/vggt_3d_wam/pointmap_decoder.py` | ray rendering 与 learned PointMap refinement |
| `groot/vla/model/vggt_3d_wam/losses.py` | LPIPS、SSIM 与图像质量损失 |
| `groot/vla/model/vggt_3d_wam/model.py` | 公共 encode/decode、loss 和 checkpoint |
| `groot/vla/model/vggt_3d_wam/visualization.py` | RGB/PointMap/3D scatter 诊断 |
| `groot/vla/experiment/vggt_3d_wam.py` | Hydra、Trainer、JSONL、resume 与 matching init |
| `groot/vla/configs/vggt_3d_wam.yaml` | 全局训练与可视化默认值 |
| `groot/vla/configs/model/vggt_3d_wam/encoder_decoder.yaml` | 模型结构与 loss 权重 |
| `groot/vla/configs/data/dreamzero/mobilemanibench_vggt.yaml` | train/val dataset |
| `scripts/data/qa_mobilemanibench_vggt_geometry.py` | 投影、coverage 与双时钟 QA |
| `scripts/train/mobilemanibench_vggt_training.sh` | 多卡训练、preflight、matching init |
| `scripts/eval/validate_vggt_3d_wam.py` | checkpoint 指标验证 |
| `scripts/eval/mobilemanibench_vggt_validate.sh` | 验证 shell 入口 |

## 5. 数据与几何代码

### `groot/vla/data/dataset/__init__.py`

新增 `MobileManiBenchVGGTDataset` 和 `MobileManiBenchVGGTDataCollator` 的 import 与
`__all__` 导出，使 Hydra 配置可以使用
`groot.vla.data.dataset.MobileManiBenchVGGTDataset` 这类稳定路径实例化对象。

### `groot/vla/data/dataset/mobilemanibench_vggt.py`

`MobileManiBenchVGGTDataset` 复用 `LeRobotSingleDataset` 的视频定位和时间戳读取，
一次加载 head/wrist 的 RGB 与 depth clip。`_build_split_indices()` 使用 seed 对
episode ID 做 90/10 划分，可防止同一 episode 的帧跨 split，但当前不读取
`meta/plan_splits.json`，因此不能保证同一 source trajectory 下的 sibling episodes
不跨 split。当前
`video_delta_indices=[0,...,32]`，只保留 episode 内能够提供完整连续 33 帧的 anchor；
不再用重复末帧补齐不完整 clip。

`__getitem__()` 从 parquet 读取 `observation.base.world` 和两个 camera pose，生成
`T_b0_camera`。内参从 `meta/calibration.json` 读取，并分别缩放到 backbone 输入分辨率
和 PointMap 分辨率。`_decode_distance()` 对三通道取中位数，按
`D = 5 * gray / 255` 解码，过滤近 0、近 255、边界和强深度边缘。未验证标定下
confidence 额外乘 `0.25`。

MobileManip 的 pose 采用 Isaac 相机 frame（X forward、Y left、Z up），深度反投影采用
OpenCV optical frame（X right、Y down、Z forward）。因此 train/val 和独立验证器统一
在相机 pose 后乘：

```text
T_camera_optical =
[[ 0,  0,  1, 0],
 [-1,  0,  0, 0],
 [ 0, -1,  0, 0],
 [ 0,  0,  0, 1]]
```

公共常量 `ISAAC_X_FORWARD_FROM_OPENCV` 定义在 dataset 文件中，避免训练、验证和 QA
各自维护一份坐标约定。

`MobileManiBenchVGGTDataCollator` 只负责 stack，不改变轴顺序或 dtype。

### `groot/vla/model/vggt_3d_wam/geometry.py`

- `pose_rpy_to_matrix()` 与 `invert_transform()`：构造和求逆刚体变换。
- `camera_rays()` / `rays_in_frame()`：从 pinhole K 生成单位 ray。
- `range_to_pointmap()`：把 camera-ray range 提升到 B0 PointMap。
- `project_points()`：把 metric query 投影为 `grid_sample` 归一化坐标和可见性。
- `metric_grid()`：生成按 `[Z,Y,X]` 排列的 voxel center。

## 6. 模型代码

### `groot/vla/model/vggt_3d_wam/configuration.py`

`VGGT3DWAMConfig` 继承 `PretrainedConfig`，集中保存 backbone、LoRA、2D latent、
metric voxel、PointMap decoder 和 loss 配置，确保 `save_pretrained()` /
`from_pretrained()` 可以完整恢复结构。生产配置通过
`patch_embed_type=dinov2_vitl14_reg`、`patch_size=14`、
`vggt_pretrain_image_size=518` 匹配源 checkpoint；单测可以切换到 `conv` patch
embed。tuple 参数统一转为 list，保证 Hugging Face JSON 序列化稳定。

本次新增/调整的关键字段为：

```text
global_temporal_window = 4
freeze_dino = true
dino_image_chunk_size = 4
backbone_gradient_checkpointing = true
latent_spatial_stride = 16
latent_temporal_stride = 4
video_decoder_dim = 256
```

### `groot/vla/model/vggt_3d_wam/backbone.py`

`VGGTBackbone` 默认使用 `timm` 的 `vit_large_patch14_reg4_dinov2` 读取 checkpoint
中的 DINOv2-L/14 前端，并恢复 VGGT camera/register special tokens。patch features
加入动态 2D patch、time 和 view sinusoidal 位置编码后，交替执行每帧 patch attention
和窗口化 global attention。生产配置使用 `global_temporal_window=4`，让 aggregator
在局部四帧窗口内融合跨视角/时间信息，避免完整 33 帧进入二次复杂度全局 attention；
完整 clip 时间建模继续由后续 2D/3D temporal branch 完成。tiny 单测可切换为轻量
conv patch 前端。

实现中的 global windows 是 `[0:4],[4:8]...`，而 `WanTemporalEncoder` 是首帧独立、
随后 `[1:5],[5:9]...`。所以二者宽度/stride 相同，但边界并非严格对齐；脚本中的
“Wan-aligned source-frame chunks”应理解为宽度对齐，而不是 exact boundary contract。

WAM 的 `160×320` 输入不能被 DINO patch size 14 整除。backbone 只在内部对右侧和
下侧补零到 `168×322`，得到 `12×23` patch grid；外部输入、2D latent 和 RGB
重建仍保持 `160×320 -> 10×20 -> 160×320`。3D sampling coordinates 使用 padding
后画布尺寸，因此无需移动相机主点；但 reference visibility 和 deformable offset
sampling validity 都限制在原始 `160×320` 有效区域，不会把右/下 padding strip 当成
真实图像证据。

本次进一步将 DINOv2 从“冻结原始权重但仍插入 LoRA”改为完全冻结：

```text
DINOv2 original parameters: frozen
DINOv2 LoRA: none
DINOv2 mode: always eval
DINOv2 forward: torch.no_grad()
DINOv2 image chunk size: 4
```

一个训练样本的 66 张图像按 4 张一组经过 DINO，输出立即 detach 后再交给 VGGT
aggregator。DINO 只提供逐图像 2D patch features；跨视角/时序适配由 frame/global
aggregator LoRA 完成，显式 3D 聚合由 `MetricTokenEncoder` 完成。

`load_vggt_checkpoint()` 使用 `weights_only + mmap`，只物化模型实际使用的权重页，
避免把无关 camera/depth/point/track heads 全部复制到 host heap。加载器会映射官方
`register_tokens` 命名，并把 DINO positional embedding 中单独保存的 class position
移除，以适配 `timm` 的 `no_embed_class` 布局。路径缺失时，除非显式
`init_random=true`，否则直接报错。

checkpoint 加载发生在 LoRA 注入前。冻结 backbone 后，`LoRALinear` 只注入
`frame_blocks` 和 `global_blocks` 的 `qkv/proj/fc1/fc2`，不会递归进入 DINO
`patch_embed`。frame/global blocks 在训练模式使用 `torch.utils.checkpoint` 且
`use_reentrant=False`，以重算换显存；这也允许输入 DINO features 已 detach 时仍正确
计算 LoRA 参数梯度。

ReconDrive 的原始实现会对整个 `aggregator` 递归调用 `apply_lora()`，因此事实上也会
进入 aggregator 内部的 DINO Transformer；但 ReconDrive 同时对 DINO、frame 和
global blocks 使用 activation checkpointing。本实现选择更严格的 DINO 完全冻结，
只复用其预训练 2D 表征。

### `groot/vla/model/vggt_3d_wam/temporal_codec.py`

新增 2D/3D 共用的 Wan-compatible 时间合同：

```text
wan_latent_time(T) = 1 + (T-1)/4
wan_video_time(T') = 1 + 4(T'-1)
33 -> 9 -> 33
```

`WanTemporalEncoder` 对首帧使用独立投影，对后续每组 4 帧使用 learnable stride-4
temporal convolution；`WanTemporalDecoder` 用独立首帧投影和 learnable channel-to-time
expansion 恢复后续 4 帧。temporal residual convolution 只做左侧 padding，归一化按
每个时空位置执行，不通过 normalization 泄漏未来信息。

### `groot/vla/model/vggt_3d_wam/video_latent.py`

`VideoLatentBranch` 执行 spatial bottleneck、逐空间位置 temporal transformer、
因果 temporal attention 和 learnable `33 -> 9` 压缩，再预测 `mu/logvar` 并重参数
采样。空间 bottleneck 自适应到严格的 `H/16 × W/16`，生产配置输出 48 channel。

`VideoDecoder` 不再使用时间插值加浅层逐帧卷积。它先执行 learnable `9 -> 33`
时间展开，再通过四级 learned spatial upsampling 恢复 16 倍空间尺寸，使用 `tanh`
输出与 Wan VAE 一致的 `[-1,1]` RGB 范围。

### `groot/vla/model/vggt_3d_wam/metric_tokens.py`

`MetricTokenEncoder` 把每个 token 绑定到固定 voxel center。本次不再使用每个 voxel
一组静态 `sample_offsets` 加普通 MHA 的简化聚合，而是参考 ReconDrive OCC GS head
改为迭代式 multi-view/multi-level deformable cross-attention：

```text
current voxel query
 -> per-head/per-level/per-point sampling offsets
 -> per-head level/point attention weights
 -> project B0 voxel center into every camera
 -> bilinear sample two-level VGGT feature pyramid
 -> masked weighted fusion over view/level/point
 -> residual voxel query
```

每层 offset 和 weight 均由当前 voxel query 动态预测。第一层由 voxel 位置引导；第一层
写入图像证据后，第二层的 offset 已经随当前输入变化。每个 deformable layer 后增加
`3×3×3` depthwise 3D local aggregation，使相邻 voxel 可以交换空间信息，再经过 FFN。
不可见相机和越界采样点在联合 softmax 前被 mask；全部不可见时图像增量为零并保留
residual query，不产生 NaN。

生产配置使用2个 deformable layers、2个 feature levels、8个 heads、每 head/level
4个 sampling points。这里使用 PyTorch `grid_sample`，不依赖 MMCV CUDA extension，
输出协议保持 `[B,T,N,C]`。

metric grid 固定在 clip 第1帧底盘坐标系 `B0`。基于 MobileManip 主要向前观察与操作的
数据假设，X 范围从 `[-1.5,3.0] m` 裁为 B0 前方 `[0.0,3.0] m`。当前 V2 grid 为
`[Z,Y,X]=[8,12,8]`：X/Y 划分保持 8/12，Z 从早期 4 层提高到 8 层，总 token 数为
768。该裁剪不会随未来底盘朝向旋转；若 clip 内明显掉头，B0 后方内容会被主动丢弃，
因此必须结合 coverage QA 解读。

由于 grid token 数和 decoder 形状已经改变，旧的 384/576-token checkpoint 不能作为
Trainer resume checkpoint。若需要复用旧训练结果，只能通过 `INIT_CHECKPOINT` 做
name-and-shape-compatible matching 初始化。当前训练脚本默认输出目录为
`work_dirs/mobilemanibench_5tasks_vggt_v2_savefix`。

本次将 geometry temporal transformer 改为 causal，并在每个 voxel 上复用
`WanTemporalEncoder`，使 3D tokens 也从 33 个 frame steps 压缩到 9 个 latent steps。
新增 `MetricTokenDecoder`，先把 `[B,9,N,C]` 学习式恢复为 `[B,33,N,C]`，再重排为
完整时间的 metric grid，供 PointMap decoder 渲染。2D/3D 因而共享相同 chunk 边界，
但保留独立的 appearance/geometry latent 分布。

### `groot/vla/model/vggt_3d_wam/pointmap_decoder.py`

decoder 为每个目标像素构造 B0 ray，在配置的 range bins 上采样 3D token volume。
`surface_head` 根据 voxel feature、候选 XYZ 和 ray direction 预测 depth-bin logits，
softmax 后对候选 B0 坐标加权求和。`ray_chunk_size` 限制峰值显存。

侧边任务新增独立 `occupancy_head`。它根据采样 voxel feature 和归一化 B0 XYZ 为
每个 ray bin 预测 occupancy logit；它不复用 `surface_head` 的 softmax，因此能够分别
监督表面之前的 free space 和表面附近的 occupied surface。decoder 同时返回
`ray_sample_valid`，网格外采样不会进入 occupancy loss。若一条 ray 的所有候选点都在
metric grid 外，PointMap 概率和坐标安全置零，不会产生 NaN。

### `groot/vla/model/vggt_3d_wam/model.py`

`VGGT3DWAMModel.forward()` 串起共享 backbone、2D 分支和 3D 分支，返回训练 loss、
latent、PointMap、occupancy logits 和可视化输出。当前总 loss 为：

```text
video_quality =
    1.0 * Charbonnier
  + 0.1 * LPIPS
  + 0.2 * SSIM
  + 0.1 * spatial gradient
  + 0.1 * temporal difference

total =
  video_quality
+ beta_2d * KL
+ warmup(0.4) * quality_weight(0.25) * (
    PointMap Huber
    + ray_weight * depth-bin CE
    + free_space_weight * (free-space BCE + surface-occupancy BCE)
    + multiview_weight * multiview consistency
    + temporal_weight * temporal geometry
    + normal_weight * surface normal
    + depth_gradient_weight * depth gradient
  )
```

PointMap coordinate loss 和 ray-surface loss 只监督 GT pseudo PointMap 落在
`[0,3]×[-2,2]×[-0.5,2] m` metric grid 内的像素。occupancy 监督沿相机 ray 生成：

```text
range < pseudo_surface - margin    -> free，target occupancy = 0
|range - pseudo_surface| <= margin -> surface，target occupancy = 1
range > pseudo_surface + margin    -> unknown，不监督
```

默认 margin 为 `0.1 m`。64 个 range bin 覆盖 `[0.05,5.0] m`，相邻 bin 约
`0.0786 m`，因此表面带通常覆盖约 2--3 个离散采样点。该目标是 coarse
free-space/surface supervision，不是完整 occupancy ground truth。

multi-view consistency 使用 GT pseudo PointMap 建立跨相机几何对应：先把源视角
B0 点投影到目标视角，再用目标视角 GT 的距离差做 visibility/occlusion gate，最后比较
源、目标视角的预测 PointMap。该项不使用同像素直接对齐，几何定义是合理的；但其可靠性
仍受 MP4 depth 和未验证相机标定约束。

可选 `masked_view_probability` 会遮掉输入视角，但仍要求重建完整 clip。
`save_pretrained()` 会移除源 VGGT 路径并保存完整 state，使训练产物可以独立加载。

模型新增明确的 tokenizer 公共接口：

```text
encode_2d(video)
decode_2d(z_2d)
encode_3d(video, camera_K, T_b0_camera)
decode_3d(z_3d, camera_K, T_b0_camera)
encode(video, camera_K, T_b0_camera)
```

`encode_2d/decode_2d` 同时接受 Wan 单视图布局 `[B,C,T,H,W]` 和本项目多视图布局。
Wan 布局下生产配置严格执行：

```text
[B,3,33,160,320]
  -> [B,48,9,10,20]
  -> [B,3,33,160,320]
```

联合 `encode()` 只运行一次共享 backbone，同时返回 `z_2d_video` 和
`z_3d_video`，避免两个独立 API 重复计算。

### `groot/vla/model/vggt_3d_wam/visualization.py`

`save_vggt_visualization()` 把同一 batch、同一 global step 的 loss 和重建结果写入：

```text
visualizations/{train|val}/step_XXXXXXXX/sample_XXX/
├── reconstruction_pointmap.png
├── pointmap_3d_scatter_b0.png
└── metrics.json
```

contact sheet 按 `time × view` 展示 RGB GT/reconstruction 和 PointMap GT/prediction。
误差列、独立 confidence 列和图顶总 loss 已移除；confidence 仍用于 3D scatter 的
有效点筛选。由于 decoder 输出 `[-1,1]` RGB，可视化前会转换回 `[0,1]`。

`pointmap_3d_scatter_b0.png` 不是独立的点云预测：它只是把
`predicted_pointmap_b0[..., Hq, Wq]` 中具有有效监督的像素展平为 `[N,3]` 后，在
B0 坐标系绘制 GT/预测散点。decoder 仍然只输出有组织的 robot-centric PointMap，
没有额外 point-cloud head 或 point-cloud loss。

### `groot/vla/model/vggt_3d_wam/__init__.py`

公开 `VGGT3DWAMConfig` 和 `VGGT3DWAMModel`，但通过 `__getattr__()` 延迟 import。
这样数据集或几何工具进程不会仅因导入 package 就初始化 `PreTrainedModel` 及可选
DeepSpeed 依赖。

## 7. 训练、配置与验证

### `groot/vla/experiment/vggt_3d_wam.py`

该入口使用 Hydra 实例化模型、dataset、collator 和
`transformers.TrainingArguments`。`VGGTTrainer` 按 DreamZero 的 batch-dict 调用模型，
记录各分支 loss，并把 HF global step 写入模型以驱动几何 loss warmup。训练和验证
可视化只在 world rank 0 执行；训练按 `visualization.train_interval` 保存，验证每次
evaluate 重置计数并受 `val_max_samples` 限制。绘图失败默认 warning，不中断训练；
`fail_on_error=true` 可改为严格模式。

新增 `VGGTJSONLLossLoggerCallback`。world rank 0 每次收到 Trainer `on_log` 事件时，
把以下标量以追加方式写入 `OUTPUT_DIR/loss_log.jsonl`：

```text
step
epoch
loss / eval_loss
learning_rate
grad_norm
video_recon_loss_avg
video_lpips_loss_avg
video_ssim_loss_avg
video_spatial_gradient_loss_avg
video_temporal_difference_loss_avg
video_quality_loss_avg
kl_2d_loss_avg
pointmap_loss_avg
ray_surface_loss_avg
free_space_loss_avg
surface_occupancy_loss_avg
multiview_consistency_loss_avg
temporal_geometry_loss_avg
surface_normal_loss_avg
depth_gradient_loss_avg
weighted_geometry_loss_avg
```

文件采用一行一个 JSON object 的格式，与 WAM loss 日志的离线分析方式一致。断点
恢复不会覆盖原文件；只有 global rank 0 写入，避免 DDP 多进程重复记录。各分支
`*_avg` 仍是最近10次 forward 的滑动平均，并按当前 Trainer 逻辑每10 step写入一次。

训练器按参数来源建立 4 个 AdamW parameter groups（backbone/head × decay/no-decay）：
预训练 VGGT aggregator 中实际可训练的 LoRA 使用独立低学习率，随机初始化的 2D/3D
codec、decoder 和 geometry heads 使用主学习率。scheduler 对两类 LR 使用相同倍率，
不会把两组重新覆盖成同一个值。日志额外记录
`backbone_learning_rate/heads_learning_rate`。

按上述独立 episode split，五任务 validation 有 24,197 个可用 33-frame clips。
训练入口使用确定性等间隔
索引将训练内 validation 限制到 `max_eval_samples=256`，避免周期性完整验证；
完整集评估仍由独立验证脚本按需运行。

这个 validation 适合监控收敛，但不是与 WAM 完全相同的 trajectory-group held-out
集合。正式比较 tokenizer 泛化或 downstream 收益前，应让 VGGT dataset 接受
`split_manifest_path` 并复用 `plan_splits.json`。

入口还会把解析后的 Hydra 配置写到 `resolved_config.yaml`，并生成
`geometry_quality.json` 标注 pseudo PointMap 的质量边界。启动时自动检测
`output_dir` 中最后一个 HF checkpoint 并续训，结束后保存模型和 Trainer state。

### `groot/vla/configs/vggt_3d_wam.yaml`

组合 model/data 配置并定义 Hydra 基础默认值：输入 `160×320`、PointMap `80×160`、
完整 33 帧偏移 `[0,...,32]`、latent 时间长度 9、batch size 1、
`max_steps=5000`、每 500 step 保存和验证。
优化器为 AdamW：新建 heads 使用 `5e-5`，VGGT aggregator LoRA 使用 `2e-5`；
前 2% warmup，随后使用带 `0.2` 最低倍率的 cosine scheduler。
启用 BF16、TF32、pinned memory。可视化默认开启：训练每 1000 step 保存一次，验证
每 5000 step 最多保存 4 个样本。数据根目录和输出目录保留为必填项，VGGT 权重默认
指向用户提供的 `model.pt`。

正式 shell launcher 会覆盖部分基础值：`max_steps=30000`、save/eval 每 5000 step、
warmup ratio `0.01`。可视化间隔仍来自 YAML：train 1000、val 5000。

### `groot/vla/configs/model/vggt_3d_wam/encoder_decoder.yaml`

定义生产模型规模：DINOv2-L/14 register-token patch embed、24 层 1024 维 backbone，
冻结基础权重并在 attention/MLP 上使用 rank-8 LoRA。2D latent 为 48 维；3D 分支使用
`[8,12,8]` B0-forward metric grid、256 维 token、64 个 ray depth bins 和
128-ray chunk。ray rendering 为 `40×80`，learned refinement 输出 `80×160`。
PointMap/geometry 总权重为 `0.4`，在 2000 step 内 warmup；ray loss 相对权重为
`0.1`。五任务 QA 后新增 `geometry_quality_weight=0.25`，用于对仍未完成官方内参/
深度标定的整个 geometry objective 做显式整体降权；warmup 完成后的有效 geometry
系数为 `0.4 × 0.25 = 0.1`。关键 3D 监督参数为：

```text
free_space_loss_weight = 0.1
free_space_surface_margin = 0.1 m
multiview_consistency_loss_weight = 0.05
multiview_occlusion_threshold = 0.15 m
```

时空接口固定为：

```text
global_temporal_window = 4
latent_spatial_stride = 16
latent_temporal_stride = 4
video_decoder_dim = 256
```

### `groot/vla/configs/data/dreamzero/mobilemanibench_vggt.yaml`

分别实例化 train/val dataset。两者共享 episode split seed、图像尺寸、PointMap 尺寸、
range 解码阈值和相机 optical transform；训练按 `sample_stride` 采样，验证使用独立的
`validation_sample_stride`。该配置同时实例化只做 batch stack 的 collator。

### `scripts/train/mobilemanibench_vggt_training.sh`

提供单机 `torchrun` 多卡入口，并在启动前检查 DreamZero 环境中的 `torchrun`、四个
dataset metadata 文件、LPIPS 依赖和 VGGT checkpoint。`PREFLIGHT_ONLY=1` 只执行检查；
`INIT_RANDOM=1` 可显式跳过预训练权重。`NUM_GPUS`、`MAX_STEPS`、`SAVE_STEPS`、
`EVAL_STEPS`、两组学习率、warmup、最低 LR 倍率、训练内验证样本数、batch size、
梯度累积和 WandB 模式均可由环境变量覆盖。
`INIT_CHECKPOINT` 可从旧 DreamZero tokenizer checkpoint 只加载名称和形状匹配的
trainable 参数；空值会完全跳过。
默认启动信息会打印 `33x160x320 -> 9x10x20 -> 33x160x320`，便于检查接口。
同时打印 DINO 完全冻结/分块 no-grad 和 aggregator LoRA/checkpointing 状态。

训练前检查示例：

```bash
PREFLIGHT_ONLY=1 \
bash scripts/train/mobilemanibench_vggt_training.sh
```

正常训练默认使用当前服务器上的权重，也可通过环境变量覆盖：

```bash
VGGT_CHECKPOINT_PATH=/mnt/yihao/codes/ReconDrive/checkpoints/model.pt \
  NUM_GPUS=8 \
  bash scripts/train/mobilemanibench_vggt_training.sh
```

### `scripts/eval/validate_vggt_3d_wam.py`

从 Hugging Face checkpoint 恢复独立模型，在 validation split 上以 inference mode
计算 RGB MAE、PointMap coordinate MAE、PointMap Euclidean error 和有效监督权重。
CUDA 不可用时自动退回 CPU；CUDA 推理使用 BF16 autocast。`_checkpoint_step()` 优先
解析 `checkpoint-N` 目录名，其次读取 `trainer_state.json`，使离线可视化沿用真实
global step。`--visualization-root` 与 `--max-visualizations` 控制诊断图输出，最终
指标可打印并保存为 JSON。

### `scripts/eval/mobilemanibench_vggt_validate.sh`

封装验证器的常用环境和默认参数。`CHECKPOINT` 为必填目录；`OUTPUT` 默认写入
checkpoint 下的 `validation_metrics.json`，`VISUALIZATION_ROOT` 默认也是 checkpoint，
并默认验证 100 个样本、保存前 4 个可视化。运行方式：

```bash
CHECKPOINT=/path/to/checkpoint \
  bash scripts/eval/mobilemanibench_vggt_validate.sh
```

## 8. 测试代码

### `tests/data/test_mobilemanibench_vggt_dataset.py`

在真实 smoke dataset 存在时检查 RGB、内外参、pseudo PointMap 和 confidence 的 shape、
dtype、finite 状态及 `≤0.25` 的未验证标定降权；同时检查 collator 增加 batch 轴后
不改变其余维度协议。数据不存在时通过 `skipUnless` 跳过。

### `tests/model/test_vggt_geometry.py`

验证 RPY 刚体变换与其逆矩阵相乘为单位阵，并验证固定 range 生成的 PointMap 可以用
同一内外参投影回原像素中心，覆盖坐标系和 `grid_sample` 归一化约定。

### `tests/model/test_vggt_3d_wam.py`

使用轻量 conv patch embed 配置覆盖三条路径：官方 DINO checkpoint 中 position/register
token 的命名和 shape 适配；完整 forward/backward 的 latent、重建和 PointMap shape
及关键梯度；保存后的 checkpoint 清除源 VGGT 文件路径并可独立恢复。

### `tests/model/test_vggt_visualization.py`

构造 tiny batch，检查 contact sheet、B0 PointMap 3D scatter 和 `metrics.json` 同目录
生成且非空，并核对 step、split、loss 及实际展示的 time/view indices。

### `tests/model/test_vggt_3d_wam_tokenizer_contract.py`

覆盖 `33 -> 9 -> 33` 时间长度与 backward gradient、因果 chunk 边界、原生多视图
2D/3D encode/decode、Wan 单视图 `[B,3,T,H,W]` drop-in 布局，以及完整训练 forward
的 loss finite 和两路 latent 时间维一致性。新增 backbone 训练范围测试，确保
`patch_embed` 没有可训练参数，trainable backbone 参数只属于 frame/global LoRA，
并覆盖 activation-checkpoint backward。侧边任务又新增 occupancy 输出 shape、
free-space/surface loss backward、multiview 几何对应和 deformable 全不可见 fallback
测试。

其余当前回归文件：

```text
tests/model/test_vggt_quality_losses.py
tests/model/test_vggt_v2_smoke.py
tests/model/test_vggt_matching_checkpoint.py
tests/experiment/test_vggt_trainer_validation.py
```

分别覆盖 V2 视觉/3D loss、768-token shape、matching-only 初始化、LPIPS
checkpoint 保存、残缺 checkpoint 跳过，以及 save-before-eval/validation batch
接口。

## 9. 已执行验证

截至 2026-07-30 已完成以下代码级检查：

- 所有新增 Python 文件通过 `py_compile`。
- 几何、temporal codec、tokenizer contract 和 V2 shape 回归通过。
- 可视化单元测试：1/1 通过。
- 真实 MobileManiBench smoke dataset：1/1 通过。
- tiny 真实数据 HF Trainer：完成 1 个 forward、backward 和 optimizer step。
- 保存后的 checkpoint 可由独立验证脚本加载并输出指标；LPIPS shared tensor 不再进入
  safetensors。
- tiny HF Trainer step 已自动生成 contact sheet、PointMap 3D scatter 和同步 loss JSON。
- matching-only 初始化和空路径跳过通过。
- 不完整 checkpoint 会被自动忽略，只从完整 checkpoint resume。
- `git diff --check` 通过。

针对 `/mnt/yihao/codes/ReconDrive/checkpoints/model.pt` 还执行了低影响集成测试：

- checkpoint 大小约 5.03 GB，全程设置 `CUDA_VISIBLE_DEVICES=""`、单 CPU 线程、
  `nice=19` 和 idle I/O priority。
- mmap/FakeTensor 检查中，生产 backbone 的 921/921 tensors 和 100% parameter
  numel 精确匹配；检查峰值 RSS 约 614 MiB。
- 实际物化 DINOv2 前端和 1 对 frame/global block 后，369/369 tensors 匹配；
  抽查 `frame_blocks.0.attn.qkv.weight` 与源 checkpoint 逐元素完全相等。
- `28x28` tiny forward 输出 `[1, 1, 1, 1024, 2, 2]`，全部 finite，单线程 forward
  约 0.56 秒；该进程峰值 RSS 约 3.1 GiB，未占用 GPU。

本次 Wan-compatible 改造额外验证：

- temporal codec 完成 `[2,8,33,2,3] -> [2,8,9,2,3] -> [2,8,33,2,3]`；
- temporal codec backward gradient 与因果 chunk 边界通过；
- tiny native multi-view、Wan 单视图 drop-in 和完整训练 forward 均通过；
- 真实五任务数据连续读取 33 帧，RGB、相机参数和 PointMap 时间维完全一致；
- Hydra resolve 确认 `num_frames=33`、`latent_frames=9`、
  `latent_spatial_stride=16`、`latent_temporal_stride=4`；
- 实际 DINOv2-L/14 前端确认外部 `160×320` 内部 padding 为 `168×322`，
  输出 patch grid `12×23`；
- 实际 DINOv2-L/14 结构确认 `model.train()` 后 DINO 仍为 eval、可训练参数为 0；
- depth=1 生产结构中只有 frame/global LoRA 可训练，共 262144 参数、16 个 tensors；
- LoRA 范围检查和 activation-checkpoint backward 通过；
- `PREFLIGHT_ONLY=1` 训练前检查通过。

这些结果证明接口和训练链路成立，不代表 30,000-step tokenizer 已达到目标视觉或
几何质量；收敛判断仍需结合 held-out validation、可视化和 downstream WAM 指标。

## 10. 2026-07-27 侧边任务 3D 监督合并审核

> 本节及后续按时间记录问题发现和修复过程，保留的旧 shape、权重和输出目录不是当前
> 默认值。阅读当前实现请以第 2–7 节及最新的 `encoder_decoder.yaml` 为准。

### 10.1 审核结论

本次审核以远程服务器
`/mnt/yihao/codes/dreamzero` 的当前文件为准，没有用本地旧副本覆盖远程代码。

结论分为两部分：

1. **训练主链路在结构上成立。** 新增 occupancy head、free-space/surface loss 和
   multiview consistency loss 已接入总 loss，tiny forward/backward 证明新增监督可以
   向 occupancy head 和 3D voxel feature 回传梯度。新增输出也会被现有
   `*_loss` 自动收集逻辑写入 `loss_log.jsonl`。
2. **当前还不能把离线验证结果当作可信的 3D 质量结论。** 验证脚本存在时间窗口、
   RGB 数值范围和 metric-grid mask 三处接口问题，并且没有统计新增监督的有效覆盖率。
   训练可以继续做链路/收敛实验，但在修复验证器和补充 coverage 指标前，不宜根据
   validation JSON 判断 3D 分支是否真正学好。

### 10.2 侧边任务逐文件改动

#### `groot/vla/model/vggt_3d_wam/pointmap_decoder.py`

- 新增 `occupancy_head`，对每个 ray-depth sample 输出独立 occupancy logit。
- head 输入为 voxel sample feature与归一化 XYZ；PointMap surface head 仍额外使用
  ray direction。
- forward 新增返回 `occupancy_logits` 和 `ray_sample_valid`。
- 全部采样点无效时使用安全 fallback，避免 masked softmax 产生 NaN。

#### `groot/vla/model/vggt_3d_wam/model.py`

- `decode_3d()` 和 `forward()` 暴露 occupancy 输出。
- PointMap 与 ray-surface loss 新增 metric-grid 内部 mask，避免要求有限 voxel grid
  拟合其表达范围外的目标点。
- 新增表面前 free-space BCE、表面带 surface-occupancy BCE。
- 新增基于投影、GT overlap 与 occlusion gate 的跨视角 PointMap consistency。
- 三项新增 loss 已纳入 geometry warmup 和 geometry 总权重内部。

#### `groot/vla/model/vggt_3d_wam/configuration.py`

新增并持久化：

```text
free_space_loss_weight
free_space_surface_margin
multiview_consistency_loss_weight
multiview_occlusion_threshold
```

#### `groot/vla/configs/model/vggt_3d_wam/encoder_decoder.yaml`

为上述四项配置生产默认值：`0.1 / 0.1 m / 0.05 / 0.15 m`。

#### `tests/model/test_vggt_3d_wam_tokenizer_contract.py`

新增 occupancy shape、loss finite、free-space backward、multiview correspondence 和
deformable invisible fallback 检查。

#### `groot/vla/experiment/vggt_3d_wam.py`

该文件没有为每一种新 loss 写死字段，而是统一收集模型输出中所有以 `_loss` 结尾的
标量。因此新增三项会自动生成：

```text
free_space_loss_avg
surface_occupancy_loss_avg
multiview_consistency_loss_avg
```

这也是本次文档需要补充日志字段、但训练 logger 不需要再次修改的原因。

### 10.3 按严重程度列出的审核问题

#### P0：现有验证脚本与当前 tokenizer 时间合同不一致

**状态：已于第一批修复中解决。**

`scripts/eval/validate_vggt_3d_wam.py` 默认：

```text
--video-delta-indices 0,8,16,24
```

即只加载 4 帧；当前 2D/3D tokenizer 强制要求输入时间长度满足 `4k+1`，生产训练窗口
为 33 帧。shell wrapper 没有覆盖这个默认值，所以按当前默认验证命令无法与训练合同
一致。验证应使用与训练完全相同的 33 帧窗口 `[0,...,32]`。

#### P0：验证 RGB MAE 的数值范围不一致

**状态：已于第一批修复中解决。**

验证脚本把模型 `reconstructed_video` 的 `[-1,1]` 直接与 GT 的 `[0,1]` 相减。
因此当前 `video_mae` 数值没有正确物理含义。计算指标前应先执行：

```text
pred_rgb_01 = (reconstructed_video + 1) / 2
```

#### P1：验证 PointMap 没有复用训练时的 metric-grid mask

**状态：已于第一批修复中解决。**

训练只监督落在 B0 grid 内的目标点；验证脚本却对所有 `pointmap_valid` 像素统计误差。
这会把模型结构上无法表示的网格外目标也算作预测失败，从而系统性抬高 PointMap error。
验证器必须复用与 `_pointmap_losses()` 相同的 inside-grid 条件，并同时报告：

```text
raw valid weight
inside-grid valid weight
inside-grid coverage ratio
```

#### P1：新增 3D loss 缺少有效监督覆盖率

**状态：已于第一批修复中解决。**

`_weighted_mean()` 在权重和为零时返回 0。因而：

```text
free_space_loss = 0
surface_occupancy_loss = 0
multiview_consistency_loss = 0
```

既可能表示预测完美，也可能表示当前 batch 没有任何有效 sample/correspondence。现有
日志只记录 loss，不记录分母。至少应补充 free samples、surface samples、multi-view
correspondences、all-invalid rays 和 inside-grid target ratio；否则无法判断新增监督
是否实际生效。

#### P1：未验证标定的 `0.25` confidence 不会降低整体 geometry 梯度

数据集在 `nominal_unverified` 标定下把 confidence 统一乘 `0.25`。但各项 loss 使用
`sum(error * weight) / sum(weight)`，统一缩放会在分子和分母中抵消。因此该处理只影响
不同像素之间的相对权重和有效 mask，**不会把整个 3D loss 降到四分之一**。

当前真正控制粗糙 pseudo-3D 监督全局强度的是 `pointmap_loss_weight=0.1`。如果设计意图
是标定未验证时进一步整体降权，需要显式的 sample/global quality multiplier；不能依赖
当前 confidence 常数缩放。

#### P1：相机标定仍是 3D 监督的主要质量上限

**状态：光学轴约定已由第二批 QA 修复；官方 K、深度尺度及 TCP 标定仍未验证。**

抽查 smoke 数据的 `meta/calibration.json`：

```text
status = nominal_unverified
K_verified = false
optical_frame_transform_verified = false
tcp_transform_verified = false
camera_optical_transform = null / identity fallback（转换产物中的历史状态）
```

第二批 QA 已通过 24 个右手坐标轴变换搜索和 head/wrist 重投影，确定上面的
Isaac-to-OpenCV 变换，并写入运行时 dataset config；没有改写历史数据文件。修正后
15 cm correspondence candidate ratio 从 identity 的约 `13.3%` 提升到约 `84.6%`。
但 K 仍为 nominal、MP4 depth 仍为有损 pseudo-range，因此 PointMap、free-space 和
multiview targets 仍只能视为 coarse pseudo labels。

#### P1：旧模型测试没有同步严格的时间/空间压缩合同

**状态：已于第一批修复中解决。**

直接运行 `tests/model/test_vggt_3d_wam.py` 时，2 个模型测试在构造阶段报错：

```text
ValueError: VGGT 2D latents must use Wan's temporal stride 4; got 2
```

其 `tiny_config()` 仍使用 `latent_temporal_stride=2` 和
`latent_spatial_stride=1`。这不是当前生产模型 forward 的错误，而是旧测试 fixture
过期；但在修复前，仓库不能宣称 VGGT 全部测试通过。

#### P2：B0 前向 grid 假设缺少数据覆盖统计

**状态：已于第二批 QA 中解决。**

所有时间帧共享第 1 帧底盘坐标系 B0 的固定网格，这是正确的时间组织；但
`X=[0,3] m` 只保留 B0 前方。机器人在 33 帧内转向后看到的内容不保证仍位于 B0 前方。
应按帧统计 GT 点的 `x<0` 比例、inside-grid ratio 和全空 ray 比例，验证前向裁剪确实
符合五任务数据分布。

#### P2：DINO padding 边缘可能被 3D 投影采样

**状态：已于第二批修复中解决。**

输入 `160×320` 被 DINO 在右/下补到 `168×322`。投影使用 padding 后尺寸，左上原点
和主点不变。当前 `project_points()` 额外接收 `valid_image_size=(160,320)`；reference
point 先按原始像素范围判可见，deformable offsets 也使用由原图/补齐图比例计算出的
normalized upper bound，因此不能进入右侧或底部 padding strip。

#### P2：checkpoint 兼容性再次发生变化

新增 `occupancy_head` 后，侧边任务修改前生成的 checkpoint 不包含该参数。它可以作为
部分初始化来源，但不应被视为严格等价的完整恢复；尤其不能假定旧 optimizer state 能
无缝续训新增参数。旧 576-token checkpoint 仍因 voxel 数和 deformable 结构变化而
不兼容。当前结构应使用新的输出目录从干净 checkpoint 开始。

#### P2：train/val 与媒体时间边界

**状态：已于第二批 QA 中量化并明确使用约定。**

- 数据集按 episode 做确定性 10% validation split，不会发生同 episode 相邻帧泄漏。
- 单 episode 数据集的 validation split 为空，不能用来验证泛化，只能在 train split
  做过拟合闭环。
- action/state 按 30 Hz 解释，MP4 编码为 25 FPS，但转换数据显式保存
  `timestamp=frame_index/25` 用于媒体 seek、`control_timestamp=frame_index/30`
  用于物理控制时间。loader 使用前者，视频帧与状态行保持一一对应；不可把 MP4 播放
  时长当成物理控制时长，也不能改用 control timestamp seek 视频。

### 10.4 本次实际执行的检查

在远程 DreamZero 环境执行：

```text
tests/model/test_vggt_geometry.py                 2/2 PASS
tests/model/test_vggt_visualization.py            1/1 PASS
tests/data/test_mobilemanibench_vggt_dataset.py   1/1 PASS
compileall                                        PASS
```

在不安装额外依赖的情况下直接调用 tokenizer contract 中与侧边改动相关的测试函数：

```text
training forward finite                           PASS
free-space/surface occupancy backward             PASS
multiview geometric correspondence                PASS
deformable all-invisible fallback + backward      PASS
```

当前远程环境未安装 `pytest`，所以没有执行完整 pytest suite。直接运行旧
`tests/model/test_vggt_3d_wam.py` 得到 3 个测试中 1 个通过、2 个因旧 stride fixture
报错。没有抢占 GPU 执行生产规模 33 帧 forward，也没有启动或修改正在进行的训练。

### 10.5 建议的修复与验收顺序

```text
1. 修复验证器 33-frame 输入与 RGB [-1,1] -> [0,1]
2. 验证 PointMap 复用 inside-grid mask
3. 为 free/surface/multiview 增加 coverage/count 日志
4. 更新旧 tiny test 的 stride=4/16 合同并跑完整测试
5. 抽样检查 head/wrist 投影、B0 前向覆盖和 multiview correspondence
6. 再做生产规模 1--2 optimizer-step GPU smoke
7. 最后才依据 validation 指标比较 loss 权重和 3D 结构
```

第 1--4 步已经由第一批修复完成，详见第 11 节。验证数值接口和 coverage 现在可用；
但在相机标定、B0 覆盖和时间同步 QA 完成前，仍不应把 PointMap 的绝对误差解读为
高精度 3D 重建能力。

## 11. 第一批问题修复记录

### 11.1 验证时间与 RGB 合同

`validate_vggt_3d_wam.py` 和 shell wrapper 默认并强制使用连续
`[0,...,32]` 共 33 帧。启动时打印实际合同：

```text
video_frames=33
latent_frames=9
temporal_stride=4
```

非 33 帧连续窗口会在构造 dataset 前直接报错。RGB MAE 现在先把
`reconstructed_video` 从 `[-1,1]` 转换并 clamp 到 `[0,1]`，再与
`video / 255` 比较。

验证器新增 `--split train|val|all`，默认仍为 `val`；单 episode 过拟合数据可以显式
使用 `SPLIT=train` 或 `SPLIT=all`，但相应结果不代表泛化能力。

### 11.2 训练和验证共用 metric-grid 判定

新增公共函数：

```text
groot/vla/model/vggt_3d_wam/geometry.py
└── points_in_metric_grid()
```

模型 PointMap loss、multiview consistency 和验证器均调用同一个 inclusive XYZ bounds
实现。验证主 PointMap 指标只使用 `inside-grid` 目标，同时保留 raw/inside count、
weight 和 coverage ratio，避免有限 voxel grid 被网格外目标不公平惩罚。

### 11.3 3D supervision coverage 日志

模型新增以下标量输出：

```text
pointmap_raw_valid_count
pointmap_inside_grid_count
pointmap_raw_valid_weight
pointmap_inside_grid_weight
ray_total_sample_count
ray_valid_sample_count
ray_supervised_pixel_count
free_space_sample_count
surface_sample_count
multiview_candidate_count
multiview_correspondence_count
multiview_correspondence_weight
```

Trainer 每个 forward 把所有 scalar loss/count/weight 打包为一次 DDP all-reduce：
loss 按 world size 取平均，count/weight 跨卡求和。日志窗口据聚合后的 numerator 和
denominator 计算：

```text
pointmap_inside_grid_ratio
pointmap_inside_grid_weight_ratio
ray_valid_ratio
ray_supervised_pixel_ratio
free_space_sample_ratio
surface_sample_ratio
multiview_correspondence_ratio
```

JSONL callback 现在保留所有 `*_avg` 和 `*_ratio`，所以 count、weight、coverage 会与
loss 一起写入 `loss_log.jsonl`。

### 11.4 ray-surface 有效 bin 修复

第一批回归测试额外发现：旧实现先在全部 range bins 中选择最近 target bin，再计算
masked logits CE；如果最近 bin 在 voxel grid 外，可能得到 `inf`。

当前实现只在 `ray_sample_valid=true` 的 bins 中选择最近标签。没有任何有效 bin 的
像素把 ray error 安全置零且不进入 loss，并通过 `ray_supervised_pixel_count/ratio`
显示实际获得 ray-surface 监督的目标数量。

### 11.5 旧测试更新

`tests/model/test_vggt_3d_wam.py` 已更新为：

```text
latent_temporal_stride = 4
latent_spatial_stride = 16
video frames = 9
latent frames = 3
```

随机 XYZ fixture 改为由相机 ray 与固定 range 生成的物理有效 PointMap；decoder gradient
断言同步到当前 `output_projection`。几何测试新增 metric-grid 边界检查，tokenizer
contract 测试新增 coverage、logger 和 DDP diagnostic 收集检查。

### 11.6 本次回归结果

远程环境实际执行：

```text
compileall                                      PASS
test_vggt_geometry.py                           3/3 PASS
test_vggt_3d_wam.py                             3/3 PASS
tokenizer training forward finite               PASS
free/surface occupancy backward                 PASS
multiview correspondence                        PASS
deformable invisible fallback                   PASS
JSONL count/ratio field smoke                    PASS
```

还使用 tiny checkpoint 和真实
`MobileManipVLA_dreamzero_smoke_v2/g1` 数据完成 1 个样本的完整 CPU 验证：

```text
video frames                         33
latent frames                         9
pointmap inside-grid count       355 / 520
ray supervised pixels           355 / 355
multiview correspondences          2 / 58
```

上述数值来自随机初始化 tiny 模型，只用于证明指标链路和分母可观测，不代表模型效果。
校验过程没有启动训练，也没有占用生产 GPU。

## 12. 第二批数据质量 QA 与修复记录

### 12.1 新增 QA 工具与运行产物

新增：

```text
scripts/data/qa_mobilemanibench_vggt_geometry.py
```

工具一次模型无关的运行即可完成：

1. 按任务抽取连续 33 帧 head/wrist RGB、depth、相机 pose 和 base pose；
2. 统计所有时刻、每个视角和每个时间 offset 的 B0 voxel coverage；
3. 做 head/wrist 双向重投影、遮挡门控和多距离阈值 correspondence 统计；
4. 搜索 24 个右手相机轴变换，检查 pose frame 与 optical frame 约定；
5. 从 parquet 的 `timestamp/control_timestamp` 和 MP4 的 ffprobe 信息分别检查媒体
   时钟与控制时钟；
6. 输出 `qa_report.json` 和可人工检查的投影误差 PNG。

五任务、50 个随机 clip 的最终报告位于：

```text
work_dirs/vggt_geometry_qa_5tasks_isaac_optical/qa_report.json
work_dirs/vggt_geometry_qa_5tasks_isaac_optical/projection_sample_*.png
```

### 12.2 相机投影与 multiview QA

在 24 个 right-handed axis rotations 中，最佳变换为：

```text
OpenCV optical (right, down, forward)
        -> Isaac camera pose (forward, left, up)

[[ 0,  0,  1],
 [-1,  0,  0],
 [ 0, -1,  0]]
```

同一抽样下的关键对比：

| 指标 | identity | Isaac-to-OpenCV |
|---|---:|---:|
| B0 inside-grid ratio | 65.0% | 91.7% |
| 15 cm candidate correspondence ratio | 约 13.3% | 84.6% |
| 15 cm source-point correspondence ratio | 约 4.3% | 39.7% |
| 有效 correspondence 平均距离 | 约 0.77 m | 0.126 m |

最终修正配置下，多阈值 candidate ratios 为：

```text
 5 cm: 64.9%
10 cm: 77.8%
15 cm: 84.6%
25 cm: 89.1%
```

投影图中共享静态表面总体对齐；较大误差主要位于遮挡边缘、机器人自遮挡和 pseudo-depth
强边缘处。结论是：**相机 optical axis 的系统性错误已经修复，但该结果不能证明 nominal
K 和 MP4 depth 已达到 metric-accurate 标定。**

### 12.3 B0 voxel 覆盖率

修正 optical transform 后，当前生产网格：

```text
X = [0, 3] m
Y = [-2, 2] m
Z = [-0.5, 2] m
grid = [4, 12, 8] = 384 voxels
```

得到：

| 范围 | inside-grid ratio |
|---|---:|
| 全部视角与全部时刻 | 91.73% |
| head | 94.20% |
| wrist | 89.14% |
| offset 0 | 92.35% |
| offset 32 | 90.94% |

越界分解中 `x < 0` 约为 `0.004%`，说明“B0 第 1 帧前方”假设成立；主要越界来自
`x > 3 m`（约 5.51%）与 `z > 2 m`（约 3.56%）。coverage 在 33 帧窗口末端仍超过
90%，因此本批不扩展到 B0 后方，也不增加 voxel 数。后续若任务集改变，应重新运行该
统计，而不是把本结论外推到所有 MobileManip 任务。

### 12.4 25/30 FPS 双时钟误差

抽查 10 条 episode：

```text
parquet rows == MP4 frame count
MP4 fps = 25
timestamp[i] = i / 25
control_timestamp[i] = i / 30
loader seek timestamp = timestamp
```

媒体和控制 timestamp 相对于各自理论 index 的 MAE 都在浮点舍入量级
（约 `1e-15`），并没有 frame-index 漂移。当前 loader 用 `timestamp=i/25` 读取 MP4，
因此视频、状态和 action 的第 i 行仍严格按 index 对齐。

但 33 帧窗口在媒体播放时钟中跨度为 `32/25=1.28 s`，在控制物理时钟中跨度为
`32/30=1.0667 s`，两者相差 `0.2133 s`。如果错误地用 control timestamp seek 25 FPS
视频，窗口末端会约早取 5 帧。因此本次不重采样视频、不修改现有 loader，只明确：

```text
timestamp         仅用于媒体 frame seek
control_timestamp 用于 action/state 的物理时间解释
```

### 12.5 DINO padding visibility 修复

DINOv2 把 `160×320` 在右/下 padding 到 `168×322`。修复后：

- `project_points()` 仍用 padded canvas 生成 `grid_sample` 坐标，保证 patch grid
  对齐，但使用原图 `valid_image_size=(160,320)` 判断 reference visibility；
- deformable offsets 的采样上界由 `valid_size/padded_size` 计算；
- 采样点必须同时满足下界 `-1` 和有效区域上界，不能进入 bottom/right padding；
- `MetricTokenEncoder` 的三个调用点均显式传入原始有效图像尺寸。

新增单测同时覆盖 reference projection 和 offset 后采样，确认 padding strip 不会成为
伪造的 2D evidence。

### 12.6 依据 coverage 与标定质量调整 3D loss

coverage 和 multiview correspondence 已足以说明几何监督不是“几乎全空”，因此不需要
扩大 grid 或关闭 3D objective；但 K、depth scale 和 TCP 仍为
`nominal_unverified`，不能让 coarse pseudo-3D 与 RGB reconstruction 同等主导训练。

模型配置新增：

```text
geometry_quality_weight = 0.25
```

geometry 总权重现在为：

```text
effective_geometry_weight
  = pointmap_loss_weight(0.1)
  * geometry_quality_weight(0.25)
  * geometry_warmup
```

warmup 完成后有效值为 `0.025`。该乘子作用于整个 geometry objective，因此不会像把
所有 pixel confidence 同比乘 `0.25` 那样在 weighted mean 中相互抵消。现有几何内部
相对权重保持不变：

```text
ray_surface_loss_weight = 0.1
free_space_loss_weight = 0.1
multiview_consistency_loss_weight = 0.05
```

`geometry_loss_weight` 已加入训练诊断与 JSONL 日志，便于确认 warmup 和最终有效权重。
后续只有在官方 K、depth scale、TCP/相机外参得到独立验证，且 validation geometry
指标与人工投影 QA 一致改善后，才应提高 `geometry_quality_weight`。

### 12.7 训练、checkpoint 与回归结论

train/val YAML 和独立验证器均使用同一个
`ISAAC_X_FORWARD_FROM_OPENCV`。训练脚本默认输出目录改为：

```text
work_dirs/mobilemanibench_5tasks_vggt_isaac_optical_q025
```

这是有意的：相机坐标约定、padding visibility 和 geometry 权重均改变了监督语义。
旧 checkpoint 的模型参数形状仍可作为初始化来源，但**不应自动恢复旧 optimizer state
并在原目录混合续训**。已经运行中的旧任务也不会热更新这些 Python/config 修改；需要
停止后从新目录重新启动才能应用。

远端最终回归：

```text
compileall                                                   PASS
test_vggt_geometry.py                                        4/4 PASS
test_vggt_3d_wam.py                                          5/5 PASS
tests/data/test_mobilemanibench_vggt_dataset.py               1/1 PASS
tokenizer training/free-surface/multiview/invisible/padding  PASS
train/val optical-transform + quality-weight YAML contract   PASS
VGGT training PRELIGHT_ONLY                                  PASS
33-frame real-data tiny-checkpoint CPU validation            PASS
```

最后一项在修正坐标系后的 smoke 样本得到 `520/520` inside-grid targets 和
`29/58` multiview correspondences；它只验证端到端接口和计数链路，不代表生产模型的
重建精度。本批没有启动训练，也没有占用正在运行的生产 GPU 作业。

## 13. 新一轮训练前最终审计与收敛参数调整

### 13.1 代码逻辑审计结论

本轮按数据到优化器的完整顺序检查：

```text
33-frame head/wrist clip + K + T_b0_camera + pseudo PointMap
  -> frozen DINOv2-L/14
  -> pretrained VGGT frame/global blocks + LoRA
  -> shared full-time features
  -> 2D causal temporal encoder 33 -> 9 -> stochastic latent
  -> learned temporal/spatial RGB decoder 9 -> 33
  -> B0 metric queries + multilevel multiview deformable attention
  -> 3D causal temporal encoder 33 -> 9
  -> 3D temporal decoder 9 -> 33 + ray PointMap decoder
  -> RGB + KL + weighted geometry losses
```

确认结果：

- ReconDrive checkpoint 与本地 backbone 为 `921/921` tensors 匹配，
  `908,999,680/908,999,680` backbone numel 匹配；
- 总参数约 `922.51M`，可训练约 `13.51M`；
- 可训练 backbone 参数约 `6.29M`，全部是 frame/global aggregator LoRA；
- 随机初始化 2D/3D codec、decoder 和 heads 约 `7.22M`，全部可训练；
- DINOv2 原始参数、VGGT 原始 aggregator 参数均保持冻结；
- 2D 和 3D 都严格执行 `33 -> 9 -> 33`，共享同一 Wan 时间 lattice；
- camera optical transform、B0 grid、padding validity、inside-grid mask 在训练与验证
  中一致；
- RGB、PointMap、ray surface、free/surface occupancy 和 multiview loss 均为 finite，
  并能向对应分支反向传播；
- checkpoint 自动恢复仍由 `get_last_checkpoint(output_dir)` 实现，新输出目录当前
  不存在，不会误恢复旧监督语义的 optimizer state。

没有发现阻止新一轮训练的结构性或梯度链路问题。剩余研究边界仍是 nominal K 和
H.264 pseudo-depth 的精度，而不是代码接口错误。

### 13.2 旧日志为什么显得收敛慢

旧单样本训练日志的窗口均值：

| step 区间 | total loss 均值 |
|---|---:|
| 1–100 | 0.519 |
| 901–1000 | 0.225 |
| 1901–2000 | 0.183 |
| 3901–4000 | 0.130 |

到 step 4000：

```text
video_recon_loss: 0.517 -> 0.035
pointmap_loss:    2.509 -> 0.583
ray_surface_loss: 4.177 -> 3.633
```

因此旧实验实际在稳定收敛。速度偏慢的主要参数原因是：约 7.22M 个随机初始化的新模块
与约 6.29M 个预训练 LoRA 共用 `1e-5`，且 10,000-step run 前 500 step 都在 LR
warmup。KL 从约 `0.10` 上升到 `1.81` 不代表发散；其权重只有 `1e-6`，encoder 在压低
posterior noise 以改善重建时 KL 上升是可预期的，仍需同时观察 recon 和 latent 数值。

### 13.3 新训练参数

默认参数调整为：

```text
heads learning rate          = 5e-5
VGGT aggregator LoRA LR      = 2e-5
warmup ratio                 = 0.02   # 200 / 10000 steps
scheduler                    = cosine_with_min_lr
minimum LR rate              = 0.2
weight decay                 = 0.01
max steps                    = 10000
per-device batch             = 1
gradient accumulation        = 1
global batch on 8 GPUs       = 8
```

对应 10,000-step LR：

| step | LoRA LR | heads LR |
|---:|---:|---:|
| 0 | 0 | 0 |
| 100 | `1.0e-5` | `2.5e-5` |
| 200 | `2.0e-5` | `5.0e-5` |
| 1,000 | `1.97e-5` | `4.93e-5` |
| 5,000 | `1.23e-5` | `3.06e-5` |
| 10,000 | `4.0e-6` | `1.0e-5` |

几何分支仍保留独立的 1,000-step loss warmup 和最终 `0.025` quality-adjusted 权重，
避免 LR 提升时粗糙 pseudo-3D 在训练初期主导共享特征。

五任务数据共有约 `853,209` 个 train clips 和 `24,197` 个 stride-4 validation
clips。8 卡 global batch 8、10,000 step 实际读取约 80,000 个训练样本，约等于全部
overlapping anchors 的 9.4%；因此 10,000 step 是第一轮效果判断点，不应被解释为完整
一 epoch 或最终收敛上限。

训练内 validation 固定均匀抽取 256 个 held-out clips；训练可视化间隔从 50 调到
200 step，启用 dataloader pinned memory。这样减少验证和绘图造成的 wall-clock 停顿，
但不改变 loss 或训练样本分布。

### 13.4 最终生产规模验收

在单张 A100 80GB 上用真实五任务样本、完整生产模型执行了 forward/backward：

```text
z_2d_video             [1,2,48,9,10,20]
z_3d_video             [1,9,384,256]
reconstructed_video    [1,33,2,3,160,320]
predicted_pointmap     [1,33,2,3,32,64]
all floating outputs   finite
LoRA gradient          present
heads gradient         present
peak allocated         33.0 GiB
peak reserved          34.1 GiB
```

随后使用相同生产模型和真实 dataset 完成一次实际 Trainer optimizer step：

```text
train loss             finite
global step            1
backbone group LR      2e-5
heads group LR         5e-5
```

此外完成：

```text
compileall                                                   PASS
geometry unit tests                                          4/4 PASS
model/save-load/freeze/grouped-optimizer tests                7/7 PASS
dataset real-data test                                        1/1 PASS
tokenizer geometry/occupancy/multiview/padding/LoRA tests     PASS
Hydra exact scheduler override composition                    PASS
training shell syntax + preflight                             PASS
validation limiting: 24197 -> 256 unique evenly spaced clips PASS
```

上述 smoke 使用临时目录，没有保存模型、没有留下测试 checkpoint，也没有启动正式
多卡训练。

## 14. VGGT step-5000 验证崩溃修复

### 14.1 现象与原因

五任务全数据 run：

```text
work_dirs/mobilemanibench_5tasks_vggt
```

正常训练到 step 5000 后，在首次 `eval_steps=5000` 验证中报：

```text
TypeError: VGGT3DWAMModel.forward() got an unexpected keyword argument 'video'
```

原因与 DreamZero VLA 的 batch-dict 问题相同：训练通过自定义 `compute_loss()` 调用
`model(inputs)`，Hugging Face 默认无标签 `prediction_step()` 却调用
`model(**inputs)`。上游 Trainer 同一步还先验证后保存，因此异常发生时没有生成
`checkpoint-5000`。

### 14.2 代码修复

`VGGTTrainer` 新增专用 `prediction_step()`：

```text
prepare nested inputs
 -> autocast + no_grad
 -> model(inputs)
 -> rank-zero validation visualization
 -> detached loss only
```

它不跨 rank gather `reconstructed_video`、PointMap 或 latents，也不把 validation
diagnostics 追加到训练滑动窗口。

同时覆盖 `_maybe_log_save_evaluate()`。当同一步同时满足 steps save/eval 时：

```text
save checkpoint -> on_save callbacks -> evaluation
```

`save_strategy=best` 保留上游顺序，因为必须先得到验证指标。

新增 `tests/experiment/test_vggt_trainer_validation.py`，分别锁定 batch-dict 调用合同和
`save -> evaluate` 事件顺序。

### 14.3 新一轮输出目录

旧 run 只有 5000-step 日志而没有 checkpoint。为避免新训练从 step 0 开始时继续追加
旧 JSONL，脚本默认输出改为：

```text
work_dirs/mobilemanibench_5tasks_vggt_evalfix
```

保留用户最新训练参数：

```text
max_steps=30000
save_steps=5000
eval_steps=5000
warmup_ratio=0.01
train visualization interval=1000
```

回归结果：

```text
VGGT batch-dict evaluation test             PASS
same-step save-before-evaluate test         PASS
VGGT model tests                            7/7 PASS
compileall / shell syntax / preflight       PASS
```

## 15. VGGT tokenizer V2：decoder、感知损失与 3D 分支增强

本轮保持既有 tokenizer/WAM 接口不变，只实施以下六项改进：

```text
1. 增强 Video decoder
2. 增加 LPIPS/SSIM/梯度/时间视觉损失
3. voxel grid 由 384 提高到 768 tokens
4. 增强 PointMap decoder，并输出 80x160 PointMap
5. 有效 3D 权重由 0.025 提高到 0.1
6. VGGT global temporal window 由 1 改为 4
```

### 15.1 `groot/vla/model/vggt_3d_wam/video_latent.py`

Video decoder 保持：

```text
[B,V,48,9,10,20] -> [B,33,V,3,160,320]
```

但隐藏通道由 128 提高为：

```text
256 -> 192 -> 128 -> 96 -> 64
```

四级 bilinear upsample 改为 `Conv2d + PixelShuffle(2)`，每一级增加两个
spatial residual blocks。Wan temporal decoder 后增加时间 residual block。没有增加
绕过 latent 的 encoder-decoder skip connection。

### 15.2 `groot/vla/model/vggt_3d_wam/losses.py`

新增：

```text
Charbonnier reconstruction
LPIPS-AlexNet
SSIM
spatial RGB gradient
temporal RGB difference
```

实际视觉目标为：

```text
1.0 Charbonnier
+ 0.1 LPIPS
+ 0.2 SSIM
+ 0.1 spatial gradient
+ 0.1 temporal difference
```

LPIPS 网络完全冻结；输入保持 `[-1,1]`。按 8 帧分块并使用 activation
checkpoint，避免 33 帧双视角的感知网络激活长期占用显存。

新增依赖：

```text
lpips==0.1.4
```

运行环境使用 `--no-deps` 安装，未替换现有 torch/CUDA 包。AlexNet 权重缓存于：

```text
/mnt/yihao/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
```

### 15.3 `groot/vla/model/vggt_3d_wam/pointmap_decoder.py`

ray prediction head 改为 256 hidden、两层 residual MLP。最终 PointMap 使用两级分辨率：

```text
ray/depth/occupancy rendering: 40x80
learned PointMap refinement:   80x160
```

高分辨率输出采用 bilinear metric baseline 加零初始化的 PixelShuffle residual，
避免训练初始阶段直接破坏 coarse metric geometry。

ray chunk 从 512 调为 128，并对每个 chunk 使用 activation checkpoint。真实尺寸
forward 初测若不使用 checkpoint，会在 pointmap residual MLP 中达到约 75 GiB 并
OOM；checkpoint 后完整 forward/backward/optimizer step 峰值降为约 34.24 GiB。

### 15.4 `groot/vla/model/vggt_3d_wam/model.py`

PointMap coordinate loss 在最终 `80x160` 上计算；ray classification、
free-space 和 occupancy loss 使用 nearest resize 后的 `40x80` pseudo target。
multiview 和 temporal geometry 使用最终高分辨率 PointMap。

新增：

```text
surface normal loss      weight 0.1
depth/range gradient     weight 0.1
temporal geometry loss   weight 0.1
```

所有 3D 子损失统一置于 quality-adjusted geometry objective 内：

```text
total =
    video_quality_loss
  + beta_2d * KL
  + geometry_weight * geometry_objective
```

不再让 temporal geometry 绕过统一的 3D 总权重。

### 15.5 配置和训练入口

涉及：

```text
groot/vla/model/vggt_3d_wam/configuration.py
groot/vla/configs/model/vggt_3d_wam/encoder_decoder.yaml
groot/vla/configs/vggt_3d_wam.yaml
scripts/train/mobilemanibench_vggt_training.sh
```

关键配置：

```text
global_temporal_window       4
video_decoder_dim            256
grid_size                    [8,12,8] = 768
pointmap_ray_size            [40,80]
pointmap_size                [80,160]
pointmap_loss_weight         0.4
geometry_quality_weight      0.25
effective geometry weight   0.1
geometry warmup              2000 steps
```

默认输出目录改为：

```text
work_dirs/mobilemanibench_5tasks_vggt_v2
```

这是必要的，因为 decoder、voxel grid 和 PointMap head 的参数形状已经变化，不能自动
恢复旧 V1 checkpoint 的 optimizer/model state。训练脚本还固定 `TORCH_HOME` 并在
preflight 检查 LPIPS 依赖。

### 15.6 验收结果

真实五任务样本、完整 33 帧双视角、单张 A100 80GB：

```text
z_2d_video             [1,2,48,9,10,20]
z_3d_video             [1,9,768,256]
reconstructed_video    [1,33,2,3,160,320]
predicted_pointmap     [1,33,2,3,80,160]
forward                PASS
backward               PASS
AdamW optimizer step   PASS
all tracked losses     finite
peak allocated         34.24 GiB
peak reserved          35.37 GiB
```

其他检查：

```text
Python compile                                  PASS
VGGT model regression tests                     7/7 PASS
LPIPS gradient + PointMap upsample smoke         2/2 PASS
training shell syntax                           PASS
training preflight                              PASS
```

`grid_size` 的实际顺序为 `[z,y,x]`。因此 `[4,12,8] -> [8,12,8]`
将垂直分辨率从 `0.625 m` 提高为 `0.3125 m`；侧向分辨率保持约
`0.333 m`，前向分辨率保持 `0.375 m`。

## 16. 从 V1 checkpoint 进行 matching-only 初始化

新增：

```text
groot/vla/model/vggt_3d_wam/checkpointing.py
tests/model/test_vggt_matching_checkpoint.py
```

训练脚本新增可选环境变量：

```text
INIT_CHECKPOINT
```

默认值为空。`None`、空字符串或仅包含空白字符时，脚本不传 Hydra override，
Python loader 也直接返回，不访问文件系统、不修改模型。

使用旧 20k checkpoint 初始化新版模型：

```bash
INIT_CHECKPOINT=/mnt/yihao/codes/dreamzero/work_dirs/mobilemanibench_5tasks_vggt/checkpoint-20000 \
bash scripts/train/mobilemanibench_vggt_training.sh
```

loader 接受 checkpoint 目录或直接的 `model.safetensors` 路径。它只加载：

```text
当前 requires_grad=True
参数名称完全一致
tensor shape 完全一致
```

冻结的 DINO/VGGT 基础参数仍从 ReconDrive checkpoint 初始化；旧 optimizer、
scheduler、global step 和 RNG 状态均不加载。safetensors 按 tensor 流式复制，
不会一次把约 3.7 GB checkpoint 完整载入额外 host memory。

真实 `checkpoint-20000` 验证结果：

```text
matched trainable tensors       589
shape-mismatched tensors          4
new/missing tensors             120
matched trainable numel       64.07%
```

被跳过的形状变化为：

```text
Video decoder input projection   128 -> 256
Video decoder output projection   32 -> 64
metric query features            384 -> 768
```

新输出目录为空时执行 matching-only 初始化。如果输出目录已有新版 checkpoint，
自动断点恢复优先，并跳过 V1 初始化，确保新版 optimizer/scheduler 可以正常续训。

检查结果：

```text
empty / whitespace no-op tests       PASS
matching/mismatch/frozen tests        2/2 PASS
real checkpoint-20000 streaming load PASS
empty/whitespace/real-path preflight  PASS
```

## 17. VGGT contact sheet 可视化精简

修改：

```text
groot/vla/model/vggt_3d_wam/visualization.py
tests/model/test_vggt_visualization.py
```

`reconstruction_pointmap.png` 从 7 列精简为 4 列：

```text
RGB GT
RGB reconstruction
PointMap GT range
PointMap predicted range
```

移除：

```text
图片顶部的全部 loss 文本
RGB absolute error
PointMap L2 error
pseudo-label confidence
```

loss 数值没有从实验记录中删除，仍完整保存在 `metrics.json`、
`loss_log.jsonl` 和 W&B 日志中。3D scatter 仍在内部使用 confidence threshold
筛选有效 pseudo points，但不再将 confidence 单独绘制为 contact-sheet 列。

验证：

```text
visualization compile       PASS
visualization tests         2/2 PASS
```

## 18. LPIPS shared tensor checkpoint 保存修复

### 18.1 报错原因

LPIPS-AlexNet 将同一组冻结 linear weights 同时注册为：

```text
lin0 ... lin4
lins[0] ... lins[4]
```

Hugging Face 使用 safetensors 保存整个模型时检测到这些共享 tensor，但模型配置未声明
它们是 tied weights，因此在 step 5000 报错并终止保存。

### 18.2 修复

`VGGT3DWAMModel.save_pretrained()` 现在从待保存 state dict 中移除：

```text
lpips_loss.*
```

LPIPS 是冻结的训练损失网络，不属于 tokenizer 推理参数。模型恢复时根据配置重新创建
LPIPS 并从固定的 AlexNet/LPIPS 权重初始化，因此排除它不会改变模型训练状态、loss
定义或后续恢复结果，同时减少 checkpoint 体积。

新增真实 LPIPS safetensors smoke：

```text
checkpoint_exists   True
lpips_keys_saved    0
```

### 18.3 残缺 checkpoint 防护

首次失败留下：

```text
work_dirs/mobilemanibench_5tasks_vggt_v2/checkpoint-5000
```

其中只有 config 和部分 rank RNG 文件，没有模型权重、trainer state、optimizer 或
scheduler，因此不能恢复。新的 `get_last_complete_checkpoint()` 只接受同时包含：

```text
model weights
trainer_state.json
optimizer.pt
scheduler.pt
```

的 checkpoint。残缺目录会发出 warning 并被忽略；若存在更早的完整 checkpoint，
则自动恢复更早的完整版本。

失败目录和原 `loss_log.jsonl` 均保留。为避免重跑日志从 step 0 追加到旧日志，训练
脚本默认输出改为：

```text
work_dirs/mobilemanibench_5tasks_vggt_v2_savefix
```

本次 step 5000 权重没有成功写盘，因此必须重新从 matching-only 的 V1
`checkpoint-20000` 初始化训练。

验证：

```text
model save tests                    8/8 PASS
Trainer/checkpoint tests            3/3 PASS
real LPIPS safetensors save         PASS
incomplete-checkpoint fallback      PASS
```
