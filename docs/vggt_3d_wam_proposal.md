# 面向 Mobile Manipulation 的 VGGT-3D WAM 方案

## 1. 研究动机

WAM 已经能够联合建模未来视觉动态和机器人动作

但在 mobile manipulation 场景中，仅依赖 2D video tokens 容易把多种物理变化混合成图像空间中的表观运动：

- 底盘运动会带来明显的相机 ego-motion 和背景变化
- 物体运动、遮挡和显露需要比局部纹理更强的空间理解

因此计划在 WAM 中引入显式的 3D spatial tokens，使 3D tokens 具备几何可解释性，并在 WAM 中承担未来空间表征的 denoising target。第一版研究重点仍是得到高质量、适合 WAM 的 2D video tokens；3D tokens 的主要职责是提供视角变化、ego-motion、遮挡和物体空间关系等粗粒度空间理解，而不是一开始就追求毫米级几何重建。

## 2. 总体方案（两阶段）

**第一阶段：基于 VGGT 的时空 tokenizer 预训练**

训练一个多视角视频 encoder-decoder：

multi-view video + camera parameters -> VGGT encoder -> 2D video latent + dynamic 3D spatial tokens

这一阶段的目标不只是单帧 RGB 压缩和静态 3D 聚合，而是同时学习 video compression 和 geometry-space temporal aggregation。2D 分支是主分支，负责得到适合 WAM denoising 的高质量压缩视频 latent；3D 分支是空间辅助分支，负责得到具备几何可解释性的 dynamic spatial tokens，并通过 robot-centric PointMap rendering、多视角一致性和跨时间一致性监督获得空间语义。第一版允许使用官方 VLA 同样读取的 `depth_image_*.mp4` 配合相机外参生成 coarse robot-centric PointMap pseudo label，先跑通完整 2D/3D tokenizer 与 WAM 链路。

**第二阶段：3D-consistent WAM 训练**

在完整 WAM 中，将 2D、3D 和 action chunks 放入同一个 causal DiT 序列，训练方式沿用 DreamZero 的 causal chunk 设计

```text
[clean history 2D tokens] + [noisy 2D chunk tokens]
[clean history 3D tokens] + [noisy 3D chunk tokens]
[noisy base action chunk tokens]
[noisy manipulator action chunk tokens]
[state / action-state tokens]
```

action tokens 采用解耦设计：base plan 和 manipulator plan 有各自独立的 token 序列、输入投影、输出投影及 token-type embedding，但共同进入 causal DiT backbone，在满足因果约束的前提下通过 self-attention 交互。manipulator plan 内部联合表示 EEF pose 与夹爪/灵巧手构型，不再为末端执行器建立第三路 action tokens。

语言和图像条件也保持 DreamZero 原有风格：language 作为 cross-attention context，timestep 通过AdaLN调制注入

## 3. 当前明确创新点

当前最明确的创新点集中在第一阶段的 2D-3D encoder：构建一个基于 VGGT 共享 backbone 的多视角时空 tokenizer，使同一组 VGGT features 同时生成 2D video latent 和几何可解释的 3D spatial tokens。

相比把 2D encoder 和 3D encoder 做成两个独立模块，这种设计有两个优势：第一，2D video latent 和 3D spatial tokens 来自同一个 3D-aware feature space，有助于保持 2D 表观信息和 3D 几何信息的一致性；第二，3D tokens 不是普通 latent，而是通过相机内外参从多视角 image tokens 中聚合，并通过 robot-centric PointMap rendering、多视角一致性和 masked-view reconstruction 获得明确的空间语义。

因此，这一阶段的核心贡献可以概括为：

```text
VGGT-shared 2D-3D encoder:
  multi-view video + camera parameters
   -> shared VGGT features
   -> compressed 2D video latent
   -> geometry-aware 3D spatial tokens on a metric grid
   -> coarse robot-centric PointMap / feature rendering supervision
```

这一设计的目标不是简单替换一个更强的 image encoder，而是训练一个同时服务于 video compression 和 geometry-aware world modeling 的统一 2D-3D tokenizer。这里“metric grid”表示 token 与真实尺度坐标 cell 绑定；使用有损 MP4 训练的第一版只能主张 coarse geometry-aware representation，不能据此主张高精度 metric reconstruction。

## 4. 第一阶段VGGT Encoder设计

### 4.1 时序聚合的总体顺序

第一阶段输入是多视角视频，而不是孤立单帧。因此 VGGT encoder 需要同时支持 2D video latent 和 dynamic 3D latent。

计划采用共享 VGGT backbone 提取每个时间步、每个视角的特征：

```text
I_{t,v} -> h_{t,v}
```

之后分成两个分支：

```text
2D video branch:
  h_{t,v}
   -> spatial bottleneck
   -> TemporalTransformer_2D
   -> z_2d_video

3D geometry branch:
  h_{t,v} + camera parameters + metric 3D queries
   -> per-frame 2D-to-3D aggregation
   -> z_{3d,t}
   -> TemporalTransformer_3D
   -> z_3d_video
```

这里的关键区别是：2D branch 的目标是压缩 video latent，不要求 token 绑定显式 3D 坐标，因此可以直接在 VGGT image-token space 上做时间聚合。VGGT 特征本身包含一定 3D-aware inductive bias，但其组织方式仍然是 2D image tokens。

3D branch 的目标是得到几何可解释的 spatial tokens，因此不能先在 image space 中混合时间信息再投影到 3D。更合理的顺序是先利用相机内外参完成 per-frame 2D-to-3D 聚合，让每个 token 绑定 metric 3D cell，再在这些 3D tokens 上做时间聚合。

### 4.2 2D Video Latent 的设计选择

计划采用 VGGT 代替DreamZero中的2D VAE，并且VGGT 的 2D latent会采用类似VAE的分布压缩形式，但不会强行对齐到原始 VAE latent space

主要考虑是：VGGT 预训练中已经包含多视角几何和 3D-aware 表征，如果直接使用 VAE latent alignment，可能会把这部分信息压回以图像重建为主的latent，丢失原本的3D表征

因此，2D latent 的设计不是单帧 RGB latent，而是面向 video chunk 的压缩表示：

```text
multi-view video
 -> VGGT backbone
 -> spatial bottleneck
 -> TemporalTransformer_2D
 -> 2D video latent z_2d_video
 -> video decoder
```

这一点需要和当前 DreamZero 的 Wan VAE 保持一致的接口理解：DreamZero 现有 VAE 是冻结的视频级 tokenizer，输入完整 video clip，输出压缩后的 latent video，再由 DiT 在 latent space 中加噪和去噪。因此，如果 VGGT 替换 2D video tokenizer，也需要承担同样的视频压缩职责，而不是只提供 per-frame image feature。

目标接口可以写成：

```text
video:        [B, V, T, 3, H, W]
z_2d_video:  [B, V or fused, T', C, H', W']
```

其中 `T'`、`H'`、`W'` 需要显式控制，使其满足 WAM 的 token budget。当前 DreamZero/Wan VAE 的参考设计是：

```text
Wan2.1: z_dim = 16, spatial stride = 8x
Wan2.2: z_dim = 48, spatial stride = 16x
temporal stride: first frame preserved, later frames roughly 4x compressed
```

VGGT 版本不必完全复刻这些数字，但需要提供类似的时空 bottleneck。例如：

```text
T -> T'    video-level temporal compression
H,W -> H',W'    spatial compression
C -> C_latent   DiT-compatible latent channels
```

这个 bottleneck 应该优先保证三点：视频重建质量足够、latent token 数可控、latent 统计适合 diffusion / flow matching。由于 mobile manipulation 对局部接触、小物体和末端执行器细节更敏感，第一版可以从较温和的压缩比开始，再根据 WAM 显存和动作预测效果调整。

`z_2d_video` 采用类似 VAE 的分布形式：

```text
mu_2d, logvar_2d = latent_head(h_vggt)
z_2d_video = mu_2d + sigma_2d * eps
```

对应训练目标为：

```text
L_2d =
    L_video_recon
  + beta_2d KL(q(z_2d_video | video) || N(0, I))
  + optional L_temporal_consistency
```

2D latent的整体设计目标不是模仿 VAE latent，而是让 VGGT 的 video latent 本身成为一个稳定、可解码、适合 diffusion / flow matching 的连续分布

如果后续发现 `z_2d_video` 与 WAM 训练不兼容，可以切回原 DreamZero 的 VAE 作为 2D encoder；此时 3D spatial tokens 仍然由 VGGT 分支提供

```text
主方案:   VGGT -> z_2d_video, VGGT -> z_3d_video
备选方案: VAE  -> z_2d_vae,  VGGT -> z_3d
```

### 4.3 3D Tokens 的参数化

3D tokens 设计为定义在 metric grid 上的learnable query：

```text
token_i <-> P_i = (x_i, y_i, z_i)
```

3D tokens的实现可以考虑 BEV 或 sparse voxel，避免 dense voxel 带来的高计算成本。

每个 token 使用空间位置初始化：

```text
F_i = MLP(pos_embed(P_i)) + learnable_feature_i
```

这样每个 token index 都对应一个真实空间 cell，这是 3D tokens 几何可解释性的基础

与 2D latent 不同，3D tokens 的首要目标是保持空间语义。因此第一版不强制把 3D latent 做成 VAE-style stochastic distribution，而是优先采用：

```text
deterministic metric 3D tokens + LayerNorm / RMSNorm
```

这样可以稳定 token 统计分布，同时避免过强 KL 约束损伤几何信息。如果第二阶段发现 3D token denoising 不稳定，再考虑引入 small-beta stochastic 3D latent：

```text
mu_3d, logvar_3d = latent_head(h_3d)
z_3d = mu_3d + sigma_3d * eps
```

### 4.4 多视角 2D-to-3D 聚合

对于每个时间步 `t` 和每个 3D query point `P_i`，利用相机内参和外参投影到每个相机视角：

```text
p_{t,i,v} = project(P_i, K_{t,v}, T_{t,v})
```

然后在投影位置附近采样 VGGT image features：

```text
f_{t,i,v} = sample(h_{t,v}, p_{t,i,v})
```

来自所有可见视角的特征再融合到共享 3D token 中：

```text
z_{3d,t,i} = deformable_attention(query=F_i, key/value={f_{t,i,v}})
```

这个设计使每个 3D token 只能从几何对应的图像区域聚合证据，多视角观测可以被融合到同一组 3D spatial tokens 中。

随后对每个 spatial token index 沿时间做 temporal transformer：

```text
{z_{3d,1,i}, ..., z_{3d,T,i}} -> TemporalTransformer_3D -> z_{3d,video,i}
```

这样 temporal aggregation 发生在 geometry space 中，建模的是同一个 metric 3D cell 在时间上的变化，而不是不同帧 image patch 之间的表观变化。

### 4.5 Robot-Centric PointMap Decoder

3D decoder 的主输出建议从 depth map 改为 robot-centric PointMap。Depth 对每个像素只预测到相机的距离标量，坐标定义在当前相机视角下；当训练数据来自不同相机视角时，同一个物理点的 depth 数值会随视角变化，策略或 WAM 还需要额外学习 camera-centric depth 到机器人动作坐标系的映射。

Robot-centric PointMap 对每个像素输出机器人基坐标系下的 3D 坐标：

```text
M_B(u, v) = [x_B(u, v), y_B(u, v), z_B(u, v)]
```

同一个物理点即使出现在不同相机的不同像素位置，只要外参和时间对齐正确，它在机器人基坐标系下的坐标应保持一致。这更符合 mobile manipulation 的动作预测需求：base plan、EEF pose、future waypoint 都已经以机器人/底盘坐标系表达，3D decoder 也应优先输出同一坐标系下的几何目标。

为了从 dynamic 3D tokens 解码某个时间步、某个目标视角的 PointMap，对每个像素构造对应的 camera ray：

```text
P(d) = O + d * r(u, v)
```

沿 ray 采样多个候选深度点，并从 3D token grid 中读取特征：

```text
P_j = O + d_j * r
g_j = trilinear_sample(3D_tokens, P_j)
logit_j = MLP(g_j, pos_embed(P_j), ray_dir)
```

这些 logits 表示该 ray 上不同候选点成为可见表面的概率：

```text
p_j = softmax(logit_j)
P_B_pred(u, v) = sum_j p_j * P_{B,j}
```

其中 `P_{B,j}` 是候选点在机器人基坐标系 `B(t)` 下的坐标。这个 decoder 仍然可以看作轻量可微渲染器。同一组 dynamic 3D tokens 表示一段时间内的场景几何，不同时间步和相机视角只需要改变 rays 和相机外参，就可以渲染对应视角下的 robot-centric PointMap video。

基础监督改为多视角 robot-centric PointMap prediction：

```text
L_pointmap = valid_mask * Huber(P_B_pred, P_B_pseudo)
           + lambda_ray * valid_mask * CE(ray_surface_logits, pseudo_surface_bin)
```

第一版的 `P_B_pseudo` 可以由官方发布数据中的 `depth_image_*.mp4`、相机内参和相机到机器人基坐标系外参生成。官方生成代码使用相机 `distance_to_camera`，将 0--5 m 截断范围线性映射到 0--255 后写入视频，因此它更接近沿单位 camera ray 的 range，而不是 optical-axis z-depth：

```text
D_pseudo(u, v) = 5.0 * Y(u, v) / 255
P_C_pseudo(u, v) = O_C + D_pseudo * normalize(r_C(u, v))
P_B_pseudo(u, v) = T_B_C @ P_C_pseudo(u, v)
```

其中 `Y` 使用解码后的 luma/灰度值或三通道中位数，不能任取一个可能受颜色空间转换影响的通道。公式还要求解码器已正确还原原始 full-range 0--255；若容器/解码器采用 limited-range YUV，必须先按视频 color-range 元数据恢复或用已知灰度标定，不能直接代入。0--5 m 的 8-bit 量化步长约为 `5/255 = 1.96 cm`，还叠加 H.264 块效应、边缘振铃和颜色范围转换误差。

因此 MP4 depth 只作为构造 `lossy_h264_pseudo_robot_centric_pointmap` 的中间量使用，并生成置信 mask：

- 接近 0 或 255 的像素设为 invalid/low-confidence；255 可能表示超过 5 m、无穷远或截断值，不能解释为精确 5 m 表面。
- 深度强边缘、遮挡边界、图像边界和明显 codec block/ringing 区域降低权重。
- K、动态外参或 camera-frame convention 未通过验证的样本，不参与依赖投影的几何损失。
- depth-bin 宽度和 voxel 分辨率不能细于数据有效精度；第一版可从不小于约 4--5 cm 的粗粒度开始，再依据实测 MP4-vs-lossless 误差调整。
- 不从有损深度生成硬接触、精细碰撞或毫米级表面标签。

PointMap target 还需要额外检查 robot-frame convention：`x/y/z` 轴方向、base link 与规划锚点 `B(t)` 的定义、外参时间戳同步和相机安装误差。若这些信息不可靠，PointMap loss 会把系统性标定误差直接注入 3D tokens，因此必须比普通 depth auxiliary loss 更严格地做 calibration QA。

如果第一版需要 occupancy，只生成带不确定带的 soft/free/unknown target：ray 上表面之前可作为 free-space 弱监督，预测表面附近保留 uncertainty band，表面之后保持 unknown。不能把 H.264 深度边界直接体素化为强 occupied ground truth。

为了避免模型退化成两个独立 autoencoder，即 image tokens 只负责重建 RGB、3D tokens 只负责预测 PointMap，需要加入跨视角和跨模态约束：

- `L_cross_view_pointmap`：将一个视角预测的 robot-centric PointMap 投影到另一个视角，与目标 PointMap 或高置信 pseudo PointMap 对齐；第一版只在有效视野重叠和高置信区域计算。
- `L_masked_view`：随机去掉一个或多个输入视角，要求共享 3D tokens 仍能重建被遮掉视角的 PointMap 或 feature。
- `L_3d_to_2d_feature`：用 `3D tokens + target camera pose` 渲染目标视角的 VGGT / DINO / VAE feature，并与对应 2D feature 对齐。
- `L_occupancy`：第一版只使用从 MP4 构造的 soft/free/unknown 弱监督；获得无损距离后再升级为更精细的 occupied / free / unknown 监督。

### 4.6 整体训练目标

```text
L_vggt =
    L_video_recon
  + beta_2d KL_2d
  + lambda_1 L_pseudo_robot_centric_pointmap
  + lambda_2 L_high_conf_cross_view_pointmap
  + lambda_3 L_masked_view
  + lambda_4 L_temporal_geometry_consistency
  + optional lambda_5 L_soft_occupancy
```

第一版采用“2D 主导、3D 辅助”的优化优先级：

1. 先确保 `z_2d_video` 的重建质量、时空压缩率和 latent 统计满足 WAM；2D-only tokenizer 是必须保留的对照基线。
2. 再逐步增加 pseudo PointMap、masked-view 和 temporal geometry loss，使 3D tokens 学到粗粒度空间结构。
3. `lambda_1/lambda_2/lambda_5` 从较小值 warm up，并监控 2D reconstruction 和下游 action 指标；如果加入 3D 监督后 2D 分支持续退化，应降低几何权重，而不是为了降低 noisy PointMap loss 牺牲主任务。
4. 共享 VGGT backbone 时可使用 gradient norm balancing、3D adapter 或在早期对部分 shared features stop-gradient，防止有损 pseudo PointMap 的噪声梯度污染 2D video latent。
5. MP4 生成的 pseudo PointMap 和未来 lossless / calibrated PointMap 必须使用不同的 quality tag、valid mask 和 loss weight，不能在训练中无区别混合。

第一版不要求 3D tokens 达到精细表面重建精度，但不能只满足“tensor shape 正确”。其最低成功标准为：

- 2D video reconstruction/denoising 不显著弱于 2D-only baseline；
- 3D tokens 能在 held-out frame/view 上解码出优于简单常数或单目无几何基线的 coarse robot-centric PointMap；
- 相机运动后，同一静态区域的 PointMap 在锚点坐标系中保持基本一致；
- 在 WAM 中加入 3D tokens 后，action/video prediction 或视角变化场景的指标相较 2D-only WAM 有可测增益；
- 通过 ablation 验证 WAM 确实使用 3D tokens，而不是完全忽略该分支。

只有在获得无损 distance/segmentation 子集并完成标定 QA 后，才评估高精度 metric depth、硬 occupancy、collision/contact 等更强几何主张。

## 5. 第二阶段 WAM 训练

第二阶段将 DreamZero 的 causal chunk 序列扩展到 3D spatial tokens

denoising targets 包括：

```text
2D video chunk tokens
3D spatial / geometry chunk tokens
base action chunk tokens
manipulator action chunk tokens
```

### 5.1 解耦的 Base / Manipulator Action Tokens

action plan 只有两个逻辑分支：

```text
base_plan[h]
  = [x, y, sin(yaw), cos(yaw)]

manipulator_plan[h]
  = [eef_x, eef_y, eef_z, eef_rotation_6d,
     hand_configuration...]
```

其中 `h` 对应一组固定的未来 waypoint offsets。两个分支都以当前规划锚点的底盘坐标系 `B(t)` 表达：

```text
T_B(t)_B(t+kh)   = inverse(T_W_B(t)) @ T_W_B(t+kh)
T_B(t)_EEF(t+kh) = inverse(T_W_B(t)) @ T_W_EEF(t+kh)
```

base branch 输出未来底盘 pose2d waypoint；manipulator branch 在同一个 token 中联合输出未来 EEF pose 和末端执行器构型。G1 的 manipulator plan 为 `9 + 1 = 10` 维，XHand 为 `9 + 12 = 21` 维。

两类 action 使用各自独立的 tokens，而不是先拼成一个 action token：

```text
base values
  -> base input projection
  -> base action tokens + base type embedding

manipulator values
  -> manipulator input projection
  -> manipulator action tokens + manipulator type embedding

base/manipulator tokens
  -> shared causal DiT backbone
  -> separate base/manipulator output projections
```

这种解耦保留了底盘导航与机械臂操作不同的维度、动态范围和控制语义，同时 shared backbone 中的 attention 仍允许两路计划互相约束。两路使用相同的 waypoint offsets 和 horizon valid mask；如果 flow matching 的噪声尺度差异明显，可以分别设置输入 normalization 或 noise preconditioning，但不应破坏其时间索引对应关系。

### 5.2 Manipulator Plan 的内部 Slice 与标签语义

manipulator 虽然只有一路 tokens 和一个输出 projection，但输出向量内部拆成三个语义 slice：

```text
manipulator_plan[h]
├── eef_position[h]       # 3, m
├── eef_rotation_6d[h]    # 6
└── hand_configuration[h] # D, joint position
```

主 action plan 使用统一的 future realized trajectory 标签：

- base waypoint 来自 future `robot_base` pose；
- EEF pose 来自 future `robot_hand` pose；
- hand configuration 来自同一未来时刻的实际夹爪/手指 joint position。

原始 EEF delta、原始 hand command 和 IK/controller joint target 只作为 baseline、auxiliary control supervision 或 tracking-error 分析，不与主 plan 的 realized trajectory 标签混用。

“Manipulator 合成一路”不等于对所有维度使用同一统计量和未加权 MSE。三个 slice 分别归一化和计算损失：

```text
L_manipulator =
    lambda_eef_pos L_eef_position
  + lambda_eef_rot L_eef_rotation
  + lambda_hand    L_hand_configuration
```

- EEF position 使用 workspace-aware normalization，可使用 L1/Huber。
- rotation-6D 解码后重新正交化，几何监督使用 SO(3) geodesic loss；不能只依赖原始 6D 欧氏误差。
- hand configuration 按每个 joint limit 归一化；G1/XHand 使用 embodiment-specific dimension mask。
- 所有 slice 都乘 horizon valid mask，episode 尾部越界 waypoint 不通过重复末帧制造伪监督。

flow-matching/denoising loss 可以在各自 action token space 中计算；上述 slice loss 则作用在模型还原出的 clean action plan 上，为物理量提供直接监督。

### 5.3 两路计划的一致性约束

base 和 manipulator 使用独立 tokens 后，单独降低两路重建误差仍不能保证组合计划可执行。对于第 `h` 个未来点，模型预测：

```text
hat_T_Bt_Bk = predicted base waypoint
hat_T_Bt_Ek = predicted EEF pose
```

将两者组合得到 EEF 相对未来底盘的预测：

```text
hat_T_Bk_Ek = inverse(hat_T_Bt_Bk) @ hat_T_Bt_Ek
```

数据真值为：

```text
T_Bk_Ek = inverse(T_W_B(t+kh)) @ T_W_EEF(t+kh)
```

加入相对 EEF 一致性损失：

```text
L_base_eef_consistency =
    L_position(hat_T_Bk_Ek, T_Bk_Ek)
  + lambda_rel_rot L_SO3(hat_T_Bk_Ek, T_Bk_Ek)
```

该损失直接约束“预测底盘走到该 waypoint 后，机械臂需要达到的相对 EEF 位姿”，使两个独立 action token streams 在物理上描述同一个整机计划。

进一步加入可达性约束：

```text
L_plan_feasibility =
    lambda_reach L_reachability(hat_T_Bk_Ek)
  + lambda_limit L_joint_limit
  + optional lambda_collision L_collision
```

- `L_reachability` 可由可微 IK、预计算 workspace SDF 或 learned reachability critic 实现。
- `L_joint_limit` 约束 IK 解或预测 hand configuration 不越关节上下限。
- `L_collision` 需要可信 robot geometry、未来 3D occupancy 和有效 mask；第一版可延后，不能用有损 depth 生成的伪碰撞标签作为强监督。
- 如果 IK/碰撞检查不可微，可在训练时作为 reranking/critic 目标，或只在推理时过滤；不能为了形式完整强行反传不可靠梯度。

一致性损失必须使用与 action plan 相同的 waypoint offsets、坐标 convention 和 horizon valid mask。base/EEF 的直接监督负责拟合示范轨迹，相对位姿与可达性损失负责限制两路独立预测的组合结果。

训练目标可以写为：

```text
L_wam =
    L_2d_denoise
  + L_3d_denoise
  + lambda_base_flow L_base_action_denoise
  + lambda_manip_flow L_manipulator_action_denoise
  + lambda_eef_pos L_eef_position
  + lambda_eef_rot L_eef_rotation
  + lambda_hand L_hand_configuration
  + lambda_plan_consistency L_base_eef_consistency
  + optional lambda_plan_feasibility L_plan_feasibility
  + lambda_geo L_future_geometry
  + lambda_proj L_video_3d_consistency
```

其中，`L_base_action_denoise` 和 `L_manipulator_action_denoise` 分别训练两路独立 action tokens；`L_3d_denoise` 用于预测未来 3D token trajectory；`L_future_geometry` 默认将预测的 future 3D tokens 解码为 robot-centric PointMap 进行监督，depth、occupancy 或 point flow 可以作为辅助几何目标；`L_video_3d_consistency` 约束预测的 future video 和 future 3D geometry 描述同一个物理未来。

这个设计保持了 WAM 的主干形式：video、两路 action tokens 和 3D geometry 在同一个 causal DiT backbone 中联合生成，同时通过独立投影与组合一致性损失保留 base/manipulator 的结构差异。

## 6. 推理流程

推理时流程如下：

```text
current / history multi-view video observations
 -> VGGT-3D encoder
 -> clean 2D history tokens + clean 3D history tokens
 -> causal WAM denoising
 -> next base action chunk + next manipulator action chunk
```

如果需要 world rollout，模型也可以同时输出 future 2D tokens 和 future 3D tokens。future 3D tokens 可以进一步解码成 robot-centric PointMap 或其他几何预测，用于 action reranking、collision checking 或 MPC-style closed-loop control。

## 7. 预期收益

- 通过绑定 metric grid 的 geometry-aware 3D tokens 提升对视角变化和 ego-motion 的建模能力；第一版监督精度按 coarse pseudo robot-centric PointMap 定位。
- 相比简单加入 current depth auxiliary loss，PointMap 让 3D tokens 直接对齐机器人动作坐标系，空间语义和生成目标更明确。
- 共享 VGGT backbone 使 2D video latent 和 3D spatial tokens 来自同一组 3D-aware features，降低 2D/3D 分支学成互不相关表示的风险。
- 将 DreamZero 从 `video + action` 自然扩展为 `video + action + 3D geometry`。
- 中间表示更可解释：每个 3D token 对应一个空间 cell，并且可以被渲染回相机视角。
- 2D latent 优先使用 VGGT-native 分布表示，保留 VGGT 的 3D-aware 表征；必要时仍可回退到原 VAE latent。
- 为 future robot-centric PointMap、occupancy、point flow 和 projection consistency 等几何监督提供统一接口。

## 8. 待确认问题

- 3D tokens 应该定义在 base frame、world frame，还是 reference camera frame？
- 第一版实现应选择 BEV、BEV-height、sparse voxel，还是 dense voxel？
- VGGT 在第一阶段和第二阶段中分别应该 freeze、partial fine-tune 还是 full fine-tune？
- `z_2d_video` 的 KL 权重、latent 维度和时序压缩比应该如何设置，才能兼顾视频重建质量和 WAM denoising 稳定性？
- 3D latent 是否需要从 deterministic tokens 升级为 small-beta stochastic tokens？
- PointMap 应该表达在当前底盘坐标系 `B(t)`、episode 初始 base frame，还是某个 world/reference frame？
- MP4 pseudo-range depth 生成 PointMap 时，valid-mask、有效分辨率、外参 QA 和 geometry loss 权重应如何设置，才能提供空间监督但不损伤 2D 主分支？
- causal attention mask 应如何扩展，才能让 2D、3D、action、state tokens 充分交互，同时避免未来信息泄漏？
