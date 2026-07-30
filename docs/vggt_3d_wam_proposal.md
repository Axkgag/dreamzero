# VGGT-3D WAM：方案与实施路线

## 1. 目标

Mobile manipulation 中，单纯的 2D video latent 容易把底盘 ego-motion、物体运动、
遮挡和视角变化混合为图像表观变化。本方案引入与机器人坐标系对齐的 3D tokens，使
WAM 同时建模：

- 多视角 2D appearance 与 future video；
- coarse robot-centric geometry；
- coarse Base mobility prior；
- refined Base 与 Manipulator action plans。

方案分两阶段：

1. 训练共享 VGGT-style backbone 的 2D/3D tokenizer；
2. 将多视角 2D latent、3D representation 和 Base Prior 接入 causal WAM。

本文只描述方案、pipeline 和实施边界。当前 tokenizer 的逐文件实现见
`docs/MM/MOBILEMANIBENCH_VGGT_CODE_CHANGES_BY_FILE.md`。

---

## 2. 当前状态

| 模块 | 状态 |
|---|---|
| VGGT-style shared backbone | 已实现 |
| 多视角 2D video tokenizer、RGB decoder | 已实现 |
| metric 3D tokenizer、PointMap decoder | 已实现 |
| tokenizer loss、训练、验证和可视化 | 已实现 |
| Base/Manipulator dual-plan WAM | 已实现 |
| clean Base Prior tokens 与 coarse waypoint head | 方案保留，当前代码尚未显式实现 |
| VGGT 多视角 `z_2d_video` 接入 WAM video stream | 未实现 |
| `z_3d_video` 接入 WAM | 未实现 |
| future 3D token/PointMap rollout | 未实现 |

当前应先验证 Stage 1 表示质量，再实施 Stage 2；不能把 tokenizer 接口已实现等同于
已经完成 WAM 集成，也不能把方案中的 Base Prior 写成当前代码已有模块。

---

## 3. 总体 pipeline

```text
Stage 1: VGGT 2D/3D tokenizer
--------------------------------
head/wrist RGB + K + T_B0_camera
  -> frozen DINOv2 + LoRA VGGT-style aggregator
  -> shared multi-view patch features
       ├─ 2D temporal VAE -> multi-view z_2d -> RGB reconstruction
       └─ metric voxel encoder -> z_3d
                                  -> PointMap/occupancy rendering

Stage 2: 3D-aware WAM
--------------------------------
clean history multi-view z_2d
+ noisy future multi-view z_2d
+ clean history z_3d
+ clean Base Prior query tokens
+ noisy Base plan tokens
+ noisy Manipulator plan tokens
(+ optional noisy future z_3d)
  -> shared causal WAM
  -> future head/wrist z_2d
  -> coarse Base prior trajectory
  -> refined Base plan + Manipulator plan
  -> optional future z_3d
  -> VGGT 2D/3D decoders
```

设计原则：

- 多视角 2D tokens 负责 appearance 和 future video；
- 3D tokens 负责 coarse metric geometry；
- Base Prior 是 clean latent queries，负责低频 mobility intention；
- Base/Manipulator plan tokens 是 flow matching 的 noisy generation variables；
- 所有 stream 使用一致的时间 lattice、planning horizon 和坐标锚点；
- MVP 先证明 3D context 与 Base Prior 有用，再增加 future 3D generation。

---

## 4. Stage 1：已实现的 tokenizer

### 4.1 输入、时间与坐标

生产协议：

```text
video              [B,33,V,3,160,320]
camera_K           [B,33,V,3,3]
T_B0_camera        [B,33,V,4,4]
pseudo_pointmap_B0 [B,33,V,3,32,64]

z_2d_video         [B,V,48,9,10,20]
z_3d_video         [B,9,384,256]
```

这里的 `z_2d_video` 已经经过：

- spatial bottleneck；
- `160x320 -> 10x20` 空间压缩；
- causal temporal Transformer；
- `33 -> 9` 时间压缩；
- VAE posterior sampling。

它是准备送入 WAM video stream 的 latent，不是还需要另一个时空 tokenizer 的
per-frame feature。

2D/3D branch 共用：

```text
33 frames -> 9 latent steps
frame 0 + 8 groups of 4 frames
```

每个 clip 固定使用第 0 帧底盘坐标系 B0：

```text
T_B0_camera(t)
  = inverse(T_world_base(frame0))
  @ T_world_camera(t)
  @ T_camera_pose_from_optical
```

voxel grid、pseudo PointMap 和预测 PointMap 都表达在 B0。未来 WAM 样本也应以当前
决策时刻为统一 `B_anchor`，保证 3D tokens、Base Prior waypoints、refined Base
waypoints 和 EEF waypoints 使用同一坐标系。

### 4.2 shared backbone

当前实现是 VGGT-style tokenizer，不是完整官方 VGGT：

```text
frozen DINOv2-L/14 patch extractor
 -> per-frame attention
 -> same-time cross-view global attention
 -> shared features
```

DINO 完全冻结且无 LoRA。LoRA 只用于 24 对 frame/global aggregator blocks。
`global_temporal_window=1`，因此 backbone 负责同一时刻的多视角融合，完整时间建模
由后续 2D/3D branch 完成。

### 4.3 2D branch

```text
shared features
 -> spatial bottleneck
 -> causal temporal Transformer
 -> learned 33->9 temporal encoder
 -> VAE mu/logvar
 -> z_2d_video [B,V,48,9,10,20]
 -> learned 9->33 RGB decoder
```

它与 Wan VAE 对齐的是每个 view 的输入输出 channel、spatial stride 16 和
`4k+1 <-> k+1` 时间 lattice。两者的 latent 统计分布未必相同，因此 WAM 集成时要
统计 scale/mean/std；这不代表 `z_2d` 还需要额外时空压缩。

`decode_2d()` 已支持多视图 6D latent：

```text
[B,V,48,9,10,20]
 -> [B,33,V,3,160,320]
```

所以目标 pipeline 是让 WAM 同时生成所有 view 的 future `z_2d`，再由同一个
VGGT 2D decoder 同时解码 head/wrist future video。

### 4.4 3D branch

生产 metric grid：

```text
frame: B0
XYZ range: X[0,3], Y[-2,2], Z[-0.5,2] m
grid [Z,Y,X]: [4,12,8]
N=384, C=256
```

每个 voxel query 投影到 head/wrist，在两级图像 feature 上做 query-conditioned
deformable sampling；随后用 `3x3x3` local aggregation 交换邻域信息。逐帧融合后，
每个固定 voxel 沿时间做 causal Transformer 和 33→9 压缩。

3D decoder 恢复 full-time metric grid。PointMap decoder 对每个相机像素构造 B0 ray，
在 `0.05..5.0 m` 采样 64 个 bins，并用 surface logits 的 softmax 期望得到 B0 XYZ。
occupancy head 监督表面前 free-space 和表面附近 occupied；表面之后保持 unknown。

### 4.5 当前训练目标

```text
L_tokenizer =
    L_RGB
  + beta_2d * L_KL
  + geometry_weight * (
        L_PointMap
      + lambda_ray * L_ray_bin
      + lambda_occ * (L_free + L_surface)
      + lambda_mv * L_multiview
    )
  + lambda_temporal * L_temporal_geometry
```

生产配置的 geometry weight 在 1000-step warmup 后为：

```text
pointmap_weight 0.1 * quality_weight 0.25 = 0.025
```

当前 temporal geometry 和 masked-view reconstruction 均未启用。pseudo range 来自
有损 H.264，K/外参仍属 nominal calibration，因此当前 3D 表示只能定位为 coarse
geometry，不能用于毫米级重建或强 collision/contact labels。

---

## 5. Stage 2：接入 WAM

### 5.1 Base、Manipulator 与 Base Prior 的关系

当前仓库已实现 dual-plan flow stream：

```text
6 noisy Base plan tokens
+ 6 noisy Manipulator plan tokens
 -> independent projections and type embeddings
 -> shared causal Wan DiT
 -> independent output projections
```

默认 plan offsets 为 `[1,4,8,12,16,24]` frames。Base token 表示
`[x,y,sin(yaw),cos(yaw)]`；Manipulator token 联合表示 EEF pose 和 hand
configuration。

目标 WAM 还应恢复一组独立的 Base Prior tokens：

```text
clean Base Prior query tokens
 -> read language + state + multi-view z_2d + z_3d
 -> coarse Base waypoint head
 -> low-frequency mobility prior
 -> condition noisy Base/Manipulator plan refinement
```

三类 token 的职责不同：

| Token | 是否加 flow noise | 作用 |
|---|---:|---|
| Base Prior | 否 | 预测 coarse 底座轨迹，提供低频移动意图 |
| Base plan | 是 | 生成 refined Base waypoints |
| Manipulator plan | 是 | 生成 EEF pose 与 hand configuration |

Base Prior 不是第二个独立模型，而是同一个 WAM/DiT 中的 clean horizon-aware queries：

```text
base_prior_i = learnable_query_i + horizon_embedding_i
```

它通过 shared attention 读取 language、state、video 和 3D scene。中间或最终 hidden
state 经轻量 MLP 输出 B_anchor 中的 coarse Base waypoints：

```text
coarse_base_prior = MLP(base_prior_hidden)
L_base_prior = SmoothL1(coarse_base_prior, GT_base_waypoints)
```

最终 noisy Base/Manipulator tokens 在同一次 DiT forward 中读取 Base Prior hidden
states并完成联合 refinement。因此该结构是端到端的“内部 coarse-to-refined
planning”，不是先运行一个 coarse planner、再运行另一个 policy。

当前远程代码中尚未发现独立 Base Prior embedding、coarse head 或
`L_base_prior`；它是 proposal 中应保留并接入现有 dual-plan WAM 的模块。

### 5.2 推荐的两步 3D 集成

#### Step A：加入 clean 3D history context

先冻结 tokenizer，让 WAM 同时处理：

```text
clean history multi-view z_2d
+ noisy future multi-view z_2d
+ clean history z_3d
+ clean Base Prior
+ noisy Base/Manipulator plans
 -> WAM
```

这一阶段同时预测 multi-view future video、coarse Base prior 和 refined dual plans，
但暂不生成 future `z_3d`。它用于验证 metric 3D context 是否改善 Base Prior、
action refinement 和 future video。

#### Step B：加入 future 3D denoising

只有 Step A 有稳定增益后，再增加：

- 3D noise/input projection；
- 3D token type/time embedding；
- 3D output projection；
- `L_3d_flow`；
- 可选的 decoded future PointMap auxiliary loss。

同一 future chunk 的多视角 2D、3D、Base plan 和 Manipulator plan tokens 应共享 flow
timestep 与 causal block index。Base Prior 保持 clean，不作为 flow generation
variable。

### 5.3 不能直接忽略的 3D token budget

raw `z_3d` 有：

```text
9 * 384 = 3456 tokens/sample
```

直接拼入 DiT self-attention 会显著增加二次注意力成本。Stage 2 必须选择：

- learned spatial resampler；
- 3D cross-attention context；
- 面向 WAM 的更粗 3D grid；
- 局部/稀疏 3D attention。

MVP 推荐 resampler 或 cross-attention，并使用 matched-compute baseline。若 future
generation 需要从压缩 tokens 恢复 384-token geometry，还需对应 expansion decoder，
不能只做单向 pooling。

### 5.4 多视角 2D latent 接口

VGGT 输出：

```text
z_2d_video [B,V,48,9,10,20]
```

其中每个 view 的 `[B,48,9,10,20]` 已与 Wan video latent 的 channel/time/space
合同一致。正确的 WAM pipeline 不再选择 primary view，也不再做一次 33→9 或
160×320→10×20 压缩，而是：

```text
[B,V,48,9,10,20]
 -> 对每个 view 使用 WAM 现有 video patch embedding
 -> 加 view embedding / camera identity
 -> 在 video token sequence 中保留 view 维
 -> WAM 联合预测所有 view 的 future latent tokens
 -> reshape 回 [B,V,48,9,10,20]
 -> VGGT decode_2d()
 -> [B,33,V,3,160,320]
```

实现上可以先 reshape 为 `[B*V,48,9,10,20]` 复用同一套 video patch embedding，
再恢复 B/V 并把各 view tokens 拼入同一个 WAM sequence。不能始终把 view 当作独立
batch，否则 head/wrist 无法在 WAM 内交互；也不应简单把 V 拼入 channel，因为会破坏
48-channel video projector 和 VGGT decoder 合同。

因此“VGGT 2D latent 只在 shape 上兼容 Wan”的准确含义是：

- 它已经完成时空压缩，可以直接作为 WAM video latent；
- 每个 view 的 tensor shape 与 Wan stream 对齐；
- 需要扩展 WAM 的 view-aware token layout；
- 需要根据统计量做 normalization/scale adaptation；
- 不需要再增加一个 video tokenizer。

所有 view 同时预测是目标设计，而不是后续可选增强。相应 token/显存成本需要通过
video patch 后的真实 sequence length 评估，但不应通过丢弃 wrist view 来规避。

### 5.5 latent normalization 与冻结

VGGT `z_2d` 的 tensor contract 与 Wan 对齐，但 latent 分布由新的 VAE 学得；
`z_3d` 则是 deterministic 新分布。接入前应统计 train/val 的：

- per-channel mean/std；
- norm 与 outlier；
- time/view/voxel 位置间方差；
- train/val drift。

WAM adapter 可以使用固定 normalization 或 learned affine projection。推荐训练顺序：

1. 冻结完整 VGGT tokenizer；
2. 训练 multi-view video adapter、Base Prior、3D adapter 和现有 WAM modules；
3. 证明 WAM 使用多视角/3D 后，再考虑解冻 aggregator LoRA；
4. DINO 始终冻结。

---

## 6. Stage 2 目标函数

Step A：

```text
L_WAM_A =
    L_multiview_video_flow
  + lambda_prior * L_base_prior
  + lambda_base * L_base_flow
  + lambda_manip * L_manipulator_flow
```

其中 `L_base_prior` 直接监督 clean Base Prior 的 coarse waypoints；
`L_base_flow/L_manipulator_flow` 监督最终 refined plans。三者不是重复目标：

- prior 学“底座大致应该去哪里”；
- Base flow 学 refined Base trajectory；
- Manipulator flow 在同一 mobility context 下学习 EEF/hand trajectory。

Step B 增加：

```text
L_WAM_B =
    L_WAM_A
  + lambda_3d * L_3d_flow
  + optional lambda_render * L_future_PointMap
```

future PointMap loss 通过冻结的 3D decoder 计算，应低权重、warmup，并使用高置信
mask。Base Prior 还可以通过预测的 base/camera motion 与 future 3D/PointMap 建立
projection consistency，但该项应在基本 coarse prior 与 3D flow 稳定后再加入。

IK、collision 和复杂 feasibility loss 属于后续 action-policy 研究，不应与 MVP
同时引入，否则难以判断增益来源。

---

## 7. 推理

### Step A

```text
head/wrist observation
 -> frozen VGGT tokenizer
 -> multi-view z_2d + history z_3d
 -> clean Base Prior queries
 -> WAM joint flow sampling
 -> future head/wrist z_2d
 -> coarse Base prior
 -> refined Base plan + Manipulator plan
 -> VGGT decode_2d() -> future head/wrist video
```

### Step B

```text
head/wrist observation
 -> frozen VGGT tokenizer
 -> clean history multi-view z_2d/z_3d
 -> Base Prior guided joint 2D/3D/action sampling
 -> future head/wrist video
 -> coarse/refined plans
 -> optional future PointMap
```

控制采用 receding horizon：每次新观测后重新建立 `B_anchor`、编码和规划，不依赖
长期 open-loop 3D map。

---

## 8. 评估

### 8.1 Stage 1

报告：

- head/wrist RGB reconstruction；
- PointMap coordinate/Euclidean error；
- inside-grid、ray-valid、surface/free、multiview coverage；
- 2D/3D latent statistics；
- 与 2D-only tokenizer 的重建和成本对比。

Stage 1 成功要求：

- 3D branch 不显著损伤多视角 2D reconstruction；
- 3D 输出优于无相机几何的简单基线；
- loss 下降不是由有效监督 coverage 变空造成。

### 8.2 Stage 2

核心 ablation：

```text
A. 原 Wan VAE + dual-plan WAM
B. VGGT multi-view z_2d + dual-plan WAM
C. B + Base Prior
D. C + history z_3d context
E. D + future z_3d denoising（可选）
```

同时报告：

- head/wrist future video quality 与 cross-view consistency；
- Base Prior coarse waypoint error；
- refined Base/Manipulator plan metrics；
- task/rollout success；
- future geometry（仅 E）；
- 显存、吞吐和推理延迟。

还应使用：

- no-Base-Prior ablation；
- shuffled/masked 3D context；
- matched-capacity 非几何 context；
- single-view 与 multi-view 对照；

验证收益来自 Base mobility prior 和几何表示，而不是单纯新增参数或 token。

---

## 9. 主要风险

- **Pseudo geometry**：H.264 range 与 nominal calibration 会引入系统性误差；
- **B0 coverage**：前向 `X[0,3]` grid 可能无法覆盖转向、后退或长 rollout；
- **Distribution mismatch**：VGGT 2D/3D latent 需要统计和归一化；
- **Multi-view token cost**：同时预测 head/wrist 会增加 video sequence length；
- **3D token cost**：raw 384-token grid 可能使注意力成本不可接受；
- **Base Prior collapse**：prior 可能退化成普通 query，或被 refined Base tokens 忽略；
- **Representation ignored**：WAM 可能完全忽略 3D context。

所有实验应同时报告 coverage、token 数、吞吐和 matched-compute baseline。

---

## 10. 实施顺序

1. 完成 Stage 1 训练，验证 head/wrist RGB、PointMap、coverage 和 latent statistics。
2. 固化 tokenizer checkpoint 与 multi-view `encode()/decode_2d()/decode_3d()` 接口。
3. 将全部 view 的 `z_2d` 接入 WAM video stream，验证 multi-view latent round-trip。
4. 在现有 dual-plan WAM 中加入 clean Base Prior queries、coarse head 和
   `L_base_prior`。
5. 实现 Step A：加入只读 history `z_3d` context。
6. 对比 single-view、multi-view、Base Prior 和 metric-3D ablations。
7. 只有 Step A 有稳定增益时，实施 Step B 的 future 3D flow。
8. 最后再考虑联合微调 LoRA、扩大 grid、精细 occupancy 或 feasibility loss。

开始 Stage 2 编码前，需要明确：

1. multi-view video tokens 在 causal block 中的排列和 attention mask；
2. Base Prior token 数量及其与 6 个 plan horizons 的对应方式；
3. 3D 使用 resampler 还是 cross-attention；
4. 可接受的显存、吞吐和推理延迟预算。
