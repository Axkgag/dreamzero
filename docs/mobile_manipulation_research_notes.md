# DreamZero 用于 Mobile Manipulation 的研究记录

> 本文是概念讨论记录，不是实现状态说明。当前代码边界见
> [MM/README.md](./MM/README.md)，正式实施顺序见
> [MM/MOBILEMANIBENCH_FINAL_RESEARCH_IMPLEMENTATION_PLAN.md](./MM/MOBILEMANIBENCH_FINAL_RESEARCH_IMPLEMENTATION_PLAN.md)。

本文档用于记录围绕 DreamZero 改进到 mobile manipulation 场景的论文方向讨论。后续讨论结果可以继续追加和修订。

## 1. 背景判断

DreamZero 当前更接近 tabletop 或 fixed-base manipulation 的 World Action Model / VLA 路线：输入多视角 RGB 视频、语言、机器人状态和动作，模型同时学习未来视频预测和动作预测。

Mobile manipulation 相比固定机械臂有几个结构性难点：

- 时间尺度不匹配：导航和接近目标通常是秒级到分钟级，局部抓取和操作则需要更高频控制。
- 动作空间异构：base velocity、arm joint / EEF delta、gripper 的频率、尺度和约束不同。
- 视觉分布非平稳：机器人底盘移动会导致背景、视角、遮挡和目标尺度剧烈变化。
- 需要空间记忆：目标、障碍物、可操作区域经常在当前视野外，需要利用历史观测。
- 安全和可行性约束更强：base collision、arm reachability、self-collision、路径可行性不能完全依赖 imitation loss。

因此，如果希望产出论文，问题不应只定义为“把 DreamZero fine-tune 到 mobile manipulation”，而应围绕 mobile manipulation 的核心难点提出方法贡献。

## 2. 当前推荐主线：Tri-Factorized Flow Matching

当前更有价值、也更可行的主线是：

> Tri-Factorized Flow Matching for 3D-Consistent World Action Models

或：

> Video-3D Consistent DreamZero for Mobile Manipulation

核心观点是：DreamZero 当前能预测未来视频和动作，但 mobile manipulation 需要的物理世界模型不应只生成视觉上合理的 2D 视频，而应理解动作导致的 3D 世界变化。Tri-Factorized Flow Matching 将 future video、robot action 和 3D scene evolution 作为三个互相关联但噪声时间尺度不同的生成目标，在共享 world latent 中联合建模，并通过 video-3D projection consistency 约束它们描述同一个物理未来。

这条路线相比“把 3D token 作为额外 condition”更强，因为 3D 分支不是为了增加输入模态，而是承担预测 action-conditioned 3D evolution 的责任，并要求模型预测的 2D 未来必须能被 3D 世界变化解释。

### 2.1 三个 factor 的职责

Tri-Factorized Flow Matching 包含三个主要分支：

- video flow：预测未来视觉变化，保留 DreamZero 的 video world model 能力。
- action flow：预测可执行控制，建议采用低频 base waypoint 和高频 EEF delta。
- 3D scene flow：预测动作导致的几何变化，例如 future depth、point flow、occupancy change。

三者共享 DreamZero 的 world latent，但承担不同角色：

```text
video: 未来看起来是什么样？
3D:    物理场景如何运动和变化？
action: 机器人应该如何执行？
```

### 2.2 与 DreamZero 当前机制的关系

DreamZero 现有代码已经有 video/action 解耦的接口雏形：video 可以使用独立 timestep，action 也可以使用独立 timestep，并且 DiT 内部将 video tokens、action tokens、state tokens 拼接后共享 transformer。

因此新方法不是另起炉灶，而是将：

```text
video + action
```

自然扩展为：

```text
video + action + 3D scene evolution
```

进一步地，action 可以采用 mobile manipulation 更合适的 factorization：

```text
base: local-frame waypoint / delta pose，低频
arm:  EEF delta + gripper，高频
```

### 2.3 Noise Schedule 设计

三类生成目标可以使用不同的 noise schedule：

```text
t_video       ~ Beta(alpha_v, beta_v)
t_3d          ~ Beta(alpha_g, beta_g) or Uniform
t_action_base ~ low-frequency flow schedule
t_action_eef  ~ high-frequency flow schedule
```

一个合理的初始设定：

- video：偏高噪声，学习粗粒度视觉动态和场景变化。
- 3D：中等噪声或偏低噪声，强调几何稳定性和精确 depth / flow。
- base waypoint：低频、长 horizon，可以偏高噪声，鼓励全局站位规划。
- EEF delta：高频、短 horizon，使用 uniform 或更充分 denoising，保证控制精度。

这样可以形成更完整的：

> Tri-Factorized Flow Matching + Multi-Rate Action Flow

为了避免论文主线过散，可以将 video/action/3D 三因子作为第一主贡献，将 base waypoint / EEF delta 分频作为 action factorization 的实现和第二贡献。

### 2.4 Video-3D Projection Consistency

Video-3D Consistency 是该方向的核心。目标是让模型预测的 future video 和 future 3D scene evolution 能互相解释。

给定当前 depth `D_t`、相机内参 `K`、相机位姿 `T_t`，以及模型预测的 3D point flow `F_3D`：

```text
P_t = unproject(D_t, K, T_t)
P_{t+k}^{pred} = P_t + F_3D
D_{t+k}^{proj}, flow_{t->t+k}^{proj} = project(P_{t+k}^{pred}, K, T_{t+k})
```

然后约束：

```text
L_depth = || D_{t+k}^{proj} - D_{t+k}^{gt} ||

L_flow = || flow_{t->t+k}^{proj} - flow_{t->t+k}^{video/gt} ||

L_warp = photometric_or_feature_warp(
    RGB_t, RGB_{t+k}, flow_{t->t+k}^{proj}
)
```

如果没有真实 scene flow，也可以从 RGB-D 序列、相机位姿和 future depth 构造弱监督。

### 2.5 推荐训练目标

整体 loss 可以写为：

```text
L = L_video
  + L_action
  + L_3d
  + L_proj_consistency
```

其中：

- `L_video`：DreamZero 原有 future video / latent flow matching loss。
- `L_action`：waypoint + EEF delta 的 action flow matching loss。
- `L_3d`：future depth、point flow 或 occupancy change 的 3D supervision。
- `L_proj_consistency`：将 predicted 3D evolution 投影回相机视角，与 future RGB / depth / flow 对齐。

该设计的关键不是多加一个 depth loss，而是要求 video future 和 3D future 在几何上描述同一个未来。

### 2.6 VGGT-Based 3D Token Pretraining

当前一个更具体的第一阶段方案是使用 VGGT 作为多视角 image encoder，并在其 image tokens 之上构建一组共享的 metric 3D tokens。VGGT 的价值在于提供更强的多视角几何先验，但核心贡献不应表述为“换了更好的 image encoder”，而应表述为：

> 用相机内外参约束 2D-to-3D lifting，使共享 3D tokens 具备明确的空间语义，并能被重新渲染回各个视角的 depth。

#### 3D Tokens 的构造

3D tokens 不应是普通 learnable latent，而应绑定到机器人 base frame、world frame 或 reference camera frame 下的固定 3D 坐标。可以定义局部 BEV / voxel / BEV-height 网格：

```text
token_i <-> cell center P_i = (x_i, y_i, z_i)
F_i = MLP(pos_embed(P_i)) + learnable_feature_i
```

这样每个 token 的 index 对应一个真实空间位置，例如机器人前方某个 `x, y, z` cell。为了控制计算量，第一版可以优先使用 BEV-height tokens，而不是高分辨率 dense voxel。

#### 2D-to-3D 聚合

对每个 3D query `P_i`，利用相机内参 `K_v` 和外参 `T_v` 投影到每个视角：

```text
p_{i,v} = project(P_i, K_v, T_v)
```

然后在该投影点附近从 VGGT image tokens 中采样特征，并通过 deformable attention 或加权融合写入同一组 3D tokens：

```text
f_{i,v} = bilinear_sample(image_tokens_v, p_{i,v})
F_i = Attention(query=F_i, key/value={f_{i,v}})
```

这一步是几何可解释性的关键：3D token 只能从其投影位置附近获得多视角证据，而不是任意 attend 全图。多视角图片因此共享同一组 3D tokens，同一个空间 cell 的证据会来自所有可见视角。

#### Ray-Based Depth Decoder

从共享 3D tokens 解码某个视角的 depth 时，不建议直接用普通 CNN / MLP 生成 depth image。更合理的方式是 ray-based decoder：对目标 view 的每个像素 `(u, v)`，根据相机参数构造 3D ray：

```text
P(d) = O + d * r(u, v)
```

沿 ray 采样多个候选深度 `d_j`，在每个采样点从 3D token grid 中插值得到特征：

```text
P_j = O + d_j * r
g_j = trilinear_sample(3D_tokens, P_j)
logit_j = MLP(g_j, pos_embed(P_j), ray_dir)
```

对所有 `logit_j` 做 softmax 得到每个候选深度是表面的概率，再用期望得到预测深度：

```text
p_j = softmax(logit_j)
D_pred(u, v) = sum_j p_j * d_j
```

监督可以使用：

```text
L_depth = L1(D_pred, D_gt)
        + CE(depth_bin_logits, gt_depth_bin)
        + optional scale-invariant log depth loss
```

这个 decoder 可以理解为一个轻量可微渲染器：同一组 3D tokens 是场景表示，不同相机的 rays 决定从哪个视角渲染 depth。

#### 避免 2D / 3D 分支退化为独立 Autoencoder

如果 image tokens 只负责重建 RGB，3D tokens 只负责预测 depth，模型可能学成两个相对独立的 autoencoder。为避免这种退化，第一阶段应加入跨视角和跨模态约束：

- masked view training：随机 drop 某些视角的 image tokens，仍要求从共享 3D tokens 解码被遮掉视角的 depth 或 feature。
- cross-view depth consistency：将一个视角预测的 depth unproject 成点云，再 project 到其他视角，与对应 depth 对齐。
- 3D-to-2D feature reconstruction：用 `3D tokens + target camera pose` 重建目标视角的 VGGT / DINO / VAE feature，而不仅是 depth。
- image-token dropout / bottleneck：限制 image decoder 直接复制完整 image tokens，迫使跨视角结构进入 3D tokens。
- occupancy / free-space supervision：如果有 RGB-D 或仿真数据，可从 depth 构造 occupied / free / unknown voxel label，对 metric 3D tokens 加空间占用监督。

一个最小可行的第一阶段目标为：

```text
L_stage1 =
    L_depth_all_views
  + lambda_1 L_cross_view_depth
  + lambda_2 L_masked_view_feature_or_depth
  + optional lambda_3 L_occupancy
```

这样训练得到的 3D tokens 既能从多视角 image tokens 中聚合信息，又能通过相机 rays 解码回各个视角的 depth，并保留明确的几何可解释性。

### 2.7 最小可行实现路径

建议分三阶段实现：

1. future depth head：训练 `future video + action + future depth`，用 `L_video + L_action + L_depth` 做 proof-of-concept。
2. point flow head：预测 per-point 3D displacement，并通过 projected depth / projected flow 约束 future video 和 future depth。
3. action-conditioned consistency：让 predicted waypoint / EEF delta 参与 3D evolution prediction，并让 3D evolution 反过来约束 action 的可行性。

初期不建议直接做完整 occupancy 或 object-level SE(3)，因为标注和工程复杂度更高。

### 2.8 为什么适合 Mobile Manipulation

Mobile manipulation 中至少有三类变化混在一起：

- camera ego-motion：base 移动导致视角变化。
- robot motion：arm / EEF 进入视野并接触物体。
- object motion：物体被推动、抓取、放置。

纯 2D video prediction 容易把这些都建模成 texture motion。3D scene evolution 分支可以强迫模型区分哪些变化来自相机移动、哪些来自机器人动作、哪些来自物体真实运动、哪些区域是遮挡或显露。

这正是 mobile manipulation 需要的物理世界模型能力。

### 2.9 Base-Prior Guided Joint Action Flow

在 action 设计上，当前更倾向于采用 **base-prior guided joint action flow**，而不是完全解耦 base 和 arm。完全解耦虽然能处理执行频率差异，但容易导致 base 和 arm 动作割裂；两段式独立模型虽然能先预测粗 base action 再预测 whole-body action，但会增加模型重量、推理延迟和训练复杂度。

更好的折中方案是在一个共享 DiT 内部加入少量 base prior tokens：

```text
[video tokens]
[3D / depth / BEV tokens]
[base prior tokens]
[noisy base waypoint tokens]
[noisy EEF delta tokens]
[state tokens]
```

其中：

- `base prior tokens`：clean latent query，不加噪声，负责从 language、3D scene、video 和 state 中提取低频移动意图。
- `noisy base waypoint tokens`：flow matching 的生成变量，最终被 denoise 成 refined base waypoint。
- `noisy EEF delta tokens`：flow matching 的生成变量，最终被 denoise 成 EEF delta 和 gripper action。

这可以理解为轻量化的“内部两段式”：

```text
base prior tokens -> coarse base waypoint -> joint base + EEF refinement
```

但它只需要一个 DiT、一次 forward、一个端到端训练过程。

#### Base Prior Tokens 如何初始化

推荐使用 learnable horizon-aware query tokens，类似 DETR object queries 或 Q-Former queries：

```text
base_prior_tokens = learnable_query + horizon_pos_embedding
```

例如设置 4 个 base prior tokens：

```text
token 0: 0.5s waypoint
token 1: 1.0s waypoint
token 2: 2.0s waypoint
token 3: 4.0s waypoint
```

batch 时扩展为：

```text
base_prior = base_prior_tokens[None].expand(B, -1, -1)
```

经过 DiT attention 后，这些 query tokens 会变成当前场景和任务下的 mobility prior。

#### Base Prior 的监督

base prior tokens 可在 DiT 中间层通过一个轻量 MLP head 预测 coarse local-frame waypoints：

```text
base_prior_tokens at layer k -> coarse_waypoints [N_prior, 3]
```

监督标签来自未来 base pose。当前时刻为 `t`，未来 base pose 为 `B_{t+i}`，转换到当前 base local frame：

```text
waypoint_i = transform_to_local(B_{t+i}, B_t)
```

使用 SmoothL1、L1 或 lightweight flow matching loss：

```text
L_base_prior = SmoothL1(pred_coarse_waypoints, gt_coarse_waypoints)
```

这样 base prior 学到的是“机器人应该大致往哪里站”，而不是直接输出最终控制。

#### 3D 引导 Base Prior

为了让 base prior 不只是普通 query，可以让它在 early layers 中更关注 3D / BEV / depth tokens、language 和 state，而减少对 noisy EEF tokens 的依赖：

```text
early layers:
  base_prior attends to language + 3D + state
  base_prior less attends to noisy EEF tokens

later layers:
  EEF/base action tokens attend to base_prior + video + 3D
```

第一版可以先使用 full attention，依靠 `L_base_prior` 和 downstream action loss 学习；后续可以加入 attention mask 或 gating，让 3D 对 base prior 的影响更明确。

#### 与 Video-3D Consistency 的连接

base prior tokens 不只用于 action head，也应参与 3D scene evolution：

```text
base prior tokens
  -> coarse waypoint
  -> predicted camera/base motion
  -> 3D scene evolution
  -> projected video/depth consistency
```

也就是说，coarse base waypoint 可以预测未来相机 / base 位姿变化；该位姿变化与当前 depth / point cloud 一起决定 future projected depth 或 projected flow。再通过 `L_proj_consistency` 与 future RGB / depth 对齐。

这样 base prior 的学习目标不仅是拟合 demonstration 轨迹，还要产生能解释未来 3D 几何和视频变化的物理意义。

#### 推荐的 Action Branch 表述

当前更合适的表述不是“完全 factorized base-arm policy”，而是：

> Base-prior guided joint action flow

即：

```text
learnable base prior tokens provide low-frequency mobility context,
while final action tokens jointly generate refined base waypoint and EEF delta.
```

这既保留了 base/arm 执行尺度不一致的建模能力，又避免了完全解耦带来的动作割裂。

## 3. 可尝试的优化方向

### 3.1 Factorized Base-Arm Action Head

Mobile manipulation 的 base action 和 arm action 不应简单拼接成一个统一 action vector。可以设计 factorized action head：

- shared world latent
- base action head：低频预测 `vx, vy, yaw_rate` 或 waypoint
- arm action head：高频预测 joint / EEF delta
- gripper / contact head：事件式或低维 binary / continuous 输出

潜在论文贡献：面向异构动作空间的 world-action model。

### 3.2 Hierarchical DreamZero

高层模块输出 subgoal、waypoint、object target 或 skill instruction，低层 DreamZero 输出连续 base+arm action。

可研究的问题：

- 高层 subgoal 如何从 language 和 scene memory 中生成？
- 低层 DreamZero 如何利用 subgoal 做 closed-loop execution？
- video rollout 能否用于判断 subgoal 是否达成？

风险是如果系统过度依赖 LLM / planner，论文容易变成工程集成。因此需要强调 DreamZero 的 world rollout 或 subgoal grounding 贡献。

### 3.3 Video Rollout as MPC / Reranking

DreamZero 能预测未来 video，这一点在 mobile manipulation 中可以作为 planning signal。

可行做法：

- sample 多条 candidate action chunks。
- 对每条 action 预测 future video 或 future geometry。
- 使用 goal scorer、VLM scorer、collision scorer 或 progress scorer 评估。
- 执行最优 action chunk。

潜在论文贡献：把 world model prediction 真正用于闭环控制，而不仅是辅助训练。

### 3.4 Geometry Auxiliary Supervision

即使部署时不一定使用完整点云，也可以在训练阶段加入几何辅助任务：

- depth prediction
- free-space / obstacle occupancy prediction
- object mask prediction
- contact prediction
- scene flow / point flow prediction
- arm reachability map prediction

这类辅助监督可以增强 video backbone 的空间和物理表征，工程风险比直接端到端学习 3D dynamics 更低。

需要注意：单纯将 3D 特征作为 condition 融入 DreamZero 不是足够强的论文贡献。更重要的是设计监督信号和一致性约束，让 3D 分支自然承担“解释 2D 视频未来”和“预测物理场景变化”的职责。因此本方向中，3D 的核心价值应来自 action-conditioned 3D scene evolution 和 video-3D projection consistency，而不是来自额外输入模态本身。

### 3.5 Curriculum / Mixture Training

建议采用分阶段训练：

1. manipulation-only pretraining：使用 DROID、RoboMIND 等数据保留抓取和局部操作能力。
2. sim mobile manipulation training：使用仿真数据学习 base motion、空间记忆和长程任务。
3. real mobile manipulation fine-tuning：使用少量真实移动机械臂数据做适配。

潜在论文贡献：适用于 world-action model 的 mobile adaptation recipe。

### 3.6 Failure-Aware Training

Mobile manipulation 的失败模式很多，例如：

- 找不到目标
- base 停位不好
- 手臂够不到
- 碰撞
- 抓取失败
- 遮挡导致目标丢失

可加入：

- success / failure classifier
- recovery action prediction
- replan trigger
- preference learning
- offline RL objective

如果使用包含 failure demonstrations 的数据集，这个方向会更自然。

### 3.7 Camera / View Robustness

移动机器人经常依赖 head camera、wrist camera 或少量外部视角。可尝试：

- camera pose embedding
- egocentric view dropout
- synthetic camera perturbation
- learned view fusion
- wrist-view importance weighting
- 替代固定 2x2 grid 拼图的 view-token fusion

该方向单独作为论文可能偏弱，但非常适合作为主方法组件。

## 4. 数据集和评测环境

当前方案对数据的理想需求是：

- mobile base + arm / EEF action，最好能拆成 base waypoint 和 EEF delta。
- RGB-D 或可生成 depth / point cloud 的多视角数据。
- camera intrinsics / extrinsics、base odometry 或相机位姿，用于 video-3D projection consistency。
- 长程任务、语言指令、subgoal / primitive annotation，便于监督 base prior tokens。
- robot / object states 或 segmentation，便于构造 future depth、point flow、occupancy change、contact / reachability 等辅助监督。

### 4.1 真实数据

- [BRMData](https://arxiv.org/abs/2405.18860)：bimanual-mobile household manipulation 数据集，包含单臂/双臂、桌面/移动操作、多视角 RGB、depth sensing、base action 和 arm action。
  - 适合点：真实 household mobile manipulation，传感器和动作字段与当前方案高度匹配，适合验证 base waypoint + EEF delta、video-depth consistency 和 3D spatial tokens。
  - 局限：规模和任务覆盖相比大规模真实数据集更有限，更适合作为真实几何验证和小规模 fine-tuning。

- [AIRoA MoMa Dataset](https://arxiv.org/abs/2509.25032)：真实 mobile manipulation 数据，包含 Human Support Robot 上采集的 25,469 episodes / 约 94 小时数据，提供 RGB、joint states、六轴 wrist force-torque、internal robot states，并带有 sub-goal 和 primitive action 的两层 annotation；数据标准化为 LeRobot v2.1。
  - 适合点：与 DreamZero 当前 LeRobot / GEAR 数据管线兼容性好；sub-goal / primitive annotation 很适合监督 base prior tokens、primitive tokens 和 failure-aware recovery。
  - 局限：公开信息中未强调 RGB-D / point cloud，因此它更适合 action coordination、hierarchical learning 和真实策略 fine-tuning，不应作为第一阶段 3D tokenizer 的唯一几何监督来源。

- [RoboMIND 2.0](https://arxiv.org/abs/2512.24653)：大规模真实多 embodiment 数据，包含 310K+ dual-arm manipulation trajectories、739 tasks、12K tactile-enhanced episodes、20K mobile manipulation trajectories，并提供 20K digital-twin simulated trajectories。
  - 适合点：规模大，包含 mobile manipulation 和 digital twin，适合真实 fine-tuning、cross-embodiment generalization、sim-to-real transfer。
  - 局限：需要进一步确认其公开数据字段是否包含足够的 camera pose / depth / point cloud，以支持精确 projection consistency。

- [Mobile ALOHA](https://arxiv.org/abs/2401.02117)：低成本 whole-body teleoperation 系统和 bimanual mobile manipulation demos，任务包括厨房、柜子、电梯、水槽等长程移动双臂操作。
  - 适合点：任务形态非常贴近 mobile manipulation，适合参考 action representation、whole-body control 和小数据 co-training 方案。
  - 局限：不是大规模通用数据集，更适合作为真实任务 demo / small-scale evaluation 参考。

- DROID：适合作为 manipulation-only pretraining 数据，不是 mobile manipulation 主数据，但可以保留 DreamZero 原有的局部操作能力。

### 4.2 仿真和 benchmark

- [MobileManiBench](https://arxiv.org/abs/2602.05233)：非常适合本方案的仿真 benchmark。它基于 NVIDIA Isaac Sim，包含 2 个 mobile platform、head / wrist 两个同步相机、RGB-depth-segmentation、多对象和机器人状态/动作、语言指令、630 objects、100 realistic scenes、100+ tasks 和 300K trajectories。
  - 适合点：几乎完整覆盖 Tri-Factorized 所需监督：RGB-D、segmentation、object / robot states、actions、language。非常适合做 future depth、point flow、occupancy、projection consistency 和 base-prior ablation。
  - 建议定位：当前方案的第一优先级主数据和主 benchmark，尤其适合 VGGT spatiotemporal tokenizer、dynamic 3D tokens 和 WAM video-action-3D 联合训练。

- [BiGym](https://proceedings.mlr.press/v270/chernyadev25a.html)：面向 bimanual mobile manipulation 的家庭任务 benchmark，包含 40 个长程家庭任务、human demonstrations，并支持 RGB、depth、多相机视角和 proprioception。
  - 适合点：任务形态与 household mobile manipulation 高度贴合，适合评估长程双臂移动操作、遮挡、base repositioning 和多视角视频建模。
  - 局限：需要确认其动作空间和数据导出格式能否直接接入 LeRobot / GEAR；更适合作为仿真评测和中等规模验证。

- ManiSkill / ManiSkill-HAB：适合快速构造可控 ablation 和大规模仿真数据。ManiSkill 支持 RGB-D、segmentation、robot/object states 和 GPU 并行采集；ManiSkill-HAB 面向 Home Assistant Benchmark 的移动操作任务，工程上比 BEHAVIOR-1K 更适合快速迭代。
  - 适合点：非常适合验证 tokenizer 压缩比、depth / occupancy supervision、projection consistency、action factorization 等模块级消融。
  - 局限：任务复杂度和真实家庭长程交互不一定覆盖完整 mobile manipulation，需要与更真实的 benchmark 搭配使用。

- BEHAVIOR-1K / OmniGibson：1000 个日常活动，长程复杂交互，适合 benchmark 和 sim pretraining。2025 BEHAVIOR Challenge 也体现了其对 long-horizon mobile manipulation 的评测价值。
  - 适合点：复杂家庭任务和长程 mobile manipulation，适合评估 generalization 和系统能力。
  - 局限：工程集成和任务复杂度较高，适合作为中后期 benchmark。

- Habitat 2.0 / Home Assistant Benchmark：长程 rearrangement、家庭场景、mobile manipulation，可用于 navigation + manipulation 层级问题。
  - 适合点：适合验证 base prior、navigation-manipulation coordination、long-horizon subgoal planning。

- EMMOE：面向 open-environment embodied mobile manipulation 的长程语言任务和评测指标。
  - 适合点：适合评估 open-environment generalization 和 instruction following。

### 4.3 3D / Flow World Model 相关数据和弱监督来源

这些不一定是 mobile manipulation 主数据集，但与 Video-3D Consistency 和 action-conditioned 3D scene evolution 高度相关，可用于设计监督、预训练或参考数据构造方式。

- [PointWorld](https://arxiv.org/abs/2601.03782)：将状态和动作统一到 3D point flow 空间，训练数据覆盖约 2M trajectories / 500 小时真实与仿真 manipulation。其核心是给定 RGB-D 和 action，预测 3D point displacement。
  - 适合点：与我们 3D scene evolution 分支高度一致，可参考其 3D point flow action/world representation。

- [3DFlowAction / ManiFlow-110k](https://arxiv.org/abs/2506.06199)：构造大规模 3D optical flow 数据，学习 language-conditioned 3D object flow world model，再用 flow 约束机器人 action planning。
  - 适合点：可参考如何从视频中构造 3D flow / object motion pseudo labels，以及如何将 3D flow 作为 action planning 约束。

- [FlowDreamer](https://arxiv.org/abs/2505.10075)：RGB-D world model，显式预测 3D scene flow，再用 scene flow 辅助 future frame prediction。
  - 适合点：与我们 “3D evolution explains video future” 的方向非常接近，可参考其 3D scene flow + diffusion future frame 的模块化训练方式。

- [RynnWorld-4D / Rynn4DDataset](https://arxiv.org/abs/2607.06559)：联合生成 future RGB、depth 和 optical flow，并构建大规模 RGB-DF 4D 数据。
  - 适合点：与 video-depth-flow tri-branch consistency 很接近，可作为 RGB-D-flow pseudo-label 体系和 4D world model 设计参考。
  - 注意：该工作很新，需要关注数据可用性和代码开放情况。

### 4.4 当前优先级建议

按照与当前 VGGT video tokenizer + dynamic 3D tokens + WAM 联合训练方案的匹配度，推荐优先级为：

1. **MobileManiBench**：最适合作为主实验和架构验证数据集。它同时提供多视角 RGB-D、segmentation、robot/object states、actions 和 language，能直接支撑 video compression、3D token 聚合、future depth / occupancy / point flow 和 projection consistency。
2. **BRMData + BiGym**：最适合验证真实或仿真 household bimanual mobile manipulation。BRMData 更偏真实多视角 RGB-D 和 base/arm action 验证；BiGym 更适合长程家庭任务 benchmark。
3. **AIRoA MoMa**：最适合接入 LeRobot / GEAR pipeline，并用于 base prior tokens、subgoal / primitive supervision、contact-rich action learning 和 failure recovery，但不是最强的几何监督来源。
4. **RoboMIND 2.0**：适合大规模真实 fine-tuning、cross-embodiment generalization 和 sim-to-real。如果确认公开字段包含稳定 depth / camera pose / point cloud，可以提升为主数据之一。
5. **ManiSkill / ManiSkill-HAB**：适合快速做模块级消融和合成数据生成，尤其是 depth、segmentation、occupancy、camera pose、action factorization 和 tokenizer 压缩比实验。
6. **BEHAVIOR-1K / OmniGibson**：适合中后期 long-horizon benchmark，用于验证复杂家庭任务、unseen layout、long-horizon spatial memory 和系统级 generalization。
7. **DROID**：只建议作为 manipulation-only pretraining 数据，帮助保留 DreamZero 原有局部操作能力，不适合作为 mobile manipulation 或 3D-consistent WAM 的主数据。
8. **PointWorld / 3DFlowAction / FlowDreamer / RynnWorld-4D**：不建议作为主 benchmark，但非常适合借鉴 3D flow supervision、pseudo-label 构造、RGB-D-flow consistency 和 video-3D consistency 训练方式。

## 5. 推荐论文方案

一个完整方案可以定义为：

> Tri-Factorized Flow Matching for 3D-Consistent Mobile Manipulation

### 5.1 主要贡献点

1. 提出 Tri-Factorized Flow Matching，在共享 DreamZero world latent 中联合建模 video、action 和 3D scene evolution，并为三者设计不同 noise schedule。
2. 提出 Video-3D Projection Consistency，使 predicted future video 必须能被 predicted 3D scene evolution 在几何上解释。
3. 提出 Base-Prior Guided Joint Action Flow，在单个共享 DiT 内部用 learnable horizon-aware base prior tokens 提供低频 mobility context，再联合生成 refined base waypoint 和 EEF delta。
4. 将 base prior 接入 3D scene evolution，使 coarse waypoint 不只是动作先验，也能通过 projected depth / flow consistency 获得物理监督。
5. 在 long-horizon mobile manipulation benchmark 上验证 3D-consistent world model 对 unseen layout、遮挡、base repositioning 和 manipulation success 的提升。

### 5.2 实验 baseline

可比较的 baseline：

- DreamZero RGB-only
- DreamZero + proprio / base state
- DreamZero + current depth only
- DreamZero + future depth auxiliary loss
- DreamZero + 3D tokens as condition
- DreamZero + 3D flow prediction without projection consistency
- hierarchical planner + BC / Diffusion Policy
- full Tri-Factorized Flow Matching

### 5.3 Ablation

建议做以下消融：

- no 3D branch
- no projection consistency
- future depth only vs point flow
- shared timestep vs tri-factorized timesteps
- concat action head vs waypoint / EEF factorized action head
- fully decoupled base/arm heads vs base-prior guided joint action flow
- no base prior tokens
- learnable base prior tokens vs horizon-aware base prior tokens
- no intermediate coarse waypoint supervision
- full attention vs 3D-guided base prior attention / gating
- same-frequency action vs multi-rate action
- no video-3D consistency during action prediction
- predicted action conditioned 3D evolution vs ground-truth action conditioned 3D evolution

### 5.4 Metrics

Mobile manipulation 评测不应只看整体 success rate。建议记录：

- task success
- subgoal success
- navigation success
- manipulation success
- collision rate
- path efficiency
- action smoothness
- recovery after failure
- unseen rooms / unseen objects / unseen instructions generalization
- future depth error
- projected flow error
- 3D Chamfer distance
- video-3D consistency error

## 6. 最现实的落地路径

按工程风险从低到高，建议路线如下：

1. 将 mobile robot 数据转成 LeRobot / GEAR 格式，至少包含 `base_state`、`arm_state`、`base_action`、`arm_action`、`video`、`language`。
2. 建立 RGB-only mobile DreamZero baseline。
3. 将 action 表示改成低频 base waypoint + 高频 EEF delta，建立 concat / factorized action baseline。
4. 加入 learnable horizon-aware base prior tokens，并用中间层 coarse waypoint loss 监督。
5. 将 base prior 作为 final joint action flow 的 condition，联合预测 refined base waypoint 和 EEF delta。
6. 加入 future depth head，验证 3D auxiliary supervision 是否提升 video/action representation。
7. 加入 point flow 或 scene flow head，开始建模 action-conditioned 3D scene evolution。
8. 将 base prior 接入 3D scene evolution，通过 coarse waypoint 预测 camera/base motion，并加入 projected depth / flow consistency。
9. 加入 tri-factorized noise schedules，比较 shared timestep 与 video/action/3D 独立 timestep。
10. 最后加入 closed-loop rollout reranking 或 MPC-style selection，形成完整系统。

如果资源有限，优先做：

- future depth + projection consistency
- point flow + video-3D consistency
- base-prior guided joint waypoint / EEF action flow

这三个组件最贴合当前主线，也最容易形成清晰的方法贡献。

## 7. 相关参考文献和可借鉴思路

本节记录与 Tri-Factorized Flow Matching 和 Video-3D Projection Consistency 思路相近的近一年左右文献。范围不局限于机器人 world model，也包括自动驾驶、视频生成、多模态生成和 flow matching，因为这些方向中有很多可借鉴的结构设计。

### 7.1 自动驾驶 World Model / 3D-Consistent Generation

这些工作和我们的 Video-3D Consistency 关系最直接。自动驾驶领域长期面对多相机、LiDAR、BEV、occupancy、trajectory planning 的跨模态一致性问题，与 mobile manipulation 中的 RGB-D / 3D scene evolution / action consistency 很接近。

- [Genesis: Multimodal Driving Scene Generation with Spatio-Temporal and Cross-Modal Consistency](https://arxiv.org/abs/2506.07497), 2025.
  - 核心思路：联合生成 multi-view driving videos 和 LiDAR sequences，并通过 shared latent space 保持视觉和几何模态的一致演化。
  - 可借鉴点：我们的 video branch 和 3D branch 也可以通过 shared world latent 耦合，而不是将 3D 仅作为 condition；评估上可参考同时报告 video metrics、LiDAR / Chamfer metrics 和 downstream task gains。

- [GenieDrive: Towards Physics-Aware Driving World Model with 4D Occupancy Guided Video Generation](https://arxiv.org/abs/2512.12751), 2025.
  - 核心思路：先预测 4D occupancy，再用 occupancy 指导 multi-view video generation，以提高物理一致性和多视角一致性。
  - 可借鉴点：这与我们“3D scene evolution 不是辅助输入，而是物理中间表示”的想法高度一致。Mobile manipulation 中可将 `4D occupancy` 替换为 local point flow、future depth、free-space / contact occupancy。

- [CoGen: 3D Consistent Video Generation via Adaptive Conditioning for Autonomous Driving](https://arxiv.org/abs/2503.22231), 2025.
  - 核心思路：用更高质量的 3D conditions 替代粗糙 2D layout condition，提高 driving video generation 的空间一致性。
  - 可借鉴点：说明 2D condition 不足以约束几何一致性。但我们的方案应进一步强调 `3D evolution prediction + projection consistency`，避免只停留在 3D condition。

- [WAM-Flow: Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving](https://arxiv.org/abs/2512.06112), 2025.
  - 核心思路：将 ego-trajectory planning 建模为 structured token space 上的 discrete flow matching，并支持 coarse-to-fine parallel denoising。
  - 可借鉴点：base waypoint prediction 可以采用 coarse-to-fine flow matching；mobile base 的 waypoint token 也可设计 metric-aligned tokenizer 或 geometry-aware objective。

- [Flow Matching-Based Autonomous Driving Planning with Advanced Interactive Behavior Modeling](https://arxiv.org/abs/2510.11083), 2025.
  - 核心思路：用 flow matching 生成多模态 driving trajectories，并通过细粒度轨迹 tokenization 和 classifier-free guidance 建模交互行为。
  - 可借鉴点：对 base waypoint / EEF delta 的 action tokenization 有参考价值，尤其是 overlapping segment tokenization 和 inference-time guidance。

### 7.2 视频生成中的 3D / View Consistency

这些工作虽然不一定包含 action，但对 Video-3D Projection Consistency 很有启发：它们都在解决“视频看起来真实但 3D 不一致”的问题。

- [GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control](https://arxiv.org/abs/2503.03751), 2025.
  - 核心思路：维护一个 3D cache，用 seed image 或已生成帧预测出的 point cloud 来渲染新视角条件，从而提升相机控制和时间 3D 一致性。
  - 可借鉴点：mobile manipulation 中可以维护 local 3D cache / scene memory，让视频分支生成未来帧时受到显式几何渲染约束。

- [View-Consistent Diffusion Representations for 3D-Consistent Video Generation](https://arxiv.org/abs/2511.18991), 2025.
  - 核心思路：研究 diffusion representation 的 multi-view consistency，并通过学习 view-consistent representations 提升 3D-consistent video generation。
  - 可借鉴点：除了像素级 projection loss，还可以在 DreamZero latent / DiT features 上做 video-3D representation consistency。

- [Voyager: Long-Range and World-Consistent Video Diffusion for Explorable 3D Scene Generation](https://arxiv.org/abs/2506.04225), 2025.
  - 核心思路：联合生成 RGB 和 depth video，并维护 world cache 来支持长程 3D scene exploration。
  - 可借鉴点：对 mobile manipulation 的 long-horizon spatial memory 很有参考价值，尤其是 `RGB + depth joint generation` 和 `world cache`。

- [DepthSync: Diffusion Guidance-Based Depth Synchronization for Scale- and Geometry-Consistent Video Depth Estimation](https://arxiv.org/abs/2507.01603), 2025.
  - 核心思路：针对长视频 depth prediction 的 scale drift 和 geometry inconsistency，引入 scale guidance 和 geometry guidance。
  - 可借鉴点：如果我们的 3D branch 先从 future depth 做起，需要特别注意跨时间窗口的 depth scale consistency，可借鉴这种 guidance / consistency 设计。

- [Video Depth Anything: Consistent Depth Estimation for Super-Long Videos](https://arxiv.org/abs/2501.12375), 2025.
  - 核心思路：提升长视频 depth estimation 的时间一致性。
  - 可借鉴点：可作为离线 pseudo-depth / future depth label 生成工具，也可参考其 temporal consistency loss。

### 7.3 机器人 / VLA 中的 Flow Matching 与异步动作生成

这些工作和 Tri-Factorized Flow Matching、multi-rate action flow 更相关，尤其适合参考 action tokenization、asynchronous denoising 和 chunked action execution。

- [AsyncVLA: Asynchronous Flow Matching for Vision-Language-Action Models](https://arxiv.org/abs/2511.14148), 2025.
  - 核心思路：传统 VLA flow matching 使用同步均匀时间表，AsyncVLA 引入 asynchronous flow matching，让不同 action tokens 按非均匀时间表生成，并通过 confidence rater 选择性 refine。
  - 可借鉴点：与我们 `t_action_base != t_action_eef` 的 multi-rate / asynchronous action flow 非常接近。可参考其训练同步/异步模式兼容的方案。

- [SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics](https://arxiv.org/abs/2506.01844), 2025.
  - 核心思路：使用 flow matching 进行连续动作生成，并用 asynchronous inference stack 解耦 perception/action prediction 与 action execution。
  - 可借鉴点：支持我们对 mobile manipulation 做低频 world perception、高频 action execution 解耦的设计。

- [3DFlowAction: Learning Cross-Embodiment Manipulation from 3D Flow World Model](https://arxiv.org/abs/2506.06199), 2025.
  - 核心思路：从人类和机器人数据学习 3D object optical flow world model，再用生成的 3D flow 指导机器人动作规划。
  - 可借鉴点：非常接近“3D scene evolution 应该指导 action”的观点。我们的区别可以是：不是单独预测 3D flow 再优化 action，而是在 DreamZero 中联合建模 video/action/3D，并用 projection consistency 绑定。

- [Action Flow Matching for Continual Robot Learning](https://arxiv.org/abs/2504.18471), 2025.
  - 核心思路：用 flow matching 在在线学习中修正 action，使其更匹配不断变化的 robot dynamics。
  - 可借鉴点：对真实 robot adaptation、model mismatch 和 continual fine-tuning 有参考价值，尤其是 mobile base / arm 动力学不一致时。

### 7.4 多模态共享潜空间和跨模态一致性

这些工作不一定面向机器人，但对 Tri-Factorized 的 shared world latent 有启发。

- [ShaLa: Multimodal Shared Latent Space Modelling](https://arxiv.org/abs/2508.17376), 2025.
  - 核心思路：学习多模态数据之间的 shared latent representation，并支持 joint multimodal synthesis 和 cross-modal inference。
  - 可借鉴点：Tri-Factorized Flow Matching 也需要 video/action/3D 在共享 latent 中对齐，同时保留各自 modality-specific details。

### 7.5 对当前方案的启发总结

从以上文献可以归纳出几个对我们最重要的设计启发：

- 3D 不应只是 condition，而应是能预测未来几何状态的中间世界表示。GenieDrive、Genesis、3DFlowAction 都支持这一点。
- Video-3D consistency 可以在多个层次做：pixel/depth projection、flow projection、latent representation alignment、shared latent coupling。
- Flow matching 不必使用统一时间表。AsyncVLA、WAM-Flow 和 driving planning 相关工作都说明，非同步、粗到细、token-wise 或 modality-wise 的 flow schedule 是合理方向。
- 自动驾驶中的 4D occupancy / LiDAR-video consistency 可以迁移到 mobile manipulation 中，替换为 local RGB-D、point flow、free-space occupancy、contact / reachability。
- 论文贡献应避免表述为“加了 3D 输入”，而应表述为“video、action、3D evolution 在共享 latent 中用不同 flow clocks 联合生成，并通过 projection consistency 形成物理约束”。

## 8. 后续待讨论问题

- 目标硬件平台是什么？例如 Stretch、Hello Robot、AgileX、双臂移动平台、自研底盘等。
- 是否有真实 mobile manipulation 数据？包含哪些传感器？
- 是否有 RGB-D、相机内外参、base odometry 或 SLAM pose？
- 主要任务是家庭长程任务、室内取放、开门开柜，还是移动到桌边后的局部操作？
- 预期论文更偏方法、系统、数据集，还是 benchmark？
- 可用算力是否足够 fine-tune DreamZero 14B，还是更适合 Wan2.2 5B / LoRA 路线？
