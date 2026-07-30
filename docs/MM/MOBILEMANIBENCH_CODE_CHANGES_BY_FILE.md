# MobileManiBench 双路 Plan：早期工作树逐文件快照

> **历史文档。** 本文记录 Phase 2 开发时相对当时 Git HEAD 的工作树，不再代表
> 2026-07-30 当前仓库状态。当前实现请阅读
> [MOBILEMANIBENCH_CODE_CHANGES_BY_FILE_PHASE0-2.md](./MOBILEMANIBENCH_CODE_CHANGES_BY_FILE_PHASE0-2.md)，
> 总体状态见 [README.md](./README.md)。

## 1. 文档目的与统计范围

本文用于帮助逐文件学习当前 MobileManiBench → DreamZero 双路 Action Plan 实现。

统计基准是远程仓库：

```text
/mnt/yihao/codes/dreamzero
```

并以该仓库当前工作树相对当前 Git `HEAD` 的差异为准。

本文明确排除：

```text
docs/
sim-evals/
```

因此：

- 不总结已有研究文档的变化；
- 不总结嵌套 `sim-evals` 仓库及其依赖锁文件；
- 不重复介绍当前 Git `HEAD` 中已经存在且没有工作树差异的文件；
- 只覆盖 DreamZero 主仓库中当前 22 个已修改或新增的代码、配置、脚本和测试文件。

文件状态含义：

```text
M = 相对当前 HEAD 已修改
N = 当前工作树新增、尚未被当前 HEAD 跟踪
```

## 2. 当前实现处于什么阶段

当前完成的是 MobileManiBench 最终研究路线中的 Phase 0、Phase 1 和 Phase 2 核心链路：

```text
转换后的 realized future state
        ↓
Base waypoint plan [6, 4]
Manipulator plan [6, 21]
        ↓
分路归一化、有效维度 mask、未来时刻 valid mask
        ↓
DreamZero 兼容打包 action [12, 21]
        ↓
6 个 Base tokens + 6 个 Manipulator tokens
        ↓
共享 Wan video DiT
        ↓
独立 Base/Manipulator decoder
        ↓
base_flow_loss + manipulator_flow_loss
```

这里的“解耦”表示：

- Base 和 Manipulator 有独立的输入投影；
- 有不同的 token type embedding；
- 有独立的输出投影；
- 分别计算 loss；
- 但两路 token 进入同一个 Wan DiT，可以在共享 Transformer 中交互。

它不是两个完全独立、互不通信的网络。

当前已经真实通过一次训练 step：

```text
输入视频            [1, 3, 33, 352, 640]
Base tokens         6
Manipulator tokens  6
总 action tokens    12
Forward             通过
Backward            通过
Optimizer step      通过
```

当前尚未实现：

- MobileManip validation dataset；
- checkpoint 批量推理评估器；
- Base ADE/FDE；
- EEF position/orientation error；
- Hand joint error；
- Base/Manipulator consistency metric；
- reachability metric；
- MobileManip 闭环仿真评测。

## 3. 必须先理解的张量协议

### 3.1 时间接口

```text
plan_horizon      = 6
plan_time_offsets = [1, 4, 8, 12, 16, 24]
control_fps       = 30
```

这 6 个 waypoint 对应的未来时间为：

```text
[1/30, 4/30, 8/30, 12/30, 16/30, 24/30] 秒
```

它们不是均匀时间采样，所以代码显式保留 `plan_time_offsets`，不能只给 token 普通序号 `0..5`。

### 3.2 Base plan

```text
shape = [6, 4]

slice:
0:2  relative XY
2:4  yaw 的 sin/cos
```

归一化策略：

```text
XY       q01/q99 → [-1, 1]
yaw      保持 sin/cos，不做 q99
```

### 3.3 Manipulator plan

统一最大形状：

```text
shape = [6, 21]
```

语义：

```text
0:3     EEF XYZ
3:9     EEF rotation6d
9:      hand configuration
```

不同机器人实际 hand 维度不同：

```text
G1     原生 manipulator_dim = 10
XHand  原生 manipulator_dim = 21
```

G1 不存在的尾部维度补零，并由 `manipulator_dim_mask` 排除。

归一化策略：

```text
EEF XYZ       q01/q99 → [-1, 1]
rotation6d    保持原值
hand joints   每关节 q01/q99 → [-1, 1]
padding       保持 0，loss mask 为 false
```

### 3.4 DreamZero 内部打包

DreamZero 原有 action head 接收一个矩形 action 张量。双路计划因此被打包为：

```text
packed action [12, 21]

token 0..5:
    Base plan
    只有前 4 维有效
    后 17 维补零且 mask=false

token 6..11:
    Manipulator plan
    最多 21 维
```

这只是存储层面的矩形打包，不表示 Base 是 21 维动作。

### 3.5 视频和 Wan block

每路原始相机先缩放到：

```text
[33, 176, 320, 3]
```

再组成 DreamZero 2×2 grid：

```text
top-left      head
top-right     wrist
bottom-left   black
bottom-right  black
```

最终 Wan 视频输入：

```text
[B, 3, 33, 352, 640]
```

经过 Wan VAE：

```text
33 RGB frames → 9 latent frames
9 = 1 condition frame + 8 future latent frames
```

因此当前双路计划使用：

```text
num_frame_per_block  = 8
num_action_per_block = 12
num_state_per_block  = 1
```

即一个完整的 8-frame future video block 对应一个 12-token plan window。

## 4. 文件总览

| 状态 | 文件 | 核心职责 |
|---|---|---|
| M | `groot/vla/data/dataset/__init__.py` | 导出双路 Plan dataset |
| M | `groot/vla/data/transform/__init__.py` | 导出 Plan normalization transform |
| M | `groot/vla/experiment/base.py` | 记录任意新增的分支 loss |
| M | `groot/vla/model/dreamzero/action_head/__init__.py` | 导出双路 action head |
| M | `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` | 为研究 action head 增加可重写 hook |
| M | `groot/vla/model/dreamzero/modules/__init__.py` | 导出双路 Wan DiT |
| M | `groot/vla/model/dreamzero/transform/__init__.py` | 导出模型 transform 和 collator |
| M | `scripts/train/mobilemanibench_training.sh` | 加固原 step-action baseline 启动脚本 |
| N | `groot/vla/configs/data/dreamzero/mobilemanibench_plan.yaml` | 双路 Plan 数据配置 |
| N | `groot/vla/configs/model/dreamzero/action_head/mobile_plan_flow_matching.yaml` | 双路 action head 配置 |
| N | `groot/vla/configs/model/dreamzero/transform/mobile_plan_cotrain.yaml` | 双路模型 transform 配置 |
| N | `groot/vla/data/dataset/mobilemanibench_plan.py` | 读取 materialized realized plans |
| N | `groot/vla/data/transform/mobile_plan.py` | slice-aware normalization、mask 和反归一化 |
| N | `groot/vla/model/dreamzero/action_head/mobile_plan_flow_matching.py` | 双路 flow loss 和时间/block 协议 |
| N | `groot/vla/model/dreamzero/modules/wan_video_dit_dual_plan.py` | 独立双路 token 编解码器 |
| N | `groot/vla/model/dreamzero/transform/mobile_plan_cotrain.py` | 数据到 DreamZero action/video/state 的适配 |
| N | `scripts/data/inspect_mobilemanibench_plan_batch.py` | 人工检查一个计划样本 |
| N | `scripts/data/prepare_mobilemanibench_plan_metadata.py` | 生成计划统计量和几何 QA |
| N | `scripts/train/mobilemanibench_plan_training.sh` | 双路 Plan smoke 训练入口 |
| N | `tests/data/test_mobilemanibench_plan_dataset.py` | dataset 和标签重建测试 |
| N | `tests/data/test_mobilemanibench_plan_transform.py` | normalization/mask/inverse 测试 |
| N | `tests/model/test_mobile_plan_phase2.py` | 双路模型投影、loss、timestep 测试 |

## 5. 数据层文件

### 5.1 `groot/vla/data/dataset/mobilemanibench_plan.py`

状态：新增。

这是双路计划数据入口，也是理解整个 Phase 1 的第一个核心文件。

#### 为什么不继续使用普通 LeRobot action loader

普通 loader 通常把 action 的第一维理解成额外的时间采样维度。但转换后的每一行 parquet 已经包含完整未来计划：

```text
action.plan.base_waypoints [6, 4]
action.plan.manipulator    [6, native_dim]
action.plan.valid          [6]
```

如果再让普通 loader 对其采样一次 horizon，就会把计划 horizon 扩展两次。

`MobileManiBenchPlanDataset` 的原则是：

```text
observation 仍交给 LeRobotSingleDataset 读取
plan 直接从当前 parquet row 取出
```

#### 初始化时读取的元数据

```text
meta/robot_schema.json
meta/extensions.json
meta/plan_stats.json（存在时）
```

从中确定：

- waypoint offsets；
- control FPS；
- hand joint indices；
- 当前机器人原生 manipulator 维度；
- Base/Manipulator 声明形状。

#### observation modalities

```text
state.eef_position
state.eef_rotation_rpy
annotation.task
video.head
video.wrist
```

dataset 使用 `xdof` embodiment tag，并将底层 metadata 暴露为：

```python
self.merged_metadata = {"xdof": self.observation_dataset.metadata}
```

这是为了兼容 `BaseExperiment` 保存训练 metadata 的接口。

#### `__getitem__` 输出

主要字段：

```text
base_plan
manipulator_plan
plan_valid
base_dim_mask
manipulator_dim_mask
plan_time_offsets
plan_time_seconds
episode_index
frame_index
hand_dim
```

原生 Manipulator 计划被复制到统一 `[6,21]` 张量前部，剩余部分补零。

#### 学习重点

重点看清三类 mask 的区别：

```text
plan_valid             某个未来时刻是否真实存在
base_dim_mask          Base 的哪些维度属于真实语义
manipulator_dim_mask   当前机器人有哪些 hand 维度
```

它们在下一层组合成最终 loss mask。

#### 当前边界

- dataset 没有 train/validation split 管理；
- `_trajectory_cache` 一次只保留一个 episode；
- 只面向当前 `xdof` MobileManiBench root；
- 依赖转换后的 parquet 已经物化 plan labels。

### 5.2 `groot/vla/data/dataset/__init__.py`

状态：修改。

新增导出：

```python
MobileManiBenchPlanDataset
```

作用是让 Hydra 配置可以使用：

```text
_target_: groot.vla.data.dataset.MobileManiBenchPlanDataset
```

这个文件没有算法逻辑，只负责 Python package public API。

### 5.3 `groot/vla/data/transform/mobile_plan.py`

状态：新增。

这是 Phase 1 的第二个核心文件，负责：

- 按 slice 归一化；
- 检查几何表示；
- 构造最终 action mask；
- 将预测值反归一化回物理量。

#### 统计量输入

必须二选一：

```text
stats_path
statistics
```

同时提供或同时不提供都会报错。

必须包含：

```text
statistics.base_xy
statistics.eef_xyz
statistics.hand
```

#### q01/q99 归一化

正向：

```text
x_norm = 2 * (x - q01) / (q99 - q01) - 1
clip 到 [-1, 1]
```

逆向：

```text
x = (x_norm + 1) / 2 * (q99 - q01) + q01
```

常量维度使用 0 作为归一化结果，逆变换恢复为 q01。

#### 不归一化的几何 slice

```text
Base yaw sin/cos
EEF rotation6d
```

原因是这两组值自身已经是有结构的旋转表示，直接逐维 q99 会破坏单位圆或正交结构。

#### 几何 QA

只在 `plan_valid=true` 的 waypoint 上检查：

```text
所有值有限
sin²(yaw)+cos²(yaw) ≈ 1
rotation6d 两行分别为单位向量
rotation6d 两行点积 ≈ 0
```

#### 最终 mask

```text
base_action_mask =
    base_dim_mask AND plan_valid[..., None]

manipulator_action_mask =
    manipulator_dim_mask AND plan_valid[..., None]
```

因此：

- episode 尾部不存在的未来 waypoint 不进 loss；
- G1 不存在的 hand padding 不进 loss。

#### `unapply`

将：

```text
base_action        → base_plan
manipulator_action → manipulator_plan
```

这将是后续离线 ADE/FDE 评估的基础，但当前还没有评估 runner 使用它。

### 5.4 `groot/vla/data/transform/__init__.py`

状态：修改。

新增导出：

```python
MobilePlanTransform
```

同样属于 package registration，没有额外算法。

## 6. 模型输入适配文件

### 6.1 `groot/vla/model/dreamzero/transform/mobile_plan_cotrain.py`

状态：新增。

这个文件连接：

```text
Phase-1 dataset sample
        ↓
DreamZero 原有 DreamTransform
        ↓
WANPolicyHead 输入
```

包含两个类：

```text
MobilePlanDataCollator
MobilePlanCotrainTransform
```

#### `MobilePlanDataCollator`

复用原 `DefaultDataCollator` 完成：

- NumPy stack；
- tokenizer；
- text attention mask；
- embodiment ID batch。

然后强制检查：

```text
base_action              [B, 6, 4]
manipulator_action       [B, 6, 21]
base_action_mask         [B, 6, 4]
manipulator_action_mask  [B, 6, 21]
plan_time_offsets        [B, 6]
```

这些检查让 shape 错误在进入 14B 模型之前尽早暴露。

#### 相机缩放

`_resize_video` 将每路 THWC 视频缩放为：

```text
[T, 176, 320, 3]
```

使用 bilinear interpolation，整数图像 round/clamp 后恢复 `uint8`。

这一步是后续修复中新增的关键逻辑。没有它时，两路 520×520 图像会拼成 1040×1040，导致：

- Wan token 数与 `frame_seqlen=880` 不一致；
- 显存和计算量异常增大。

#### 视频 canonicalization

构造三个 view：

```text
view 0 = head
view 1 = black
view 2 = wrist
```

再交给 `DreamTransform` 的普通 2×2 grid 逻辑，形成：

```text
top-left=head, top-right=wrist
bottom row=black
```

#### state canonicalization

当前 state 由：

```text
EEF position [3]
EEF RPY      [3]
```

拼接为 6 维，用 `meta/stats.json` 的 `observation.state` q01/q99 归一化。

模型配置保持：

```text
max_state_dim = 64
```

因此原 `DreamTransform` 会把有效 6 维 state padding 到 checkpoint 兼容的 64 维。不能把网络 state projector 改成 44 或 6，否则无法加载 DreamZero-AgiBot checkpoint。

#### action packing

```text
Base [6,4]
  ↓ pad
Base packed [6,21]

Base packed [6,21]
+ Manipulator [6,21]
  ↓ concat
action [12,21]
```

mask 使用完全相同的布局。

#### 保留语义分支字段

即使 DreamZero 主 action 入口读取 packed `action`，transform 仍额外保留：

```text
base_action
manipulator_action
base_action_mask
manipulator_action_mask
plan_valid
plan_time_offsets
plan_time_seconds
```

这样 action head 可以分路算 loss，后续评估也不必从 packed tensor 猜语义。

### 6.2 `groot/vla/model/dreamzero/transform/__init__.py`

状态：修改。

导出：

```python
MobilePlanCotrainTransform
MobilePlanDataCollator
```

### 6.3 `groot/vla/configs/model/dreamzero/transform/mobile_plan_cotrain.yaml`

状态：新增。

该配置继承：

```text
dreamzero_cotrain
```

并替换：

- data collator；
- model-specific transform。

关键配置：

```text
max_action_dim = 21
max_state_dim  = 64
```

`max_state_dim=64` 是预训练 checkpoint 的结构要求，不表示 MobileManip 实际提供 64 个有意义 state。

配置还把以下参数传入 transform：

```text
plan_horizon
base_action_dim
manipulator_action_dim
state_stats_path
control_fps
image_resolution_height
image_resolution_width
```

## 7. 双路 Wan DiT 文件

### 7.1 `groot/vla/model/dreamzero/modules/wan_video_dit_dual_plan.py`

状态：新增。

这是 Phase 2 最核心的 token architecture 文件。

包含：

```text
PlanOffsetEmbedding
DualPlanActionEncoder
DualPlanActionDecoder
WanVideoDiTDualPlan
```

#### `PlanOffsetEmbedding`

输入不是 token ordinal index，而是物理秒：

```text
offset_seconds = plan_time_offsets / control_fps
```

先经过 sinusoidal positional encoding，再经过：

```text
Linear → SiLU → Linear
```

这使模型能够区分不均匀的真实时间间隔。

#### `DualPlanActionEncoder`

两路分别使用：

```text
base_encoder
manipulator_encoder
```

还使用两个 learnable type embeddings：

```text
type_embedding[0] = Base
type_embedding[1] = Manipulator
```

每路 token 最终为：

```text
action projection
+ physical offset embedding
+ branch type embedding
```

输出仍按：

```text
[6 Base tokens, 6 Manipulator tokens]
```

拼接后进入共享 Wan blocks。

#### `DualPlanActionDecoder`

共享 Transformer hidden states 再次分成两路：

```text
前 6 token → base_decoder → [B,6,4]
后 6 token → manipulator_decoder → [B,6,21]
```

为了适配统一输出，Base decoder 的 `[B,6,4]` 被补零为 `[B,6,21]`，然后与 Manipulator 输出拼回 `[B,12,21]`。

#### `WanVideoDiTDualPlan`

继承原：

```text
CausalWanModel
```

但替换 action encoder/decoder。

构造时强制：

```text
action_dim = manipulator_action_dim
num_action_per_block = 2 * plan_horizon
```

forward 时必须显式传入 `plan_time_offsets`，且必须与模型注册的 offsets 完全一致；错误 offsets 会立即报错。

#### 当前架构边界

- 两路只在 projection 和 loss 层面解耦，Transformer 主干共享；
- `num_embodiments` 当前固定为 1；
- 还没有加入 Base/Manipulator consistency loss；
- 还没有 VGGT 2D/3D token；
- 还没有 reachability/collision 模块。

### 7.2 `groot/vla/model/dreamzero/modules/__init__.py`

状态：修改。

导出：

```python
WanVideoDiTDualPlan
```

## 8. Action head 文件

### 8.1 `groot/vla/model/dreamzero/action_head/mobile_plan_flow_matching.py`

状态：新增。

这个文件负责双路计划在 diffusion/flow matching 训练层面的协议。

#### `MobilePlanPolicyHeadConfig`

新增配置字段：

```text
plan_horizon
base_action_dim
manipulator_action_dim
plan_time_offsets
control_fps
base_flow_loss_weight
manipulator_flow_loss_weight
```

初始化时检查：

```text
action_horizon == 2 * plan_horizon
action_dim == manipulator_action_dim
```

#### 显式时间 offsets

`prepare_action_model_kwargs`：

- 从 batch 读取 `plan_time_offsets`；
- 与 config 中 offsets 做逐元素严格比较；
- 通过 keyword argument 传给双路 DiT。

#### Base/Manipulator timestep 对齐

当前使用同一 plan window 的 coupled diffusion timestep：

```text
t_base(h) = t_manipulator(h)
```

在当前 one-block 设计中，12 个 action tokens 都使用该 block 的 timestep。

#### action/video block 校验

双路计划要求：

```text
future latent frames = num_frame_per_block = 8
action tokens        = num_action_per_block = 12
state tokens         = num_state_per_block = 1
```

这取代了原 action head 假设的“每个小视频块分配若干 action token”的比例校验。

#### 分路 masked loss

Base：

```text
prediction[:, :6, :4]
```

Manipulator：

```text
prediction[:, 6:, :21]
```

每个 token 的 MSE 只对 active dims 求平均，再结合：

```text
action mask
has_real_action
diffusion timestep weight
```

最终：

```text
action_loss =
    base_weight * base_flow_loss
  + manipulator_weight * manipulator_flow_loss
```

输出日志字段：

```text
action_loss
base_flow_loss
manipulator_flow_loss
```

#### 推理输出拆分

`get_action` 在原 `action_pred` 之外增加：

```text
base_plan_pred
manipulator_plan_pred
```

注意：这里只完成模型输出拆分；完整 checkpoint 推理 runner、反归一化和离线指标尚未实现。

### 8.2 `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`

状态：修改。

这个文件是原 DreamZero WAN action head。修改原则是：

```text
不把 legacy 路径硬改成 MobileManip
而是增加可由研究 head override 的 hooks
```

新增 hooks：

```text
prepare_action_model_kwargs
align_action_timestep_ids
validate_action_video_layout
build_coupled_action_timestep_ids
compute_action_losses
```

默认实现保持原单路行为；`MobilePlanFlowMatchingActionHead` 覆盖需要变化的部分。

forward 的结构变化：

```text
准备额外 model kwargs
→ 调用可重写 layout validator
→ 调用可重写 timestep builder
→ 调用 timestep aligner
→ 将额外 kwargs 传给 DiT
→ 调用可重写 action loss
→ 把分支 loss 加入输出
```

这是一个重要的可扩展性改造：未来 3D tokens 或其他 embodiment 可以继续增加子类，而不必复制整段 WAN forward。

#### 与原行为的兼容性

默认：

- `prepare_action_model_kwargs` 返回空 dict；
- timestep 不额外对齐；
- 使用 legacy layout assertion；
- 使用原单流 masked MSE。

因此原 DreamZero action head 理论上保持旧行为。

### 8.3 `groot/vla/model/dreamzero/action_head/__init__.py`

状态：修改。

导出：

```python
MobilePlanFlowMatchingActionHead
MobilePlanPolicyHeadConfig
```

### 8.4 `groot/vla/configs/model/dreamzero/action_head/mobile_plan_flow_matching.yaml`

状态：新增。

继承：

```text
wan_flow_matching_action_tf
```

关键全局配置：

```text
plan_horizon              = 6
action_horizon            = 12
max_action_dim            = 21
num_action_per_block      = 12
base_action_dim           = 4
max_manipulator_action_dim= 21
```

替换：

```text
action_head_cfg._target_
diffusion_model_cfg._target_
```

分别指向：

```text
MobilePlanFlowMatchingActionHead
WanVideoDiTDualPlan
```

两路 loss 默认权重均为 1。

`max_state_dim` 显式传给双路 DiT，以保证 state encoder 与 transform、checkpoint 一致。

## 9. 数据 Hydra 配置

### 9.1 `groot/vla/configs/data/dreamzero/mobilemanibench_plan.yaml`

状态：新增。

该文件选择双路 Plan dataset，而不是原 step-action dataset。

核心参数：

```text
plan_horizon      = 6
plan_time_offsets = [1,4,8,12,16,24]
num_frames        = 33
action_horizon    = 12
state_horizon     = 1
```

`train_dataset` 的视频 delta indices 是：

```text
0..32
```

即一次取 33 帧用于 video dynamics。

transform 链：

```text
MobilePlanTransform
→ MobilePlanCotrainTransform
```

第一层处理真实计划的归一化和 mask，第二层将其适配到 DreamZero。

#### 当前边界

该配置只声明 `train_dataset`，没有 MobileManip `val_dataset`。这就是当前训练日志显示：

```text
eval dataloader length: no eval dataloader
```

的原因。

## 10. 统计与人工检查脚本

### 10.1 `scripts/data/prepare_mobilemanibench_plan_metadata.py`

状态：新增。

用途是从转换后的 parquet 计算：

```text
meta/plan_stats.json
```

支持输入：

- 单个机器人 root；
- 同时含 `g1/` 和 `xhand/` 的父目录。

只使用 `plan_valid=true` 的 waypoint 拟合统计量。

输出：

- mean/std/min/max；
- q01/q99；
- normalization policy；
- waypoint 数量；
- hand joint names；
- 几何 QA。

几何 QA 包括：

```text
finite 检查
yaw sin/cos 单位范数误差
rotation6d 行单位范数误差
rotation6d 两行点积误差
```

#### 当前注意事项

`fit_split` 当前字符串写的是：

```text
train (all episodes selected by the converted smoke dataset)
```

它准确描述 smoke 数据，但将来用于正式全量 train split 时应同步更新元数据描述。

### 10.2 `scripts/data/inspect_mobilemanibench_plan_batch.py`

状态：新增。

用于从 dataset 读取一个样本并生成：

```text
PNG  人工可视化
JSON shape/offset/mask/range 摘要
```

PNG 横向拼接：

```text
head image
wrist image
Base XY 与 EEF XY 轨迹面板
```

轨迹点用 horizon index 标注，方便检查：

- 坐标系方向；
- 未来 waypoint 顺序；
- terminal mask；
- Base 与 EEF 轨迹是否明显异常。

这个脚本检查的是真实 label 和 normalization，不是模型 prediction。

## 11. 训练脚本

### 11.1 `scripts/train/mobilemanibench_plan_training.sh`

状态：新增。

这是当前双路 Plan 的正式启动入口。

#### 自包含环境

脚本自动：

- 解析 repo root；
- 使用 `/mnt/yihao/envs/dreamzero`；
- 找到该环境的 `torchrun`；
- 设置 Hydra full error；
- 禁止 Albumentations 更新提示。

因此可以从任意目录直接运行：

```bash
bash /mnt/yihao/codes/dreamzero/scripts/train/mobilemanibench_plan_training.sh
```

#### 当前默认是 smoke overfit

```text
data root       smoke_v2/g1
GPU             1
max steps       200
save steps      100
batch size      1
learning rate   1e-5
```

脚本显式检查：

- dataset metadata；
- plan stats；
- checkpoint directories；
- `save_total_limit >= 5`。

支持：

```text
PREFLIGHT_ONLY=1
```

只检查环境和文件，不启动训练。

#### 双路核心 override

```text
data=dreamzero/mobilemanibench_plan
action_head=mobile_plan_flow_matching
transform=mobile_plan_cotrain
action_horizon=12
num_frame_per_block=8
num_action_per_block=12
num_state_per_block=1
frame_seqlen=880
save_lora_only=true
```

#### 已知限制 1：默认自动续训不成立

默认：

```text
RUN_ID = 当前时间
OUTPUT_DIR = ..._${RUN_ID}
```

DreamZero 底层会在同一个 `OUTPUT_DIR` 中自动查找最后 checkpoint，但脚本每次默认创建新时间戳目录，所以直接重复运行命令不会命中上一次目录。

#### 已知限制 2：最终保存很大

一次真实 1-step 验证发现最终 root save 可能写出约 30GB，并在 NFS 上耗时很长。虽然配置是：

```text
save_lora_only=true
```

但当前完整保存路径仍需要单独审计。全量训练前不应忽略该问题。

#### 已知限制 3：当前没有 evaluation

脚本没有开启 MobileManip validation dataset 或离线轨迹指标。

### 11.2 `scripts/train/mobilemanibench_training.sh`

状态：修改。

这是原 step-action baseline，不是双路 Plan 路径。

修改主要是启动健壮性：

- checkpoint、tokenizer、pretrained path 改为可由环境变量覆盖；
- 增加 `REPORT_TO`、`WANDB_PROJECT`；
- 增加可配置 `MAX_STEPS`、`SAVE_STEPS`、batch size、logging steps；
- 增加 `SAVE_TOTAL_LIMIT >= 5` 检查；
- 增加配置摘要打印；
- 增加 `PREFLIGHT_ONLY`；
- Hydra 参数改为引用上述变量。

学习时必须区分：

```text
mobilemanibench_training.sh
    = 原 DreamZero step-action baseline

mobilemanibench_plan_training.sh
    = 当前 Base/Manipulator dual-plan research path
```

baseline 脚本没有被改写成双路计划。

## 12. 实验日志文件

### 12.1 `groot/vla/experiment/base.py`

状态：修改。

只修改 `LossLoggerCallback`。

原来只记录固定字段：

```text
loss
dynamics_loss_avg
action_loss_avg
learning_rate
```

现在改为：

```text
loss
learning_rate
所有以 _loss_avg 结尾的日志
```

因此新字段：

```text
base_flow_loss_avg
manipulator_flow_loss_avg
```

能够自动写入：

```text
OUTPUT_DIR/loss_log.jsonl
```

这个修改并不负责计算 loss，只负责将 Trainer 已经 log 出来的所有分支 loss 持久化。

## 13. 测试文件

### 13.1 `tests/data/test_mobilemanibench_plan_dataset.py`

状态：新增。

使用远程 smoke_v2 数据执行集成测试。

覆盖：

#### G1

- Base shape `[6,4]`；
- Manipulator padding 后 `[6,21]`；
- 只有前 10 维有效；
- 尾部 padding 为 0；
- horizon 没有被重复扩展；
- terminal sample 的未来计划全部 invalid。

#### XHand

- Manipulator shape `[6,21]`；
- 21 维全部有效；
- hand dimension 为 12。

#### 标签重建

测试重新调用 converter 的 `build_plan_labels`，从保留的：

```text
observation.base.world
observation.eef.world
observation.robot_joint
```

重建 Base/Manipulator realized plans，并与 parquet 存储标签比较。

这项测试证明 label 不是不可追踪的二次产物，而是可以由保留的真实状态确定性重建。

### 13.2 `tests/data/test_mobilemanibench_plan_transform.py`

状态：新增。

对 G1 和 XHand 同时检查：

- action shape；
- q99 slice 落在 `[-1,1]`；
- yaw sin/cos 未被改变；
- rotation6d 未被改变；
- valid mask 与 dimension mask 正确合并；
- `unapply` 能恢复物理量。

因为正向会 clip 到 q01/q99，所以 inverse 的期望值也是 raw value clip 到对应 quantile 范围，而不是无条件恢复所有极端 outlier。

### 13.3 `tests/model/test_mobile_plan_phase2.py`

状态：新增。

这是不依赖 14B checkpoint 的轻量模型单元测试。

覆盖：

#### 双路投影

- encoder/decoder 输出 `[B,12,21]`；
- Base padding 维严格为 0；
- Base encoder、Manipulator encoder 和两个 decoder 都能收到梯度。

#### 分路 loss

- padding 维不影响 loss；
- invalid waypoint 不影响 loss；
- Base 和 Manipulator loss 分别计算。

#### timestep

- Base/Manipulator 使用一致 timestep；
- one video block 的 timestep 能扩展为全部 12 个 plan tokens。

## 14. 推荐学习顺序

建议不要按目录字母顺序阅读，而按数据流阅读。

### 第一阶段：理解标签和归一化

```text
1. groot/vla/data/dataset/mobilemanibench_plan.py
2. groot/vla/data/transform/mobile_plan.py
3. scripts/data/prepare_mobilemanibench_plan_metadata.py
4. scripts/data/inspect_mobilemanibench_plan_batch.py
5. tests/data/test_mobilemanibench_plan_dataset.py
6. tests/data/test_mobilemanibench_plan_transform.py
```

读完应能回答：

- realized plan 从哪里来；
- G1/XHand 维度为何不同；
- 哪些 slice 归一化；
- mask 如何组合；
- terminal waypoint 如何处理。

### 第二阶段：理解 DreamZero 打包

```text
7. groot/vla/model/dreamzero/transform/mobile_plan_cotrain.py
8. groot/vla/configs/model/dreamzero/transform/mobile_plan_cotrain.yaml
9. groot/vla/configs/data/dreamzero/mobilemanibench_plan.yaml
```

读完应能回答：

- `[6,4] + [6,21]` 如何变成 `[12,21]`；
- state 为什么是 64 维；
- 两路相机为何必须先 resize；
- 33 帧和 880 tokens/frame 如何对应。

### 第三阶段：理解双路 tokens

```text
10. groot/vla/model/dreamzero/modules/wan_video_dit_dual_plan.py
11. groot/vla/model/dreamzero/action_head/mobile_plan_flow_matching.py
12. groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py
13. groot/vla/configs/model/dreamzero/action_head/mobile_plan_flow_matching.yaml
14. tests/model/test_mobile_plan_phase2.py
```

读完应能回答：

- 两路在哪里解耦；
- 两路在哪里重新共享计算；
- physical time offsets 如何注入；
- flow loss 如何按 slice 计算；
- 为什么当前 `num_frame_per_block=8`。

### 第四阶段：理解训练入口

```text
15. scripts/train/mobilemanibench_plan_training.sh
16. groot/vla/experiment/base.py
17. scripts/train/mobilemanibench_training.sh
```

读完应能回答：

- baseline 和 research path 的区别；
- 哪些参数来自 Hydra，哪些来自 shell 环境变量；
- checkpoint 保存频率；
- loss 日志保存位置；
- 当前自动续训和保存路径的限制。

最后再阅读各个 `__init__.py`，理解 Hydra target 的 Python import 路径即可。

## 15. 当前代码的完整训练数据流

```text
MobileManiBench converted parquet row
│
├── observation images/state/language
│     └── LeRobotSingleDataset
│
├── action.plan.base_waypoints
├── action.plan.manipulator
└── action.plan.valid
      │
      ▼
MobileManiBenchPlanDataset
      │
      ├── pad manipulator to 21 dims
      ├── dimension masks
      └── physical plan offsets
      │
      ▼
MobilePlanTransform
      │
      ├── q99 normalization by slice
      ├── geometry validation
      └── valid × dimension masks
      │
      ▼
MobilePlanCotrainTransform
      │
      ├── resize/compose video
      ├── normalize/pad state
      └── pack [6 Base + 6 Manipulator] → [12,21]
      │
      ▼
MobilePlanDataCollator
      │
      ▼
MobilePlanFlowMatchingActionHead
      │
      ├── shared plan timestep
      ├── explicit physical offsets
      └── one 8-frame video block ↔ one plan window
      │
      ▼
WanVideoDiTDualPlan
      │
      ├── Base encoder
      ├── Manipulator encoder
      ├── type + offset embeddings
      ├── shared Wan DiT
      ├── Base decoder
      └── Manipulator decoder
      │
      ▼
base_flow_loss + manipulator_flow_loss + dynamics_loss
```

## 16. 现阶段最值得继续检查的风险

这些不是本轮新增修改，而是阅读当前实现时必须知道的边界：

1. 当前没有 validation dataloader，训练 loss 下降不等于泛化有效。
2. 当前没有离线 trajectory metrics。
3. `RUN_ID` 默认时间戳会让直接重启脚本创建新目录，底层自动 resume 无法命中旧 checkpoint。
4. `save_lora_only=true` 下最终保存仍曾产生约 30GB 输出，需要审计。
5. 全量数据转换完成前，脚本默认仍指向两 episode smoke_v2/G1。
6. `plan_stats.json` 必须只用正式 train split 拟合，不能泄漏 validation/test。
7. reachability、collision 和 task success 仍需要机器人模型或仿真环境。
8. 当前实现还没有 Phase 3 的 Base/Manipulator consistency loss。
9. 当前实现还没有 VGGT 2D/3D tokenizer。
10. 双路计划推理输出已能拆分，但还缺完整 checkpoint inference/反归一化 runner。

## 17. 最小验证命令

数据和模型轻量测试：

```bash
cd /mnt/yihao/codes/dreamzero
/mnt/yihao/envs/dreamzero/bin/python -m unittest \
  tests.model.test_mobile_plan_phase2 \
  tests.data.test_mobilemanibench_plan_transform \
  tests.data.test_mobilemanibench_plan_dataset \
  -v
```

训练前文件检查：

```bash
PREFLIGHT_ONLY=1 \
bash /mnt/yihao/codes/dreamzero/scripts/train/mobilemanibench_plan_training.sh
```

当前 smoke 训练：

```bash
bash /mnt/yihao/codes/dreamzero/scripts/train/mobilemanibench_plan_training.sh
```

观察：

```text
base_flow_loss
manipulator_flow_loss
action_loss
dynamics_loss
loss
```

其中应近似满足：

```text
action_loss =
    base_flow_loss
  + manipulator_flow_loss

loss =
    dynamics_loss
  + action_loss
```

默认两个分支 loss 权重均为 1。

## 18. 训练内验证接口与 checkpoint 顺序修复

### 18.1 问题

五任务 Wan2.2-5B 双路 Plan 训练在 step 2000 进入 Hugging Face 默认验证路径时，
`prediction_step()` 把 batch mapping 展开为：

```python
model(state=..., action=..., video=..., ...)
```

DreamZero `VLA.forward()` 的真实合同是：

```python
model(inputs)
```

因此训练正常而验证报：

```text
TypeError: VLA.forward() got an unexpected keyword argument 'state'
```

### 18.2 修复

`groot/vla/experiment/base.py` 新增 DreamZero 专用 `prediction_step()`：

- 复用 Trainer 的 `_prepare_inputs()` 和 autocast context；
- 以单个 positional batch dictionary 调用 `model(inputs)`；
- 只返回 detached validation loss，不跨 rank 收集巨大的视频/action logits。

同文件同时覆盖 `_maybe_log_save_evaluate()`。当 step 同时满足 save/eval 条件且
`save_strategy != best` 时，执行顺序改为：

```text
checkpoint save -> callback on_save -> logging/evaluation
```

因此 validation 即使报错，该 step 的 checkpoint 也已经保存。`best` 策略仍保留上游
顺序，因为它必须先获得验证指标才能决定是否保存。

新增：

```text
tests/experiment/test_base_trainer_prediction_step.py
```

覆盖 batch-dict validation 调用和同 step `save -> evaluate` 顺序。

### 18.3 训练内 validation 规模

五任务 validation split 当前有 56,376 个 anchors。训练脚本默认：

```text
MAX_EVAL_SAMPLES=1024
```

`MobileManiBenchPlanDataset` 在完整 validation 序列上使用 `np.linspace` 取 1,024 个
确定性、等间隔索引。因此训练内验证不是全量验证，也不是每次重新随机抽样；每个
checkpoint 比较的是同一组覆盖完整 validation 顺序的固定子集。完整 validation 和
轨迹指标仍应使用独立离线验证脚本执行。

### 18.4 `no_grad` 下 DiT block 返回 tuple 的验证崩溃

修复 batch-dict 调用后，验证能够真正进入主网络，但在第一层之后报：

```text
AttributeError: 'tuple' object has no attribute 'shape'
```

`CausalWanAttentionBlock.forward()` 的固定返回合同为：

```python
(hidden_states, updated_kv_cache)
```

训练时 gradient checkpointing 的 wrapper 会取出 `hidden_states`；验证运行在
`torch.no_grad()` 下，不走 checkpointing 分支，旧代码却把完整 tuple 赋给 `x`。
下一层读取 `x.shape` 时因此崩溃。

`groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`
现在通过 `_training_block_hidden_states()` 统一两条路径：

- 只把 `hidden_states` 传给下一层；
- 明确断言训练/验证路径不得产生 KV cache；
- 若意外产生 cache，立即抛出带语义的 `RuntimeError`。

`tests/model/test_mobile_plan_phase2.py` 增加了空 cache 正常解包和非空 cache 拒绝
两项回归测试。由于 18.2 已经将保存顺序改成 `save -> evaluate`，本次首次验证失败
仍完整保留了 `checkpoint-2000`，可由训练入口自动恢复。

### 18.5 `BatchFeature` loss 读取兼容

主网络返回类型声明为 Transformers `BatchFeature`。它实现 mapping 接口并包含
`"loss"`，但不是 Python 内置 `dict`。旧版 `prediction_step()` 使用
`isinstance(outputs, dict)` 判断，因而误走 tuple 的 `outputs[0]` 分支并报：

```text
KeyError: Indexing with integers is not available when using Python based feature extractors
```

`groot/vla/experiment/base.py` 现在使用
`collections.abc.Mapping` 判断，统一支持 `dict`、`BatchFeature` 和其他标准 mapping
输出；真正的 tuple 输出仍保留索引兼容。回归测试增加真实 `BatchFeature` 返回模型，
覆盖完整 `Trainer.evaluate()` 路径。本次报错前已保存的 `checkpoint-4000` 包含八卡
DeepSpeed optimizer shards，可自动恢复。
