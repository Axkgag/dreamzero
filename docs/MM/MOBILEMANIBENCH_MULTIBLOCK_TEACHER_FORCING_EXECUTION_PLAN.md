# MobileManiBench WAM 多 Block Teacher Forcing 完整执行方案

> 文档状态：已实现路径的规范与验收记录；训练收敛和仿真成功率仍待实验确认
> 目标仓库：`/mnt/yihao/codes/dreamzero`
> 目标任务：MobileManiBench 5-task、Wan2.2-TI2V-5B、Base/Manipulator 双分支 waypoint 预测
> 实现核对：2026-08-17，`multiblock-wam` 分支提交 `4242553`

## 1. 目标架构与不可变约束

本方案将当前 Mobile WAM 的“单个 8-latent future block + 一次预测完整稀疏计划”改为与 DreamZero 固定 block 自回归训练一致的结构：

- 33 个 RGB 帧经 Wan VAE 编码为 `z0...z8` 共 9 个 latent；
- `num_frame_per_block=2`，未来 8 个 latent 被拆成 4 个 video block；
- 每个 video block 覆盖 8 个 RGB 帧；
- 默认每个 block 预测 3 个 waypoint，局部 offsets 为 `[2, 4, 8]`；
- block anchors 与 local offsets 是实验配置，不固化进 parquet；
- 每个 block 包含 `3 Base + 3 Manipulator = 6` 个 flow action token；
- 4 个 block 共 24 个 flow action token；
- 每个 block 使用该 block 起点处的 1 个 state token，共 4 个 state token；
- 训练使用 GT future video 的 block-causal teacher forcing；
- 部署时每次只使用当前真实观测和当前 state，预测当前 block 的局部计划
  `[+2,+4,+8]`，不依赖未来 GT；
- primary inference 不使用 RGB repeat 补足 block。

这里的“与原版 DreamZero 一致”有严格边界：clean/noisy 双路
tensor、block-causal teacher forcing、video/action/state block 对齐、
coupled per-block timestep 以及训练时按实际 block 数派生 register
长度必须复用原版语义。Base/Manipulator 双分支稀疏 waypoint、
physical consistency 和 clean prior 是 Mobile 任务扩展；它们不得
改变 clean future 的可见性、block 边界或 teacher-forcing 历史路由。

核心配置为：

```yaml
num_frames: 33
num_frame_per_block: 2       # 历史键名；单位是 VAE latent frame，不是 RGB frame
max_chunk_size: 4
num_plan_blocks: 4
plan_waypoints_per_block: 3
plan_local_offsets: [2, 4, 8]
# derived video/block stride: 8 RGB frames
# DreamZero 语义：horizon 表示单个 block/chunk 的对外长度。
action_horizon: 6
num_flow_action_per_block: 6
# P = num_prior_tokens_per_block；未启用 prior 时 P=0，endpoint prior 时 P=1
# DiT 内部实际分块宽度：num_action_per_block = 6 + P
state_horizon: 1
num_state_per_block: 1
# 4-block 完整窗口的派生不变量（不要作为独立 Hydra override）：
# training_flow_action_tokens = 4 * 6 = 24
# training_internal_action_registers = 4 * (6 + P)
# training_state_tokens = 4 * 1 = 4
```

这里必须保持原版 DreamZero 的命名语义：`action_horizon` 和
`state_horizon` 表示单个 block 在推理时产生/消费的 chunk 长度；训练
tensor 的总长度由实际 `num_training_blocks` 动态派生。对于 4-block
完整窗口，flow action/state tensor 分别为 24/4 tokens，但不能因此把配置中的
单-block horizon 写成 24/4。

全局 waypoint 时间点为：

```text
[2, 4, 8, 10, 12, 16, 18, 20, 24, 26, 28, 32]
```

当前默认在每个 8-frame block 内使用 `[2,4,8]`，兼顾中间监督与 endpoint。不要继续
使用单 block baseline 的 `[1,4,8,12,16,24]` 再按 video block 生硬切分；这种切分会
导致每个 block token 数不同，也不符合固定 block action register 的设计。

## 2. 当前实现核对结果

以下结论以当前代码为准，而不是以旧文档为准。

### 2.1 legacy Mobile WAM baseline 是单 block 设计

以下内容用于解释为什么不能只改旧脚本参数。当前仓库已经另行实现 multiblock
Dataset/Transform/Collator、DiT、policy head、训练脚本和 evaluator；旧单 block 路径仍
保留用于公平 baseline。

当前训练脚本 `scripts/train/mobilemanibench_plan_training_wan22_5b.sh` 使用：

```text
num_frames=33
num_frame_per_block=8
num_action_per_block=12
action_horizon=12
state_horizon=1
```

Wan VAE 的 33 个 RGB 帧对应 9 个 latent：

```text
z0: RGB frame 0
z1: RGB frames 1-4
z2: RGB frames 5-8
...
z8: RGB frames 29-32
```

因此当前 attention/block 语义是：

```text
clean context: [z0]
one noisy future block: [z1,z2,z3,z4,z5,z6,z7,z8]
one action block: [6 Base tokens, 6 Manipulator tokens]
one state block: [state at t]
```

这同样是可见关系的简写；当前底层 DreamZero forward 仍会构造等长的
`clean_x=[z0...z8]` 与 `noisy_x=[z0...z8]`。问题不在于物理 tensor 缺少
latent，而在于 `num_frame_per_block=8` 使 `z1...z8` 全部被视为同一个
future block，只能对应一组 action/state register。

当前专用代码也显式锁死了这个语义：

- `MobilePlanFlowMatchingActionHead.validate_action_video_layout()` 要求 future latent 数等于 `num_frame_per_block`，即只能有 1 个 video block；
- `build_coupled_action_timestep_ids()` 要求 `timestep_id_block.shape[1] == 1`；
- `WanVideoDiTDualPlan` 把 `num_action_per_block` 强制设为 `2 * plan_horizon`；
- `DualPlanActionEncoder/Decoder` 假设全局 branch-major 布局：先全部 Base，再全部 Manipulator；
- `MobilePlanCotrainTransform` 把 6 Base 与 6 Manipulator 拼成 12 个 token；
- `MobileManiBenchPlanDataset` 每行只读取一份相对当前 `B(t)` 的 6-waypoint 长计划；
- `scripts/eval/mobilemanibench_plan_eval.sh` 每个样本 reset，只输入当前观测帧，不执行多 block continuation。

所以只修改训练脚本中的 `num_frame_per_block=2` 会直接报 shape/layout 错误，且即使绕开断言，action 的坐标系和 block 对齐仍然是错的。

### 2.2 目标 teacher forcing 语义

目标训练的可见 clean context 必须是：

```text
预测 noisy [z1,z2]：可见 clean [z0]
预测 noisy [z3,z4]：可见 clean [z0,z1,z2]
预测 noisy [z5,z6]：可见 clean [z0,z1,z2,z3,z4]
预测 noisy [z7,z8]：可见 clean [z0,z1,z2,z3,z4,z5,z6]
```

上述写法描述的是可见关系，不代表物理输入 tensor 只包含 8 个 noisy
future latent。为了与当前发布的 DreamZero 训练实现一致，实际 forward
合同应明确为：

```text
clean_x: [z0,z1,z2,z3,z4,z5,z6,z7,z8]
noisy_x: [z0,z1,z2,z3,z4,z5,z6,z7,z8]
future video blocks: [z1,z2], [z3,z4], [z5,z6], [z7,z8]
```

原版实现对完整 latent tensor 构造 clean/noisy 双路，未来 block 数通过
`(noisy_frames - 1) / num_frame_per_block` 得到；`z0` 不属于 action block，
但当前 released implementation 的 video dynamics loss 仍包含 noisy `z0`
预测项。目标实现保留该语义；不构造 noisy `z0` 或 mask 掉其
dynamics loss 都会偏离本文档定义的原版 teacher-forcing 合同。

attention mask 必须保证 block `b` 只能读取 clean `z0` 和此前 block 的
clean GT，不能读取本 block或未来 block 的 clean GT。否则会发生 future
leakage。

### 2.3 原版 DreamZero 的 horizon 与动态 register 语义

原版 DROID 训练脚本设置：

```text
action_horizon=24
num_action_per_block=24
state_horizon=1
num_state_per_block=1
```

对应核对入口为：

- `scripts/train/droid_training_lora.sh`、`scripts/train/droid_training_wan22.sh`；
- `groot/vla/data/dataset/lerobot_sharded.py` 中 DROID 的多 chunk
  video/action/state 采样；
- `wan_flow_matching_action_tf.py` 中 action timestep 与 layout 校验；
- `wan_video_dit_action_casual_chunk.py` 中动态 register 长度和
  teacher-forcing attention。

这里的 `action_horizon=24` 是单个 block 的 action chunk 长度。Dataset
可以返回 1～4 个 block，因此训练 forward 的真实 tensor 可以是：

```text
action: [B, num_training_blocks * 24, action_dim]
state:  [B, num_training_blocks * 1, state_dim]
```

底层 `CausalWanModel` 再根据实际 video block 数和 register tensor 长度
动态得到训练 horizon。Mobile multiblock 必须保留同一语义：

```text
配置/单次推理 action_horizon = 6
配置/单次推理 state_horizon  = 1
4-block 训练 action tokens   = 4 * 6 = 24
4-block 训练 state tokens    = 4 * 1 = 4
```

不要让同一个 `action_horizon` 同时表示“单 block 推理 chunk”和“四 block
训练总长度”。如果某个新模块确实需要总长度，使用
`training_action_tokens`、`num_training_action_tokens` 或直接从 tensor shape
派生。

### 2.4 与原版一致的范围

本方案所说的“一致”是指固定 block 的 video/action/state 对齐、联合 flow
matching、block-causal teacher forcing、coupled per-block timestep，以及
单 block KV-cache 闭环推理一致。Mobile 任务的双 Base/Manipulator 稀疏
waypoint、physical-time embedding、physical consistency 和 clean prior
仍然是原版 DreamZero 之上的任务专用扩展，不应描述成原版 action head 的
逐参数复现。

## 3. 目标时间、token 与坐标布局

### 3.1 精确 block 对齐

| Block | Video latent | RGB 范围 | Block anchor | 局部 waypoint offsets | 全局目标帧 | Action token |
|---|---|---:|---:|---|---|---|
| 0 | `z1,z2` | `t+1...t+8` | `t` | `[2,4,8]` | `t+2,t+4,t+8` | `B0_2,B0_4,B0_8,M0_2,M0_4,M0_8` |
| 1 | `z3,z4` | `t+9...t+16` | `t+8` | `[2,4,8]` | `t+10,t+12,t+16` | `B1_2,B1_4,B1_8,M1_2,M1_4,M1_8` |
| 2 | `z5,z6` | `t+17...t+24` | `t+16` | `[2,4,8]` | `t+18,t+20,t+24` | `B2_2,B2_4,B2_8,M2_2,M2_4,M2_8` |
| 3 | `z7,z8` | `t+25...t+32` | `t+24` | `[2,4,8]` | `t+26,t+28,t+32` | `B3_2,B3_4,B3_8,M3_2,M3_4,M3_8` |

其中 `B` 表示 Base waypoint token，`M` 表示 Manipulator waypoint token。

### 3.2 必须采用 block-major packing

推荐规范张量先保留结构：

```text
base_plan:        [batch, 4 blocks, 3 waypoints, 4]
manipulator_plan: [batch, 4 blocks, 3 waypoints, 21]
plan_valid:       [batch, 4 blocks, 3 waypoints]
state:            [batch, 4 blocks, 64]
```

进入 DiT 前再 flatten 为：

```text
block 0: Base[3], Manipulator[3]
block 1: Base[3], Manipulator[3]
block 2: Base[3], Manipulator[3]
block 3: Base[3], Manipulator[3]
```

最终 action tensor：

```text
[batch, 24, 21]
```

不要采用以下布局：

```text
[全部 12 个 Base][全部 12 个 Manipulator]
```

因为底层 causal attention 按连续的单 block 宽度切 action
register（无 prior 时为 6，启用 endpoint prior 时为 7）；
branch-major 会把前两个 block 错当成纯 Base block，并破坏
video/action/state block 对齐。

### 3.3 waypoint 必须相对每个 block 的 anchor 重新编码

令世界系中的 Base pose 为 `T_W_B(k)`，EEF pose 为 `T_W_E(k)`。对 block `b`：

```text
anchor_index = t + 8*b
target_index = anchor_index + local_offset
```

Base target：

```text
T_Banchor_Btarget = inverse(T_W_B(anchor_index)) @ T_W_B(target_index)
```

Manipulator target：

```text
T_Banchor_Etarget = inverse(T_W_B(anchor_index)) @ T_W_E(target_index)
```

Base 仍编码为：

```text
[dx, dy, sin(dyaw), cos(dyaw)]
```

Manipulator 仍编码为：

```text
[eef_xyz(3), eef_rotation6d(6), hand_configuration]
```

block state 使用 anchor 时刻的真实 state：

```text
raw_state[b] = EEF position(3) + EEF rotation RPY(3),
               relative to Base at t + 8*b
model_state[b] = normalize(raw_state[b]) and pad from 6 to max_state_dim=64
```

严禁把全部 target 都相对初始 `B(t)` 计算后直接 reshape 为 `[4,2,...]`。那样 block 1-3 的监督坐标系与推理时当前真实 Base 坐标系不一致。

重新编码只作用于当前 block 的 action/state label。例如 block 1
使用 `B(t+8)`，但 block 0 action 仍保持在 `B(t)` 中；不会把
历史 action 统一重算到 `B(t+8)`。`z0...z8` 是视频 latent，只有
时间和 attention block 语义，不对它们施加 SE(2)/SE(3) Base
坐标变换。

## 4. 动态标签数据方案

### 4.1 Canonical 数据根目录保持不变

直接使用现有五任务数据：

```text
/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_g1_5tasks/g1
```

每个 parquet 已包含逐帧 canonical trajectory：

```text
observation.base.world
observation.eef.world
observation.robot_joint
```

multiblock label 必须从这些世界坐标轨迹动态派生。修改 anchors 或
local offsets 不得要求重新转换 parquet、复制视频或创建新的数据根目录。
旧的 `action.plan.*` 列保留给 baseline，但不是动态 multiblock 路径的
source of truth。

### 4.2 YAML 是 label specification 的唯一来源

默认配置为：

```yaml
mobilemanibench_plan_label_source: dynamic
block_anchor_offsets: [0, 8, 16, 24]
plan_local_offsets: [2, 4, 8]
```

Dataset 使用公共 geometry helper 对每条 trajectory 向量化计算：

```text
target_index = sample_index + block_anchor_offset + local_waypoint_offset
Base target = inv(B_anchor) * B_target
EEF target  = inv(B_anchor) * EEF_target
```

`global_waypoint_offsets`、block 数和每 block waypoint 数均由该 spec
派生，不作为 parquet metadata 的硬约束。保留 materialized label mode
只用于兼容和数值对照，不作为默认训练路径。

当前 video layout 下 `block_anchor_offsets=[0,8,16,24]` 是由 4 个
8-RGB-frame video block 决定的结构不变量，不作为 action sampling
消融参数。`plan_local_offsets` 的数量 K 可以配置；collator、packing、
flow register 和 clean-prior internal register 必须分别动态派生为
`K`、`2K` 和 `2K+1`，不得硬编码 K=3 或 24 个总 flow token。

公共动态 helper 必须与 converter 的 `build_block_plan_labels()` 做逐值
一致性测试，防止在线与离线几何语义分叉。

### 4.3 完整 33-frame window 训练样本约束

目标 Mobile 配置使用 4 个完整 block，并过滤 episode 尾部
padding。这是样本选择约束，不改变原版 DreamZero 的
block-causal teacher-forcing 语义。模型实现不得硬编码总
action/state horizon；即使当前数据都是 4 block，register 长度仍必须
由当次 forward 的实际 video block 数动态派生。

当前 LeRobot loader 会对 episode 尾部越界视频做末帧 padding，而 video dynamics loss 没有 per-latent valid mask。本配置必须过滤：

```text
frame_index + 32 < episode_length
```

这样可同时保证：

- 33 个 RGB 帧全部真实存在；
- 4 个 block anchor state 全部存在；
- 8 个 waypoint 全部有效；
- dynamics loss 不会把重复末帧当作真实运动监督。

partial window 不属于当前目标配置。任何启用 partial window 的
实现都必须同时给 video dynamics loss 增加 latent/block validity
mask，不得仅依赖 action `plan_valid`。

### 4.4 统计量按 label spec 缓存

新标签改成 block-anchor-relative，分布与当前初始-anchor长计划不同，因此不能复用旧 `plan_stats.json`。

使用 train split 单独拟合：

```text
base_xy q01/q99
eef_xyz q01/q99
hand q01/q99
```

统计文件使用 canonical label spec 的 hash 命名：

```text
meta/dynamic_plan_stats/block_plan_<spec_hash>.json
```

它必须记录 block 配置、`fit_split=train`、split manifest SHA256 和
geometry QA。训练 preflight 必须校验这些字段与 YAML 完全一致。修改
offset 后只生成一个新的小型 statistics 文件，不重写 parquet。

## 5. Dataset 与 Transform 修改

### 5.1 `MobileManiBenchBlockPlanDataset`

使用独立的 `MobileManiBenchBlockPlanDataset`：

- 默认读取 Base/EEF/joint canonical world-state columns；
- 按 YAML anchors/local offsets 动态构造 block labels；
- 输出 `[4,2,...]` 结构，不要在 Dataset 内过早 flatten；
- 输出 `block_anchor_offsets`、`plan_local_offsets`、`global_plan_offsets`；
- 输出 4 个 anchor state；
- 加入 `require_full_video_window=true` 的 step filter；
- 严格检查 YAML spec、statistics spec 与模型 token layout；
- worker 内按 trajectory 缓存一次向量化几何结果；
- 旧 `MobileManiBenchPlanDataset` 保持不变。

### 5.2 `MobilePlanTransform`

修改 `groot/vla/data/transform/mobile_plan.py` 或新增 `MobileBlockPlanTransform`：

- 支持 `[4,2,4]` Base 与 `[4,2,21]` Manipulator；
- normalization 在最后一维执行，保留 block/waypoint 两级结构；
- `plan_valid` 广播到 branch dimension mask；
- geometry QA 对所有有效 `(block, waypoint)` 检查；
- `unapply()` 恢复相同结构；
- 不再用固定错误消息 `Expected base plan [6,4]`；
- 校验新 `plan_stats.json` 的 local offsets、block stride 和 anchor frame。

### 5.3 `MobilePlanCotrainTransform`

`groot/vla/model/dreamzero/transform/mobile_plan_cotrain.py` 已新增 multiblock 子类：

```text
输入：
base              [4,2,4]
manipulator       [4,2,21]
base_mask         [4,2,4]
manipulator_mask  [4,2,21]

每个 block pack：
[padded Base(2), Manipulator(2)] -> [4,21]

最终 flatten：
action       [16,21]
action_mask  [16,21]
state        [4,64]
```

`MobilePlanDataCollator` 应同时保留结构化 semantic keys 供日志、physical loss 和 evaluator 使用，并检查 batch shape：

```text
base_action              [B,4,2,4]
manipulator_action       [B,4,2,21]
base_action_mask         [B,4,2,4]
manipulator_action_mask  [B,4,2,21]
plan_local_offsets       [B,2]
block_anchor_offsets     [B,4]
```

## 6. DiT action encoder/decoder 修改

### 6.1 新增 multiblock 模型类

实现没有改写旧 `WanVideoDiTDualPlan`，而是在
`groot/vla/model/dreamzero/modules/` 新增：

```text
wan_video_dit_dual_plan_multiblock.py
```

推荐类名：

```text
MultiBlockDualPlanActionEncoder
MultiBlockDualPlanActionDecoder
WanVideoDiTMultiBlockDualPlan
```

`WanVideoDiTMultiBlockDualPlan` 必须继承现有 `CausalWanModel`（或继承
`WanVideoDiTDualPlan` 后继续复用其 `CausalWanModel` 主干），只替换
multiblock action encoder/decoder 和必要的 token metadata。不要复制重写
原版 clean/noisy teacher-forcing attention、RoPE、KV cache 或
training/inference forward；否则即使 shape 一致，也无法保证 WAM 行为与
原版一致。

显式参数：

```text
num_plan_blocks=4
plan_waypoints_per_block=3
plan_local_offsets=(2,4,8)
num_flow_action_per_block=6
num_prior_tokens_per_block=P
num_action_per_block=6+P      # DiT 内部 action register 分块宽度
action_horizon=6              # 单 block/单次推理的 flow 输入输出
# runtime: action.shape[1] = actual_num_blocks * 6
```

不要继续让 `plan_horizon` 同时表示“每个 block waypoint 数”和“全窗口 waypoint 总数”。这两个量必须拆开命名。

### 6.2 encoder 行为

encoder 输入先 reshape：

```text
[B,24,21] -> [B,4 blocks,6 tokens,21]
```

每个 block 内：

```text
token 0-2: Base，取前 4 维
token 3-5: Manipulator，取前 21 维
```

继续复用：

- Base/Manipulator 独立 projector；
- branch type embedding；
- physical time offset embedding。

offset embedding 使用局部时间 `[2/30, 4/30, 8/30]` 秒，并在每个 block 重复。
block 身份由 action register 的 block 位置、RoPE 和 causal attention 对齐提供，不需要
把全局 offsets 误当作相对当前 block 的时间。

### 6.3 decoder 行为

decoder 对每个 block 独立拆出 3 个 Base hidden 与 3 个 Manipulator hidden，使用共享
branch decoder，随后恢复 block-major layout。

输出 flow tensor仍为：

```text
[B,24,21]
```

当前实现由 encoder、decoder、action head 共享同一个 block-major shape 公式，但没有
导出以下独立公共 helper：

```text
pack_block_plan(...)
unpack_block_plan(...)
```

若后续再增加新的 consumer，应先提取这两个 helper，避免继续复制 slicing；这不是当前
提交中已经存在的 API。

### 6.4 权重初始化

结构变化不应直接 `resume` 当前 checkpoint。推荐 matching initialization：

- 载入 Wan2.2 backbone；
- 载入当前 checkpoint 中 shape 相同的 LoRA 权重；
- 复用 Base/Manipulator encoder projector；
- 复用 Base/Manipulator decoder projector；
- 复用 type embedding 与 offset MLP；
- 跳过旧 `expected_plan_time_offsets` buffer；
- 目标配置未启用 clean prior 时跳过所有 `prior_*` 参数；
- 输出 missing/unexpected/shape-mismatch 的完整报告并保存到新 output dir。

projection 权重在各 block 间共享，因此当前 6-waypoint 模型的 projector 是有价值的初始化；但 token layout、offset buffer 和 attention register 语义不能按 checkpoint 原样恢复。

## 7. Policy head 与 timestep 修改

已新增：

```text
groot/vla/model/dreamzero/action_head/mobile_plan_multiblock_flow_matching.py
```

不要放宽旧单 block class 的断言后继续使用旧 slicing。

### 7.1 layout validation

必须验证：

```text
noisy_latents = 9             # z0...z8，与 clean_x 等长
future_latents = 8
num_video_blocks = 8 / 2 = 4
training_flow_action_tokens = action.shape[1] = 4 * 6 = 24
training_state_tokens = state.shape[1] = 4 * 1 = 4
per_block_flow_action_horizon = num_flow_action_per_block = 6
num_prior_tokens_per_block = P
per_block_internal_action_registers = num_action_per_block = 6 + P
training_internal_action_registers = action_features.shape[1] = 4 * (6 + P)
per_block_state_horizon = num_state_per_block = 1
```

即：

```text
num_video_blocks
== action.shape[1] / num_flow_action_per_block
== action_features.shape[1] / num_action_per_block
== state.shape[1] / num_state_per_block
== 4
```

这里必须使用当前 training tensor 的真实长度做验证，不能用单 block 配置
`action_horizon=6`、`state_horizon=1` 去计算训练 block 数。

### 7.2 action timestep

video timestep block shape：

```text
[B,4,2]
```

同一个 video block 内 2 个 latent 共用 timestep。coupled action mode 下，对应的 6 个
action tokens 也共用该 block timestep：

```text
[tv0,tv0,tv0,tv0,tv0,tv0,
 tv1,tv1,tv1,tv1,tv1,tv1,
 tv2,tv2,tv2,tv2,tv2,tv2,
 tv3,tv3,tv3,tv3,tv3,tv3]
```

目标训练模式固定为 coupled/per-block：每个 video block 独立
采样一个 timestep，该 block 的 2 个 video latent 与 6 个 action
token 共用这个 timestep。不在本实现中引入 per-token action
timestep，以保持原版 DreamZero 的 block-coupled flow-matching 语义。
启用 clean prior 时，每个 prior register 使用 clean timestep `0`；它不是
额外 noisy flow token。例如 `P=1` 时单 block 内部 timestep 布局为
`[0, tv_b, tv_b, tv_b, tv_b, tv_b, tv_b]`。

### 7.3 branch loss

旧 loss 通过 `[:horizon]` 与 `[horizon:]` 切 Base/Manipulator，不适用于 block-major。必须先 reshape：

```text
[B,24,21] -> [B,4,6,21]
base = block[...,0:3,0:4]
manipulator = block[...,3:6,0:21]
```

至少记录：

```text
base_flow_loss/block_0...3
manipulator_flow_loss/block_0...3
base_flow_loss/offset_2, offset_4, offset_8
manipulator_flow_loss/offset_2, offset_4, offset_8
valid_ratio/block_0...3
dynamics_loss/block_0...3
```

全局均值会掩盖后段 block 的 exposure-bias 和漂移问题，因此 per-block 日志是验收必需项。

## 8. Attention mask 与防泄漏验收

底层 `CausalWanModel` 已有 clean/noisy 双路和 block-causal attention 基础，但新的 Mobile layout 必须用测试证明，而不是凭配置推断。

目标依赖关系：

```text
flow action block b 可见：
- z0 clean condition
- block 0...b-1 的 clean GT video
- block b 的 noisy video
- block b 的 state
- block b 的 action peers

action block b 不可见：
- block b 的 clean GT video
- block b+1...3 的任何 clean/noisy video
- 其他 block 的 action/state token
```

video block 同样不能读取本 block/future block 的 clean GT。

当前 dense-mask 单元测试位于 `tests/model/test_mobile_multiblock_plan.py`：

1. block 0 只能读取 clean `z0`；
2. block 1 能读取 clean `z1,z2`，不能读取 clean `z3...z8`；
3. block 3 能读取 clean `z1...z6`，不能读取 clean `z7,z8`；
4. action block 与 video/state block index 一一对应；
5. 改动 future clean block 的值不会影响更早 block 输出；
6. 改动 earlier clean block 会影响其后的 block 输出。

如果第 5 项失败，训练指标即使很好也属于 GT leakage，
该实现不符合本文档的 teacher-forcing 验收标准。

## 9. Teacher-forcing 主干与 Mobile 任务扩展的边界

### 9.1 原版 DreamZero teacher-forcing 合同

Mobile WAM 的训练 forward 必须直接复用 `CausalWanModel` 的
clean/noisy 双路和 block 路由，不新建另一套 teacher-forcing
forward。对任意 block `b`：

```text
clean history = z0 + blocks [0, b)
prediction target = noisy video block b + action block b
state condition = state at block b anchor
forbidden = clean block b, future clean/noisy blocks, other action/state blocks
```

一次 training forward 同时预测全部 noisy block，而不是每个 block
单独调用一次模型。clean 分支输出不作为未来预测结果；
action/video loss 只能使用合法的 noisy 输出与 validity mask。

### 9.2 Physical consistency 的最终合同

Physical consistency 只从已经按 block-major 解码的 flow 输出计算
附加 loss，不得改变 clean/noisy tensor、attention mask、timestep
或 register 分块。physical loss 先 unpack 到 `[B,4,3,...]`：

- Base yaw unit loss：逐 `(block,waypoint)`；
- EEF rotation loss：逐 `(block,waypoint)`；
- Base/EEF consistency：必须使用同一个 block anchor；
- validity mask：同时应用 block 与 waypoint validity；
- 汇报 per-block 以及总体值。

不要跨 block 直接比较两个不同 anchor 坐标系中的 Base/EEF 数值。

### 9.3 Clean prior 的最终合同

开启 clean prior 时，每个 block 放 1 个 endpoint prior token，对应
local offset `8`：

```text
每 block: [1 prior, 3 Base flow, 3 Manipulator flow]
internal num_action_per_block = 7
flow action_horizon = 6                    # 单 block 对外 flow chunk
training flow action tokens = 24           # 4 blocks * 6
training internal action registers = 28    # 4 blocks * 7
```

该 prior token 可通过独立 decoder head 输出 Base 和 EEF target，但
仍只占用一个 attention register。Base prior 使用当前 block anchor
坐标系。EEF prior 的坐标系必须由配置显式指定：

```yaml
prior:
  time_offsets: [8]          # 在每个 block 内解释为 local offset
  predict_base: true
  predict_eef: false
  eef_frame: future_base
```

`eef_frame` 只定义 EEF prior target；它不改变主 Manipulator flow
相对 block anchor Base 的定义。`predict_eef=false` 时不计算 EEF
prior loss，`eef_frame` 也不参与目标构造。

clean-prior encoder/decoder 也必须 block-major；不能把 4 个 prior 全放在全局 flow tokens 前面。

Prior query 只能读取合法 clean history、当前 anchor state 和当前
block 的 prior peers；不能读取当前 clean/noisy future video。Flow
与 noisy video 可读取当前 prior。这条定向 Prior attention 是 Mobile
扩展，但 clean-history teacher forcing 路由仍必须与原版一致。

## 10. 推理实现

### 10.1 部署语义：局部闭环 block rollout

与训练保持 block 接口、时间 anchor 和因果可见性一致的推理方式：

```text
t：输入当前真实 RGB + 当前真实 state
   预测 video t+1...t+8 对应的 z1,z2
   预测局部 Base/Manipulator waypoint at t+2,t+4,t+8
   执行动作

t+8：获得新的真实 RGB/history + 新的真实 state
     将新 observation latents 写入 KV cache
     预测下一 block 的 [+2,+4,+8]
```

每次模型对外只输出当前 block 的 3 个 Base waypoint 和 3 个 Manipulator waypoint。
训练窗口里存在 4 个 action block，是为了让模型学习有历史条件时后续 block 的稳定性，
不代表部署第一次调用就有未来 state。

正式推理应复用原版 `lazy_joint_video_action()`/cached rollout 的“一次一个
block”算法语义，但不能原样复用当前 Mobile 单-block dual-plan 的固定
6-waypoint long-plan unpack。multiblock action head 设置 `self.action_horizon=6`，
普通单次推理只创建当前 block 的 6 个 noisy flow action tokens；
DiT 内部使用 `num_action_per_block=6+P`，由 encoder 注入 `P` 个 clean
prior register。最后由专用 block-major slicing 还原 `Base[3] + Manipulator[3]`。

如果实现离线 4-block oracle 或变长 block 诊断，noise shape 必须由本次
`num_inference_blocks * num_flow_action_per_block` 动态派生，不能把训练总长度 24
写回 `self.action_horizon`。优先继承并扩展原版 sampler/KV-cache 路径，不要
另写一套不同的去噪与缓存语义。

新 sampler 的双态 shape contract 必须是：

```text
训练：
clean/noisy video latents [B,9,...] = z0...z8
block-aligned future latents [B,8,...] = z1...z8
action               [B,24,21] = 4 blocks * 6 tokens
internal action regs  [B,4*(6+P),D]
state                [B,4,64]  = 4 blocks * 1 token

单次缓存推理：
video future latents [B,2,...]
action noise/output  [B,6,21]  = current block only
internal action regs [B,6+P,D]
state                [B,1,64]  = current real state only
```

对应修改要求：

- action encoder/decoder 在 training forward 接受 4 blocks，在 cached inference 接受 1 block；
- primary cached inference 的 noise action shape 固定为单 block
  `action_horizon=num_flow_action_per_block=6`；变长/oracle 模式才按本次
  `num_inference_blocks * num_flow_action_per_block` 动态计算；
- DiT 内部 action register 数按 `num_inference_blocks * (6+P)` 派生；
- state register 同理按本次 block 数计算；
- cached attention 用 `current_block_index`/`current_start_frame` 选择正确 RoPE 与 KV 位置；
- `get_action()` 只拆分当前 6 个 token 为 `Base[3] + Manipulator[3]`；
- 单元测试必须确保 cached inference 从未静默 repeat/pad 到 24 个 action token或 4 个 state token。

### 10.2 不使用 repeat 的约束

缓存 continuation 每次收集完整 8 个新 RGB 帧，再编码为 2 个新 reference latent。VAE 边界处理可能复制首帧一次以满足 causal 编码长度，但不应把 4 个 RGB 帧重复扩展成 8-frame block。

如果控制系统要求每 4 帧重新规划，有两个合法选择：

1. 每次 reset，把最新真实帧作为新 `z0`，做独立 one-shot block 预测；
2. 另行训练 `num_frame_per_block=1` 的模型。

不要在 primary deployment 中用 4 帧 repeat 冒充 8 帧真实历史。

### 10.3 不可部署但有价值的诊断模式

可实现 `gt_history_cached`：按 episode 顺序每 8 帧调用一次，输入真实新增 RGB 和真实当前 state，保留 KV cache。未来 GT 只用于下一时刻已经到达后的历史输入与误差计算，不在当前预测前泄漏。

另可实现 `gt_future_state_teacher_forced`，一次离线生成 4 block，用 GT future state；它只能衡量模型在理想 state 条件下的上限，必须明确标为 oracle diagnostic，不能作为部署指标。

### 10.4 全局轨迹拼接

每个 block 的输出位于自己的 anchor Base 坐标系。为了绘制 32-frame 全局计划，要递推组合 Base endpoint：

```text
T_B0_Banchor(0) = I
T_B0_Banchor(b+1) = T_B0_Banchor(b) @ T_Banchor(b)_Bendpoint(b)
```

block `b` 的 Base/EEF waypoint 再左乘 `T_B0_Banchor(b)`，统一转换到初始 Base 坐标系。直接把 4 组局部 XY 连接作图是错误的。

## 11. 评估方案

### 11.1 evaluator 分层

已新增独立 evaluator，不改变单 block 的 `scripts/eval/mobilemanibench_plan_eval.sh`：

```text
scripts/eval/mobilemanibench_multiblock_plan_eval.sh
scripts/eval/evaluate_mobilemanibench_multiblock_plan.py
```

当前支持：

```text
single_block_reset
episode_ordered_reset
gt_history_cached
oracle_four_block_teacher_forced
```

四种模式的实现边界为：

- `single_block_reset`：单个当前 RGB/state，reset cache，只预测 block 0 的
  `[+2,+4,+8]`；
- `episode_ordered_reset`：每个 block 都使用对应当前 RGB/state，但每个 block reset；
- `gt_history_cached`：真实历史、每 8 帧滚动、无 future leakage；
- `oracle_four_block_teacher_forced`：完整 33 帧 clean window，只报告 teacher-forced
  loss，不是部署 waypoint metric。

脚本支持 `INSPECT_ONLY=1`，可在不加载 checkpoint 的情况下核对 split、动态 labels、
offsets、stats path 和 task-balanced root windows。closed-loop simulator success 尚未实现。

### 11.2 waypoint 指标

按 block、offset、task 分别统计：

```text
Base XY L1/L2
Base endpoint displacement error
Base yaw angular error
EEF position L1/L2
EEF rotation geodesic error
hand MAE
plan ADE/FDE
相邻 block endpoint/start continuity error
有效 waypoint 数与 mask ratio
```

### 11.3 video 指标和可视化

当前 multiblock evaluator 只保存 waypoint/prior 指标，不评价或保存生成 video。现有输出为：

```text
summary.json
per_prediction_metrics.jsonl
per_block_metrics.csv
predictions.npz
```

生成 video 的以下产物仍是后续工作：

- GT 与 predicted future video 并排 mp4；
- head/wrist composite view；
- 每 block 边界标记；
- predicted plan 叠加图；
- episode 连续滚动视频，而不是随机单帧 montage。

建议指标：

```text
PSNR / SSIM
LPIPS
temporal LPIPS 或 frame-difference error
block boundary discontinuity
可选 FVD（仅在评估样本数足够时报告）
```

### 11.4 与当前模型的公平对比

当前单 block 模型 offsets 为 `[1,4,8,12,16,24]`，multiblock 默认全局 offsets 为
`[2,4,8,10,12,16,18,20,24,26,28,32]`。公平直接对比使用交集：

```text
[4,8,12,16,24]
```

同时分别报告：

- 当前模型原生 6-point 指标；
- multiblock 模型原生 12-point 指标；
- common-offset 指标；
- 目标 multiblock 模型 per-block 连续性与 cached-history 收益。

不要把不同 offset 集合的简单均值直接比较。

## 12. 配置与训练脚本

已新增且不覆盖单 block 路径：

```text
groot/vla/configs/data/dreamzero/mobilemanibench_multiblock_plan.yaml
groot/vla/configs/model/dreamzero/transform/mobile_plan_multiblock_cotrain.yaml
groot/vla/configs/model/dreamzero/action_head/mobile_plan_multiblock_flow_matching.yaml
groot/vla/configs/model/dreamzero/action_head/mobile_plan_multiblock_flow_matching_wan22.yaml
scripts/train/mobilemanibench_multiblock_plan_training_wan22_5b.sh
```

目标训练命令必须体现：

```text
num_frames=33
num_frame_per_block=2
num_flow_action_per_block=6
num_prior_tokens_per_block=P
num_action_per_block=6+P
num_state_per_block=1
action_horizon=6
state_horizon=1
num_plan_blocks=4
max_chunk_size=4
```

其中 `training_action_tokens=24`、`training_state_tokens=4` 是完整 4-block
batch 的 shape 不变量，应从实际 tensor 和 block 数派生，不应作为独立 Hydra
override 传入。模型始终按 runtime tensor shape 派生 block 数，
不将 4-block 总长度写固到单 block horizon 配置中。

此时：

```text
local_attn_size = max_chunk_size * num_frame_per_block + 1
                = 4 * 2 + 1
                = 9 latent frames
```

正好覆盖 `z0...z8`，即整个 33-RGB-frame 训练窗口。

### 12.1 output 目录

当前脚本默认目录仍为：

```text
work_dirs/mobilemanibench_g1_5tasks_wan22_5b_multiblock_k2_wp2
```

该目录名保留了早期 `K=2/wp2` 命名，但当前 resolved config 实际是
`K=3`、`plan_local_offsets=[2,4,8]`。新实验应通过 `OUTPUT_DIR` 覆盖为无歧义名称，
例如 `..._multiblock_k3_offsets_2_4_8`；判断合同必须读
`experiment_cfg/conf.yaml`，不能从目录名反推。

禁止在当前单 block output dir 中自动 resume。新结构的 action/state horizon 与 scheduler 都不同。

### 12.2 优化器与初始化约束

这不是对原 checkpoint 的同构 continuation。当前训练脚本使用 `1e-5`，从 raw Wan2.2
components 构造模型，并没有实现从旧单 block checkpoint 做专用 matching-load；也不能
resume 旧 optimizer/scheduler。只有同一 multiblock 结构、同一 resolved plan spec 的
checkpoint 才可作为恢复候选。

如果实现参数组：

```text
新/重排后的 action head: 1e-5
已匹配加载的 Wan LoRA: 4e-6 ~ 1e-5
```

当前 action head 会记录 per-block Base/Manipulator flow loss，physical/prior 路径也有
各自 scalar diagnostics；dynamics/action 梯度比例与生成视频尚未形成完整自动报告，
不能在实验结论中假定这些验收已经完成。

## 13. 完整实施范围

当前提交已经具备核心 train/eval 链路。若要把它称为“完成实验验收的 multiblock WAM”，
还必须同时满足以下完整范围；代码存在不等于所有实验项已经完成：

- canonical world trajectory 上的动态 block-label provider；
- spec-hashed train-only plan statistics 与 normalization round trip；
- block-major Dataset/Transform/Collator 以及公共 pack/unpack helper；
- 复用 `CausalWanModel` 的 multiblock Base/Manipulator encoder/decoder；
- 按实际 video block 数派生 action/state register 的 policy head；
- 与原版一致的 clean/noisy teacher forcing、block-causal attention 和
  coupled per-block timestep；
- 当前配置启用的 physical consistency 与 clean prior，且它们必须
  遵守第 9 节的可见性和坐标系合同；
- 单 block cached inference、episode-ordered reset 和 GT-history cached evaluator；
- 数据重建、shape、attention 防泄漏、forward/backward、checkpoint 和
  rolling evaluation 测试。

当前尚缺专用旧 checkpoint matching initializer、生成 video 评估与 simulator
closed-loop，因此准确状态是“核心结构和离线 waypoint evaluator 已实现，完整研究验收未完成”。

## 14. 测试清单

当前实际测试文件：

```text
tests/data/test_mobilemanibench_block_plan_labels.py
tests/model/test_mobile_multiblock_plan.py
tests/eval/test_mobilemanibench_multiblock_eval.py
```

后两个文件分别合并覆盖 collator/shape、三 waypoint encoder/decoder、per-block prior、
coupled timestep、attention 防泄漏，以及局部 plan composition、零误差 metric 和推理 batch
去除 future state/action。独立 checkpoint matching test 当前不存在。

2026-08-17 在远程现有环境以 `CUDA_VISIBLE_DEVICES=0` 运行上述三个文件的
`unittest`，15/15 通过。

必须覆盖：

- exact shapes；
- block-major round trip；
- Base/Manipulator branch slicing；
- local/global offset mapping；
- per-block anchor 坐标重建；
- action/video/state block 数一致；
- timestep block coupling；
- attention future leakage；
- invalid/episode boundary 行为；
- normalization round trip；
- checkpoint matching load；
- local plan 到初始 Base frame 的 SE(2)/SE(3) composition；
- 旧单 block evaluator 与训练配置不回归。

## 15. 验收标准

目标实现必须同时满足以下条件：

### 正确性

- converter 语义重建最大误差 `<2e-5`；
- 训练样本无未来 padding；
- 4 video/action/state block 严格对齐；
- attention leakage tests 全通过；
- pack/unpack 与 normalize/unapply 为可逆；
- block 1-3 label 均相对各自 anchor，而非初始 anchor。

### 训练稳定性

- 小数据 overfit 中各项可训练 loss 可明显下降；
- 4 个 block 均有非零有效梯度；
- 无 NaN/Inf；
- Base 与 Manipulator gradient ratio 不出现数量级失衡；
- checkpoint save/load 后相同输入输出误差在数值容差内。

### 效果

- common offsets `[4,8,12,16,24]` 不劣于当前单 block baseline；
- block boundary discontinuity 下降；
- `gt_history_cached` 不低于 `episode_ordered_reset`；
- 后段 block 的误差增幅可控；
- video 可视化没有明显 block 拼接跳变；
- 最终以 simulator/任务成功率作为主要指标，不只看 flow loss。

## 16. 已知边界与兼容性

### 风险 1：teacher forcing exposure bias

训练后段 block 使用 GT previous video，部署使用真实新观测时接口
和因果语义一致，但仍存在闭环访问状态的分布偏移；若做纯生成
4-block rollout，模型会看到自己的预测，误差会累积。评估必须区分
真实历史 rolling 与纯生成 rollout。

### 风险 2：未来 state 不可用

训练 4 block 使用 4 个 anchor state，部署闭环在每个 block 边界能获得真实 state；一次性 one-shot 生成 4 block 时拿不到未来 state。不要把 oracle future-state 结果当作部署能力。

### 风险 3：改变 waypoint 任务定义

目标模型没有 offset 1，而增加了 20、28、32。这会降低 1-frame
紧急响应的直接监督密度，是严格固定 block 和全局均匀采样带来的
明确权衡。

### 风险 4：当前 clean-prior checkpoint 不完全兼容

只能 matching-load 共享参数，不能 resume optimizer/scheduler。初始化报告必须可审计。

## 17. 最终实施检查表

- [x] 动态 block-label spec 与 geometry helper
- [x] 动态/物化 geometry 数值一致性测试
- [x] full-window step filter
- [x] spec-hashed train-only plan stats
- [x] structured Dataset/Transform/Collator
- [ ] block-major pack/unpack helper
- [x] multiblock dual-plan encoder/decoder
- [x] multiblock policy head
- [x] 4-block action timestep mapping
- [x] attention leakage tests
- [ ] matching checkpoint initialization
- [ ] 完整 teacher-forcing forward/backward 与 overfit 验证
- [ ] video generation保存与可视化
- [x] episode-ordered reset evaluator
- [x] GT-history cached evaluator
- [ ] common-offset baseline comparison
- [x] 当前配置启用的 structured physical loss
- [x] 当前配置启用的 per-block clean endpoint prior
- [ ] simulator closed-loop evaluation

## 18. 不应采用的捷径

以下做法看似改动小，但都是错误或不可验证的：

- 只把 `num_frame_per_block` 从 8 改为 2；
- 仍用 12 个 action token，却让 4 个 video block 共用；
- 把旧 `[1,4,8,12,16,24]` 不均匀分给 4 个 block；
- 让每个 block 的 waypoint 数不同并依赖 padding token；
- 把全部 waypoint 继续相对 `B(t)` 编码；
- 使用 branch-major action layout 配合任何单 block register 宽度；
- 训练 block 1-3 读取本 block 的 clean GT video；
- 用未来 GT state 的 one-shot 指标代表真实推理；
- 每 4 个新 RGB 帧 repeat 成 8 帧作为 primary cached inference；
- 复用旧 plan stats；
- 在旧 output dir 中自动 resume optimizer/scheduler；
- 只看总 action loss，不记录 per-block loss 与 video 可视化。

本方案定义一个单一的 multiblock dual-plan WAM 目标：它的
clean/noisy tensor、block-causal attention、coupled timestep、动态
register 长度和单 block cached inference 与原版 DreamZero teacher-forcing
模式一致；Mobile 任务专用的 Base/Manipulator、physical consistency
与 clean prior 必须严格遵守同一 block 边界和因果可见性。
