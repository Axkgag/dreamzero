# MobileManiBench 1.6 秒对齐 WAM 与阶段均衡最终方案

> 目标仓库：`/mnt/yihao/codes/dreamzero`
>
> 目标分支：`multiblock-wamv2`
>
> 文档性质：`multiblock-wamv2` 的实现合同。本文描述的训练配置、阶段 sidecar、variable-block/success-hold mask、多 Prior 和真实历史 cache rebuild 已在该分支落地。

## 1. 目标与设计结论

本方案同时解决两个问题：

1. 当前 MobileManiBench WAM 的单 block 时间跨度只有 `8 / 30 = 0.267 s`，不足以表达 Base 导航、EEF 接近、抓取和持续操作之间的长时协调；
2. 原始轨迹中导航阶段持续时间长，抓取、接触和物体操作阶段短，普通 root-frame 均匀采样会使导航窗口主导训练。

最终设计采用以下时间尺度：

```text
控制频率                 30 Hz
视频建模频率              5 Hz
RGB 采样间隔              6 control ticks
单 block 物理时长         48 ticks = 1.6 s
单 block 未来 RGB         8 frames
单 block video latent     2 latent frames
训练 block 数             4
完整训练时间范围          192 ticks = 6.4 s
动作预测时间范围          48 ticks = 1.6 s
默认闭环执行时间          16 ticks = 0.533 s
```

核心原则为：

- video 和 action 在训练目标上都覆盖 1.6 秒；
- action 使用近端密、远端疏的 waypoint；
- 每次只执行预测计划的前三分之一，然后基于真实观测重新规划；
- 预测视频永远不写入长期 KV cache；
- 在 `rebuild_cache_from_real_history=true` 时，每次重规划从真实历史重新构造 causal context，避免将 0.533 秒的执行错误地当作一个 1.6 秒 block 写入 cache；
- 训练以 0.7/0.3 混合 phase-balanced 分布与自然分布；task balance 保留为可配置项，默认关闭；
- phase-balanced 分支还必须同时均衡 block slot 和 waypoint horizon。

## 2. 最终目标配置

目标顶层配置如下：

```yaml
control_fps: 30
video_fps: 5
video_sample_stride: 6

num_plan_blocks: 4
num_frame_per_block: 2
block_anchor_offsets: [0, 48, 96, 144]
plan_local_offsets: [1, 2, 4, 8, 16, 32, 48]

prior:
  time_offsets: [4, 16, 32, 48]
  predict_base: true
  predict_eef: true
  # 沿用当前 clean-prior 的 EEF 目标定义；由 clean action 在线构造，
  # 不要求重新转换 parquet。
  eef_frame: future_base

sampling:
  task_balanced: false
  phase_balanced_ratio: 0.7
  natural_ratio: 0.3
  balance_block_slots: true
  balance_waypoint_horizons: true

inference:
  prediction_horizon_ticks: 48
  execution_horizon_ticks: 16
  rebuild_cache_from_real_history: true
```

以下字段继续由现有配置保留或动态派生：

```yaml
num_frames: 33
max_chunk_size: 4
plan_waypoints_per_block: 7       # len(plan_local_offsets)
num_flow_action_per_block: 14     # 2 * plan_waypoints_per_block
action_horizon: 14                # 单 block Flow token 数，不是 48 ticks
state_horizon: 1
num_state_per_block: 1
```

`48` 是物理预测 horizon；`action_horizon=14` 是单 block 内 `7 Base + 7 EEF` Flow token 的数量。二者不能混用。

配置加载时必须验证：

```text
control_fps % video_fps == 0
video_sample_stride == control_fps / video_fps == 6
len(block_anchor_offsets) == num_plan_blocks == 4
block_anchor_offsets == [i * 48 for i in range(num_plan_blocks)]
plan_waypoints_per_block == len(plan_local_offsets) == 7
max(plan_local_offsets) == prediction_horizon_ticks == 48
all offsets strictly increase and are within [1, 48]
all prior.time_offsets strictly increase and are within [1, 48]
phase_balanced_ratio + natural_ratio == 1.0
num_flow_action_per_block == 2 * plan_waypoints_per_block == 14
```

不应让 `num_action_per_block`、`action_horizon` 或其他 shape 参数在多个 YAML 中分别手写不同数值；必须从 waypoint 数量和 prior token 数动态派生并在启动时 fail fast。

## 3. Video 时间采样与 block 对齐

### 3.1 原始数据不重新转换

现有 `g1` parquet 和视频保存的是逐帧 episode 数据。新方案只改变 dataset loader 的索引生成和动态 label 构造，不生成另一份固定-offset parquet。

训练样本以 root tick `r` 为起点，动态读取：

```text
video_delta_indices = [0, 6, 12, ..., 192]
                    = 33 RGB frames
```

33 张 RGB 经 Wan VAE 编码后仍为：

```text
[z0, z1, z2, z3, z4, z5, z6, z7, z8]
```

其中 `z0` 是初始条件 latent，未来 latent 分成四个 block：

| Block | 真实时间范围 | RGB 采样 tick | Video latent | State anchor |
|---|---:|---|---|---:|
| 0 | `(r, r+48]` | `r+6,12,...,48` | `[z1,z2]` | `r` |
| 1 | `(r+48, r+96]` | `r+54,60,...,96` | `[z3,z4]` | `r+48` |
| 2 | `(r+96, r+144]` | `r+102,108,...,144` | `[z5,z6]` | `r+96` |
| 3 | `(r+144, r+192]` | `r+150,156,...,192` | `[z7,z8]` | `r+144` |

`num_frame_per_block=2` 的单位始终是 VAE latent frame，不是 RGB frame。每个 block 的两个 latent 联合表示 8 个以 5 Hz 采样的未来 RGB，也就是 1.6 秒。

### 3.2 Teacher forcing 可见关系不变

新时间尺度只改变帧索引，不改变当前 multiblock teacher forcing 合同：

```text
预测 noisy [z1,z2]：可见 clean [z0]
预测 noisy [z3,z4]：可见 clean [z0,z1,z2]
预测 noisy [z5,z6]：可见 clean [z0,z1,z2,z3,z4]
预测 noisy [z7,z8]：可见 clean [z0,z1,z2,z3,z4,z5,z6]
```

当前 block 或未来 block 的 clean target 不可见。以前一 block 的模型预测代替 clean GT 也不属于本训练方案。

## 4. Base/EEF waypoint 与 token 布局

### 4.1 单 block waypoint

每个 block 的局部 waypoint 为：

```text
[1, 2, 4, 8, 16, 32, 48] ticks
```

对应物理时间：

| Offset | 时间 |
|---:|---:|
| 1 | 0.033 s |
| 2 | 0.067 s |
| 4 | 0.133 s |
| 8 | 0.267 s |
| 16 | 0.533 s |
| 32 | 1.067 s |
| 48 | 1.600 s |

四个 block 的全局目标为：

```text
Block 0: r + [1,2,4,8,16,32,48]
Block 1: r + [49,50,52,56,64,80,96]
Block 2: r + [97,98,100,104,112,128,144]
Block 3: r + [145,146,148,152,160,176,192]
```

`offset=1` 主要提供当前轨迹切线和短期平滑监督。它只有 33 ms，在异步实机上可能早于完整 WAM 推理结果返回，因此不能把它作为保证会被执行的远程目标；求解器应从当前实测状态连续插值到后续 waypoint。

### 4.2 Packing 和 shape

Dataset/transform 保留结构化张量：

```text
base_plan:        [B, 4, 7, 4]
manipulator_plan: [B, 4, 7, 21]
plan_valid:       [B, 4, 7]
state:            [B, 4, 64]
block_valid:      [B, 4]
```

进入 DiT 前使用 block-major packing：

```text
Block 0: Base[7], EEF[7]
Block 1: Base[7], EEF[7]
Block 2: Base[7], EEF[7]
Block 3: Base[7], EEF[7]
```

无 prior 时，单 block Flow action token 数为 14，四 block 训练 Flow tensor 为 `[B, 56, D]`。不能打包为 `[全部 Base][全部 EEF]`。

Base 和 EEF 目标继续相对各自 block anchor 构造。Block `b` 的状态与坐标 anchor 是 `r + block_anchor_offsets[b]`，不能仍然全部相对 `B(r)`。

### 4.3 Prior

每个 block 使用四个 prior offsets：

```text
[4, 16, 32, 48]
```

Base prior 和 EEF prior 使用解耦输出 head，但共享相同的 block 条件和 clean-prior token 交互。`future_base` EEF prior 目标从当前样本的 clean Base/EEF action 动态组合，不在 parquet 中固化。

Prior、Flow 和 consistency loss 的现有 warmup/ramp 机制保持独立；改变 offset 数量后必须按有效 prior 数归一化，不能因为从 1 个 prior 增加到 4 个就把总 prior 梯度放大四倍。

## 5. 阶段标签

### 5.1 阶段集合

phase-balanced 分支使用四个粗粒度阶段：

```text
navigation
approach
grasp
manipulation
```

默认在 balanced 分支内四阶段等概率，不额外引入未出现在目标配置中的 phase weight：

```text
P(phase | balanced branch) = 0.25
```

成功后的稳定保持窗口作为 `manipulation` 的一个子类记录 `success_hold=true`，并在统计与验证中单独报告；如果后续确认其数据量足够，再决定是否提升为独立采样 phase。

### 5.2 自动阶段锚点

阶段解析器优先从 `reference/episode_xxxxxx/state_infos.pkl` 读取 object、success、robot base、hand/EEF 和时间戳。字段适配放在 task adapter 中，采样器只消费统一事件：

```text
t_object_move
t_grasp
t_success
```

第一层稳定规则以物体持续运动为锚点：

```text
navigation:   episode start -> t_object_move - 2.0 s
approach:     t_object_move - 2.0 s -> t_object_move - 0.5 s
grasp:        t_object_move - 0.5 s -> t_object_move
manipulation: t_object_move -> episode end
```

物体开始运动要求方向归一化 progress 超过阈值并持续若干帧；初始默认值为：

```yaml
phase_index:
  progress_threshold: 0.02
  persistent_frames: 4
  approach_lead_seconds: 2.0
  grasp_lead_seconds: 0.5
```

hand 闭合事件可靠时，用检测到的 `t_grasp` 替代固定的 `t_object_move - 0.5 s`。没有有效物体运动的 episode 标记为 `no_object_motion`，不静默并入 manipulation。

### 5.3 独立索引，不修改数据

阶段扫描只生成 sidecar index，例如：

```text
meta/phase_index.jsonl
meta/phase_index_summary.json
```

索引至少包含：

```json
{
  "episode_index": 71243,
  "task": "open_door",
  "phase": "grasp",
  "target_frame": 180,
  "object_move_frame": 186,
  "grasp_frame": 180,
  "success_frame": 186,
  "has_object_motion": true,
  "is_success_episode": true,
  "success_hold": false
}
```

train/val split 继续按 episode 隔离。sidecar index 不得跨 split 共享窗口。

## 6. Task/Phase/Block/Horizon 联合均衡采样

### 6.1 顶层混合分布

`task_balanced` 默认关闭，以便先独立验证 phase/block/horizon 均衡是否有效；开启后对 balanced 和 natural 两个分支都生效。开启时五任务首先以相同概率选择：

```text
P(task) = 1 / 5
```

然后选择采样分支：

```text
P(phase-balanced branch) = 0.7
P(natural branch)        = 0.3
```

balanced 分支：

```text
1. 均匀选择 phase；
2. 在 task × phase 下均匀选择 episode；
3. 均匀选择有效 block slot；
4. 均匀选择有效 waypoint horizon；
5. 选择该 episode/phase 的目标时刻 tau；
6. 反推出 root。
```

natural 分支：

```text
1. task 仍保持均匀；
2. 在该 task 的原始合法 root 分布中采样；
3. 不人为均衡 phase/block/horizon。
```

开启 task balance 时，0.3 natural 不会重新引入五任务数量失衡，只恢复每个任务内部的自然阶段和转移比例；默认关闭时则保留任务的经验分布。

### 6.2 目标中心的 root 构造

阶段标签由模型实际监督的目标时刻决定，而不是只由 root 所在阶段决定。

对目标时刻 `tau`、block slot `b` 和 waypoint index `j`：

```text
root = tau - block_anchor_offsets[b] - plan_local_offsets[j]
```

候选项必须同时满足：

```text
0 <= root <= episode_length - 1 - 48；
被选择的 target 确实有效；
所需 RGB/state/action index 可构造；
不会跨 language/task segment；
episode 属于当前 split。
```

这里的合法 anchor 与原版 DreamZero 一致，只要求至少存在一个完整的未来 block，而不要求完整四 block。采样器应枚举目标时刻 `tau` 对应的所有合法 `(root, block_slot, waypoint_horizon)`，再按 block/horizon 均衡策略选择候选，不能把尾部 root 对齐或 clamp 到少数固定位置。

对于非成功 episode，被选择的目标 block 必须完整；对于成功 episode，允许目标 block 在真实终止帧处结束，并按照第 7 节显式补为 `success_hold`。若某个 task × phase 下没有合法 pair，应在同一 task × phase 内重新选择有效 pair，不得退化为导航样本。

### 6.3 Episode balance 与重叠窗口

balanced 分支必须先选 episode，再选窗口，避免长 episode 和 stride-1 重叠窗口获得更高概率。

推荐候选 root stride：

```yaml
window_stride:
  navigation: 4
  approach: 2
  grasp: 1
  manipulation: 1
```

同一 batch 默认限制同一 episode 至多出现一次；若某个 task × phase 的 episode 数小于 batch 配额，再允许有控制的重复。采样日志必须输出 unique episode 数和最大重复倍率。

## 7. 原版式合法 anchor、可变 block 与成功保持

### 7.1 合法 anchor

长度为 `T`、真实帧为 `0...T-1` 的 episode，只要求 root 后至少存在一个完整的 48-tick block：

```text
root + 48 <= T - 1
```

不要求 `root + 192 <= T - 1`，也不把 root 向前移动到让终止帧恰好落在某个 block endpoint。这样末段接触、抓取和保持可以出现在任意 block slot；终止帧也可以自然落在某个 block 的中间，训练分布不会把“操作末段”与 Block 3 或最远 waypoint 人为绑定。

### 7.2 真实完整 block 与部分终止 block

对给定 root：

```text
remaining = (T - 1) - root
num_real_blocks = min(4, floor(remaining / 48))
terminal_local_offset = remaining mod 48
```

前 `num_real_blocks` 是完全来自数据的真实 block。若 `terminal_local_offset > 0`，真实终止帧位于下一个 block 内部；代码不得平移窗口，也不得把这部分误认为普通完整 block。

固定 tensor shape 保持为：

```text
num_valid_blocks in [1, 4]
block_valid: [B, 4]
video_valid: [B, 33]
plan_valid:  [B, 4, 7]
state_valid: [B, 4]
```

### 7.3 显式 absorbing `success_hold`

只有 episode 尾部具有持续成功标记时，才允许把终止状态重复到当前部分 block 的边界：

```text
hold_block_slot = num_real_blocks
hold_ticks = 48 - terminal_local_offset
```

补全过程为：

- RGB 重复最后真实帧；
- Base、EEF 和 hand 的终止后 waypoint 重复最后真实状态，并用现有动态几何编码生成 target；该部分 block 的 anchor state 仍取真实 block 起点，后续无效 block state 不做重复监督；
- 终止前目标标记为 `real`，终止后到 block endpoint 的目标标记为 `success_hold`；
- 该 block 之后仍为普通 `padding`，不得参与任何 loss；
- 非成功、失败或尾部成功不稳定的 episode 不生成 hold，部分 block 整体无效。

例如 `episode_length=171`、`root=0`，真实终止帧 170 位于 Block 3 的 local offset 26：

```text
num_real_blocks = 3
hold_block_slot = 3
terminal_local_offset = 26
hold_ticks = 22
```

因此 Block 0--2 是纯真实监督，Block 3 由真实前缀和显式 hold 后缀共同组成；无需把 root 移到 26。若同一 episode 取 `root=100`，终止则位于 Block 1 的 local offset 22，末段操作自然进入更近的 block，而不是固定在 Block 3。

### 7.4 Mask 与统计要求

- `plan_valid`、`state_valid`、`video_latent_valid` 必须共同收紧为同一个连续 `block_valid` 前缀；
- 有效 block 可以是全真实，也可以是“真实前缀 + success_hold 后缀”，但不能包含未标记 padding；
- 无效 block 不参与 Flow、dynamics、prior 或 consistency loss；
- 所有 loss 按有效元素数量归一化；
- attention 中有效的较早 block 不得读取无效的 padded future；
- 训练与验证分别报告 real/hold/padding 比例、有效 block 数和 full-window 子集；
- `success_hold` 只表达成功后的 absorbing state，不能被阶段均衡器当成新演示或恢复轨迹。

阶段均衡不能创造不存在的末段多样性。单个成功 episode 的过采样倍率仍需受控；若 contact/manipulation 的 unique episode 不足，应补充演示或 recovery rollout，而不是继续提高 sampler 权重。

## 8. Loss 归一化与训练日志

任意带 mask 的 loss 均使用：

```text
sum(valid_mask * element_loss) / max(sum(valid_mask), 1)
```

不能先对包含 padding 的完整 tensor 求 mean 再乘 mask。Base/EEF Flow、video dynamics、Base prior、EEF prior、joint consistency 和 physical consistency 都必须遵循相同原则。

训练日志至少记录：

```text
sampled_task_ratio/*
sampled_phase_ratio/*
sampled_block_slot_ratio/*
sampled_waypoint_horizon_ratio/*
unique_episode_count/*
valid_video_ratio/*
valid_plan_ratio/*
valid_block_ratio/*
loss_by_phase/*
base_loss_by_horizon/*
eef_position_loss_by_horizon/*
eef_rotation_loss_by_horizon/*
prior_loss_by_offset/*
```

必须在训练启动后的前若干百个 batch 内验证实际采样比例，而不是只打印配置概率。

## 9. 闭环推理与 cache 合同

### 9.1 预测与执行

每次规划预测：

```text
未来 48 ticks = 1.6 s
```

默认只执行：

```text
前 16 ticks = 0.533 s
```

求解器在 30 Hz 下从当前实测状态插值并跟踪 `[1,2,4,8,16]` 的 waypoint；`[32,48]` 只提供远期意图。本轮执行结束后，丢弃未执行的远期计划并重新观测、重新规划。

### 9.2 真实历史 buffer

runtime 必须持续收集 30 Hz 真实 RGB 和 state，而不是只在 WAM 调用时保存一帧。重规划时根据 timestamp 选择最近的 5 Hz 观测；由于 16 不能被 6 整除，使用最近时间戳匹配而不是假设固定整数 index，最大时间误差不得超过半个 video period。

预测视频只用于 dynamics/可视化/辅助评估，不进入下一轮真实历史。

### 9.3 rebuild_cache_from_real_history

该模式每次重规划都执行：

```text
1. 清空本轮临时 self-attention KV cache 和 current_start_frame；
2. 从当前时刻向前选择最多 3 个完整的 1.6 秒真实视频 block；
3. 按 5 Hz 编码这些真实历史 block，并以 timestep=0 prefill cache；
4. 使用当前真实 state 预测一个新的 1.6 秒 future block 和 plan；
5. 执行前 16 ticks；
6. 下一轮重新从真实历史构造，不 append 本轮预测 block。
```

最多使用 3 个完成的历史 block，是因为训练最多 4 个 block，当前待预测 block 必须占用其中一个 slot。完整时间关系为：

```text
最多 4.8 秒真实历史 + 1.6 秒当前预测 = 6.4 秒模型窗口
```

episode 开始时允许 0、1 或 2 个历史 block，不使用重复旧帧伪装成完整历史。缺失历史通过有效长度和正确 RoPE 起点处理。

此模式计算量高于 persistent cache，但时间语义明确。只有在其闭环正确性验证通过后，才考虑把已经完整结束、且不会与下一轮 rolling window 重叠的 GT block 做增量缓存优化；优化不得改变模型输入和输出。

## 10. 验证与验收指标

### 10.1 数据与采样验收

必须生成：

```text
task × phase 计数
phase × block_slot 计数
phase × waypoint_horizon 计数
每个 episode frame 作为 root/block anchor/waypoint/RGB 的覆盖次数
full-window、variable-block 与 success-hold 样本数量
no_object_motion 与 parse_failed episode 数量
```

70% balanced 分支的四阶段比例应接近 25%/25%/25%/25%；只有启用 `task_balanced` 时，五任务总采样比例才应接近 20%/task。统计容差需按观测 batch 数给出，不以单个小 batch 判断失败。

### 10.2 Open-loop 指标

除总 ADE/FDE 外，至少报告：

```text
Base position/yaw error @ [1,2,4,8,16,32,48]
EEF position/rotation error @ [1,2,4,8,16,32,48]
prior error @ [4,16,32,48]
上述指标按 task 和 phase 分组
full-window、variable-block 与 success-hold 分组
```

不能只用导航占主导的全局平均数判断改进。

### 10.3 Cache 与闭环指标

固定场景、固定 episode、固定 seed 对比：

```text
cache reset 单窗口 open-loop
真实历史 cache rebuild rollout
Base GT + WAM EEF/hand 回放
WAM Base + EEF/hand GT 回放
完整 WAM 闭环
```

同时记录 waypoint-to-control 转换后的实际速度、角速度、跨 plan 跳变、接触保持、物体进度和最终 success。闭环成功率是最终指标；开环 loss 下降不能替代闭环验收。

## 11. 实现落点

当前实现涉及：

- `groot/vla/configs/data/dreamzero/mobilemanibench_multiblock_plan.yaml`
  - 增加视频物理采样频率、stride、48-tick anchors、7-waypoint offsets、sampling 配置；
- `groot/vla/configs/model/dreamzero/action_head/mobile_plan_multiblock_clean_prior_physical_consistency.yaml`
  - prior offsets 改为 `[4,16,32,48]`，启用 Base+EEF，并保持动态 token shape；
- `groot/vla/data/dataset/mobilemanibench_block_plan.py`
  - 动态 5 Hz 视频索引、原版式至少一完整 block 的合法 anchor、阶段 sidecar、可变 block mask，以及显式 absorbing `success_hold`；
- `groot/vla/model/dreamzero/transform/mobile_plan_cotrain.py`
  - 4×7 plan、real/hold/padding 的 video/plan/state/block mask 和动态 shape 校验；
- `groot/vla/model/dreamzero/action_head/mobile_plan_multiblock_flow_matching.py`
  - 动态 14-token Flow block、所有辅助 loss 的有效元素归一化；
- `groot/vla/model/dreamzero/modules/wan_video_dit_dual_plan_multiblock.py`
  - 动态 register 宽度、variable-block mask 和 teacher-forcing attention 验证；
- `scripts/data/`
  - 新增 phase index 生成与统计工具，但不改写原始 parquet；
- `scripts/train/mobilemanibench_multiblock_plan_training_wan22_5b.sh`
  - 使用新配置并在启动时打印全部派生时间/token 不变量；
- `scripts/eval/evaluate_mobilemanibench_multiblock_plan.py`
  - 导出 task/phase/block/horizon/full/variable/hold 分组指标；
- `groot/vla/utils/mobilemanibench_receding_horizon.py`
  - 连续真实观测 buffer、最近时间戳 5 Hz sampling、可执行 waypoint 选择和 cache reset；
- `eval_utils/serve_mobilemanibench_wam.py`
  - 单 GPU websocket runtime；每轮清空临时 cache，从最多三个完整真实历史 block 重建，再输出 48-tick plan 与前 16-tick 可执行索引。

实现不得修改 legacy single-block baseline 的数据、模型或验证入口。

## 12. 风险与明确边界

1. 合法 root 只要求一个完整 1.6 秒 block；四 block 是最大上下文而非采样门槛，成功 episode 的部分终止 block通过显式 hold 补全。
2. `offset=1` 对真实部署可能早于推理返回，保留它是为了局部形状监督，不代表 33 ms 控制承诺。
3. 每次重建真实历史 cache 会增加推理成本；它首先保证时序正确，性能优化不应通过重复帧或写入预测视频实现。
4. 阶段均衡只解决专家轨迹中的阶段覆盖，不会自动教会模型从 EEF overshoot、丢失接触或 Base/EEF 不一致中恢复；recovery 数据仍是独立问题。
5. 对稀缺 manipulation episode 过采样会造成记忆化，必须同时监控 unique episode 数和固定场景闭环泛化。
6. 物体运动不等同于正确抓取，阶段索引必须经过每任务可视化抽检。
7. 训练、验证和推理均必须使用同一 `control_fps`、`video_fps`、block duration、offset 单位和坐标 anchor 定义。

## 13. 最终不变量摘要

```text
1 video block = 8 RGB @ 5 Hz = 2 latent = 48 ticks @ 30 Hz = 1.6 s
1 action block = 7 Base + 7 EEF Flow tokens, max offset 48 = 1.6 s
1 prior block = 4 offsets [4,16,32,48], Base+EEF jointly conditioned
4 training blocks = 33 sampled RGB = 9 latent = 192 ticks = 6.4 s
70% phase/block/horizon balanced + 30% natural sampling；task balance 默认关闭、可配置开启
1 inference call predicts 48 ticks, executes 16 ticks, then replans
cache rebuild consumes only real history; predicted video is never persistent context
```

以上不变量共同构成本方案的实现与验收合同。任何单项变化都必须同步检查视频采样、action/state anchor、token shape、loss mask、阶段索引和闭环 cache 时间轴。
