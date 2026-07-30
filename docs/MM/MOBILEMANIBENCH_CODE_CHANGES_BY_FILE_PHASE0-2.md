# MobileManiBench WAM Phase 0–2 当前实现指南

## 1. 文档定位

本文是当前 MobileManiBench WAM（Wan Action Model）Phase 0–2 代码的学习入口。

它回答的是：

- 训练标签如何从真实轨迹构造；
- Base 与 Manipulator plan 的坐标系、shape 和 mask 是什么；
- 数据如何进入 DreamZero/Wan；
- 6 个 Base tokens 和 6 个 Manipulator tokens 如何编码、交互和解码；
- video dynamics loss 与两路 action flow loss 如何计算；
- 当前有哪些训练、validation 和离线评估入口；
- 哪些能力已经实现，哪些仍只是后续计划。

本文以服务器当前代码为准：

```text
/mnt/yihao/codes/dreamzero
```

本文不再按“相对 Git HEAD 修改了哪些文件”组织，也不使用 `M/N` 工作树状态。Git
状态会随提交变化，不适合作为理解当前模型实现的长期入口。

本文只介绍纯 WAM Phase 0–2 baseline。仓库中已经存在独立的 VGGT tokenizer 和
VGGT-3D-WAM 路径，但纯 WAM baseline 不消费 VGGT 2D/3D tokens。

---

## 2. 一页总览

### 2.1 Phase 0–2 的当前目标

Phase 0–2 已经把原 DreamZero 的单路 step-action 路径扩展为双路、长时域 plan
预测：

```text
MobileManiBench realized future states
        │
        ├── Base plan        [6, 4]
        └── Manipulator plan [6, native_dim]
                    │
                    ▼
          train-only statistics
          normalization + masks
                    │
                    ▼
       packed action [12, 21]
       ├── token 0..5  : Base
       └── token 6..11 : Manipulator
                    │
                    ▼
       independent branch encoders
       + branch type embeddings
       + physical-time embeddings
                    │
                    ▼
          shared causal Wan DiT
        video/action/state joint attention
                    │
                    ▼
       independent branch decoders
                    │
                    ▼
       base_flow_loss
       + manipulator_flow_loss
       + dynamics_loss
```

“双路”不是两个互不通信的网络。两路只在输入投影、type embedding、输出投影和
loss 统计上解耦；12 个 action tokens 位于同一个 Wan action block，在共享
Transformer 中可以完整交互。

### 2.2 当前关键常量

```text
plan_horizon              = 6
plan_time_offsets         = [1, 4, 8, 12, 16, 24]
control_fps               = 30
base_action_dim           = 4
max_manipulator_action_dim= 21
packed_action_horizon     = 12
state_horizon             = 1
max_state_dim             = 64
num_frame_per_block       = 8
num_action_per_block      = 12
num_state_per_block       = 1
```

6 个 waypoint 的物理时间为：

```text
[1/30, 4/30, 8/30, 12/30, 16/30, 24/30] 秒
```

这些时间点不均匀，因此模型不能只依赖普通 token ordinal `0..5`。

### 2.3 当前没有单独的 base prior token stream

当前 `CausalWanModel` 的 action register 是：

```text
[12 noisy plan tokens] + [1 state token]
```

前 6 个 Base plan tokens 本身是 diffusion 预测目标，不是额外的 coarse base
prior。当前代码中没有一组独立的“先预测 coarse base，再作为 Base/Manipulator
先验”的 base-prior tokens。若 proposal 中保留该设计，应明确标记为后续方案，
不能描述成 Phase 0–2 已实现代码。

---

## 3. 当前实现边界

### 3.1 已实现

- 从 realized future state 构造 anchor-relative Base/Manipulator labels；
- G1 与 XHand 的统一 21 维 Manipulator 表示；
- terminal waypoint valid mask；
- train/val episode split；
- train-only 流式统计量；
- slice-aware normalization 和 inverse normalization；
- 双路 action packing、encoder、type embedding、physical-time embedding；
- 共享 Wan video/action/state Transformer；
- 独立 Base/Manipulator decoder；
- 分路 masked flow-matching loss；
- Wan2.1 smoke/兼容训练入口；
- Wan2.2-TI2V-5B 五任务正式训练入口；
- 训练内 validation dataset、固定 validation 子集；
- DreamZero 专用 Trainer validation 调用；
- checkpoint-before-eval 保存顺序；
- checkpoint 离线 sampling 和轨迹指标代码；
- normalization、dataset、双路投影、loss、timestep 和 validation 回归测试。

### 3.2 尚未实现或尚未完整验证

- 独立的 Base coarse prior token stream；
- Base/Manipulator consistency training loss；
- 基于机器人模型的 reachability/collision loss；
- MobileManip 闭环 task-success 仿真；
- 纯 WAM baseline 中的 VGGT 2D/3D conditioning；
- 当前 Wan2.2-5B 正式 checkpoint 的离线 evaluator 兼容修复；
- validation 修复后的下一次正式大模型 eval 结果。

离线 evaluator 虽然已经实现，但 `scripts/eval/evaluate_mobilemanibench_plan.py`
当前会无条件把 `cfg.pretrained_model_path` 当作基础 checkpoint 加载。Wan2.2 正式
配置中该值是 `null`，因此当前 evaluator 不能直接用于该 Wan2.2 checkpoint；
Wan2.2 需要按 raw Wan component + LoRA overlay 的实际加载方式修复。

---

## 4. 文件地图

### 4.1 标签、数据与统计

```text
scripts/data/convert_mobilemanibench_to_gear.py
    原始 MobileManiBench → DreamZero/LeRobot 格式；
    构造 realized Base/Manipulator plan labels。

scripts/data/prepare_mobilemanibench_splits.py
    生成稳定的 train/val episode split。

scripts/data/prepare_mobilemanibench_plan_metadata.py
    只在选定 split 上计算 plan_stats.json；
    mean/std/min/max 精确，q01/q99 流式采样估计。

groot/vla/data/dataset/mobilemanibench_plan.py
    observation 交给 LeRobotSingleDataset；
    plan 直接从当前 parquet row 读取。

groot/vla/data/transform/mobile_plan.py
    normalization、geometry QA、mask 合成和反归一化。

scripts/data/inspect_mobilemanibench_plan_batch.py
    人工检查视频、Base XY、EEF XY 和 mask。
```

### 4.2 模型输入与训练

```text
groot/vla/model/dreamzero/transform/mobile_plan_cotrain.py
    视频 grid、state normalization、action packing、collator。

groot/vla/model/dreamzero/base_vla.py
    VLA 顶层编排：prepare_input → backbone → action_head。

groot/vla/model/dreamzero/action_head/mobile_plan_flow_matching.py
    双路 timestep/layout 协议、分路 flow loss、推理输出拆分。

groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py
    Wan video/action flow-matching 主训练与 sampling 流程。

groot/vla/model/dreamzero/modules/wan_video_dit_dual_plan.py
    双路 action encoder/decoder 和 physical offset embedding。

groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py
    Wan action/state register、attention mask、teacher forcing 和 DiT blocks。

groot/vla/experiment/base.py
    Trainer、loss logging、validation positional batch 调用和保存顺序。
```

### 4.3 Hydra 与启动入口

```text
groot/vla/configs/data/dreamzero/mobilemanibench_plan.yaml
groot/vla/configs/model/dreamzero/transform/mobile_plan_cotrain.yaml
groot/vla/configs/model/dreamzero/action_head/mobile_plan_flow_matching.yaml
groot/vla/configs/model/dreamzero/action_head/mobile_plan_flow_matching_wan22.yaml

scripts/train/mobilemanibench_training.sh
    原 DreamZero step-action baseline，不是 dual-plan。

scripts/train/mobilemanibench_plan_training.sh
    Wan2.1-I2V-14B smoke/兼容 dual-plan。

scripts/train/mobilemanibench_plan_training_wan22_5b.sh
    当前五任务 Wan2.2-TI2V-5B 正式 dual-plan baseline。
```

### 4.4 推理与分析

```text
scripts/eval/evaluate_mobilemanibench_plan.py
scripts/eval/mobilemanibench_plan_eval.sh
    checkpoint flow sampling、inverse normalization、轨迹指标和结果保存。

scripts/eval/evaluate_mobilemanibench_fixed_timestep_flow.py
scripts/eval/mobilemanibench_fixed_timestep_flow_eval.sh
    固定 timestep 的 flow 诊断。

scripts/eval/analyze_mobilemanibench_plan_predictions.py
    prediction 分布、slice regression、per-horizon 和样本可视化。
```

---

## 5. Phase 1：标签是如何构造的

核心函数：

```text
scripts/data/convert_mobilemanibench_to_gear.py::build_plan_labels
```

对于当前 anchor 时刻 `t`，future indices 为：

```text
t + [1, 4, 8, 12, 16, 24]
```

超出 episode 末尾的 index 先 clamp 到最后一帧以保证数组读取安全，随后对应 label
被置零，并由：

```text
action.plan.valid [6]
```

标记为无效。置零本身不是 loss 保护，真正的保护来自后续 valid mask。

### 5.1 坐标系

所有未来位姿都表达在 anchor 时刻的 Base frame `B(t)` 中。

设：

```text
R_B(t), p_B(t)       anchor Base world pose
R_target, p_target   future target world pose
```

则：

```text
p_rel = R_B(t)^T · (p_target - p_B(t))
R_rel = R_B(t)^T · R_target
```

所以 Base 与 EEF 两路共享同一个 anchor frame。这使两路几何关系可比较，也为后续
consistency metric/loss 留出接口。

### 5.2 Base plan

```text
base_plan shape = [6, 4]

0:2  future Base relative XY in B(t)
2    sin(relative yaw)
3    cos(relative yaw)
```

只预测平面移动与 yaw，不包含 Base Z、roll、pitch。

### 5.3 Manipulator plan

```text
manipulator_plan native shape = [6, 9 + hand_dim]

0:3   future EEF XYZ in B(t)
3:9   future EEF rotation6d in B(t)
9:    future hand joint configuration
```

rotation6d 使用相对旋转矩阵的前两行：

```text
R_rel[:2, :].reshape(6)
```

当前机器人：

```text
G1:
    hand_dim = 1
    native manipulator_dim = 10

XHand:
    hand_dim = 12
    native manipulator_dim = 21
```

Dataset 将 G1 从 `[6,10]` pad 到 `[6,21]`，同时保留
`manipulator_dim_mask`，因此 padding 不进入 loss。

### 5.4 当前 state

`build_core_state` 将当前 EEF pose 表达在当前 Base frame 中：

```text
state = [EEF relative XYZ, EEF relative RPY]  # 6 dims
```

该 state 是 condition，不是未来监督目标。

---

## 6. Dataset、split 与统计量

### 6.1 为什么需要专用 Dataset

每个 parquet row 已经存有一个完整的 `[6,...]` future plan。若继续让普通 action
loader 把第一维当成待采样时间轴，就会二次扩展 horizon。

`MobileManiBenchPlanDataset` 因此采用：

```text
observation:
    LeRobotSingleDataset 按 video_delta_indices 读取

plan:
    直接读取 anchor row 中已物化的三个 action.plan.* 列
```

主要输出：

```text
base_plan                [6,4]
manipulator_plan         [6,21]
plan_valid               [6]
base_dim_mask            [6,4]
manipulator_dim_mask     [6,21]
plan_time_offsets        [6]
plan_time_seconds        [6]
episode_index
frame_index
hand_dim
```

### 6.2 train/val split

Dataset 当前支持：

```text
split = train | val | all
split_manifest_path = meta/plan_splits.json
```

episode IDs 在构造底层 `LeRobotSingleDataset` 时过滤，避免同一 episode 的 anchors
跨 train/val 泄漏。

validation 可设置：

```text
max_samples = 1024
```

当完整 validation 大于该值时，代码使用 `np.linspace` 选择固定、等间隔 anchors。
它不是每次随机抽样，所以不同 checkpoint 使用相同 validation 子集。

### 6.3 统计量

`prepare_mobilemanibench_plan_metadata.py` 支持：

```text
--split train|val|all
--split-manifest ...
--write-core-stats
```

正式训练必须用：

```text
--split train
```

Wan2.2 启动脚本会检查：

```text
plan_stats.json["fit_split"] == "train"
```

统计策略：

```text
mean/std/min/max  流式精确统计
q01/q99           确定性均匀 Bernoulli sample 估计
```

只有 `plan_valid=true` 的 waypoint 进入 plan statistics。

---

## 7. Normalization、geometry QA 与 mask

核心类：

```text
groot/vla/data/transform/mobile_plan.py::MobilePlanTransform
```

### 7.1 q01/q99 slice

需要归一化的量：

```text
Base XY
EEF XYZ
每个 hand joint
```

正向：

```text
x_norm = 2 * (x - q01) / (q99 - q01) - 1
x_norm = clip(x_norm, -1, 1)
```

逆向：

```text
x = (x_norm + 1) / 2 * (q99 - q01) + q01
```

当 `q01 == q99` 时，正向固定为 0，逆向恢复 q01，避免除零。

### 7.2 保持原值的几何 slice

```text
Base yaw sin/cos
EEF rotation6d
```

它们不做逐维 quantile normalization，否则会破坏单位圆或旋转正交结构。

### 7.3 geometry QA

有效 waypoint 上检查：

```text
所有 Base/Manipulator 值有限
||[sin(yaw), cos(yaw)]|| ≈ 1
rotation6d 两行范数 ≈ 1
rotation6d 两行点积 ≈ 0
```

### 7.4 最终 loss mask

```text
base_action_mask =
    base_dim_mask AND plan_valid[..., None]

manipulator_action_mask =
    manipulator_dim_mask AND plan_valid[..., None]
```

因此：

- terminal 不存在的 future waypoint 不进 loss；
- G1 为对齐 21 维而额外补出的 11 个 channels 不进 loss；
- Base packed tensor 的后 17 维不进 loss。

---

## 8. 模型输入适配

核心类：

```text
MobilePlanCotrainTransform
MobilePlanDataCollator
```

### 8.1 视频 grid

输入相机：

```text
video.head
video.wrist
```

canonical view 顺序：

```text
view 0 = head
view 1 = black
view 2 = wrist
```

DreamTransform 的普通 2×2 布局是：

```text
+----------------+----------------+
| head           | wrist          |
+----------------+----------------+
| black          | black          |
+----------------+----------------+
```

### 8.2 state

6 维 EEF-relative state 使用 `meta/stats.json["observation.state"]` 的 q01/q99
归一化，然后由 DreamTransform pad 到：

```text
[1,64]
```

`max_state_dim=64` 是 checkpoint/projector 结构兼容要求，不表示有 64 个真实 state
维度。

### 8.3 action packing

```text
Base [6,4]
    → 在 channel 维 pad
Base packed [6,21]

[Base packed, Manipulator]
    → 在 token 维 concat
packed action [12,21]
```

布局：

```text
token 0..5:
    channel 0..3  = Base
    channel 4..20 = 0, mask=false

token 6..11:
    channel 0..20 = Manipulator
    G1 padding channels mask=false
```

Transform 同时保留 semantic branch 字段：

```text
base_action
manipulator_action
base_action_mask
manipulator_action_mask
plan_valid
plan_time_offsets
plan_time_seconds
```

Collator 强制检查 batch shape，防止错误进入大模型后才暴露。

---

## 9. 两个 Wan 运行 profile

### 9.1 Wan2.1-I2V-14B smoke/兼容 profile

入口：

```text
scripts/train/mobilemanibench_plan_training.sh
```

当前默认：

```text
dataset               smoke_v2/g1
GPUs                  8
max_steps             5000
save_steps            500
per-device batch      1
learning rate         1e-5
pretrained base       DreamZero-AgiBot
```

视频：

```text
每视角                 176×320
2×2 grid              352×640
33 RGB frames         → 9 latent frames
latent                9×44×80
DiT patch grid/frame  22×40
frame_seqlen          880
```

### 9.2 Wan2.2-TI2V-5B 正式 profile

入口：

```text
scripts/train/mobilemanibench_plan_training_wan22_5b.sh
```

当前默认：

```text
dataset               MobileManipVLA_dreamzero_g1_5tasks/g1
GPUs                  8
per-device batch      32
global batch          256
max_steps             10000
save_steps            2000
eval_steps            2000
max_eval_samples      1024
learning rate         1e-5
LR scheduler          cosine_with_min_lr
min_lr_rate           0.1
```

视频存在 transform grid 与最终 DiT 输入两级尺寸：

```text
每视角                       160×320
MobilePlan 2×2 grid         320×640
action head target resize   160×320
33 RGB frames               → 9 latent frames
Wan2.2 latent               9×10×20
DiT patch grid/frame        5×10
frame_seqlen                50
```

Wan2.2 没有单独 CLIP 文件，当前脚本从 Wan2.1 checkpoint 目录加载 image encoder。

### 9.3 两个 profile 的共同 block 协议

```text
9 latent frames =
    1 condition latent frame
  + 8 future latent frames

num_frame_per_block  = 8
num_action_per_block = 12
num_state_per_block  = 1
```

因此一次训练 sample 恰好对应一个完整 plan window。

---

## 10. 双路 token encoder/decoder

核心文件：

```text
groot/vla/model/dreamzero/modules/wan_video_dit_dual_plan.py
```

### 10.1 Physical-time embedding

```text
offset_seconds = plan_time_offsets / control_fps
```

经过：

```text
SinusoidalPositionalEncoding
→ Linear
→ SiLU
→ Linear
```

每路第 `h` 个 token 都加上相同的 physical offset embedding。

注意：Wan 内部仍有 action RoPE/token position；physical-time embedding 是额外显式
信息，用来表达 waypoint 间隔不均匀，而不是完全替换 Wan RoPE。

### 10.2 DualPlanActionEncoder

```text
Base:
    base_encoder([B,6,4], timestep)
    + offset_embedding
    + type_embedding[BASE]

Manipulator:
    manipulator_encoder([B,6,21], timestep)
    + offset_embedding
    + type_embedding[MANIPULATOR]
```

随后：

```text
concat → [B,12,dim]
```

当前 `num_embodiments` 在双路 module 中固定为 1。

### 10.3 DualPlanActionDecoder

```text
hidden[:, :6]  → base_decoder        → [B,6,4]
hidden[:, 6:]  → manipulator_decoder → [B,6,21]
```

Base 输出再 pad 为 `[B,6,21]`，最终返回统一：

```text
action_noise_pred [B,12,21]
```

### 10.4 offset contract

`WanVideoDiTDualPlan.forward()` 要求 batch 显式携带 `plan_time_offsets`，并与模型
buffer 中的 `[1,4,8,12,16,24]` 完全一致。缺失或错序会立即报错。

---

## 11. Wan 内部 action/video/state attention

核心文件：

```text
groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py
```

训练时 noisy half 的逻辑 register 为：

```text
[video tokens] [12 action tokens] [1 state token]
```

对于当前唯一的 future block：

### Video query 可以看到

```text
condition clean frame
当前 noisy 8-frame video block
当前 12 action tokens
当前 state token
```

### Action query 可以看到

```text
condition clean frame
当前 noisy 8-frame video block
同一 block 的全部 12 action tokens
当前 state token
```

所以 Base 和 Manipulator tokens 可以相互建模，不是仅按相同 horizon 一对一连接。

### State query

state block 只做自身 attention；它作为 condition 被 video/action queries 读取。

### clean target 是否泄漏

训练调用会把 `clean_x` 与 noisy stream 一起送进 teacher-forcing 路径，但 causal
attention 对第一个 future block 只开放 condition clean frame，不开放该 future
block 的 clean target。当前 one-block 设置因此不会把未来 clean video 直接泄漏给
action tokens。

---

## 12. Flow-matching 训练与 loss

核心文件：

```text
groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py
groot/vla/model/dreamzero/action_head/mobile_plan_flow_matching.py
```

### 12.1 Video 与 action 加噪

训练时：

```text
video latent  + video noise  → noisy video latent
packed action + action noise → noisy action
```

scheduler 同时提供：

```text
add_noise(...)
training_target(...)
training_weight(timestep)
```

模型预测的是 scheduler 定义的 flow/noise target，而不是直接回归 clean plan。

### 12.2 timestep coupling

默认配置：

```text
decouple_video_action_noise = false
```

一个 future video block 采样一个 timestep，并扩展给全部 12 action tokens：

```text
t_base(0..5) = t_manipulator(0..5) = t_block
```

`align_action_timestep_ids` 还会强制 Manipulator 复制 Base 的 6 个 timestep，保护
双路对应关系。

若将 `decouple_video_action_noise=true`，action timestep 会独立采样；这不是当前
正式 baseline 默认行为。

### 12.3 分路 masked loss

Base：

```text
prediction = action_noise_pred[:, :6, :4]
target     = training_target_action[:, :6, :4]
mask       = action_mask[:, :6, :4]
```

Manipulator：

```text
prediction = action_noise_pred[:, 6:, :21]
target     = training_target_action[:, 6:, :21]
mask       = action_mask[:, 6:, :21]
```

每个 token 的计算顺序：

```text
1. 对每个 active channel 计算 squared error
2. 除以该 token 的 active dimension 数
3. 排除 active_dims == 0 的 token
4. 乘 has_real_action
5. 乘 scheduler.training_weight(timestep)
6. 对有效 tokens 求平均
```

这意味着 G1 和 XHand 都先对各自真实维度取 token mean，不会因为 XHand 有更多
hand channels 就简单获得更大的 loss 权重。

最终：

```text
action_loss =
    base_flow_loss_weight * base_flow_loss
  + manipulator_flow_loss_weight * manipulator_flow_loss

loss = dynamics_loss + action_loss
```

当前两个 branch weight 都是 1。

### 12.4 训练输出

```text
loss
dynamics_loss
action_loss
base_flow_loss
manipulator_flow_loss
```

`BaseTrainer.compute_loss()` 对所有 `*_loss` 维护最近 10 step 的 moving average，
`LossLoggerCallback` 将 `*_loss_avg` 写入：

```text
OUTPUT_DIR/loss_log.jsonl
```

---

## 13. LoRA、预训练权重与 checkpoint

两个 profile 都使用：

```text
train_architecture=lora
lora rank=4
lora alpha=4
targets=q,k,v,o,ffn.0,ffn.2
```

LoRA 注入后，当前代码显式训练：

```text
Wan LoRA adapters
state_encoder
dual action_encoder
dual action_decoder
type/offset embedding 所在 action module 参数
```

冻结：

```text
text encoder
image encoder
VAE
非 LoRA Wan 主干参数
```

`save_lora_only=true` 只使 `model.safetensors` overlay 较小，不代表整个 Trainer
checkpoint 很小。当前 Wan2.2 `checkpoint-2000`：

```text
model.safetensors 约 190 MB
完整 checkpoint 约 49 GB
```

大头来自 8 个 ZeRO optimizer state。容量规划必须按完整 checkpoint 计算。

---

## 14. 训练内 validation

### 14.1 数据

`mobilemanibench_plan.yaml` 同时定义：

```text
train_dataset split=train
val_dataset   split=val, max_samples=${max_eval_samples}
```

只有 `do_eval=true` 时，`BaseExperiment.create_val_dataset()` 才实例化 validation。

Wan2.1 smoke 脚本未显式开启 eval；Wan2.2 正式脚本默认：

```text
do_eval=true
eval_strategy=steps
eval_steps=2000
per_device_eval_batch_size=1
max_eval_samples=1024
```

### 14.2 DreamZero positional batch contract

DreamZero VLA 接口是：

```python
model(inputs)
```

而不是：

```python
model(**inputs)
```

`BaseTrainer.prediction_step()` 因此覆盖 Hugging Face 默认行为，用单个 positional
batch dictionary 调用模型，只返回 detached validation loss，不收集巨大
video/action logits。

### 14.3 save-before-eval

当同一 step 同时需要 save/eval 且：

```text
save_strategy != best
```

执行顺序为：

```text
checkpoint save
→ callback on_save
→ logging/evaluation
```

这样 expensive validation 即使失败，该 step checkpoint 也已保存。`best` 策略仍
必须先 eval 再决定是否保存。

### 14.4 no_grad block tuple 修复

Wan attention block 固定返回：

```text
(hidden_states, updated_kv_cache)
```

validation 在 `torch.no_grad()` 下不走 gradient checkpointing。当前
`_training_block_hidden_states()` 统一解包两条路径，并断言训练/validation
不得产生非空 KV cache。

---

## 15. 推理与离线指标

`MobilePlanFlowMatchingActionHead.get_action()` 调用 joint video/action flow sampler，
并把 packed prediction 拆成：

```text
base_plan_pred        [B,6,4]
manipulator_plan_pred [B,6,21]
```

离线 evaluator 的正确推理输入只有当前 RGB observation：

```text
video_delta_indices = [0]
```

训练中的未来 32 RGB frames 是 dynamics targets，不能在部署推理时作为 clean
context 输入。

sampling 完成后使用 `MobilePlanTransform.unapply()` 回到物理单位，再计算：

```text
Base:
    ADE/FDE [m]
    yaw MAE/final error [deg]

Manipulator:
    EEF position ADE/FDE [m]
    EEF orientation geodesic error [deg]
    hand joint MAE

Combined geometry:
    Base-relative EEF position/orientation error

Aggregation:
    mean/median/p90
    per-horizon
    per-sample
```

输出包括：

```text
summary.json
per_sample_metrics.jsonl
per_horizon_metrics.csv
predictions.npz
```

重要限制：当前 evaluator 的模型加载流程适配带
`pretrained_model_path` 的 DreamZero overlay。Wan2.2 正式配置将该字段设为
`null`，所以不能把“评估代码存在”理解为“Wan2.2 checkpoint 已端到端评估通过”。

---

## 16. 当前验证证据

### 16.1 轻量测试

以下 10 个 Phase 1–2 tests 当前通过：

```bash
cd /mnt/yihao/codes/dreamzero
NO_ALBUMENTATIONS_UPDATE=1 \
/mnt/yihao/envs/dreamzero/bin/python -m unittest \
  tests.model.test_mobile_plan_phase2 \
  tests.data.test_mobilemanibench_plan_transform \
  tests.data.test_mobilemanibench_plan_dataset \
  -v
```

覆盖：

- G1/XHand dataset shape、padding、terminal mask；
- stored labels 可由 preserved realized state 重建；
- normalization、geometry slice、mask 和 inverse；
- 双路 projection shape 与 gradients；
- invalid/padding 不影响 branch loss；
- Base/Manipulator timestep 对齐；
- validation block tuple 解包。

Trainer validation tests 建议在外部关闭可见 GPU，避免同一 unittest 进程中先导入
Torch 后触发 DataParallel 测试副本状态问题：

```bash
CUDA_VISIBLE_DEVICES='' \
/mnt/yihao/envs/dreamzero/bin/python -m unittest \
  tests.experiment.test_base_trainer_prediction_step \
  -v
```

它覆盖：

- validation 以 `model(inputs)` 调用；
- 同 step 的 `save → evaluate` 顺序。

### 16.2 preflight

```bash
PREFLIGHT_ONLY=1 \
bash scripts/train/mobilemanibench_plan_training.sh

PREFLIGHT_ONLY=1 \
bash scripts/train/mobilemanibench_plan_training_wan22_5b.sh
```

当前两者都能通过文件和配置前置检查。

### 16.3 Wan2.2 正式训练证据

当前五任务 run 已产生：

```text
checkpoint-2000
loss_log.jsonl 已记录到 step 3020
```

观测到：

```text
step 0:
    dynamics_loss_avg = 1.4600
    action_loss_avg   = 5.3409

step 3020:
    dynamics_loss_avg = 0.2297
    action_loss_avg   = 0.1259
```

这证明当前双路 forward、backward、optimizer 和 loss logging 已在正式 5B 训练中
工作。

但 `checkpoint-2000/trainer_state.json` 尚无成功 `eval_loss`：首次 step-2000
validation 在上述接口修复前失败。当前修复已通过轻量回归测试，仍需用下一次正式
validation 结果完成大模型端到端确认。

---

## 17. 阅读顺序

### 第一步：理解 label 和坐标系

```text
scripts/data/convert_mobilemanibench_to_gear.py::build_plan_labels
scripts/data/convert_mobilemanibench_to_gear.py::build_core_state
tests/data/test_mobilemanibench_plan_dataset.py
```

应能回答：

- 为什么所有 future pose 都在 `B(t)`；
- Base yaw 为什么用 sin/cos；
- EEF rotation6d 采用哪两行；
- terminal plan 为什么置零且 mask=false。

### 第二步：理解 normalization 与 packing

```text
groot/vla/data/dataset/mobilemanibench_plan.py
groot/vla/data/transform/mobile_plan.py
groot/vla/model/dreamzero/transform/mobile_plan_cotrain.py
```

应能回答：

- G1 `[6,10]` 如何统一为 `[6,21]`；
- 哪些 slice 做 q01/q99；
- `[6,4]+[6,21]` 如何变成 `[12,21]`；
- video grid 和 64 维 state 从哪里来。

### 第三步：理解双路 DiT

```text
groot/vla/model/dreamzero/modules/wan_video_dit_dual_plan.py
groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py
tests/model/test_mobile_plan_phase2.py
```

应能回答：

- 两路在哪里独立、在哪里交互；
- physical-time embedding 与 Wan RoPE 的区别；
- action/video/state attention 可见性；
- 为什么没有 future clean target leakage。

### 第四步：理解 flow loss 和训练

```text
groot/vla/model/dreamzero/action_head/mobile_plan_flow_matching.py
groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py
scripts/train/mobilemanibench_plan_training_wan22_5b.sh
groot/vla/experiment/base.py
```

应能回答：

- shared timestep 如何构造；
- mask 如何进入分路 loss；
- dynamics/action loss 如何相加；
- LoRA 与双路 projector 哪些参数可训练；
- validation 和 checkpoint 为什么按当前顺序执行。

### 第五步：理解部署评估

```text
scripts/eval/evaluate_mobilemanibench_plan.py
scripts/eval/analyze_mobilemanibench_plan_predictions.py
```

应能回答：

- 为什么推理只能提供当前 observation；
- prediction 如何反归一化；
- ADE/FDE、orientation 和 relative EEF 指标如何汇总；
- 当前 Wan2.2 evaluator 还缺哪一步兼容修复。

---

## 18. 最容易混淆的结论

1. `[12,21]` 是存储/接口矩形，不表示 Base 是 21 维。
2. Base 和 Manipulator 有独立投影，但共享 Wan blocks，并能完整交互。
3. `plan_time_offsets` 是物理 future 时间，不是普通 ordinal。
4. 33 帧训练视频包含 dynamics target；部署推理不能输入未来 32 帧。
5. 6 维真实 state pad 到 64 维是 projector/checkpoint 兼容，不是新增状态语义。
6. Base plan tokens 是待去噪目标，不是独立 coarse base-prior tokens。
7. validation loss 已接入，但不是完整 trajectory metric。
8. trajectory evaluator 已实现，不代表当前 Wan2.2 checkpoint 已跑通。
9. `save_lora_only=true` 不会消除 ZeRO optimizer checkpoint 的大容量。
10. VGGT 已在仓库独立路径实现，但不属于本文件的纯 WAM Phase 0–2 输入。
