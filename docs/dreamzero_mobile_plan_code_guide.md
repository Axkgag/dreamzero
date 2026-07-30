# DreamZero Mobile 双计划 VLA 代码导读

> 本文解释远端服务器 `yihao@A100-gpu003` 上
> `/mnt/yihao/codes/dreamzero` 当前工作区中的实现。
>
> - 分支：`main`
> - 基准提交：`251bb43`
> - 阅读日期：2026-07-28
> - 注意：`wan_flow_matching_action_tf.py`、`wan_video_dit_action_casual_chunk.py`
>   和 `experiment/base.py` 在阅读时包含未提交修改，本文以这些文件的**当前工作区内容**
>   为准。

## 1. 一句话总览

这套代码把 MobileManiBench 的一个样本整理成“视频上下文 + 机器人状态 +
两条语义不同但共享未来时间点的计划”：

- Base plan：6 个未来时刻，每个时刻 4 维；
- Manipulator plan：同样 6 个未来时刻，每个时刻 21 维；
- 为了复用 Wan 原有的定长 action register，两路被打包为
  `[B, 12, 21]`，前 6 个 token 是补零后的 Base，后 6 个 token 是
  Manipulator；
- 两路计划使用相同的 6 个物理时间偏移和相同的 diffusion timestep，
  但使用独立的 encoder、decoder、type embedding 和 flow loss；
- Wan DiT 将视频 patch token、action token 和 state token 放进同一套
  causal/teacher-forcing Transformer 中，联合预测 video flow 和 action flow；
- `BaseTrainer` 负责把多个 loss 做滑动平均、以正确的 positional batch
  接口做验证，并在同一步同时触发保存和验证时先保存再验证。

核心数据流如下：

```text
原始样本
  ├─ video.head / video.wrist
  ├─ eef position / rotation
  ├─ base plan      [6, 4]
  └─ manip plan     [6, 21]
          │
          ▼
MobilePlanCotrainTransform
  ├─ 2×2 video grid
  ├─ state q01/q99 normalization
  └─ packed action [12, 21] + mask
          │
          ▼
MobilePlanDataCollator → batch
          │
          ▼
VLA.prepare_input
  ├─ backbone.prepare_input
  └─ action_head.prepare_input
          │
          ▼
backbone → BatchFeature(backbone_features, ...)
          │
          ▼
MobilePlanFlowMatchingActionHead / WANPolicyHead
  ├─ VAE/CLIP/T5 条件
  ├─ video/action 加噪
  ├─ WanVideoDiTDualPlan 联合预测 flow
  └─ dynamics loss + base loss + manipulator loss
```

## 2. 关键张量和布局约定

以下记号贯穿所有文件：

- `B`：batch size；
- `H = 6`：单路 plan horizon；
- `D_base = 4`；
- `D_manip = 21`；
- packed action horizon 为 `2H = 12`；
- packed action dim 为 `D_manip = 21`；
- `S`：state token 数；
- `F`：VAE latent frame 数；
- `P`：每帧视频 patch token 数，当前 patch stride 为 `(1, 2, 2)`，
  所以 `P = (H_latent // 2) * (W_latent // 2)`。

重要布局：

| 名称 | 形状 | 含义 |
|---|---:|---|
| `base_action` | `[B, 6, 4]` | 底盘/基座计划 |
| `manipulator_action` | `[B, 6, 21]` | 操作臂计划 |
| packed `action` | `[B, 12, 21]` | `[Base-padded ; Manipulator]` |
| `action_mask` | `[B, 12, 21]` | Base 后 17 维为 `False` |
| `plan_time_offsets` | `[B, 6]` | 默认 `[1,4,8,12,16,24]` 控制 tick |
| physical seconds | `[B, 6]` | offsets / 30，即约 `[0.033,0.133,0.267,0.4,0.533,0.8]` 秒 |
| DiT action register | `[B, 12, dim]` | 前 6 Base token，后 6 Manipulator token |
| state register | `[B, S, dim]` | 状态条件 token |

packed action 的具体结构：

```text
token 0..5:   Base(t1..t6), 每个 token 只有维度 0..3 有效，4..20 补零
token 6..11:  Manipulator(t1..t6), 21 维全部按 mask 决定是否有效
```

这不是“12 个连续控制时刻”。它是“两条各长 6 的 typed plan stream”。
后续代码必须保留这个语义，否则很容易错误地把第 7 个 token 理解成更远的时间点。

## 3. `transform/mobile_plan_cotrain.py`

### 3.1 文件职责

该文件是 Mobile 双计划数据进入 DreamZero 的专用适配层，负责四件事：

1. 将多相机图像整理成 Wan 接收的单张 2×2 grid；
2. 将末端位姿 state 用离线统计量归一化；
3. 将 Base/Manipulator 两路 action 打包到统一 register 形状；
4. collate 后检查关键语义字段没有被通用 collator 丢失或变形。

### 3.2 `_numpy`

`_numpy(value)` 是一个小型边界适配器：

- 输入是 `torch.Tensor` 时，执行 `detach().cpu().numpy()`；
- 其他输入执行 `np.asarray`。

因此 transform 层统一在 NumPy 上工作，且不会把上游 tensor 的计算图带入
DataLoader worker。

### 3.3 视频 resize 和 2×2 grid

`_resize_video` 要求单个 camera stream 是 `THWC` RGB：

```text
[T, H, W, 3] → [T, 176, 320, 3]
```

实现先转成 `TCHW` float，用 bilinear interpolation resize，再转回
`THWC`。若原输入是整数类型，结果会 round、clip 到 `[0,255]`，最终返回
`uint8`。

`_canonicalize` 在没有统一 `video` 字段时读取：

- `video.head`；
- `video.wrist`；
- 一个与 head 同形状的全黑 view。

然后构造：

```python
result["video"] = np.stack([head, black, wrist], axis=1)
```

得到 `[T, 3, 176, 320, 3]`。父类 `DreamTransform._prepare_video` 对非 DROID
embodiment 使用以下布局：

```text
┌──────────────┬──────────────┐
│ view 0: head │ view 2: wrist│
├──────────────┼──────────────┤
│ view 1: black│ black        │
└──────────────┴──────────────┘
```

因此最终每个视频帧是 `352×640` 的 grid。这里必须先 resize 单个 view，
再拼 grid；若先拼再 resize，相机区域的采样比例和边界会改变。

父类随后把 grid 整理成模型侧 `images`。经 VLM processing 后，单样本通常
以 `(t*v)` 合并时间和 view；本专用路径拼完 grid 后只剩一个逻辑 view，
所以 batch 后的 `images` 可以按 `[B,T,H_grid,W_grid,C]` 被 action head 使用。

### 3.4 state 组合与 normalization

当没有统一 `state` 字段时，代码拼接：

```text
state.eef_position + state.eef_rotation_rpy
```

即把末端位置与 RPY 旋转沿最后一维合并。归一化统计从
`state_stats_path` 指向的 JSON 中读取 `observation.state`：

```text
x_norm = 2 * (x - q01) / (q99 - q01) - 1
```

随后 clip 到 `[-1,1]`。这是 robust range normalization：使用 1%/99%
分位数，而不是容易受异常值影响的绝对 min/max。

对于 `q99 == q01` 的恒定维度：

- `varying=False`；
- 输出保持初始化值 0；
- 避免除零。

需要注意，统计向量的维数和拼接后的 state 最后一维必须可广播匹配，当前函数
没有额外显式 shape 校验。

### 3.5 action packing

`_prepare_action` 先严格检查：

```text
base_action.shape        == [6, 4]
manipulator_action.shape == [6, 21]
```

然后把 Base 右侧补零到 21 维：

```text
packed_base       [6,21] = base 写入 [:,:4]，其余为 0
packed_base_mask  [6,21] = base_mask 写入 [:,:4]，其余为 False
```

最后沿 token 轴拼接：

```text
action      = concat(packed_base, manipulator)       → [12,21]
action_mask = concat(packed_base_mask, manip_mask)   → [12,21]
```

返回的 token 数是 12。构造函数同时强制父类配置
`action_horizon == 2 * plan_horizon`，即必须为 12。

补零不是为了让 Base 伪装成 21 维控制，而是为了让两类 token 能放入同一个
dense tensor。真正的有效维度由 mask、独立 encoder/decoder 和独立 loss
共同保证。

### 3.6 `apply_single` 与语义字段保留

执行顺序为：

1. `_canonicalize` 生成统一 video/state；
2. 调父类 `apply_single`，完成 grid、文本、state padding、action packing、
   embodiment id 等通用处理；
3. 把原始双路字段重新放回 transformed sample：
   `base_action`、`manipulator_action`、两路 mask、`plan_valid`、
   `plan_time_offsets`、`plan_time_seconds`。

父类训练路径实际使用 packed `action/action_mask`；额外保留的 typed 字段主要
用于 shape 校验、debug、离线分析和防止语义信息在 collate 时消失。

`apply` 明确只处理单样本，batching 交给 DataLoader collator，避免 transform
内部又做一层 batch 推断。

### 3.7 `MobilePlanDataCollator`

它先调用通用 `DefaultDataCollator`，完成 NumPy stack、Tensor 转换和文本
tokenization，然后严格验证：

```text
base_action             [B,6,4]
manipulator_action      [B,6,21]
base_action_mask        [B,6,4]
manipulator_action_mask [B,6,21]
plan_time_offsets       [B,6]
```

这里是一道“语义防线”：只要某个 dataset item 少了一维、时间轴被错误展平，
或 collator 改变了 key 的布局，就会在模型前立即报错。

## 4. `base_vla.py`

### 4.1 顶层职责

`VLA` 是组合器，而不是具体网络。Hydra 根据两个 config 实例化：

- `self.backbone`；
- `self.action_head`。

它规定统一调用协议：

```text
raw inputs
  → prepare_input
  → backbone(backbone_inputs)
  → action_head(backbone_outputs, action_inputs)
  → BatchFeature
```

### 4.2 `prepare_input`

`prepare_input` 是 raw batch 到两个子系统输入的唯一分叉点：

```python
backbone_inputs = self.backbone.prepare_input(inputs)
action_inputs   = self.action_head.prepare_input(inputs)
```

之后用 `tree.map_structure` 递归搬运所有叶子 tensor：

- 浮点 tensor：搬到 `self.device`，并 cast 到 `self.action_head.dtype`；
- 整数/bool tensor：只搬设备，不改 dtype。

虽然 config 中保存了 `compute_dtype`，当前具体 cast 使用的是
`action_head.dtype`，不是 `self.compute_dtype`。这保证 action head 内部 Wan
模块的 dtype 是实际准绳，但也意味着仅修改 `compute_dtype` 字符串不会直接
改变这里的 cast 结果。

### 4.3 `forward`

训练路径非常直接：

```python
backbone_inputs, action_inputs = self.prepare_input(inputs)
backbone_outputs = self.backbone(backbone_inputs)
action_head_outputs = self.action_head(backbone_outputs, action_inputs)
return action_head_outputs
```

当前 `WANPolicyHead.forward` 并不消费 `backbone_output` 的内容，主要从
`action_inputs` 取图像、文本、state/action；但保留该参数使 action head 接口
能与其他 VLA head 统一，也允许未来融合 `backbone_features`。

训练 `forward` 没有调用 `validate_data`，因此训练输出约束主要由 Trainer
读取 `outputs["loss"]` 和各具体 head 自己保证。

### 4.4 推理入口

`get_action`、`joint_video_action`、`lazy_joint_video_action` 及若干 causal/
efficient 变体都遵循相同编排：

```text
prepare_input → backbone → action_head 的对应推理方法 → validate_data
```

`get_language` 改用 `backbone.generate`；`get_video` 只使用 action-head 侧输入。

对 Mobile 双计划，`MobilePlanFlowMatchingActionHead.get_action` 特别覆盖了
默认行为，确保进入 joint video/action sampler，而不是误调用训练 forward。

### 4.5 输入和输出校验

`validate_inputs` 的主要约束：

- `action` 必须是三维 Tensor；
- action token 数需能被 `self.action_horizon` 整除；
- 最后一维等于 `self.action_dim`；
- `video` 若存在，要求 NumPy `uint8`、六维，并且第 4 维为 RGB channel。

`validate_data` 要求：

- backbone 输出是含 `backbone_features` 的 `BatchFeature`；
- 训练时 action head 输出可只含 `loss`；
- 推理时必须含 `[B, action_horizon, action_dim]` 的 `action_pred`。

双计划配置中，这意味着统一 `action_pred` 是 `[B,12,21]`，typed 输出
`base_plan_pred` 与 `manipulator_plan_pred` 是额外字段。

### 4.6 其他顶层能力

文件后半部分主要处理工程生命周期：

- 从 pretrained checkpoint 构造/加载；
- LoRA key 重写、延迟注入与单独加载；
- `post_initialize` 把初始化交给 action head；
- `parallelize` 把 device mesh 交给 action head；
- `CotrainVLA` 在 batch 标记 `cotrain` 时绕过常规 action head，
  直接调用 `backbone.cotrain(inputs)`。

这些逻辑不改变主数据流，但决定 checkpoint/LoRA 是否能正确恢复。

## 5. `action_head/mobile_plan_flow_matching.py`

### 5.1 为什么要继承 `WANPolicyHead`

Mobile head 复用 Wan 的全部视频编码、flow scheduler、训练 forward 和联合采样，
只覆盖双计划特有的协议：

- packed shape；
- physical offset 元数据；
- 两路 diffusion timestep 对齐；
- 单个 plan window 对单个 video block 的布局；
- 分路 masked loss；
- 推理结果拆分。

### 5.2 配置不变量

初始化时强制：

```text
action_horizon == 2 * plan_horizon == 12
action_dim     == manipulator_action_dim == 21
```

默认 offset 是 `(1,4,8,12,16,24)`，控制频率 30 Hz。Base 和 Manipulator
loss 权重分别由 `base_flow_loss_weight`、`manipulator_flow_loss_weight`
控制。

### 5.3 physical offset 协议

`prepare_action_model_kwargs` 读取 batch 中 `[B,6]` 的
`plan_time_offsets`，要求每个样本都与 config 中的预期 offset 完全相等。
验证通过后将它作为额外 kwarg 传给 DiT。

这形成两层检查：

1. action head 检查 batch/config 一致；
2. `WanVideoDiTDualPlan.forward` 再检查传入 DiT 的 offsets。

offset 目前不是动态可变输入：虽然 batch 显式携带它，但只接受固定的预设序列。
显式携带的价值是防止数据集和模型悄悄使用不同时间定义。

### 5.4 双路 timestep 协议

标准 Wan head 可以为每个 action token 独立或按 video block 构造 timestep。
Mobile head 的 `align_action_timestep_ids` 强制：

```text
t_action = [t_base(6), t_base(6)]
```

也就是第 `i` 个 Base token 与第 `i` 个 Manipulator token 使用同一个
diffusion timestep。

在 coupled 模式，`build_coupled_action_timestep_ids` 还要求整个 dual plan
只对应一个 video block，然后将该 block timestep 扩展到全部 12 个 token。

因此：

- 同一未来物理时刻的两路计划 diffusion time 对齐；
- coupled 模式下整个窗口的 12 个 token 甚至共享同一 block diffusion time；
- decoupled video/action noise 模式下，action timestep 可先独立随机采样，
  但两路之间仍会由 `align_action_timestep_ids` 成对对齐。

### 5.5 layout 协议

`validate_action_video_layout` 将一个 Mobile dual plan 定义为一个完整 register
block：

- `noise.shape[1] - 1 == num_frame_per_block`；
- action token 数等于 DiT 的 `num_action_per_block`，即 12；
- state token 数等于 `num_state_per_block`。

减 1 是因为 latent 序列第一个 frame 是 conditioning frame，剩余 future
latent frames 才组成当前预测 block。

### 5.6 分路 flow loss

`_masked_branch_loss` 对每个 token 做以下计算：

```text
squared_error = (prediction - target)^2
token_loss = sum(squared_error * mask) / max(active_dims, 1)
valid = (active_dims > 0) AND has_real_action
weighted = token_loss * scheduler.training_weight(timestep) * valid
branch_loss = sum(weighted) / max(number_of_valid_tokens, 1)
```

与基础 Wan loss 直接对 action dim 求 mean 相比，这里有两个重要改进：

- 只按真实有效维度归一化，Base 的 4 维不会被补出的 17 个零维“稀释”；
- 没有任何有效维度的 token 和 `has_real_action=False` 的样本不进入分母。

`compute_action_losses` 明确切片：

```text
Base:        tokens [:6], dims [:4]
Manipulator: tokens [6:], dims [:21]
```

总 action loss：

```text
action_loss =
    base_flow_loss_weight * base_flow_loss
  + manipulator_flow_loss_weight * manipulator_flow_loss
```

返回三个可日志化字段：`action_loss`、`base_flow_loss`、
`manipulator_flow_loss`。

### 5.7 推理拆分

`get_action` 调用父类 `lazy_joint_video_action` 完成联合采样，得到：

```text
packed action_pred [B,12,21]
```

然后添加：

```text
base_plan_pred        = action_pred[:, :6, :4]  → [B,6,4]
manipulator_plan_pred = action_pred[:, 6:]      → [B,6,21]
```

原 packed `action_pred` 仍保留，以满足 `VLA.validate_data` 的统一输出协议。

## 6. `action_head/wan_flow_matching_action_tf.py`

### 6.1 文件职责

这是 video/action 联合 flow-matching 的主控制器，位于数据/Trainer 与底层 DiT
之间，主要负责：

- 加载 T5、CLIP、VAE、Wan DiT 和可选 LoRA；
- 视频预处理、resize、VAE/CLIP/text encoding；
- 训练时采样 video/action timestep、加噪、调用 DiT、计算 loss；
- 推理时维护 conditional/unconditional context、KV cache 和两套 scheduler；
- 输出 `BatchFeature` 供 VLA/Trainer 使用。

### 6.2 训练前处理

`forward` 先将冻结模块切到 eval，然后从 `action_input` 取得：

```text
images, text, text_attention_mask,
state, action, action_mask, has_real_action, embodiment_id
```

视频从 `[B,T,H,W,C]` 变为 `[B,C,T,H,W]`。若是 `uint8`：

```text
uint8 [0,255] → float [0,1] → Normalize(0.5,0.5) → [-1,1]
```

对于 Wan 5B，若 config 未明确 target size 且 `frame_seqlen` 为 50/55，使用
`176×320`。这样 VAE latent 与 DiT 每帧 token 数匹配。随后：

- T5 编码文本；
- VAE 编码完整 video 为 `latents`；
- CLIP + VAE 编码首帧为 image condition `clip_feas`、`ys`。

### 6.3 video timestep 采样

代码有三种模式：

1. `decouple_video_action_noise=True`：
   video 使用 `video_beta_dist`，action 独立 uniform；
2. `use_high_noise_emphasis=True`：
   video 使用 `high_noise_beta_dist`，action 从 video block timestep 派生；
3. 默认：
   video timestep id uniform，action 与 video block coupled。

首 conditioning latent frame 保留自己的 timestep；未来帧按
`num_frame_per_block` reshape，同一 block 内后续帧 timestep 被改成 block
第一个 timestep。

Mobile override 会进一步要求这里只有一个 future video block。

### 6.4 加噪和 flow target

视频与 action 都走同一个 `FlowMatchScheduler` 接口：

```text
noisy = scheduler.add_noise(clean, noise, timestep)
target = scheduler.training_target(clean, noise, timestep)
```

视频在调用 scheduler 时临时 flatten `B,F`，action 临时 flatten `B,H`，
随后恢复原形状。

DiT 接收：

```text
noisy video latents
video timestep
CLIP condition / VAE image condition / text context
state register / embodiment id
noisy action / action timestep
clean_x=clean video latents
额外 action_model_kwargs（Mobile 中是 plan_time_offsets）
```

`clean_x` 触发底层 teacher forcing：同一 forward 同时含 clean context
视频 token 与 noisy target 视频/action/state token。

### 6.5 训练 loss

DiT 返回：

```text
video_noise_pred, action_noise_pred
```

视频 loss：

- 对 channel、height、width 求 MSE mean，保留 batch/frame；
- 乘 scheduler 的 timestep training weight；
- 再整体 mean 得到 `dynamics_loss`。

当 patch/unpatch 因奇数 latent 空间尺寸导致输出略小，代码会把 video target
裁到 prediction 的空间大小。

action loss 通过虚函数 `compute_action_losses`：

- 基础 Wan head：单路 masked MSE；
- Mobile head：Base/Manipulator 分路有效维度归一化。

最终：

```text
loss = dynamics_loss + action_loss
```

输出至少包含 `loss`、`dynamics_loss`、`action_loss`，Mobile 路径还包含
`base_flow_loss` 和 `manipulator_flow_loss`。

### 6.6 lazy joint sampling

`lazy_joint_video_action` 是主要推理入口。其过程可分为：

1. 视频/state 转 dtype 和目标分辨率；
2. 根据 language 是否变化、输入帧数和 local-attention window 决定是否重置
   `current_start_frame`；
3. 编码 positive/negative text，准备 classifier-free guidance；
4. 首次序列编码首帧并创建每层 self-attention KV cache 与 cross-attention
   cache；
5. 将已知 context latent 以 timestep 0 写入 KV cache；
6. 为 future video block 和 action 分别生成高斯噪声；
7. 建立 video 与 action 两套 `FlowUniPCMultistepScheduler`；
8. 每个 diffusion step 联合预测 video flow/action flow并分别 step；
9. 返回 action latent 和 video latent。

每一步构造：

```text
video timestep  [B, num_frame_per_block]
action timestep [B, action_horizon]
```

conditional/unconditional DiT 输出中：

- video 使用 CFG：
  `uncond + cfg_scale * (cond - uncond)`；
- action 直接使用 conditional action flow，没有做同样的 CFG 混合。

`dit_step_mask` 和 `should_run_model` 可跳过部分 DiT step，并复用/外推缓存的
flow prediction，减少实际 DiT 计算次数。

### 6.7 推理时 video/action 解耦

当 `decouple_inference_noise=True`：

- video scheduler 的 sigma 轨迹被重新映射，使最终仍停在
  `video_inference_final_noise`；
- action scheduler 保持标准完整去噪到 0。

所以模型可以把视频保留在较高噪声状态作为辅助 latent dynamics，同时仍输出
完全去噪的 action。这两套 scheduler 在同一个循环中使用各自 timestep 和
step，不应混用。

最终返回：

```python
BatchFeature({
    "action_pred": latents_action,
    "video_pred": output.transpose(1, 2),
})
```

## 7. `modules/wan_video_dit_dual_plan.py`

### 7.1 `PlanOffsetEmbedding`

它把“物理未来秒数”而不是 token ordinal 编入 hidden：

```text
seconds → sinusoidal positional encoding → Linear → SiLU → Linear
```

使用秒数意味着若控制频率或 offset 改变，embedding 表达的仍是实际未来时间，
而不是“第几个 plan token”。

### 7.2 `DualPlanActionEncoder`

输入仍是 packed `[B,12,21]`，但立即按语义拆开：

```text
base        = [:,:6,:4]
manipulator = [:,6:,:]
```

两路分别通过独立 `MultiEmbodimentActionEncoder`。每个 encoder 内部将：

- action value 通过 category-specific linear；
- diffusion timestep 做 sinusoidal encoding；
- 两者 concat 后再经两层 category-specific projection。

之后每条支路都加：

```text
该未来时刻的 physical offset embedding
+ 该支路的 type embedding
```

最后再拼回 `[Base tokens ; Manipulator tokens]`。因此模型能够区分：

- 同一物理未来时刻；
- token 属于 Base 还是 Manipulator；
- 当前 action diffusion timestep；
- action 数值本身。

### 7.3 `DualPlanActionDecoder`

Transformer hidden 也按前 6/后 6 分路，用两个独立
`CategorySpecificMLP` 解码：

- Base decoder 输出 4 维；
- Manipulator decoder 输出 21 维。

为恢复统一 packed 输出，Base 右侧 pad 17 个零，再与 Manipulator 拼成
`[B,12,21]`。因此 Base 的无效维度不是由共享 decoder 学出来的，而是 decoder
之后确定性补零。

### 7.4 `WanVideoDiTDualPlan`

它继承 `CausalWanModel`，但在构造时覆盖：

```text
action_dim = 21
num_action_per_block = 12
action_encoder = DualPlanActionEncoder
action_decoder = DualPlanActionDecoder
```

当前 embodiment category 数被固定为 1，底层代码也会把 embodiment id 重置为
0，所以 category-specific 层在这个实现里只有一个实际类别。

`forward` 强制显式提供 `plan_time_offsets` 并与持久化 buffer 完全一致，然后才
调用父类。值得注意的是，offset 的实际 embedding 使用构造时保存的
`offset_seconds`；forward 参数承担的是一致性证明，而不是运行时动态改变
embedding。

## 8. `modules/wan_video_dit_action_casual_chunk.py`

### 8.1 文件职责

这是联合 video/action/state DiT 的主体，包含：

- embodiment-specific action/state encoder/decoder；
- video/action/state 的 RoPE；
- teacher-forcing 与 KV-cache 两套 self-attention 路径；
- blockwise causal attention；
- Wan Transformer blocks；
- train/inference 两套 forward。

文件名中的 `casual` 是现有拼写，逻辑含义是 `causal`。

### 8.2 action/state register

`CausalWanModel` 将视频 latent 通过 3D Conv patch embedding 后变成 video
token。若提供 action：

```text
action_features = action_encoder(action, timestep_action)
state_features  = state_encoder(state)
action_register = concat(action_features, state_features)
x = concat(video_tokens, action_register)
```

对应时间 embedding 也按同样顺序拼：

```text
[video timestep per patch token ;
 action timestep per action token ;
 subsampled action timestep per state token]
```

state timestep 通过按 stride 从 action timestep 中取样获得，所以每个 state
token 与相应 action block 的 diffusion time 对齐。

Transformer blocks 完成后：

- video token 送进 `CausalHead` 并 unpatchify 为 video flow；
- action token 切出来送进 `action_decoder` 为 action flow；
- state token 只作为条件，不单独解码。

### 8.3 RoPE

视频使用三维频率（时间、高、宽），action/state 使用各自的一维 RoPE
`freqs_action`、`freqs_state`。推理带 KV cache 时根据
`current_start_frame` 计算 `action_state_index`，只取当前 block 对应的
action/state 频率 slice。

这样 action/state register 虽然追加在 video sequence 后面，仍有独立、稳定
的序列位置定义，不会误用视频空间位置。

TensorRT 路径使用显式 sin/cos 实数运算；普通路径使用 complex polar
乘法，数学作用相同。

### 8.4 训练 teacher forcing 序列

`WANPolicyHead.forward` 传入 `clean_x` 后，`_forward_train` 构造：

```text
[clean video tokens]
[noisy video tokens]
[noisy action tokens]
[state tokens]
```

clean video 使用 `aug_t`（默认全 0）的时间 embedding；noisy video/action/
state 使用各自 diffusion time。

每个 attention block 都收到 `is_tf=True`，并把 Q/K/V 拆成 clean 与 noisy
部分。处理规则为：

- clean image：只做 image-only blockwise causal attention；
- noisy image block：可看 clean 历史图像、当前 noisy image、当前 action、
  当前 state；
- noisy action block：可看 clean 历史图像、当前 noisy image、同 block
  action、同 block state；
- state block：只在自己的 state block 内 self-attend。

同一 block 内 video 与 action 可以双向耦合，但看不到未来 block；这既允许
联合建模当前计划窗口，又防止 future leakage。

所有 Transformer block 后会丢掉 clean half 的 hidden，只对 noisy video 和
action 产生训练预测。

### 8.5 block layout 与 attention mask

非 teacher-forcing 的完整布局写成：

```text
[first conditioning image]
[future image block 0 ... block N-1]
[action block 0 ... block N-1]
[state block 0 ... block N-1]
```

要求三类 block 数一致：

```text
(num_frames - 1) / num_frame_per_block
  == action_horizon / num_action_per_block
  == state_horizon / num_state_per_block
```

允许关系可总结为：

| Query | 可见 Key |
|---|---|
| first image | 自身 |
| image block `i` | first image、image `≤i`、action `i`、state `i` |
| action block `i` | first image、image `≤i`、action `i`、state `i` |
| state block `i` | 自身 state block |

实际 teacher-forcing 路径主要使用分块 Flash Attention helper，而文件中
`_prepare_blockwise_causal_attn_mask` 的 FlexAttention `BlockMask` 还承担
规则的显式表达与 debug/可视化用途。

布局检查会直接比较实际 sequence length 与
`image + action + state` 的理论长度。Wan 5B 路径特别提示使用 `320×176`，
因为该分辨率与 latent `frame_seqlen=55` 的 register 布局匹配。

### 8.6 Transformer block

每个 `CausalWanAttentionBlock` 大体执行：

1. time-conditioned self-attention；
2. text/CLIP cross-attention；
3. time-conditioned FFN；
4. residual connection 和相应 norm/modulation。

训练时默认开启 gradient checkpointing。block 的统一返回值是
`(hidden_states, updated_kv_cache)`；训练契约要求 cache 必须为 `None`，
`_training_block_hidden_states` 会显式检查，避免训练分支意外写 KV cache。

### 8.7 推理与 KV cache

当 `forward` 收到 `kv_cache` kwarg 时路由到 `_forward_inference`，否则路由
到 `_forward_train`。

推理时：

- 新视频 K/V 追加到每层历史 cache；
- cache 超过 local attention 最大 token 数后从左侧裁剪；
- action/state register 参与当前 attention，但不被永久追加为视频历史 cache；
- 若当前 step 同时有视频和 action，query 可以联合读取历史 video cache、
  当前 video 和当前 action/state。

这使 autoregressive video block 能复用历史计算，同时 action 始终对应当前
待生成 block。

## 9. `experiment/base.py`

### 9.1 `BaseTrainer` 的模型接口

DreamZero 模型要求完整 batch dict 作为一个 positional 参数：

```python
outputs = model(inputs)
```

而不是 Hugging Face 常见的：

```python
outputs = model(**inputs)
```

`compute_loss` 和自定义 `prediction_step` 都显式遵循前一种接口。这是验证能否
工作的关键；若使用默认无 label prediction path，mapping 会被展开，VLA
签名不匹配。

### 9.2 loss logging

`compute_loss` 遍历所有输出 key：

- key 以 `_loss` 结尾；
- 且 key 不等于顶层 `loss`。

例如：

```text
dynamics_loss
action_loss
base_flow_loss
manipulator_flow_loss
```

每种 loss 各维护最近 10 次的 Python list，并在
`current_step % 10 == 0` 时调用：

```text
self.log({"<key>_avg": avg_loss})
```

`LossLoggerCallback` 只在 world process zero 把以下字段追加到
`loss_log.jsonl`：

- `loss`；
- `learning_rate`；
- 以 `_loss_avg` 结尾的滑动平均。

所以日志是 JSONL，每行带 `step`，适合训练后直接用 pandas/脚本分析。

需要理解两个细节：

- `current_step` 在 `training_step` 结束后才加 1，因此第一次 forward
  处于 step 0，也满足 `%10==0`；
- gradient accumulation 下 `compute_loss` 的调用单位可能是 micro-batch，
  因而 queue 更接近“最近 10 次 forward”，不必然是最近 10 个 optimizer
  update。

### 9.3 validation positional batch 调用

`prediction_step`：

1. `_prepare_inputs(inputs)` 搬设备；
2. `torch.no_grad()`；
3. `compute_loss_context_manager()`；
4. `outputs = model(inputs)`；
5. 从 mapping 的 `outputs["loss"]` 或 tuple 的第 0 项取 loss；
6. 返回 `(loss.mean().detach(), None, None)`。

训练期 validation 当前只报告 loss，不 gather action/video prediction，避免在
多卡间搬运巨大视频张量。它也没有调用 `compute_loss`，所以 validation loss
不会混入训练的辅助 loss 滑动队列。

### 9.4 保存与验证顺序

上游 Hugging Face Trainer 在同一步通常先 evaluate 再 save。这里重写
`_maybe_log_save_evaluate`：

```text
若 should_save AND should_evaluate AND save_strategy != BEST：
    先 _save_checkpoint
    再触发 on_save callbacks
    清掉 should_save，防止父类重复保存
然后调用父类，让它继续 log/evaluate
```

效果是：昂贵验证即使失败，该 step 的参数也已经持久化。

`SaveStrategy.BEST` 被排除，因为是否保存 best 必须先拿到新 validation
metric，不能预先决定。

checkpoint 保存完成后，`CheckpointFormatCallback.on_save` 再把：

- `experiment_cfg/`；
- processor 目录（若配置）；
- `wandb_config.json`

复制到 `checkpoint-{step}`，使 checkpoint 更接近可独立恢复的目录。

整个 experiment 训练结束时的顺序是：

```text
trainer.train(...)
→ trainer.save_state()
→ safe_save_model_for_hf_trainer(..., final output_dir)
```

即先保存 Trainer state，再做最终模型安全保存。

### 9.5 其他 Trainer 行为

- 使用 `BaseSampler` 控制 train shuffle 与 eval non-shuffle；
- 对特定 `ShardedLeRobotMixtureDataset` 自建 DataLoader，并在 resume 时按
  global step 改 seed、设置 `ignore_data_skip=True`；
- optimizer 分 weight-decay/non-decay 两组；
- DeepSpeed CPU Adam 参数组补 `bias_correction=True`；
- 支持只保存 trainable/LoRA 参数，以及单独保存 LLM/value model；
- 可选 per-step profiler 和独立 `ProfCallback`。

## 10. 端到端训练时序

把七个文件合起来，一次训练 forward 的真实时序是：

1. dataset sample 进入 `MobilePlanCotrainTransform`；
2. head/wrist resize 后组成 2×2 grid；
3. EEF position/RPY 拼成 state，并用 q01/q99 归一化；
4. Base `[6,4]` 补零为 `[6,21]`，与 Manipulator `[6,21]` 拼成
   action `[12,21]`；
5. collator stack 成 batch，并验证 typed plan shape；
6. `BaseTrainer.compute_loss` 调用 `model(inputs)`；
7. `VLA.prepare_input` 分别调用 backbone/action-head prepare_input，递归搬
   device/dtype；
8. backbone 先运行，输出 `BatchFeature`；
9. Mobile action head 验证固定 plan offsets，并进入 Wan policy forward；
10. T5/CLIP/VAE 构造文本、首帧与视频 latent 条件；
11. video/action 分别采 timestep 和噪声，Mobile head 对齐两路 action
    timestep；
12. dual-plan DiT 将 noisy video、Base、Manipulator、state 放进联合
    teacher-forcing Transformer；
13. DiT 输出 video flow 与 packed action flow；
14. policy head 计算 dynamics loss；Mobile head 分别计算 Base/Manipulator
    masked flow loss；
15. 总 loss 反传；Trainer 维护辅助 loss 的 10 次滑动平均；
16. 若同一步触发 checkpoint 和 validation，先保存 checkpoint，再以
    `model(inputs)` 方式验证。

## 11. 最重要的不变量与排错清单

### 数据层

- 单路相机输入必须是 `THWC` RGB；
- Base 必须严格 `[6,4]`，Manipulator 必须严格 `[6,21]`；
- `action_horizon=12`，`action_dim=21`；
- `plan_time_offsets` 每个样本必须严格等于 `[1,4,8,12,16,24]`；
- state stats 的维数必须匹配 EEF position + RPY；
- action/state 值应已在模型要求的 `[-1,1]` 范围内。

### video/register layout

- Mobile dual plan 对应且只对应一个 future video block；
- `num_action_per_block=12`；
- state token 数必须等于 `num_state_per_block`；
- 分辨率、VAE downsample、patch stride 和 `frame_seqlen` 必须一致；
- conditioning 首帧不计入 future block 数。

### 双路语义

- packed token 轴是 `[6 Base ; 6 Manipulator]`，不是 12 个连续未来时刻；
- Base 有效 dim 只有 4，补零维必须 mask 掉；
- 两路相同 future offset 的 diffusion timestep 必须一致；
- encoder、decoder 和 loss 都要按两路拆分，不能只在输出末端拆。

### Trainer

- 模型调用必须是 `model(inputs)`；
- validation 只返回 loss，不返回/gather video/action；
- `BEST` 保存策略仍然必须 evaluate 后再决定保存；
- `loss_log.jsonl` 的辅助项是最近 10 次 forward 的平均，不一定等价于最近
  10 个 optimizer step。

## 12. 当前实现中值得留意的细节

以下不是必然的 bug，但阅读或修改时应特别留意：

1. `VLA.validate_inputs` 对训练 action token 轴检查的是
   `shape[1] % action_horizon == 0`，推理输出校验则要求严格等于
   `action_horizon`；二者宽松程度不同。
2. `VLA.forward` 训练路径没有调用 `validate_data`，新增 head 时要自行保证
   输出至少有标量 `loss`。
3. `encode_prompt` 的 padding 清零循环使用 `prompt_emb[:, v:] = 0`，
   看起来会对整个 batch 使用当前样本的长度切片；batch 内文本长度不同时应
   额外确认这是否符合预期。
4. Mobile offset 在 forward 时必须显式传递，但实际 embedding 来自构造时的
   buffer；当前协议支持“验证固定 offset”，不是每样本动态 offset。
5. Base loss 正确按 4 个有效维度归一化；如果退回基础
   `WANPolicyHead.compute_action_losses`，补零/mask 维度的 mean 语义会改变。
6. `max_num_embodiments` 在 `CausalWanModel` 内被实际重置为 1，且 forward
   中 embodiment id 被改成 0；当前双计划实现并未真正使用多 embodiment
   category 参数。

## 13. 文件版本校验

本文阅读时七个文件的 SHA-256：

```text
355911c2b17f46c85ae2764c51f74f64111b3b69867e86778d6baeec77343643  mobile_plan_cotrain.py
8d75c380a3e833b47c9a3de86ae33546013823bc3e7ca13b00f8890dfd7656ed  base_vla.py
f2be744f3f2b62d44d2a57842f10750d4afbb830233b5807e9a7da5b4dbcb0fb  mobile_plan_flow_matching.py
c84f387af1176cbfe35484f0dade09d8d8b46a143c44bfeb88afe369151e5e38  wan_flow_matching_action_tf.py
00f647b6e24e45d124aa3013cfd12094b6d8732dc2d4c2e2d0ae9aa82b4b5011  wan_video_dit_dual_plan.py
1d4abc776acc086abaa5d33870999518cccbbcc4343b98f64d3fde2705fd1143  wan_video_dit_action_casual_chunk.py
23050aa1a1a99742522033316dd9752cd0508a94a159aef54ea5e36e369cf414  experiment/base.py
```

若远端代码继续修改，可用这些 hash 判断本文与代码是否仍完全对应。
