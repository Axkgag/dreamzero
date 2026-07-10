# `wan_video_dit_action_casual_chunk.py` 代码解析

本文档解析
`groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`
这个文件的主要结构、关键类/函数职责，以及训练和推理时的简化执行流程。

这个文件实现的是 DreamZero 中经过改造的 causal WAN DiT。它在原本的视频
diffusion transformer 中插入 robot action token 和 state token，使模型能够同时预测：

- video latent noise：用于 dynamics / video denoising 目标
- action noise：用于 robot policy denoising 目标

核心序列可以理解为：

```text
video tokens + action tokens + state tokens
```

训练时使用 teacher forcing，实际进入 self-attention 的序列更接近：

```text
[clean video tokens] [noisy video tokens] [noisy action tokens] [state tokens]
```

其中 clean video tokens 提供历史上下文，noisy video/action tokens 是当前要去噪预测的目标，state tokens 是条件信息。

## 整体结构

文件中的主要组件是：

```text
CategorySpecificLinear
CategorySpecificMLP
MultiEmbodimentActionEncoder
causal_rope_action_apply / no_polar / polar
CausalWanSelfAttention
CausalWanAttentionBlock
CausalHead
CausalWanModel
```

其中最核心的是：

- `class CausalWanSelfAttention`：实现带 action/state token 的 causal attention 规则
- `class CausalWanAttentionBlock`：一个完整的 WAN transformer block
- `class CausalWanModel`：顶层 DiT backbone，负责训练/推理路径调度

## `class CategorySpecificLinear`

这是一个按 category/embodiment 选择参数的线性层。

输入形状：

```text
x:       [B, T, input_dim]
cat_ids: [B]
```

它为每个 category 保存一套独立的权重和 bias：

```text
W: [num_categories, input_dim, hidden_dim]
b: [num_categories, hidden_dim]
```

forward 时根据 `cat_ids` 为 batch 中每个样本选择对应的权重，然后用 `torch.bmm` 做投影。

在当前文件中，它主要服务于 state/action 的 category-specific MLP。不过需要注意，`CausalWanModel.__init__` 里当前把 `max_num_embodiments` 重置成了 `1`，所以默认实际上只有一个 category。

## `class CategorySpecificMLP`

两层 category-specific MLP，由两个 `CategorySpecificLinear` 组成。

主要用于：

- `state_encoder`：把 robot state 编码到 DiT hidden dimension
- `action_decoder`：把 DiT 输出的 action token 解码成 action noise prediction

简化结构：

```text
x
  -> CategorySpecificLinear
  -> ReLU
  -> CategorySpecificLinear
```

## `class MultiEmbodimentActionEncoder`

负责把 noisy action 编码成 transformer token。

输入：

```text
actions:   [B, action_horizon, action_dim]
timesteps: [B, action_horizon]
cat_ids:   [B]
```

输出：

```text
action_features: [B, action_horizon, dim]
```

它做了三件事：

1. 用 category-specific linear 投影 action
2. 给 action timestep 加 sinusoidal positional/timestep encoding
3. 通过 MLP 混合 action embedding 和 timestep embedding

输出的 `action_features` 后续会和 `state_features` 拼接，形成 action/state register。

## `def causal_rope_action_apply`

这是 action/state aware 的 RoPE 入口函数。

它根据环境变量选择不同实现：

```python
ENABLE_TENSORRT=true  -> causal_rope_action_apply_no_polar
否则                  -> causal_rope_action_apply_polar
```

作用是给当前 token 序列施加 rotary position embedding。

这里有两类位置编码：

- video token：使用 3D RoPE，对应 time / height / width
- action/state token：使用 1D RoPE，对应 action/state register 中的时间位置

## `def causal_rope_action_apply_no_polar`

非复数极坐标版本的 RoPE 实现，主要用于 TensorRT 兼容路径。

它把最后一维拆成 real/imag 两部分，显式使用 sin/cos 做旋转：

```text
x_real_rotated = x_real * cos - x_imag * sin
x_imag_rotated = x_real * sin + x_imag * cos
```

当存在 action/state register 时，它会根据 `action_state_index` 取出当前 block 对应的 action/state RoPE 频率，并拼到 video RoPE 频率后面。

## `def causal_rope_action_apply_polar`

复数极坐标版本的 RoPE 实现。

它先把 tensor reshape 成 complex view，然后直接做复数乘法完成旋转。

逻辑上和 `causal_rope_action_apply_no_polar` 等价，只是实现方式不同。

## `class CausalWanSelfAttention`

这是文件中最关键的 attention 实现。

它在 WAN self-attention 基础上加入了：

- clean/noisy teacher-forcing 训练结构
- video block 的 causal attention
- action/state register 的 attention 规则
- 推理时的 KV cache
- action/state RoPE

### `def CausalWanSelfAttention.__init__`

初始化 self-attention 所需的模块和布局参数。

主要创建：

- `q/k/v/o` projection
- Q/K RMSNorm
- 普通 attention：`self.attn`
- causal attention：`self.causal_attn`
- block/token 布局参数：
  - `frame_seqlen`
  - `num_frame_per_block`
  - `num_action_per_block`
  - `num_state_per_block`
  - `local_attn_size`

### `def CausalWanSelfAttention._visualize_attention_mask`

调试辅助函数，用于构造并打印 attention mask 的可视化矩阵。

它不是主训练路径的核心逻辑，主要用于检查 blockwise causal mask 是否符合预期。

### `def CausalWanSelfAttention._blockwise_causal_flash_attn`

用于非 teacher-forcing 路径的 blockwise causal attention。

它假设序列布局是：

```text
[first image] [image blocks] [action blocks] [state blocks]
```

主要规则：

- first image 是 conditioning frame
- image block `i` 可以看 first image、过去/当前 image blocks、当前 action block、当前 state block
- action block `i` 可以看 first image、过去/当前 image blocks、当前 action block、当前 state block
- state block 只作为条件，基本不向其它 state block 泄漏信息

这个函数主要在 `is_tf=False` 且没有 KV cache 的路径中使用。

### `def CausalWanSelfAttention._process_clean_image_only`

处理 teacher-forcing 训练中的 clean video half。

规则：

```text
first clean frame:
  self-attention only

后续 clean blocks:
  causal attention over first frame + previous/current clean tokens
```

如果 `local_attn_size == -1`，后续 clean blocks 可以用一次 causal attention 处理。

如果开启 local attention，则按 block 循环，每个 block 手动拼接可见上下文。

### `def CausalWanSelfAttention._process_noisy_image_blocks`

处理 teacher-forcing 训练中的 noisy video tokens。

对于 noisy video block `i`，可见上下文是：

```text
first clean frame
+ clean blocks before i
+ current noisy image block
+ current noisy action block
+ current state block
```

它不能看未来 clean block、未来 noisy image block、未来 action block 或未来 state block。

这对应训练时对 closed-loop 推理的近似：历史是 clean/observed，当前 chunk 是 noisy，需要去噪。

### `def CausalWanSelfAttention._process_noisy_action_blocks`

处理 teacher-forcing 训练中的 noisy action tokens。

对于 action block `i`，可见上下文是：

```text
first clean frame
+ clean blocks before i
+ current noisy image block
+ current noisy action block
+ current state block
```

因此 action block 只能利用历史视觉上下文和当前 block 的 noisy video/action/state，不会看到未来视频或未来动作。

### `def CausalWanSelfAttention._process_state_blocks`

处理 state tokens。

state block 基本只做本 block 内 self-attention：

```text
state block i -> only state block i
```

state 作为当前 noisy video/action block 的条件信息使用。

### `def CausalWanSelfAttention.forward`

这是 self-attention 的主入口，按是否 teacher forcing、是否有 KV cache 分成三条路径。

#### 路径 1：teacher-forcing 训练

条件：

```python
kv_cache is None and is_tf is True
```

输入序列结构：

```text
[clean video tokens] [noisy video tokens] [noisy action tokens] [state tokens]
```

执行流程：

```text
1. 计算 q/k/v
2. 根据 action_register_length 拆分 clean half 和 noisy half
3. 对 clean 和 noisy 区域分别施加 RoPE
4. 拆成 clean image / noisy image / noisy action / noisy state
5. 分别调用 _process_clean_image_only、_process_noisy_image_blocks、
   _process_noisy_action_blocks、_process_state_blocks
6. 把输出重新拼成同样顺序
```

#### 路径 2：非 teacher-forcing 的完整序列

条件：

```python
kv_cache is None and is_tf is False
```

执行流程：

```text
1. 对完整序列施加 RoPE
2. 计算 action/state horizon
3. 调用 _blockwise_causal_flash_attn
```

#### 路径 3：KV-cache 推理

条件：

```python
kv_cache is not None
```

执行流程：

```text
1. 根据 current_start_frame 施加 causal RoPE
2. 把 action/state register 从 video token 后面拆出来
3. 把当前 video K/V 追加到历史 KV cache
4. 当前 video/action/state token attend 到 cached history + 当前 token
5. 返回更新后的 KV cache
```

这条路径用于 closed-loop 推理。

## `class CausalWanAttentionBlock`

一个完整的 WAN transformer block。

它包含：

- timestep-conditioned modulation
- causal self-attention：`CausalWanSelfAttention`
- text/image cross-attention
- FFN

简化结构：

```text
x
  -> WanLayerNorm + timestep modulation
  -> CausalWanSelfAttention
  -> residual
  -> cross-attention(context)
  -> FFN
  -> residual
```

### `def CausalWanAttentionBlock.__init__`

初始化一个 transformer block。

主要创建：

- `norm1`
- `self_attn`
- `cross_attn`
- `norm2`
- `ffn`
- `modulation`

其中 `cross_attn` 的类型由 `model_type` 决定：

```python
t2v -> t2v_cross_attn
i2v/ti2v -> i2v_cross_attn
```

### `def CausalWanAttentionBlock.forward`

执行一个 transformer block。

输入包括：

- `x`：当前 token 序列
- `e`：timestep modulation
- `freqs` / `freqs_action` / `freqs_state`
- `action_register_length`
- `context`：text/image context
- 可选 `kv_cache`
- `current_start_frame`
- `is_tf`

输出：

```text
x, updated_kv_cache
```

## `class CausalHead`

DiT 的最终输出头。

它负责把 transformer hidden states 投影到 video latent patch 的输出维度。

### `def CausalHead.__init__`

创建：

- `WanLayerNorm`
- linear head
- timestep modulation 参数

输出维度是：

```text
prod(patch_size) * out_dim
```

### `def CausalHead.forward`

执行 timestep-conditioned output projection。

流程：

```text
x
  -> WanLayerNorm + timestep modulation
  -> Linear
```

输出还不是完整 video latent，而是 patch token 形式。后续会由 `unpatchify` 还原成 `[B, C, F, H, W]`。

## `class CausalWanModel`

顶层 diffusion backbone。

它是 DreamZero 中 `WANPolicyHead` 内部真正调用的 DiT 模型，支持：

- `t2v`
- `i2v`
- `ti2v`

### `def CausalWanModel.__init__`

初始化完整模型。

主要创建：

- `patch_embedding`：把 video latent patchify 成 token
- `text_embedding`：把文本 embedding 投影到 DiT hidden dimension
- `time_embedding` / `time_projection`：生成 timestep modulation
- `img_emb`：I2V/TI2V 图像条件投影
- `blocks`：多层 `CausalWanAttentionBlock`
- `head`：`CausalHead`
- `state_encoder`
- `action_encoder`
- `action_decoder`
- video/action/state RoPE 频率表

关键配置：

- `frame_seqlen`：每个 latent frame 经过 patch embedding 后的 token 数
- `num_frame_per_block`：每个 video block 包含多少帧
- `num_action_per_block`：每个 block 对应多少个 action token
- `num_state_per_block`：每个 block 对应多少个 state token
- `concat_first_frame_latent`：是否把 first-frame latent condition `y` 拼到输入 channel 上

### `def CausalWanModel._set_gradient_checkpointing`

设置是否启用 gradient checkpointing。

### `def CausalWanModel._prepare_blockwise_causal_attn_mask`

构造 blockwise causal mask 的静态辅助函数。

它描述的完整布局是：

```text
[first image] [image blocks] [action blocks] [state blocks]
```

当前主 teacher-forcing 路径更多使用 `_process_*` 手写分块 attention；这个函数更像是完整 mask 版本和调试/备用路径。

### `def CausalWanModel._prepare_teacher_forcing_mask`

构造 teacher-forcing 场景下 image-only 的 attention mask。

布局是：

```text
[clean frames] [noisy frames]
```

noisy frame 可以看对应的 clean history 和自身 noisy block。

### `def CausalWanModel._prepare_blockwise_causal_attn_mask_i2v`

构造 I2V 场景下的 blockwise causal mask。

它会单独处理 first frame，使 first frame 成为 image-to-video generation 的条件帧。

### `def CausalWanModel._forward_train`

训练 forward 路径。没有传 `kv_cache` 时会走这里。

输入包括：

```text
x:               noisy video latents
clean_x:         clean video latents
action:          noisy actions
state:           clean state
timestep:        video diffusion timestep
timestep_action: action diffusion timestep
context:         text embeddings
clip_feature:    image condition
y:               first-frame latent condition
```

简化执行流程：

```text
1. 如果需要，拼接 first-frame latent condition y
2. patchify noisy video latent
3. encode noisy action
4. encode state
5. 拼接 noisy video tokens + action/state register
6. 为 video/action/state token 构造 timestep embedding
7. 投影 text context 和 optional CLIP image context
8. patchify clean_x
9. 把 clean video tokens 拼到最前面：
   [clean video][noisy video][noisy action][state]
10. 逐层执行 CausalWanAttentionBlock，且 is_tf=True
11. 丢弃 clean prefix 的输出
12. action token slice -> action_decoder -> action_noise_pred
13. video token slice -> CausalHead -> unpatchify -> video_noise_pred
```

输出：

```text
video_noise_pred, action_noise_pred
```

### `def CausalWanModel._forward_inference`

带 KV cache 的推理路径。

输入是当前 noisy video block、当前 noisy action chunk、当前 state，以及历史 KV cache。

简化执行流程：

```text
1. 如果需要，拼接 first-frame latent condition y
2. patchify 当前 noisy video block
3. 根据 current_start_frame 创建 video RoPE
4. 调用 _forward_blocks
5. unpatchify video token 输出
6. 返回 video_noise_pred、action_noise_pred、updated_kv_caches
```

推理时历史上下文不是通过 `clean_x` 传入，而是通过 `kv_cache` 提供。

### `def CausalWanModel._forward_blocks`

推理式 block 执行的共享 helper。

它做的事情：

```text
1. flatten patchified video tokens
2. 如果有 action，encode action 和 state，并拼成 action/state register
3. 拼接 video tokens + action/state register
4. 构造 timestep modulation
5. 构造 text/image context
6. 逐层执行 CausalWanAttentionBlock，并传入 KV cache
7. 提取 action token slice，经过 action_decoder
8. 提取 video token slice，经过 CausalHead
```

输出：

```text
x_video, action_noise_pred, updated_kv_caches
```

### `def CausalWanModel._forward_inference_trt`

TensorRT 推理包装函数。

它从 packed KV cache 中恢复每层 cache，并根据 cache 长度估计 `current_start_frame`，然后调用 `_forward_inference`。

### `def CausalWanModel._forward_inference_trt_droid`

DROID/TensorRT 相关的推理包装函数，逻辑和 `_forward_inference_trt` 基本一致。

### `def CausalWanModel.forward`

顶层 dispatch 函数。

逻辑很简单：

```python
if kwargs.get("kv_cache", None) is not None:
    return self._forward_inference(...)
else:
    return self._forward_train(...)
```

也就是说：

- 没有 `kv_cache`：训练路径
- 有 `kv_cache`：推理路径

### `def CausalWanModel.unpatchify`

把 patch token 还原成 video latent tensor。

输入：

```text
[B, num_patches, patch_volume * out_dim]
```

输出：

```text
[B, out_dim, F, H, W]
```

### `def CausalWanModel._create_freqs`

创建 video token 的 3D RoPE 频率。

它组合：

```text
time freqs
height freqs
width freqs
```

推理时会使用 `start_frame=current_start_frame`，保证当前 block 的时间位置和 KV cache 中的历史位置连续。

### `def CausalWanModel.init_weights`

初始化模型权重，包括 linear、embedding、norm、conv 等模块。

## 训练流程简化版

完整训练流程从 `WANPolicyHead.forward` 开始，这里只描述进入 `CausalWanModel` 后的部分。

```text
WANPolicyHead.forward
  准备 noisy_latents、clean_latents、noisy_actions、state、text/image condition
    |
    v
CausalWanModel.forward
  没有 kv_cache -> _forward_train
    |
    v
patchify noisy video
encode noisy action
encode state
构造 action/state register
patchify clean video
拼接:
  [clean video][noisy video][noisy action][state]
    |
    v
逐层 CausalWanAttentionBlock
  CausalWanSelfAttention(is_tf=True)
    clean image:
      first frame self-attn，后续 clean blocks causal attention
    noisy image:
      attend to clean history + current noisy image/action/state block
    noisy action:
      attend to clean history + current noisy image/action/state block
    state:
      block 内 self-attn
    |
    v
cross-attention to text/image context
FFN
    |
    v
丢弃 clean prefix
解码:
  video tokens  -> video_noise_pred
  action tokens -> action_noise_pred
```

## 推理流程简化版

```text
CausalWanModel.forward
  有 kv_cache -> _forward_inference
    |
    v
patchify 当前 noisy video block
encode 当前 noisy action chunk
encode 当前 state
拼接 action/state register
根据 current_start_frame 创建 RoPE
    |
    v
逐层 CausalWanAttentionBlock
  CausalWanSelfAttention(kv_cache != None)
    当前 video K/V 追加进 cache
    当前 video/action/state token attend 到 cached history + 当前 token
    更新 cache
    |
    v
cross-attention to text/image context
FFN
    |
    v
解码:
  video tokens  -> video_noise_pred
  action tokens -> action_noise_pred
返回 updated KV caches
```

## Attention 规则总结

最重要的理解是：

```text
历史帧是 clean context
当前 chunk 是 noisy target
未来 chunk 不可见
```

训练时通过 teacher-forcing split 实现：

```text
[clean video][noisy video][noisy action][state]
```

推理时通过 KV cache 实现：

```text
cached history + current noisy video/action/state
```

对于 block `i`：

```text
noisy video/action block i 可以看:
  first clean frame
  clean video blocks before i
  noisy video block i
  noisy action block i
  state block i

noisy video/action block i 不能看:
  future clean blocks
  future noisy video blocks
  future action blocks
  future state blocks
```

## First Frame 的作用

first frame 是输入视频的第 0 帧，作为条件帧使用。

它的作用包括：

- 在 `WANPolicyHead` 外部被编码成 image condition，传入 `clip_feature`
- 在部分配置下通过 `y` 作为 first-frame latent condition
- 在 teacher-forcing 中作为 clean video context 的第一帧
- 为后续 video/action blocks 提供 causal 起点

action chunk 通常和 first frame 之后的 video block 对齐。

例如：

```text
frame 0:
  conditioning frame

frames 1..N:
  第一个 video block，对应 action chunk 0
```

## 实现注意点

- 文件名里是 `casual`，但实现语义是 `causal`。
- `max_num_embodiments` 参数虽然存在，但当前构造 encoder/decoder 前被重置为 `1`。
- teacher-forcing 训练路径主要使用 `_process_*` 这些手写 block attention helper，而不是单个全局 causal mask。
- 训练和推理的历史条件来源不同：
  - 训练：`clean_x` 拼到序列最前面
  - 推理：历史信息来自 KV cache
- `concat_first_frame_latent` 区分不同 WAN 配置：
  - Wan2.1 I2V 风格可能把 `y` 拼到 latent channel
  - Wan2.2 TI2V 风格通常只用 latent 输入，first frame 主要通过 CLIP/image context 条件化

