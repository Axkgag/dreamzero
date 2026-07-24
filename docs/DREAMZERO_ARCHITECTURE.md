# DreamZero Network Architecture

本文档用一张简化计算图说明 DreamZero 的整体网络结构。这里的 DreamZero 不是一个单纯的图像到视频模型，而是一个 **视频预测 + 动作预测** 的联合模型：

- 视频分支：在 Wan VAE latent 空间里做 flow matching / diffusion，预测未来视频 latent。
- 动作分支：把机器人 action 作为 register token 放进同一个 DiT，预测 action flow。
- 条件分支：语言指令和首帧图像通过 cross-attention 注入 DiT。

## 最简计算图

```text
输入条件
  语言指令
    -> Wan Text Encoder
    -> 文本条件 tokens
    -> Cross-Attention context

  参考/首帧图像
    -> CLIP Image Encoder
    -> 图像条件 tokens
    -> Cross-Attention context

  参考/首帧图像
    -> Wan VAE Encoder
    -> 首帧 latent 条件 y

扩散主路
  视频帧序列
    -> Wan VAE Encoder
    -> 干净视频 latent x0
    -> 加随机视频噪声
    -> noisy video latent
    -> 3D Patch Embedding
    -> video tokens

机器人控制分支
  当前机器人状态
    -> State Encoder
    -> state register tokens

  训练: GT action / 推理: action noise
    -> Action Encoder
    -> action register tokens

时间条件
  扩散时间步 t
    -> Time Embedding
    -> DiT 中的 AdaLN / modulation

核心网络
  video tokens
  action register tokens
  state register tokens
  Cross-Attention context = 图像条件 tokens + 文本条件 tokens
  首帧 latent 条件 y 的处理方式
    - Wan2.1/14B: y 可 concat 到 video latent 输入
    - Wan2.2/5B: 当前配置不 concat 到输入，首帧主要走 CLIP cross-attn
  时间条件
    -> Causal Wan DiT

输出
  Causal Wan DiT
    -> Video Head + Unpatchify
    -> video flow/noise pred
    -> Scheduler 去噪
    -> 去噪后视频 latent
    -> Wan VAE Decoder
    -> 预测/生成视频

  Causal Wan DiT
    -> Action Decoder
    -> action flow/noise pred
    -> Action Scheduler 去噪
    -> 预测动作 action_pred
```

## Mermaid 版本

如果 Markdown 查看器支持 Mermaid，可以看下面这张更结构化的图：

```mermaid
flowchart TD
  Text[语言指令] --> TextEnc[Wan Text Encoder]
  TextEnc --> TextTok[文本条件 Tokens]

  Ref[参考/首帧图像] --> CLIP[CLIP Image Encoder]
  CLIP --> ImgTok[图像条件 Tokens]
  Ref --> RefVAE[Wan VAE Encoder]
  RefVAE --> Y[首帧 latent 条件 y]

  Video[视频帧序列] --> VAE[Wan VAE Encoder]
  VAE --> X0[干净视频 latent x0]
  Noise[随机视频噪声] --> AddNoise[加噪 / Flow Matching]
  X0 --> AddNoise
  AddNoise --> Xn[noisy video latent]

  State[机器人状态 state] --> StateEnc[State Encoder]
  StateEnc --> StateTok[state register]

  Action[训练: GT action<br/>推理: action noise] --> ActEnc[Action Encoder]
  ActEnc --> ActTok[action register]

  Step[扩散时间步 t] --> TimeEmb[Time Embedding]

  TextTok --> Context[Cross-Attention Context]
  ImgTok --> Context

  Xn --> Patch[3D Patch Embedding]
  Y -.Wan2.1/14B 可 concat 到输入.-> Patch
  Y -.Wan2.2/5B 当前配置不进入 patch embedding.-> YNote[首帧主要通过 CLIP cross-attn 注入]

  Patch --> DiT[Causal Wan DiT Blocks]
  StateTok --> DiT
  ActTok --> DiT
  TimeEmb --> DiT
  Context --> DiT

  DiT --> VideoHead[Video Head + Unpatchify]
  DiT --> ActionHead[Action Decoder]

  VideoHead --> VPred[video flow/noise pred]
  ActionHead --> APred[action flow/noise pred]

  VPred --> Sched[Scheduler 去噪]
  Sched --> Z[去噪后视频 latent]
  Z --> Dec[Wan VAE Decoder]
  Dec --> OutVideo[预测/生成视频]

  APred --> ASched[Action Scheduler 去噪]
  ASched --> OutAction[预测动作 action_pred]
```

## 各模块做什么

### 1. VAE：把视频压到 latent 空间

DreamZero 使用 Wan 的视频 VAE。输入图像/视频先被 resize、归一化，然后编码成视频 latent。后续 DiT 不直接在 RGB 像素上扩散，而是在 latent 上预测 flow/noise。

当前 Wan2.2/5B 配置中使用 `WanVideoVAE38`，latent 通道数是 48，空间下采样约为 16 倍。例如 160x320 的输入会变成 10x20 的 latent；DiT 的 patch embedding 再按 `(1, 2, 2)` 切 patch，所以每帧 token 数是 `(10/2) * (20/2) = 50`。

### 2. 首帧图像条件：两条路径

首帧/参考图像有两条用途：

- 旁路：送入 CLIP image encoder，得到图像语义 tokens。
- 主路条件：送入 VAE，得到首帧 latent 条件 `y`，里面还包含一个 mask，标记第一帧是条件帧。

需要注意：不同 Wan backbone 处理 `y` 的方式不同。

- Wan2.1/14B I2V 风格：可以把 noisy latent `x` 和首帧条件 `y` 在通道维 concat 后再 patch embedding。
- Wan2.2/5B 当前配置：`concat_first_frame_latent: false`，也就是不做 `[x; y]` 通道拼接；latent 输入保持 48 通道，首帧更多通过 CLIP/I2V 条件进入 DiT。

### 3. 文本和图像条件：通过 Cross-Attention 注入

语言指令先经过 Wan text encoder，得到 text tokens。首帧经过 CLIP image encoder，得到 image tokens。

DiT 内部会把它们拼成：

```text
context = image tokens + text tokens
```

每个 DiT block 都有 cross-attention，视频/action/state tokens 会通过 cross-attention 读取这些条件信息。

### 4. Action/State register：把机器人控制问题并入 DiT

DreamZero 的关键改动是：它不只预测视频，还把机器人状态和动作也变成 transformer token。

```text
state  ─► State Encoder  ─► state register
action ─► Action Encoder ─► action register
```

这些 register 会和视频 tokens 放进同一个 Causal Wan DiT 中。DiT 的 self-attention mask 规定了它们如何互相看见：

- 视频 token 可以看历史视频、当前动作、当前状态。
- 动作 token 可以看相关视频上下文、当前状态和自身动作块。
- 状态 token 主要作为条件 token。

因此，动作预测不是外接一个简单 MLP，而是和视频预测在同一个时空 DiT 里联合建模。

### 5. DiT block 内部结构

每个 Causal Wan DiT block 大致是：

```text
输入 tokens
   │
   ├─ AdaLN / timestep modulation
   ▼
Causal Self-Attention
   │
   ▼
Cross-Attention(context = CLIP image tokens + text tokens)
   │
   ▼
FFN
   │
   ▼
输出 tokens
```

其中 timestep embedding 会调制每层的归一化和残差分支；视频 token 使用 3D RoPE，action/state register 使用单独的位置编码。

### 6. 位置编码设计

DreamZero 的 Causal Wan DiT 里有三类 token：视频 token、action register token、state register token。它们不会共用一套简单的绝对位置编码，而是分别使用 RoPE 形式的位置编码。

视频 token 使用 3D RoPE，对应 latent patch grid 的三个轴：

```text
video token position = (frame_index, latent_h_index, latent_w_index)

3D RoPE =
  temporal RoPE(frame_index)
  + height RoPE(latent_h_index)
  + width RoPE(latent_w_index)
```

代码里对应三组 `self.freqs`：

```text
self.freqs[0]: 时间轴 frame
self.freqs[1]: latent height
self.freqs[2]: latent width
```

每次进入 DiT 前，会根据 patch embedding 后的 grid size 生成当前视频 token 的 RoPE：

```text
grid_size = (F_patch, H_patch, W_patch)
freqs = create_3d_rope(grid_size, start_frame)
```

其中 `start_frame` 很重要：

- 训练时通常从 `start_frame = 0` 开始，因为一次 forward 里能看到完整训练片段。
- 自回归推理时会传入 `current_start_frame`，让当前 block 的时间位置接在 KV cache 中已有历史帧后面。

action 和 state register 使用单独的一维 RoPE：

```text
self.freqs_action: action token 的 1D RoPE
self.freqs_state:  state token 的 1D RoPE
```

当 DiT 输入中包含 action/state register 时，代码会把视频 RoPE 和 action/state RoPE 拼到一起，使 self-attention 里的 Q/K 都带有位置：

```text
video tokens:
  使用 3D RoPE(frame, h, w)

action register tokens:
  使用 action 1D RoPE(action_index)

state register tokens:
  使用 state 1D RoPE(state_index)
```

训练和推理中 action/state 的 RoPE 使用方式略有差异：

- 训练 teacher-forcing：clean video tokens 和 noisy video/action/state tokens 一起进入 DiT。clean/noisy 视频部分复用同一段视频 RoPE；noisy action/state register 使用对应的一维 RoPE。
- 推理 KV cache：每次只处理当前生成 block，action/state RoPE 会根据 `current_start_frame` 推出当前 action/state block 的 index，保证它和当前视频 block 对齐。

所以位置编码的核心设计是：

```text
视频 token:
  用 3D 时空位置，表达“第几帧、latent 图上的哪个 patch”

动作/状态 token:
  用 1D register 位置，表达“当前 block 中第几个 action/state token”

自回归推理:
  用 current_start_frame 把新 block 的时间位置接到历史 KV cache 后面
```

## 训练时的数据流

训练时有真实视频和真实动作：

```text
真实视频 -> VAE -> clean latent x0
随机噪声 + x0 -> noisy video latent

真实动作 -> 加噪 -> noisy action
机器人状态 -> state register

noisy video latent + noisy action + state + 文本/图像条件 -> DiT

DiT 输出：
  1. video flow/noise pred
  2. action flow/noise pred

loss = 视频 latent flow loss + action flow loss
```

代码中训练还会把 `clean_x` 作为 teacher-forcing 上下文拼进去，让 noisy 部分在因果注意力下参考干净历史视频 token。

## 推理时的数据流

推理时没有未来视频和真实动作：

```text
首帧/历史观测 -> VAE/CLIP -> 条件和 KV cache
随机视频噪声 -> noisy video latent
随机动作噪声 -> noisy action

for 每个扩散步:
    DiT(noisy video, noisy action, state, text, image condition, KV cache)
      -> video flow pred
      -> action flow pred
    scheduler 更新 video latent
    scheduler 更新 action

最终：
    action_pred 直接作为动作输出
    video latent 可送入 VAE Decoder 得到预测视频
```

为了流式/自回归推理，DreamZero 会维护 KV cache。第一帧或历史 latent 先写入 cache，后续只对当前 block 做去噪预测。

## 一句话总结

DreamZero 的主体是一个改造过的 Wan 视频 DiT：它在 latent 视频扩散的基础上，把语言、首帧图像、机器人状态和动作 token 都放进同一个 transformer 计算图里，通过 cross-attention 注入语义条件，通过 causal self-attention 联合建模视频和动作，最后同时输出预测视频 latent 和机器人动作。
