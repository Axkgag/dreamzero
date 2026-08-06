# MobileManiBench VGGT Tokenizer 优化计划

## 1. 目标与固定合同

本轮目标是先得到一个可冻结、可重建、可用于重新训练 WAM，并支持 chunk-online 推理的
VGGT 2D/3D tokenizer。暂不要求复用原 Wan-WAM 的 latent 数值语义，也不要求逐帧在线。

固定输入输出：

```text
video              [B,33,V,3,160,320]
camera_K           [B,33,V,3,3]
T_B0_camera        [B,33,V,4,4]
pseudo_pointmap_B0 [B,33,V,3,80,160]

z_2d [B,V,48,9,10,20]
z_3d [B,9,768,256]
```

本轮固定选择：

```text
在线方案                 A：完整四帧 chunk 到达后产生一个 latent
global attention         chunk 内双向，不加 causal attention mask
temporal compression     frame 0 + 两级 stride-2 causal downsample
temporal decoding        两级 causal upsample；2D 保留 PixelShuffle，3D 保留 metric grid
feature taps             [4,11,17,23]
per-layer projection     frame 1024->128，global 1024->128，直接 concat
2D fusion                four-level concat 1024->256->48
2D spatial resampler     learned 10x20 queries
3D levels                layer 11 fine + layer 23 coarse
geometry auxiliary loss  inside-grid masks
feature fusion gate      不使用 frame/global gate 或 layer gate
```

空间 RGB PixelShuffle decoder、PointMap ray decoder、video loss、KL 权重和 WAM 结构不在
本轮修改；2D/3D temporal decoder 与 encoder 同步升级。Tokenizer 完成后冻结，Stage 2
WAM 全程使用该 tokenizer 生成的 latent 重新训练。

## 2. 目标数据流

```text
RGB multi-view video
 -> frozen DINOv2 patch features
 -> VGGT frame/global aggregator
      main path: 1024-dimensional tokens
      taps: [4,11,17,23]
        each tap:
          frame 1024 -> 128
          global 1024 -> 128
          direct concat -> 256
             |
             +-> 2D
             |    four taps concat -> 1024
             |    -> residual bottleneck 1024 -> 256
             |    -> learned query resampler 12x23 -> 10x20
             |    -> residual projection 256 -> 48
             |    -> temporal modeling and 33 -> 9 compression
             |    -> z_2d [B,V,48,9,10,20]
             |    -> two-stage causal temporal upsample 9 -> 33
             |    -> existing spatial PixelShuffle -> RGB
             |
             +-> 3D
                  layer 11 fine + layer 23 coarse
                  -> camera-aware deformable sampling
                  -> B0 voxel fusion
                  -> temporal modeling and 33 -> 9 compression
                  -> z_3d [B,9,768,256]
                  -> two-stage per-voxel causal temporal upsample 9 -> 33
                  -> existing causal refiner and PointMap ray decoder
```

## 3. 优化点一：方案 A aligned chunk 与 Wan-style temporal codec

### 3.1 当前实现

Backbone global windows：

```text
[0:4], [4:8], [8:12], ..., [28:32], [32:33]
```

Temporal codec：

```text
frame 0 单独投影
frames [1:5], [5:9], ... 使用一次 kernel_t=4,stride_t=4 Conv3d
33 -> 9
```

问题是 global window 和 latent chunk 错位，而且一次 stride-4 卷积没有复现 Wan VAE 的
两级下采样与跨 chunk 时间感受野。

### 3.2 本轮计划实现

**方案 A 的在线协议**

```text
frame 0 到达       -> emit z0，timestamp=0
frames 1..4 到达   -> emit z1，timestamp=4
frames 5..8 到达   -> emit z2，timestamp=8
...
frames 29..32 到达 -> emit z8，timestamp=32
```

global windows 改为：

```text
[0], [1:5], [5:9], ..., [29:33]
```

每个四帧窗口完整到达后才运行该窗口的 global attention。窗口内部保持双向 attention，
但不能访问未来窗口。因此本轮保证 chunk/latent-level causality，不保证 chunk 内的
frame-level causality。

Stage 2 WAM 训练和在线推理必须使用同一协议：

- latent timestamp 取 chunk 末帧，不取 chunk 首帧；
- 只把已经闭合的 chunk 送给 WAM；
- 新 clip、episode reset 或 batch sequence 切换时清空 tokenizer state；
- 训练中不能把包含未来 chunk 的 latent 错标到更早 action timestamp。

**Wan-style temporal encoder**

```text
[B,C,33,H,W]
 -> split [0], [1:5], [5:9], ..., [29:33]
 -> causal residual blocks
 -> CausalConv3d kernel_t=3,stride_t=2
 -> causal residual blocks
 -> CausalConv3d kernel_t=3,stride_t=2
 -> causal residual blocks
 -> [B,C,9,H,W]
```

Causal residual block 采用：

```text
main:
  RMSNorm -> SiLU -> CausalConv3d(k=3)
  -> RMSNorm -> SiLU -> Dropout -> CausalConv3d(k=3)
shortcut:
  Identity；channel 改变时使用 1x1x1 Conv
output:
  main + shortcut
```

2D branch 使用 Wan 风格的时空 causal block；3D branch 复用同样的两级时间拓扑和 cache
协议，但不把扁平 voxel index 当作图像邻域，时间卷积可使用 `(3,1,1)`。

**跨 chunk cache**

- 每个 causal convolution 独立缓存其输入端最近两个 temporal positions；
- 两个 stride-2 downsample stages 各自保存边界状态；
- 下一个 chunk 使用对应层 cache 补充左侧上下文，不能在每个 chunk 重新 zero-pad；
- cache 在同一 clip 内连续，在 reset 时清空；
- 训练时默认不 detach cache，保留跨 chunk 梯度；
- 2D/3D branch 不共享权重和 cache。

### 3.3 后续可进一步优化

方案 B 作为后续独立消融：

- global attention 改为按 frame 的 block-causal mask；
- 同一 frame 的所有 views/patches 互相可见，query frame 只能访问当前及过去 frames；
- 增加每层 global attention KV/cache，支持逐帧增量计算；
- 比较方案 A chunk-online 与方案 B frame-online 的重建、动作效果、延迟和显存。

### 3.4 收益与风险

收益：

- global window、2D latent 和 3D latent 使用同一时间索引；
- 保留 Wan 式首帧规则、两级 `/2` 感受野和跨块历史；
- 方案 A 不需要实现复杂的 global KV cache，可以先得到可用 tokenizer；
- chunk 内双向 VGGT attention 有利于多视角和短时几何聚合。

风险：

- 方案 A 只能在四帧 chunk 闭合后产生新 latent；
- 如果 action timestamp 错绑到 chunk 首帧，会产生训练标签的未来泄漏；
- cache layer index、reset 或 stride 奇偶位置错误时，shape 可能正确但时间语义错误；
- 双向 global attention 使其不能用于要求逐帧输出的在线策略。

## 4. 优化点二：`[4,11,17,23]` frame/global feature taps

### 4.1 当前实现

Backbone 主链每层是：

```text
tokens 1024 -> frame block -> frame 1024 -> global block -> global 1024
```

当前 branch 只使用最后层 global feature `[B,T,V,1024,12,23]`，没有保留 frame feature，
也没有使用 transformer-depth 中间层。

### 4.2 本轮计划实现

主链继续保持 1024 维，不把 frame/global concat 后送入下一层。

在 `[4,11,17,23]` 保存旁路：

```text
frame_i  [1024] -> LayerNorm -> Linear/1x1 Conv -> [128]
global_i [1024] -> LayerNorm -> Linear/1x1 Conv -> [128]
direct concat                                  -> [256]
```

Backbone 输出：

```text
feature_levels[0] layer 4  [B,T,V,256,12,23]
feature_levels[1] layer 11 [B,T,V,256,12,23]
feature_levels[2] layer 17 [B,T,V,256,12,23]
feature_levels[3] layer 23 [B,T,V,256,12,23]
```

不增加 frame/global gate 或 layer gate。原版 Transformer block 内的 LayerScale 保持
checkpoint 兼容，但不扩展为 branch fusion gate。

### 4.3 后续可进一步优化

- 消融 `64+64`、`128+128`、`256+256` projection width；
- 尝试深层分配更大 channel 的非对称 projection；
- 若直接 concat 不足，再测试 DPT-style top-down fusion；
- 根据 2D/3D gradient 冲突决定是否使用 branch-specific projection。

### 4.4 收益与风险

收益：

- 2D 同时获得浅层纹理、中层结构和深层跨视角信息；
- 3D 可以使用真正不同 transformer depth，而不是最后层的空间池化副本；
- 每层先降到 256，避免缓存四层完整 2048-channel features。

风险：

- `2048 -> 256` 是有损压缩，128+128 可能成为瓶颈；
- 四层 activation 增加显存，必须保留 checkpointing；
- 新 projection 随机初始化会造成初期 loss spike。

## 5. 优化点三：2D four-level fusion 与 learned spatial resampler

### 5.1 当前实现

```text
last global [B,T,V,1024,12,23]
 -> single-step 1024 -> 48
 -> adaptive_avg_pool2d 12x23 -> 10x20
 -> temporal Transformer
 -> temporal codec 33 -> 9
```

单步通道压缩和固定平均池化可能丢失边缘、小物体与机械臂细节。

### 5.2 本轮计划实现

四层按固定顺序直接 concat：

```text
[F4,F11,F17,F23]
 -> [B,T,V,1024,12,23]
 -> residual bottleneck 1024 -> 256
 -> learned query resampler 12x23 -> 10x20, width=256
 -> residual projection 256 -> 48
 -> [B,T,V,48,10,20]
```

Residual projection 使用普通 residual addition 和 1x1 skip projection，不使用 GLU、
channel gate 或 layer gate。

Learned query resampler：

```text
source K/V  [B*T*V,12*23,256] + 2D positional encoding
query Q     [B*T*V,10*20,256] + learned query/2D position
output      [B*T*V,10*20,256]
```

第一版使用 2～4 heads、1 layer cross-attention，并保持输出 `10x20` 固定。

### 5.3 后续可进一步优化

- 与 adaptive average pool、bilinear resize + Conv 做 matched-compute 对照；
- full cross-attention 过重时改为 deformable query sampling；
- 若 latent 保留细节但 RGB 仍模糊，再单独优化空间 RGB decoder 和 video loss；
- 后续测试更深的 2D fusion block，不与本轮同时引入。

### 5.4 收益与风险

收益：

- 先在 256 维空间重组四层信息，再压到 48 维；
- learned queries 可以主动选择非整数缩放下的重要 patch；
- 输出 shape 和 WAM token budget 不变。

风险：

- cross-attention 增加显存和计算；
- learned queries 可能忽略部分 source tokens 或发生位置漂移；
- 四层 concat 与新 resampler 同时随机初始化，需 warmup 和分组 LR。

## 6. 优化点四：3D two-level true-depth features

### 6.1 当前实现

```text
level 0 = layer 23 feature，12x23
level 1 = avg_pool(layer 23)，6x11
```

两个 level 只有空间尺度差异，没有 transformer-depth 差异。

### 6.2 本轮计划实现

固定使用：

```text
level 0 = layer 11 projected frame/global，12x23，fine
level 1 = layer 23 projected frame/global，6x11，coarse
```

layer 23 coarse feature 在 3D branch 内通过可学习 stride-2 adapter 或受控 pooling 产生；
Backbone 仍输出原始 12x23 feature。

随后保持现有几何路径：

```text
camera projection
 -> multi-view/two-level deformable sampling
 -> B0 voxel fusion [B,T,768,256]
 -> causal temporal modeling
 -> 33 -> 9
```

本轮不把 layer 4/17 接入 3D sampler。

### 6.3 后续可进一步优化

- two-level 稳定增益后，再消融 `[4,11,17,23]` 四个 3D levels；
- 比较 learned stride-2 adapter 与 average pooling；
- 增加每层独立 fine/coarse spatial pyramid；
- pseudo geometry 质量提升后再增加更精细 occupancy/flow 监督。

### 6.4 收益与风险

收益：

- layer 11 提供定位和中层结构，layer 23 提供深层跨视角上下文；
- 保持两个 deformable levels，控制 sampling 成本；
- 不再把同一最后层 feature 误称为 transformer-depth pyramid。

风险：

- layer 11 未必天然比 layer 23 更适合 metric projection；
- 两层 feature 统计不同，需要独立 adapter/normalization；
- pseudo PointMap 和标定误差仍会限制 3D 上限。

## 7. 优化点五：inside-grid auxiliary masks

### 7.1 当前实现

PointMap coordinate、ray/surface、occupancy 和 multiview correspondence 已过滤 metric grid
外 GT；以下辅助 loss 仍主要依赖 `pointmap_valid`：

```text
temporal geometry loss
surface normal loss
depth gradient loss
```

因此 grid 外 GT 仍可能进入辅助监督。

### 7.2 本轮计划实现

统一计算：

```python
inside = points_in_metric_grid(target_xyz, x_range, y_range, z_range)
inside_valid = pointmap_valid & inside
```

使用规则：

```text
temporal:       t 和 t-1 都 inside_valid
surface normal: 构成法向量的所有邻点都 inside_valid
depth grad x:   左右邻点都 inside_valid
depth grad y:   上下邻点都 inside_valid
```

`free_space_loss` 保持例外：只要 ray sample 自身位于 grid 内且处于目标表面之前，即使
目标表面在 grid 外仍是有效 free-space 监督。

记录：

```text
temporal_inside_valid_ratio
normal_inside_valid_ratio
depth_gradient_inside_valid_ratio
auxiliary_outside_rejected_count
```

### 7.3 后续可进一步优化

- 根据 pseudo depth confidence 对辅助 loss 加权；
- 对相机标定异常和时间不同步样本增加 sample-level mask；
- 对 normal/gradient 使用 robust weighting，而不是继续扩大监督范围。

### 7.4 收益与风险

收益：

- 所有表面相关 loss 与生产 metric grid 定义一致；
- 避免模型被迫拟合无法由固定 B0 grid 表示的坐标；
- diagnostics 可以区分真实优化与监督量下降。

风险：

- 有效像素减少，辅助 loss 数值会发生跳变；
- 邻域 mask 实现错误可能把 normal/gradient 监督清空；
- free-space 若误乘 target-inside mask，会丢失合法射线监督。

## 8. 优化点六：2D/3D two-stage causal temporal decoder

### 8.1 当前实现

2D 和 3D decoder 当前共用简化 `WanTemporalDecoder`：

```text
latent 0 单独 1x1 projection
后续每个 latent 通过 1x1 projection 扩为 4C
channel-to-time reshape，一次展开为4帧
temporal residual refinement
9 -> 33
```

之后两条分支不同：

```text
2D: decoded frame features -> four-stage PixelShuffle -> RGB
3D: decoded per-voxel tokens -> causal Transformer refiner
    -> fixed B0 token grid -> PointMap ray decoder
```

当前 temporal encoder/decoder 是粗粒度 `stride-4 / latent-to-4` 配对；当本轮 encoder
升级为两级 stride-2 后，保留一次展开会造成明显的时间层次不对称。

### 8.2 本轮计划实现

2D 和 3D 使用相同的时间拓扑和 state API，但各自持有权重和 cache：

```text
latent [T'=9]
 -> causal residual blocks
 -> first-frame-aware learned temporal upsample x2
 -> causal residual blocks
 -> first-frame-aware learned temporal upsample x2
 -> causal residual blocks
 -> frame features [T=33]
```

每个 upsample stage 使用 causal temporal convolution 产生 `2C` channels，再执行
channel-to-time rearrange；first latent 保持首帧规则，最终严格满足：

```text
T = 1 + 4 * (T' - 1)
```

每个 causal convolution 和两个 upsample stages 都维护独立跨 chunk cache，并与 encoder
共用 `init_state/reset_state/forward_chunk/forward_full` 协议。

2D decoder：

- 在 `10x20` 低分辨率 feature 上完成两级 causal temporal upsample；
- 保留当前 temporal refinement；
- 保留现有四级空间 PixelShuffle 和 RGB output projection；
- 不在 `160x320` RGB feature 上增加昂贵的 3D convolution。

3D decoder 也同步修改，因为 `MetricTokenDecoder` 当前直接复用同一个简化 temporal
decoder：

- 对每个 B0 voxel 独立执行两级 temporal upsample；
- temporal kernel 使用 `(3,1,1)`，可采用 depthwise temporal conv + pointwise projection；
- 不在扁平 voxel index 之间做伪空间卷积；
- 第一版保留现有 causal Transformer refiner；
- 固定 `768` voxel grid、PointMap ray decoder 和 occupancy/surface heads 全部不变。

### 8.3 后续可进一步优化

- 消融 3D causal Transformer refiner：保留、减层或由 residual blocks 完全替代；
- 对比简化 depthwise temporal upsample 与 Wan full-channel causal convolution；
- 方案 B 中加入 decoder stateful incremental API，验证 prefix decode 一致性；
- 只有 temporal decoder 仍限制重建时，才考虑更完整的 Wan-style spatial decoder。

### 8.4 收益与风险

收益：

- encoder `/2,/2` 与 decoder `x2,x2` 时间层次对称；
- 比一次 latent-to-4 展开更容易恢复 chunk 内运动和跨 chunk 连续性；
- 2D/3D 继续共享同一时间 lattice，生成的未来 appearance/geometry 对齐；
- 保留 PixelShuffle 和 PointMap decoder，控制本轮改动范围。

风险：

- 3D 对 768 个 voxel 独立解码，full-channel temporal conv 成本较高；
- decoder cache 与 encoder cache 不能混用，reset/index 错误会破坏 prefix consistency；
- 新 temporal decoder 随机初始化可能暂时降低现有重建；
- 两级 causal upsample 与已有 temporal refiner 可能功能重叠，需要后续消融。

## 9. 实施与消融顺序

```text
E0 baseline
  固化当前 checkpoint、重建指标、3D 指标、显存和速度

E1 inside-grid masks
  只修改辅助 loss mask

E2 方案 A aligned temporal path
  E2a aligned global windows
  E2b two-stage causal temporal codec + per-layer cache
  E2c E2a + E2b

E3 frame/global feature taps
  先 layer 23，再扩展 [4,11,17,23]

E4 2D fusion/resampler
  four-level concat -> 1024->256 -> learned 10x20 -> 48

E5 3D true-depth levels
  layer 11 fine + layer 23 coarse

E6 two-stage temporal decoders
  2D: x2,x2 causal temporal upsample + existing PixelShuffle
  3D: per-voxel x2,x2 causal temporal upsample + existing refiner/ray decoder

E7 combined
  只组合前面有稳定收益的项

Future E8 方案 B
  block-causal global attention + per-layer KV cache + frame-online
```

结构改动从当前最佳 Stage 1 checkpoint 做 name-and-shape matching 初始化。DINO 保持冻结；
新增 projections、resampler、3D adapters 和 temporal codec 随机初始化。新模块先 warmup，
必要时短暂冻结 aggregator LoRA，再以较低 LR 解冻。

## 10. 代码改动范围

```text
backbone.py
  aligned windows
  [4,11,17,23] frame/global taps and projections
  structured BackboneOutput

temporal_codec.py
  Wan-style causal residual blocks
  two stride-2 temporal downsample stages
  two causal temporal upsample stages
  encoder/decoder per-layer cache/state/reset/forward_chunk/forward_full

video_latent.py
  four-level concat
  1024->256->48 residual bottleneck
  learned 10x20 query resampler
  connect two-stage temporal decoder while preserving spatial PixelShuffle

metric_tokens.py
  layer 11 fine + layer 23 coarse
  two-level deformable sampling input
  per-voxel two-stage temporal decoder
  preserve causal refiner and PointMap decoder

model.py / losses
  inside-grid auxiliary masks and diagnostics

configuration.py / YAML
  feature layers/dims, temporal codec, resampler, geometry levels and mask switches
```

建议固定配置：

```yaml
online_tokenizer_mode: chunk
align_global_windows_to_codec: true
global_attention_causal: false
temporal_codec_num_downsample_stages: 2
temporal_codec_use_layer_cache: true
temporal_decoder_num_upsample_stages: 2
temporal_decoder_use_layer_cache: true
feature_tap_layers: [4, 11, 17, 23]
feature_tap_frame_dim: 128
feature_tap_global_dim: 128
video_fusion_dim: 256
video_latent_dim: 48
video_latent_size: [10, 20]
geometry_feature_layers: [11, 23]
mask_auxiliary_losses_to_grid: true
```

## 11. 必须通过的验收

接口与时间：

- `33 -> 9`、`9 -> 33` 和 `z_2d/z_3d` shape 不变；
- chunk mapping 为 `[0],[1:5],...,[29:33]`；
- streaming 与 full-clip reference 数值一致；
- 加入未来 chunk 后，已经闭合的 latent 不改变；
- cache reset 后不残留上一 clip，跨 chunk 梯度可达；
- Stage 2 latent timestamp 固定在 `0,4,8,...,32`。
- 2D/3D decoder 都满足 first-frame-aware `9 -> 33`，且 prefix/full decode 一致；
- decoder 与 encoder 使用独立 cache，2D/3D decoder 也不共享 cache。

特征与分支：

- 四层 tap shape、顺序和 gradient 正确；
- 架构中不存在新增 frame/global gate 或 layer gate；
- learned resampler 输出固定 `10x20`；
- 3D sampler 使用 layer 11 + layer 23，而不是 layer 23 + pool(layer 23)。
- 2D temporal upsample 后仍使用原空间 PixelShuffle；
- 3D temporal upsample 后 voxel 数仍为 768，PointMap ray decoder 接口不变。

质量与成本：

- 报告 head/wrist PSNR、SSIM、LPIPS、edge/temporal error；
- 报告 PointMap inside-grid error 和 auxiliary valid ratios；
- 固定样本可视化 chunk 边界、机械臂、小物体和 3D scatter；
- 同时报告 peak memory、step time 和 feature cache 大小。
