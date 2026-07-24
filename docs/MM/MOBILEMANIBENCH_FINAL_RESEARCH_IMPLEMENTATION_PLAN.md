# MobileManiBench 最终研究方案实施修改计划

> 状态：**Review Draft，仅规划，尚未按本文修改代码**  
> 日期：2026-07-23  
> 目标仓库：`/mnt/yihao/codes/dreamzero`  
> 数据集：`/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_smoke_v2/{g1,xhand}`  
> 相关文档：[vggt_3d_wam_proposal.md](./vggt_3d_wam_proposal.md)、[MOBILEMANIBENCH_TO_DREAMZERO.md](./MOBILEMANIBENCH_TO_DREAMZERO.md)

## 1. 目标

将当前 DreamZero 的单路 step-action 基线：

```text
Observation
├── head/wrist RGB
├── current EEF state
└── language
        ↓
single action tokens
        ↓
future 24-step EEF delta + hand command
```

扩展为最终研究接口：

```text
Observation + 2D/3D tokens
        ↓
Base tokens
└── future base waypoints

Manipulator tokens
└── future EEF pose + hand configuration
```

最终模型需要同时满足：

1. Base 和 Manipulator 使用独立 token 序列、输入投影、输出投影和 token-type embedding。
2. 两路 token 共同进入 DreamZero causal DiT，通过 attention 交互。
3. Manipulator 只有一路 token，但内部按 EEF position、EEF rotation、hand configuration 三个 slice 归一化和计算 loss。
4. 两路计划使用同一锚点坐标系 `B(t)`、同一 future offsets 和同一 horizon valid mask。
5. 2D tokens 是主视觉表示，3D tokens 提供 coarse geometry-aware spatial understanding。
6. 当前可运行的 `mobilemanibench_training.sh` 保留为 baseline，不被研究版修改破坏。

## 2. 本计划不包含的工作

以下内容不在第一轮实现范围内：

- 不立即删除或替换当前 7/18 维 step-action baseline。
- 不在第一轮实现中加入强 collision/contact supervision。
- 不使用有损 depth MP4 声称厘米级或毫米级几何精度。
- 不在双路 plan 尚未过拟合前同时引入 VGGT、3D grid、collision、IK critic 等全部模块。
- 不把 future `robot_base`、future `robot_hand`、success 或 object goal 泄漏到 observation condition。
- 不在同一个 batch 中混合 G1 和 XHand，直到两种 embodiment 分别通过独立过拟合测试。

## 3. 当前基线与已具备数据

### 3.1 当前训练入口

当前配置读取：

```text
video.head
video.wrist
state.eef_position
state.eef_rotation_rpy
action.eef_delta_position_normalized
action.eef_delta_rotation_rpy_normalized
action.hand_target_normalized
annotation.task
```

当前 action shape：

```text
G1:    [24, 7]
XHand: [24, 18]
```

当前 action 是 next-step aligned、表达在当前底盘坐标系中的 EEF delta + hand command。

### 3.2 已生成的最终计划标签

每个 Parquet row 已保存：

```text
action.plan.base_waypoints
action.plan.manipulator
action.plan.valid
```

固定 future offsets：

```text
[1, 4, 8, 12, 16, 24]
```

原始存储 shape：

| 字段 | G1 | XHand |
|---|---:|---:|
| `action.plan.base_waypoints` | `[24]` | `[24]` |
| `action.plan.manipulator` | `[60]` | `[126]` |
| `action.plan.valid` | `[6]` | `[6]` |

目标 reshape：

| 分支 | G1 | XHand |
|---|---:|---:|
| Base | `[6,4]` | `[6,4]` |
| Manipulator | `[6,10]` | `[6,21]` |
| Valid | `[6]` | `[6]` |

标签语义：

```text
base[h]
= [x, y, sin(yaw), cos(yaw)]

manipulator[h]
= [eef_xyz, eef_rotation_6d, hand_configuration]
```

两路均表达在当前规划锚点底盘坐标系 `B(t)` 中，并来自同一未来时刻的真实状态。

## 4. 冻结的数据和模型接口

实施前先冻结以下接口，后续阶段不得静默改变。

### 4.1 时间接口

```python
PLAN_HORIZON = 6
PLAN_OFFSETS = [1, 4, 8, 12, 16, 24]
CONTROL_FPS = 30.0
```

必须向模型显式提供 `PLAN_OFFSETS`。这6个 waypoint 不是均匀采样，不能只使用普通序号 `0..5` 代替真实时间间隔。

### 4.2 Base 接口

```text
base_plan:      float32 [B, 6, 4]
base_dim_mask:  bool    [B, 6, 4]
```

slice：

```text
[0:2] base_xy
[2:4] base_yaw_sincos
```

### 4.3 Manipulator 接口

统一使用最大21维：

```text
manipulator_plan:     float32 [B, 6, 21]
manipulator_dim_mask: bool    [B, 6, 21]
```

G1：

```text
[0:3]  EEF position
[3:9]  EEF rotation6d
[9:10] gripper actual position
[10:21] padding, mask=false
```

XHand：

```text
[0:3]  EEF position
[3:9]  EEF rotation6d
[9:21] 12 right-hand actual joint positions
```

### 4.4 Horizon mask

```text
plan_valid: bool [B,6]
```

最终 loss mask：

```text
base_loss_mask
= plan_valid[...,None] & base_dim_mask

manipulator_loss_mask
= plan_valid[...,None] & manipulator_dim_mask
```

### 4.5 模型 batch 接口

研究版 batch 至少包含：

```python
{
    "images": ...,
    "state": ...,
    "state_mask": ...,
    "base_action": ...,
    "base_action_mask": ...,
    "manipulator_action": ...,
    "manipulator_action_mask": ...,
    "plan_valid": ...,
    "plan_time_offsets": ...,
    "language": ...,
    "embodiment_id": ...,
}
```

接入 VGGT 后扩展：

```python
{
    "tokens_2d": ...,
    "tokens_3d": ...,
    "tokens_3d_valid": ...,
    "camera_intrinsics": ...,
    "camera_extrinsics": ...,
}
```

## 5. 总体实施阶段

计划分为六个阶段。每一阶段通过验收后才能开始下一阶段。

```text
Phase 0  固化 baseline 与测试基线
Phase 1  Plan 数据读取、reshape、mask、stats
Phase 2  双路 Base/Manipulator action tokens
Phase 3  分 slice loss 与两路一致性约束
Phase 4  VGGT 2D/3D tokenizer 独立训练
Phase 5  2D/3D tokens 接入 WAM
Phase 6  推理、控制接口与完整评估
```

## 6. Phase 0：固化当前 Baseline

### 6.1 目的

保证研究版开发过程中始终有一个可运行的 RGB + step-action 对照组。

### 6.2 工作项

1. 保留当前文件：

```text
groot/vla/configs/data/dreamzero/mobilemanibench_relative.yaml
scripts/train/mobilemanibench_training.sh
```

2. 修正 baseline 脚本中已经发现的配置问题：

```text
save_total_limit >= 5
```

3. 将训练步数、保存间隔、W&B 模式改成环境变量可覆盖，避免每次 smoke test 修改脚本。
4. 保存一次 G1 和一次 XHand 的 baseline loss 曲线：

```text
loss
dynamics_loss_avg
action_loss_avg
```

5. 记录 baseline 的 dataset sample shape、显存、单步耗时和 checkpoint 加载方式。

### 6.3 验收标准

- G1 和 XHand 均可完成至少10个 training steps。
- `action_loss_avg` 和 `dynamics_loss_avg` 是有限数。
- 小批量运行中 loss 有下降趋势。
- 研究版新增文件不会改变 baseline Hydra composition。

## 7. Phase 1：Plan 数据读取与归一化

### 7.1 新增 Plan Dataset

建议新增：

```text
groot/vla/data/dataset/mobilemanibench_plan.py
```

职责：

1. 读取当前 row 的三个计划字段。
2. Base reshape 为 `[6,4]`。
3. Manipulator 按 robot schema reshape 为 `[6,10]` 或 `[6,21]`。
4. G1 pad 到 `[6,21]` 并生成 dimension mask。
5. 读取 `[6]` horizon valid mask。
6. 输出 `[1,4,8,12,16,24]` time offsets。
7. 不对计划字段再次应用 `[0..23]` 的 future sampling。

计划字段自身已经包含完整 horizon，因此数据配置必须使用：

```yaml
delta_indices: [0]
```

### 7.2 新增 Plan Stats

建议新增：

```text
meta/plan_stats.json
```

仅使用 train split 计算：

```text
base_xy
eef_position
hand_configuration per joint
```

以下几何表示不执行普通 q99 归一化：

```text
base sin/cos
EEF rotation6d
valid mask
```

推荐归一化：

```text
base_xy         -> train q99 symmetric normalization
eef_xyz         -> train q99/workspace normalization
base_yaw_sincos -> keep [-1,1]
rotation6d      -> keep [-1,1]
hand            -> per-joint q99 or joint-limit normalization
```

### 7.3 新增 Plan Transform

建议新增：

```text
groot/vla/data/transform/mobile_plan.py
```

职责：

- 分 slice normalize/un-normalize；
- G1/XHand dimension padding；
- valid/dimension mask 合成；
- 保留 Base/Manipulator 两个独立 key；
- 提供 inverse transform 给推理输出反归一化；
- 检查 sin/cos norm、rotation6d 有限值和 shape。

不应直接复用当前 `ConcatTransform` 将两路合并为一个 `action`。

### 7.4 数据配置

新增：

```text
groot/vla/configs/data/dreamzero/mobilemanibench_plan.yaml
```

核心配置：

```yaml
relative_action: false
plan_horizon: 6
plan_time_offsets: [1, 4, 8, 12, 16, 24]
base_action_dim: 4
max_manipulator_action_dim: 21
```

### 7.5 QA

新增测试：

```text
tests/data/test_mobilemanibench_plan_dataset.py
tests/data/test_mobilemanibench_plan_transform.py
```

必须覆盖：

- G1/XHand reshape；
- G1 padding mask；
- episode 尾部 valid mask；
- normalize → unnormalize round trip；
- 同一 `h` 的 base/EEF/hand 来源索引一致；
- 不发生 double-horizon sampling；
- world state 独立回算 plan 的最大误差小于阈值；
- future target 不出现在 condition。

### 7.6 验收标准

- 随机抽样 batch 的数值与 Parquet 原字段一致。
- G1 输出 `[6,4]` 和 `[6,21]`，后11维 mask=false。
- XHand 输出 `[6,4]` 和 `[6,21]`，所有有效手部维 mask=true。
- episode 尾部无效 waypoint 的 loss mask=false。
- 可视化 base XY 与 EEF XY 轨迹和已有 validation sample 一致。

## 8. Phase 2：双路 Action Tokens

### 8.1 DreamTransform

建议不要直接破坏现有 `DreamTransform`，而是新增：

```text
groot/vla/model/dreamzero/transform/mobile_plan_cotrain.py
```

可复用当前 video/language/state 处理，但将 `_prepare_action` 替换为：

```text
_prepare_base_action
_prepare_manipulator_action
_prepare_plan_masks
_prepare_plan_time_offsets
```

输出：

```text
base_action
base_action_mask
manipulator_action
manipulator_action_mask
plan_valid
plan_time_offsets
```

### 8.2 Collator

新增研究版 collator，batch 后保持：

```text
base_action:             [B,6,4]
manipulator_action:      [B,6,21]
base_action_mask:        [B,6,4]
manipulator_action_mask: [B,6,21]
plan_time_offsets:       [B,6] or shared [6]
```

### 8.3 双路输入/输出投影

修改或派生：

```text
groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py
groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py
```

建议第一版新增研究类，避免在原类中大量 `if mobile_plan`：

```text
MobilePlanFlowMatchingActionHead
WanVideoDiTDualPlan
```

模块：

```text
BasePlanEncoder       4 -> hidden
BasePlanDecoder       hidden -> 4

ManipulatorEncoder   21 -> hidden
ManipulatorDecoder   hidden -> 21

BaseTypeEmbedding
ManipulatorTypeEmbedding
PlanOffsetEmbedding
```

每个 future offset 对应一个 token：

```text
6 base tokens
6 manipulator tokens
total 12 plan tokens
```

### 8.4 时间编码

普通 token index 无法表达非均匀 offsets，新增：

```text
offset_seconds = offsets / 30
```

并通过 sinusoidal/MLP embedding 加到两路对应 token：

```text
token[h] += PlanOffsetEmbedding(offset_seconds[h])
```

Base[h] 与 Manipulator[h] 使用相同的 offset embedding。

### 8.5 Flow Matching

第一版共享同一个 action diffusion timestep：

```text
t_base = t_manipulator
```

分别生成：

```text
noisy_base
target_base
base_noise_pred

noisy_manipulator
target_manipulator
manipulator_noise_pred
```

共享 timestep 有利于两路同步去噪，也减少第一版变量。独立 timestep 只作为后续 ablation。

### 8.6 Token Layout 和 Attention

需要显式记录 token slice：

```text
video token range
state token range
base token range
manipulator token range
```

第一版采用同一计划窗口内双路 plan token 全连接：

```text
Base tokens <-> Manipulator tokens
```

但只允许读取合法的历史/当前 observation tokens，不允许读取 future clean video/3D target。

如后续改为 action autoregressive mask，需要单独实验，不在第一版同时引入。

### 8.7 配置

新增：

```text
groot/vla/configs/model/dreamzero/action_head/mobile_plan_flow_matching.yaml
groot/vla/configs/model/dreamzero/transform/mobile_plan_cotrain.yaml
scripts/train/mobilemanibench_plan_training.sh
```

保留：

```text
mobilemanibench_training.sh       # step-action baseline
mobilemanibench_plan_training.sh  # final plan path
```

### 8.8 Checkpoint 兼容

现有 DreamZero checkpoint 不包含双路 plan encoder/decoder。

加载策略：

1. 原 Wan/video/text/VAE 参数按原 checkpoint 加载。
2. 新 Base/Manipulator encoder、decoder、type embedding、offset embedding 随机初始化。
3. 输出明确 missing/unexpected key 报告。
4. 新模块必须加入 optimizer，即使 backbone 使用 LoRA。
5. 不允许因为 `strict=False` 静默遗漏本应加载的原模块。

### 8.9 Phase 2 验收标准

先关闭 VGGT 和一致性 loss，仅训练：

```text
video dynamics
base flow matching
manipulator flow matching
```

验收：

- 1 batch forward/backward 通过。
- Base 和 Manipulator projection 都有非零梯度。
- padding/invalid 部分梯度为零。
- 2-episode smoke 数据上两路 flow loss 均明显下降。
- 模型可生成 `[B,6,4]` 和 `[B,6,21]`。
- G1 padding 维不会影响 loss。

## 9. Phase 3：分 Slice Loss 与两路一致性

### 9.1 Manipulator Slice Loss

在模型还原出的 clean plan 上计算：

```text
L_manipulator =
    lambda_eef_pos * L_eef_position
  + lambda_eef_rot * L_eef_rotation
  + lambda_hand    * L_hand
```

建议第一版：

```text
EEF position: Huber or L1
EEF rotation: rotation6d -> SO(3) geodesic
Hand: masked per-joint Huber
```

rotation6d 解码后先正交化，再计算 geodesic loss。

### 9.2 Base Slice Loss

```text
L_base =
    lambda_base_xy  * L_xy
  + lambda_base_yaw * L_yaw
```

其中：

```text
L_xy: Huber/L1
L_yaw: sincos loss + unit-circle regularization
```

### 9.3 两路一致性

对于每个 `h`：

```text
hat_T_Bt_Bk = predicted base waypoint
hat_T_Bt_Ek = predicted EEF pose

hat_T_Bk_Ek
= inverse(hat_T_Bt_Bk) @ hat_T_Bt_Ek
```

真值：

```text
T_Bk_Ek
= inverse(T_W_B(t+kh)) @ T_W_EEF(t+kh)
```

一致性 loss：

```text
L_consistency =
    L_position(hat_T_Bk_Ek, T_Bk_Ek)
  + lambda_rel_rot * L_SO3(hat_T_Bk_Ek, T_Bk_Ek)
```

### 9.4 暂缓项

第一版暂不反传：

```text
collision loss
contact loss
differentiable IK loss
```

先作为离线指标或 inference reranking。只有坐标系、相机标定和几何监督通过 QA 后再启用。

### 9.5 Loss 日志

必须分别记录：

```text
base_flow_loss
manipulator_flow_loss
base_xy_loss
base_yaw_loss
eef_position_loss
eef_rotation_loss
hand_loss
base_eef_consistency_loss
dynamics_loss
total_loss
```

### 9.6 验收标准

- 每个 slice loss 在 smoke overfit 中下降。
- `plan_valid=false` 的位置不改变 loss。
- Base/Manipulator 单独预测误差下降时，一致性误差也下降。
- rotation 输出经过正交化后 determinant 接近1。
- 反归一化后的 hand 不超出合理 joint range。

## 10. Phase 4：VGGT 2D/3D Tokenizer

当前仓库没有 VGGT 实现，本阶段为新增模块。

### 10.1 建议目录

```text
groot/vla/model/dreamzero/vggt/
├── backbone.py
├── tokenizer_2d.py
├── tokenizer_3d.py
├── geometry_adapter.py
├── depth_decoder.py
└── losses.py
```

### 10.2 输入

```text
head RGB video
wrist RGB video
camera intrinsics K
dynamic camera pose/extrinsics
coarse depth MP4
depth confidence/valid mask
segmentation/dynamic mask
```

### 10.3 2D 分支

主方案：

```text
multi-view RGB
-> shared VGGT backbone
-> per-view 2D features
-> TemporalTransformer_2D
-> z_2d_video
```

`z_2d_video` 必须具备：

- 稳定 latent statistics；
- video reconstruction/feature reconstruction 能力；
- 可加噪、可去噪；
- 可解码或映射回视频/feature target；
- 不强行对齐旧 VAE latent。

保留回退方案：

```text
Wan VAE 2D latent + VGGT 3D tokens
```

### 10.4 3D 分支

推荐第一版锚点坐标系：

```text
current base frame B(t)
```

原因：

- 与 Base/Manipulator plan 使用同一坐标系；
- action/geometry consistency 更直接；
- 避免世界系场景原点差异；
- 推理时可由当前 base pose 恢复世界系。

候选表示需要 Review 确认：

```text
A. BEV-height tokens
B. sparse voxel tokens
C. dense voxel tokens
```

第一版推荐 A 或 B，不推荐 dense voxel。

流程：

```text
VGGT image features
+ camera K/extrinsics
+ metric grid queries in B(t)
-> per-frame 2D-to-3D aggregation
-> TemporalTransformer_3D
-> deterministic z_3d
```

### 10.5 MP4 Depth 使用边界

当前 depth 是 lossy H.264 pseudo-range：

- 只用于 coarse geometry supervision；
- 必须使用 valid/confidence mask；
- voxel/bin 分辨率不能高于数据有效精度；
- 第一版 geometry loss 小权重 warm up；
- 不启用强 collision/contact 标签；
- 不声称高精度 metric reconstruction。

### 10.6 Tokenizer Loss

```text
L_tokenizer =
    lambda_rgb * L_2d_reconstruction
  + lambda_feature * L_2d_feature
  + lambda_depth * L_coarse_depth
  + lambda_masked_view * L_masked_view
  + lambda_temporal * L_temporal_geometry
  + lambda_cross_view * L_cross_view
```

训练顺序：

1. 先训练/验证 2D-only。
2. 加入低权重 depth。
3. 加入 masked-view。
4. 加入 temporal/cross-view consistency。

### 10.7 验收标准

- 2D-only 重建质量达到可用于 WAM 的水平。
- 加入3D后2D指标不显著退化。
- 3D tokens 可解码出优于常数/单目无几何基线的 coarse depth。
- 相机移动下静态区域的3D token基本一致。
- 打乱相机参数会显著降低3D指标，证明模型使用了几何输入。

## 11. Phase 5：2D/3D Tokens 接入 WAM

### 11.1 模型输入

扩展研究版 action head 输入：

```text
clean history 2D tokens
noisy future 2D tokens
clean history 3D tokens
noisy future 3D tokens
state tokens
base plan tokens
manipulator plan tokens
language tokens
```

### 11.2 Adapter

新增：

```text
2DTokenAdapter: VGGT 2D dim -> DiT hidden dim
3DTokenAdapter: VGGT 3D dim -> DiT hidden dim
```

并加入：

```text
view embedding
time embedding
2D/3D type embedding
metric-grid positional encoding
```

### 11.3 Attention 防泄漏

必须用单测验证：

- history 2D/3D tokens 可作为 condition；
- future clean 2D/3D 不可进入 condition；
- future target 只能以加噪形式进入 denoising；
- action tokens 可与同一预测窗口的2D/3D tokens交互；
- future `robot_base/hand/joint` 只用于 target/loss。

### 11.4 联合 Loss

```text
L_total =
    lambda_2d * L_2d_denoise
  + lambda_3d * L_3d_denoise
  + lambda_base_flow * L_base_flow
  + lambda_manip_flow * L_manipulator_flow
  + lambda_eef_pos * L_eef_position
  + lambda_eef_rot * L_eef_rotation
  + lambda_hand * L_hand
  + lambda_consistency * L_base_eef_consistency
  + lambda_geometry * L_future_geometry
  + lambda_video_3d * L_video_3d_consistency
```

训练时监控各分支 gradient norm。3D loss 不得通过共享 backbone 持续破坏2D主分支。

### 11.5 Ablation

至少训练：

```text
A. RGB/VAE + dual plan
B. VGGT 2D + dual plan
C. VGGT 2D + VGGT 3D + dual plan
D. C 去掉 consistency loss
E. C 打乱/屏蔽3D tokens
```

只有 C 相比 B 有稳定收益，才能支持“3D tokens 改善空间理解”的结论。

## 12. Phase 6：推理与控制接口

### 12.1 推理输出

研究版 policy 返回：

```python
{
    "base_plan": ...,             # [6,4]
    "manipulator_plan": ...,      # [6,10/21]
    "plan_time_offsets": ...,
}
```

后处理：

1. Base xy 和 EEF xyz 反归一化。
2. Base yaw sin/cos 重新单位化。
3. EEF rotation6d 投影到合法 SO(3)。
4. Hand configuration 反归一化并裁剪到 joint limits。
5. 将 `B(t)` 中的计划转换到控制器需要的坐标系。

### 12.2 执行器

```text
Base plan
-> waypoint tracker
-> base velocity/pose command

Manipulator plan
-> trajectory interpolator
-> IK / whole-body controller
-> arm joints + gripper/hand targets
```

### 12.3 Receding Horizon

建议：

```text
每次预测6个 waypoint
只执行前1--2个
重新观测
重新规划
```

需要定义：

- replanning frequency；
- waypoint interpolation；
- base/manipulator 同步策略；
- stale plan 丢弃规则；
- invalid/unsafe plan fallback。

### 12.4 推理指标

Base：

```text
ADE/FDE
yaw error
waypoint smoothness
tracking error
```

Manipulator：

```text
EEF position ADE/FDE
SO(3) geodesic error
hand joint error
trajectory smoothness
```

联合：

```text
base/EEF relative consistency
reachability
collision rate
task success
```

## 13. 文件修改清单

### 13.1 保留不动的 Baseline

```text
groot/vla/configs/data/dreamzero/mobilemanibench_relative.yaml
scripts/train/mobilemanibench_training.sh
```

baseline 脚本只允许做独立的健壮性修复，不改成 research path。

### 13.2 建议新增

```text
groot/vla/data/dataset/mobilemanibench_plan.py
groot/vla/data/transform/mobile_plan.py

groot/vla/model/dreamzero/transform/mobile_plan_cotrain.py
groot/vla/model/dreamzero/action_head/mobile_plan_flow_matching.py
groot/vla/model/dreamzero/modules/wan_video_dit_dual_plan.py

groot/vla/model/dreamzero/vggt/backbone.py
groot/vla/model/dreamzero/vggt/tokenizer_2d.py
groot/vla/model/dreamzero/vggt/tokenizer_3d.py
groot/vla/model/dreamzero/vggt/geometry_adapter.py
groot/vla/model/dreamzero/vggt/depth_decoder.py
groot/vla/model/dreamzero/vggt/losses.py

groot/vla/configs/data/dreamzero/mobilemanibench_plan.yaml
groot/vla/configs/model/dreamzero/action_head/mobile_plan_flow_matching.yaml
groot/vla/configs/model/dreamzero/transform/mobile_plan_cotrain.yaml

scripts/train/mobilemanibench_plan_training.sh
scripts/inference/mobilemanibench_plan_policy.py

tests/data/test_mobilemanibench_plan_dataset.py
tests/data/test_mobilemanibench_plan_transform.py
tests/model/test_dual_plan_shapes.py
tests/model/test_dual_plan_masks.py
tests/model/test_dual_plan_gradients.py
tests/model/test_plan_attention_no_leakage.py
tests/model/test_vggt_token_shapes.py
```

### 13.3 可能需要小范围修改

```text
groot/vla/model/dreamzero/base_vla.py
groot/vla/data/dataset/__init__.py
groot/vla/data/transform/__init__.py
groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml
scripts/data/convert_mobilemanibench_to_gear.py
```

原则：优先注册新类/新配置，不在现有 baseline 类中堆叠大量条件分支。

## 14. 测试矩阵

### 14.1 数据测试

| 测试 | G1 | XHand |
|---|---:|---:|
| reshape | 必须 | 必须 |
| valid mask | 必须 | 必须 |
| dimension mask | 必须 | 必须 |
| round-trip normalization | 必须 | 必须 |
| world-state plan reconstruction | 必须 | 必须 |

### 14.2 模型测试

| 测试 | 预期 |
|---|---|
| Base forward | `[B,6,4]` |
| Manipulator forward | `[B,6,21]` |
| G1 padding gradient | 0 |
| invalid horizon gradient | 0 |
| Base encoder gradient | non-zero |
| Manipulator encoder gradient | non-zero |
| checkpoint missing keys | 仅新模块 |
| attention leakage | 不可读取 future clean target |

### 14.3 Overfit 测试

顺序：

```text
1 episode G1
2 episodes G1
1 episode XHand
2 episodes XHand
```

必须观察：

```text
base_flow_loss
manipulator_flow_loss
base_xy_loss
base_yaw_loss
eef_position_loss
eef_rotation_loss
hand_loss
consistency_loss
```

## 15. 风险和缓解

### 15.1 Plan horizon 与现有 action block 不兼容

风险：

```text
当前 action_horizon=24
最终 plan_horizon=6
```

缓解：

- 新建 dual-plan module；
- 不复用 `num_action_per_block=24` 的隐式假设；
- 显式测试 token ranges 和 attention mask。

### 15.2 Double-horizon sampling

风险：对已经包含完整未来计划的 row 再使用 `[0..23]` delta indices。

缓解：

```text
plan delta_indices=[0]
```

并增加 source-index 单测。

### 15.3 G1/XHand hand 维度不同

缓解：

- 最大21维统一 head；
- embodiment-specific dimension mask；
- 第一阶段分开训练；
- padding loss/gradient 单测。

### 15.4 两路 token 各自拟合但组合不可执行

缓解：

- shared offset embedding；
- shared backbone attention；
- relative EEF consistency loss；
- reachability 先做 metric/reranking，后决定是否可微。

### 15.5 Rotation 表示退化

缓解：

- yaw sin/cos unit-circle regularization；
- rotation6d 正交化；
- SO(3) geodesic loss；
- inference 后投影到合法旋转。

### 15.6 3D 噪声破坏2D主分支

缓解：

- 2D-only baseline；
- geometry loss warm up；
- gradient norm monitoring；
- 3D adapter；
- 必要时对 shared features 部分 stop-gradient。

### 15.7 相机标定未验证

缓解：

- 第一版只做 coarse geometry；
- projection/collision 强 loss 保持关闭；
- 将 K/optical convention QA 设为启用强几何 loss 的前置门槛。

### 15.8 新模块无法从原 checkpoint 加载

缓解：

- 新模块显式随机初始化；
- 原模块 strict audit；
- missing/unexpected key 白名单；
- 确认新模块进入 optimizer。

## 16. 回滚和兼容策略

研究版使用独立：

```text
dataset class
transform
action head
DiT module
Hydra config
training script
inference policy
```

通过配置选择：

```text
data=dreamzero/mobilemanibench_relative
  -> 当前 step-action baseline

data=dreamzero/mobilemanibench_plan
  -> dual-plan research path
```

任何阶段失败时，可以直接切回 baseline，不需要回滚数据转换结果。

## 17. 实施里程碑与停止条件

### Milestone A：Plan Data Ready

- 数据 reshape、mask、stats、可视化全部通过。
- 不涉及模型修改。

### Milestone B：Dual Plan Overfit

- 不使用 VGGT。
- 两路 flow loss 和各 slice loss 在 smoke 数据上下降。
- 能输出正确 shape 并反归一化。

### Milestone C：Plan Consistency

- consistency loss 下降。
- 组合后的相对 EEF 误差低于无一致性版本。

### Milestone D：VGGT 2D Tokenizer

- 2D-only tokenizer 达到 WAM 可用重建质量。

### Milestone E：VGGT Coarse 3D

- coarse depth/cross-view 指标优于简单基线。
- 2D指标没有不可接受退化。

### Milestone F：2D/3D WAM

- 2D+3D 相比2D-only在 action 或强视角变化场景有稳定收益。

任一里程碑未通过时停止扩展，先修复当前阶段，不继续叠加下一阶段。

## 18. 需要 Review 确认的设计选择

请在执行前确认以下项目。

### 18.1 计划锚点

推荐：

```text
Base/Manipulator/3D metric grid 全部使用当前底盘 B(t)
```

待确认：是否同意。

回复：同意

### 18.2 第一版视觉路径

推荐实施顺序：

```text
现有 VAE + dual plan
-> VGGT 2D + dual plan
-> VGGT 2D/3D + dual plan
```

待确认：是否同意先完成 plan-only 模型改造，再接 VGGT。

回复：同意

### 18.3 3D 表示

推荐第一版：

```text
BEV-height 或 sparse voxel
```

待确认：优先选择哪一种。

回复：BEV-height

### 18.4 Flow timestep

推荐第一版：

```text
Base/Manipulator 共享 action diffusion timestep
```

待确认：是否同意。

回复：同意

### 18.5 Attention

推荐第一版：

```text
同一预测窗口内 Base/Manipulator 全连接
+ 严格屏蔽 future clean observation targets
```

待确认：是否同意。

回复：同意

### 18.6 Hand normalization

候选：

```text
A. train q99 per joint
B. robot joint-limit normalization
```

推荐：如果官方 joint limits 可完整验证，使用 B；否则第一版使用 A。

回复：第一版先使用A，并同步验证官方是否包含joint limits

### 18.7 G1/XHand 训练方式

推荐：

```text
先分别过拟合和训练，再做混合 embodiment 实验
```

待确认：是否同意。

回复：同意

### 18.8 一致性约束启用顺序

推荐：

```text
先 relative EEF pose consistency
后 reachability metric/critic
最后才考虑 collision
```

待确认：是否同意。

回复：同意

## 19. Review 后的执行顺序

本文 Review 通过后，下一轮执行只从 Phase 0 和 Phase 1 开始：

1. 修复并冻结 baseline。
2. 新增 Plan Dataset/Transform/Stats。
3. 完成数据单测和 batch 可视化。
4. 输出 Phase 1 修改报告。
5. 等待确认后再进入双路模型结构修改。

不会在一次修改中同时实现所有 Phase，也不会在 Plan Dataset 尚未验收时直接改 DiT/VGGT。
