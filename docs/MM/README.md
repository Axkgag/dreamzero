# MobileManiBench / VGGT 文档入口与当前实现状态

> 校对日期：2026-07-30
> 代码基准：远程当前工作树（包含可配置 sparse clean prior 与 physical-consistency 实现）
> 远程仓库：`/mnt/yihao/codes/dreamzero`

本页是 MobileManiBench 相关文档的状态入口。代码、配置和脚本是实现事实的最终来源；
proposal 与 implementation plan 描述目标，历史报告只说明当时的实验状态。

## 1. 当前实现边界

| 能力 | 状态 | 权威入口 |
|---|---|---|
| 原始 MobileManip → LeRobot/GEAR 转换 | 已实现 | `scripts/data/convert_mobilemanibench_to_gear.py` |
| realized Base/Manipulator plan labels | 已实现 | `action.plan.*`、`mobilemanibench_plan.py` |
| WAM train/val trajectory-group split | 已实现 | `prepare_mobilemanibench_splits.py`、`meta/plan_splits.json` |
| VGGT train/val split | 已实现但待统一 | 当前为 seeded episode-level 90/10，不读取 `plan_splits.json` |
| 五任务平衡子集 | 已实现 | `create_mobilemanibench_task_subset.py` |
| 双路 Base/Manipulator flow tokens | 已实现 | `mobile_plan_flow_matching.py`、`wan_video_dit_dual_plan.py` |
| 分支 flow loss | 已实现 | `base_flow_loss`、`manipulator_flow_loss` |
| Phase 3 physical slice training loss | 已实现 | `mobile_plan_physical_losses.py` |
| Base/EEF relative-pose consistency training loss | 已实现 | physical-consistency action-head configs |
| Wan2.2-5B 训练内 `eval_loss` | 已实现 | `BaseTrainer`、`mobilemanibench_plan_training_wan22_5b.sh` |
| Wan2.2 checkpoint 离线轨迹 evaluator | 已修复 | `evaluate_mobilemanibench_plan.py` |
| VGGT 2D/3D tokenizer | 已实现 | `groot/vla/model/vggt_3d_wam/` |
| VGGT 独立训练、日志、验证、可视化 | 已实现 | `vggt_3d_wam.py`、对应 train/eval shell |
| configurable sparse clean Base/EEF Prior | 已实现 | `mobile_plan_clean_prior_flow_matching.py`、`wan_video_dit_dual_plan_prior.py` |
| VGGT `z_2d/z_3d` 接入 WAM | **未实现** | Phase 6 计划 |
| future 3D flow、控制器与仿真成功率 | **未实现** | 后续研究阶段 |

“已实现”表示代码接口和相应轻量测试存在，不自动等价于全量训练已经收敛或任务成功率
已经验证。Wan2.2 evaluator 的 `pretrained_model_path=null` 加载逻辑已经修复。
Phase 3/4 的轻量单测与训练启动链路已经通过，但 prior 是否改善最终轨迹指标，仍必须
用相同 checkpoint 预算、相同验证样本和相同采样参数做消融后判断。

## 2. 当前关键合同

### 双路 Action Plan

```text
PLAN_OFFSETS = [1, 4, 8, 12, 16, 24] at 30 Hz

Base plan        [B,6,4]   = x, y, sin(yaw), cos(yaw)
Manipulator plan [B,6,21]  = EEF xyz, rotation6d, hand, embodiment padding

DreamZero packed action [B,12,21]
token 0..5   = Base，只有前4维有效
token 6..11  = Manipulator，按 embodiment mask 有效
```

两路标签均表示同一 `B_anchor=B(t)` 坐标系中的 future realized state。训练目标可由
`MOBILE_PLAN_LOSS_PROFILE` 选择：

- `flow_only`：Base/Manipulator masked flow matching；
- `physical_consistency`：在 flow loss 之外，从预测 velocity 恢复 clean plan，增加
  Base XY/yaw、EEF position/SO(3)、hand slice loss，以及 Base–EEF relative-pose
  consistency。

physical loss 只作用于有效 horizon 和有效 embodiment 维；collision、contact 与
differentiable IK 仍未进入训练目标。

### Sparse clean prior

当前 `clean_prior` architecture 在 12 个 noisy flow tokens 之前加入 `K` 个不加 flow
noise 的 clean prior tokens。`K` 由 `prior.time_offsets` 的长度决定，当前配置为：

```yaml
prior:
  time_offsets: [8, 16, 24]
  predict_base: true
  predict_eef: true
  eef_frame: future_base
```

因此当前内部布局是 `3 clean prior + 6 Base flow + 6 Manipulator flow = 15` tokens，
而不是为六个 flow horizon 各复制一个 prior。`time_offsets` 必须是
`PLAN_OFFSETS` 的严格递增子集。若要做 Base-only 消融，将 `predict_eef` 设为
`false`；token 数不变，EEF 与 joint prior loss 自动不参与。

一组共享的 prior hidden tokens 使用相互解耦的 Base head 与 EEF head：

```text
Base prior [B,K,4] = x, y, sin(yaw), cos(yaw)
EEF prior  [B,K,9] = xyz + rotation6d（不预测 hand）
```

Base prior 表达在 `B(t)`；当 `eef_frame=future_base` 时，EEF prior 表达在
`B(t+h)`。该 EEF target 在训练时由现有 clean Base/EEF action 动态构造：

```text
T_B(t+h)_EEF(t+h)
  = inverse(T_B(t)_B(t+h)) @ T_B(t)_EEF(t+h)
```

不需要重新转换数据集。联合 composition loss 再把预测的 Base 与 future-base EEF
组合回 `B(t)`，与 clean EEF target 比较。当前 direct loss 权重为 Base `0.1`、
EEF `0.1`，joint composition 为 `0.05`，并分别 warm up/ramp；这些是初始安全值，
正式实验前应结合 gradient diagnostics 校准。

信息流为单向：

```text
clean context -> clean prior -> noisy Base/Manipulator flow
```

prior 不能读取 noisy flow hidden states。最终控制输出仍只使用 flow 采样得到的
refined Base/Manipulator plan；`base_prior_pred` 和 `eef_prior_pred` 仅作为辅助监督、
诊断与可视化输出。

### VGGT tokenizer

```text
RGB input          [B,33,V,3,160,320]
z_2d               [B,V,48,9,10,20]
z_3d               [B,9,768,256]
decoded 3D grid    [B,33,256,8,12,8]
RGB reconstruction [B,33,V,3,160,320]
PointMap           [B,33,V,3,80,160]
```

2D/3D 共享 Wan 时间 lattice：`frame0 + 8×4-frame chunks`，即 `33→9→33`。
当前 metric grid 是 `B0` 前向范围：

```text
X [0,3] m, Y [-2,2] m, Z [-0.5,2] m
grid [Z,Y,X] = [8,12,8] = 768 voxels
```

当前生产配置：

- DINOv2 完全冻结、无 LoRA；
- VGGT frame/global aggregator 使用 rank-8 LoRA；
- `global_temporal_window=4`，当前 global windows 为 `[0:4],[4:8]...`；
- 2-level、2-layer、8-head deformable image-to-voxel aggregation；
- PointMap ray rendering 为 `40×80`，learned refinement 输出 `80×160`；
- geometry 主权重 `0.4`，质量系数 `0.25`，warmup 后有效权重 `0.1`；
- `masked_view_probability=0`，其余 temporal/multiview/normal/gradient 几何项已启用。

深度来自有损 H.264 pseudo-range，内外参仍按 nominal calibration 使用。因此这些
3D tokens 只应表述为 coarse geometry，不能支持毫米级重建或强碰撞/接触结论。

注意 temporal codec 的边界是 `frame0 + [1:5],[5:9]...`，与当前 aggregator 的
`[0:4],[4:8]...` 相差一帧。两者 stride 都是 4，但尚不能描述为严格 chunk-boundary
对齐；这是接入 WAM 前需要决定是否修正的实现问题。

## 3. 数据集状态

| 数据集 | Train | Validation | 用途 |
|---|---:|---:|---|
| G1 full | 130,799 episodes / 23,313,897 frames | 6,911 / 1,236,759 | WAM full split |
| G1 five-task | 5,856 / 1,090,154 | 286 / 56,376 | WAM five-task split |
| G1 smoke v2 | 1 / 171 | 1 / 207 | 链路与过拟合 smoke |

五任务为：

```text
close box
pull cart
open faucet
open window
open dishwasher
```

五任务数据的 `meta/info.json.total_tasks` 仍继承原 registry 值 `39`；实际选择任务必须
以 `meta/plan_splits.json.tasks` 和 `selection` 为准。训练/验证统计只使用 split
manifest 中的 episode。

上表是 WAM 的 `plan_splits.json` 统计。VGGT dataset 当前另行按 episode ID 做
seeded 90/10 split，未复用 trajectory-group manifest；同一 source trajectory 的
sibling episodes 可能跨 split。VGGT validation 可用于训练监控，但在统一 split 前
不应作为严格泛化结论。

控制状态按 30 Hz 解释，MP4 metadata 为 25 FPS。当前转换保持 frame-index 对齐，而不
把媒体时间戳作为控制时钟；涉及严格物理时间同步的结论必须保留该限制。

## 4. 文档阅读顺序

### 当前实现

1. [Phase 0–2 当前实现指南](./MOBILEMANIBENCH_CODE_CHANGES_BY_FILE_PHASE0-2.md)
2. [VGGT 当前代码指南](./MOBILEMANIBENCH_VGGT_CODE_CHANGES_BY_FILE.md)
3. [Wan2.2-5B 数据、训练与验证命令](./MOBILEMANIBENCH_WAN22_5B_FULL_BASELINE_COMMANDS.md)
4. [数据转换方案](./MOBILEMANIBENCH_TO_DREAMZERO.md)

### 目标与计划

1. [最终研究实施计划](./MOBILEMANIBENCH_FINAL_RESEARCH_IMPLEMENTATION_PLAN.md)
2. [VGGT-3D WAM proposal](../vggt_3d_wam_proposal.md)

### 历史记录

- `MOBILEMANIBENCH_CODE_CHANGES_BY_FILE.md`：早期工作树逐文件快照；
- `MOBILEMANIBENCH_PHASE0_PHASE1_REPORT.md`：Phase 0/1 当时的验收记录；
- `../dreamzero_mobile_plan_code_guide.md`：2026-07-28 代码阅读快照；
- `../mobile_manipulation_research_notes.md`：概念讨论，不代表当前实现。

历史文档中的 checkpoint、工作树状态和当时尚未修复的问题不应覆盖本页的当前状态。

## 5. 常用入口

```bash
# 当前默认：3-token Base+EEF prior + physical consistency
MOBILE_PLAN_ARCHITECTURE=clean_prior \
MOBILE_PLAN_LOSS_PROFILE=physical_consistency \
  bash scripts/train/mobilemanibench_plan_training_wan22_5b.sh

# 无 prior 的双路 flow baseline
MOBILE_PLAN_ARCHITECTURE=dual_plan \
MOBILE_PLAN_LOSS_PROFILE=flow_only \
  bash scripts/train/mobilemanibench_plan_training_wan22_5b.sh

# 双路 checkpoint 离线轨迹评估
bash scripts/eval/mobilemanibench_plan_eval.sh

# VGGT tokenizer
bash scripts/train/mobilemanibench_vggt_training.sh

# VGGT checkpoint 验证
CHECKPOINT=/absolute/path/to/checkpoint-N \
  bash scripts/eval/mobilemanibench_vggt_validate.sh
```

prior 的 offsets、预测目标、坐标系和三个 loss 权重在
`groot/vla/configs/model/dreamzero/action_head/mobile_plan_flow_matching_clean_prior.yaml`
中配置；architecture/loss profile、数据路径、输出目录和训练规模优先使用脚本公开的
环境变量覆盖，不要复制整段 Hydra 参数维护第二套启动命令。
