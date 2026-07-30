# MobileManiBench / VGGT 文档入口与当前实现状态

> 校对日期：2026-07-30
> 代码基准：`092f247`，并包含当前工作树中的文档与验证器修复
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
| Phase 3 slice training loss | **未实现** | 仍只有计划文档 |
| Base/Manipulator consistency training loss | **未实现** | 离线相对位姿指标已实现，但不反传 |
| Wan2.2-5B 训练内 `eval_loss` | 已实现 | `BaseTrainer`、`mobilemanibench_plan_training_wan22_5b.sh` |
| Wan2.2 checkpoint 离线轨迹 evaluator | 已修复 | `evaluate_mobilemanibench_plan.py` |
| VGGT 2D/3D tokenizer | 已实现 | `groot/vla/model/vggt_3d_wam/` |
| VGGT 独立训练、日志、验证、可视化 | 已实现 | `vggt_3d_wam.py`、对应 train/eval shell |
| clean Base Prior tokens / coarse head | **未实现** | Phase 4 计划 |
| VGGT `z_2d/z_3d` 接入 WAM | **未实现** | Phase 6 计划 |
| future 3D flow、控制器与仿真成功率 | **未实现** | 后续研究阶段 |

“已实现”表示代码接口和相应轻量测试存在，不自动等价于全量训练已经收敛或任务成功率
已经验证。Wan2.2 evaluator 的 `pretrained_model_path=null` 加载逻辑已经修复，但新的
完整离线评估结果仍应由实际 checkpoint 运行后记录。

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

两路标签均表示同一 `B_anchor` 坐标系中的 future realized state。当前训练只计算两路
masked flow MSE；slice loss 与两路 consistency 目前仅存在于计划和离线指标中。

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
# 五任务 Wan2.2-5B 双路 baseline
bash scripts/train/mobilemanibench_plan_training_wan22_5b.sh

# 双路 checkpoint 离线轨迹评估
bash scripts/eval/mobilemanibench_plan_eval.sh

# VGGT tokenizer
bash scripts/train/mobilemanibench_vggt_training.sh

# VGGT checkpoint 验证
CHECKPOINT=/absolute/path/to/checkpoint-N \
  bash scripts/eval/mobilemanibench_vggt_validate.sh
```

需要覆盖默认值时使用脚本已经公开的环境变量；不要复制整段 Hydra 参数重新维护第二套
启动命令。
