# MobileManiBench Final Research Implementation：Phase 0 / Phase 1 报告

日期：2026-07-23  
远端仓库：`/mnt/yihao/codes/dreamzero`  
运行环境：`/mnt/yihao/envs/dreamzero`  
小批量数据：`/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_smoke_v2`

## 1. 完成范围

本轮仅完成
`MOBILEMANIBENCH_FINAL_RESEARCH_IMPLEMENTATION_PLAN.md` 中的 Phase 0 和
Phase 1：

- Phase 0：固化现有 step-action 过拟合基线和可复现启动入口；
- Phase 1：读取已保存的 realized Action Plan，完成 reshape、padding、mask、
  time offsets、统计、分 slice 归一化、反归一化和 QA；
- 未修改 DreamZero 模型头、loss、collator 或推理控制器；
- 未下载 checkpoint；
- 未启动新的正式训练。

因此，Phase 1 的 Plan batch 已经可验证，但在 Phase 2 双路 action head 接入前，
`mobilemanibench_plan.yaml` 不能直接作为最终研究训练入口。

## 2. Phase 0：基线固化

### 2.1 已有基线

远端已有 G1 小批量 1000-step 过拟合结果：

```text
work_dirs/dreamzero_mobilemanibench_overfit
```

该结果训练的是 DreamZero 原有单路 step action，不是最终的双路 realized plan。
现有日志的代表性结果为：

| 指标 | 起始值 | 末尾值 | 最小值 |
|---|---:|---:|---:|
| total loss | step 10: 0.2075 | step 1000: 0.0726 | step 900: 0.0352 |
| action_loss_avg | step 0: 0.166006 | step 990: 0.040699 | step 900: 0.025969 |
| dynamics_loss_avg | step 0: 0.026257 | step 990: 0.021297 | step 900: 0.015660 |

这些结果只用于确认旧链路能够加载数据、反向传播并对小数据过拟合，不能作为最终
Base/Manipulator Plan head 的效果结论。

### 2.2 训练入口加固

`scripts/train/mobilemanibench_training.sh` 已增加：

- checkpoint、数据、输出目录和主要训练参数的环境变量覆盖；
- `SAVE_TOTAL_LIMIT >= 5` 的显式检查；
- `PREFLIGHT_ONLY=1` 纯预检查模式；
- 明确打印最终解析的数据目录、GPU 数、步数、保存频率和 W&B 模式；
- 保持 `WANDB_MODE=offline` 默认值；
- 保持“不自动下载 checkpoint”。

远端已执行：

```bash
cd /mnt/yihao/codes/dreamzero
export PATH=/mnt/yihao/envs/dreamzero/bin:$PATH
PREFLIGHT_ONLY=1 NUM_GPUS=1 \
  bash scripts/train/mobilemanibench_training.sh
```

结果：所有数据元信息和 checkpoint 目录检查通过，脚本在启动 `torchrun` 前正常退出。

## 3. Phase 1：Plan 数据接口

### 3.1 Dataset

新增 `MobileManiBenchPlanDataset`：

- 每个 parquet row 直接读取一个完整 Action Plan；
- `action.plan.base_waypoints` reshape 为 `[6,4]`；
- G1 manipulator 从 `[6,10]` pad 为 `[6,21]`；
- XHand manipulator 保持 `[6,21]`；
- 同时输出 `plan_valid`、两路 dimension mask；
- 显式输出 `[1,4,8,12,16,24]` 和对应秒数；
- observation 仍复用现有 LeRobot 视频、state、language 加载器；
- 不对 Plan 再应用一次 future delta sampling，避免产生“双重 horizon”。

输出关键字段：

```text
base_plan:                  [6,4]
manipulator_plan:           [6,21]
plan_valid:                 [6]
base_dim_mask:              [6,4]
manipulator_dim_mask:       [6,21]
plan_time_offsets:          [6]
plan_time_seconds:          [6]
```

### 3.2 Plan statistics

新增可复现命令：

```bash
python scripts/data/prepare_mobilemanibench_plan_metadata.py \
  --dataset-root \
  /mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_smoke_v2 \
  --force
```

分别生成：

```text
g1/meta/plan_stats.json
xhand/meta/plan_stats.json
```

统计只使用 `plan_valid=true` 的 waypoint：

| embodiment | 有效 waypoint | manipulator dim | hand dim |
|---|---:|---:|---:|
| G1 | 2138 | 10（pad 后 21） | 1 |
| XHand | 1874 | 21 | 12 |

归一化策略：

- base xy：train q01/q99；
- EEF xyz：train q01/q99；
- hand configuration：每个关节独立 q01/q99；
- base yaw sin/cos：保持原值；
- EEF rotation6d：保持原值；
- valid/dimension mask：不归一化。

### 3.3 Transform

新增 `MobilePlanTransform`：

- 独立输出 `base_action` 和 `manipulator_action`；
- 只对上述三个连续 slice 执行 q99 归一化并 clip 到 `[-1,1]`；
- G1 padding 维不参与 loss；
- 合成 horizon valid mask 与 dimension mask；
- 提供 `unapply()`，将推理输出反归一化回物理量；
- 对有效 waypoint 检查 finite、yaw sin/cos 单位范数、rotation6d 两行单位范数和正交性。

统计阶段的几何 QA：

| embodiment | yaw 最大单位范数误差 | rotation6d 最大单位范数误差 | rotation6d 最大行点积 |
|---|---:|---:|---:|
| G1 | 2.98e-8 | 4.26e-8 | 4.59e-8 |
| XHand | 2.99e-8 | 4.02e-8 | 5.78e-8 |

所有输入均为有限值。

### 3.4 Hydra 配置实例化

新增 `mobilemanibench_plan.yaml`，并分别以 G1/XHand 根目录通过 Hydra
实例化。验证结果：

```text
G1:    len=378, base_action=(6,4), manipulator_action=(6,21)
XHand: len=334, base_action=(6,4), manipulator_action=(6,21)
```

样例中有效的 manipulator loss-mask 元素数：

```text
G1:    6 * 10 = 60
XHand: 6 * 21 = 126
```

这确认了 padding 维不会进入 G1 loss。

## 4. 自动测试

远端执行：

```bash
MOBILEMANIBENCH_SMOKE_ROOT=\
/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_smoke_v2 \
python -m unittest discover \
  -s tests/data \
  -p "test_mobilemanibench_plan_*.py" \
  -v
```

结果：4/4 通过。覆盖：

- G1/XHand reshape 和维度；
- G1 padding 与 dimension mask；
- terminal frame 的 horizon valid mask；
- 防止 horizon 被重复展开；
- 从 preserved `robot_base`、`robot_hand`、`robot_joint` 重建标签并与存储值比较；
- q99 slice 范围；
- identity geometry slice；
- horizon/dimension 组合 mask；
- inverse transform。

## 5. 可视化校验样例

已对 G1/XHand 的 episode 0、frame 30 生成 head RGB、wrist RGB 和
Base/EEF XY 计划叠加图：

```text
g1/validation_samples/phase1_plan_batch_ep000000_frame000030.{png,json}
xhand/validation_samples/phase1_plan_batch_ep000000_frame000030.{png,json}
```

两个样例均有 6 个有效 waypoint。JSON 同时记录原始/归一化 shape、time offsets、
time seconds 和归一化数值范围。PNG 已人工检查，两个相机画面可正常解码，轨迹图可见。

## 6. 本轮新增/修改文件

远端仓库新增：

```text
groot/vla/configs/data/dreamzero/mobilemanibench_plan.yaml
groot/vla/data/dataset/mobilemanibench_plan.py
groot/vla/data/transform/mobile_plan.py
scripts/data/prepare_mobilemanibench_plan_metadata.py
scripts/data/inspect_mobilemanibench_plan_batch.py
tests/data/test_mobilemanibench_plan_dataset.py
tests/data/test_mobilemanibench_plan_transform.py
docs/MOBILEMANIBENCH_PHASE0_PHASE1_REPORT.md
```

远端仓库修改：

```text
groot/vla/data/dataset/__init__.py
groot/vla/data/transform/__init__.py
scripts/train/mobilemanibench_training.sh
```

数据目录新增：

```text
g1/meta/plan_stats.json
xhand/meta/plan_stats.json
g1/validation_samples/phase1_plan_batch_ep000000_frame000030.{png,json}
xhand/validation_samples/phase1_plan_batch_ep000000_frame000030.{png,json}
```

## 7. 下一阶段边界

Phase 2 才应开始：

- research collator；
- base/manipulator 独立 action tokens；
- dual-plan action head；
- 两路独立 loss 及 manipulator slice loss；
- 将 `plan_time_offsets` 编码进模型；
- 用最终 Plan 目标重新做小批量过拟合。

在这些内容完成前，现有 `mobilemanibench_training.sh` 仍训练旧的 step action，
不应误认为已经训练了最终 Base waypoint + Manipulator plan。
