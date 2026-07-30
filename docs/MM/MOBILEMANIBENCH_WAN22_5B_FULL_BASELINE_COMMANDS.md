# MobileManiBench → DreamZero WAN2.2-5B 全量 Baseline 命令记录

## 1. 实验目标与固定路径

本实验将 DreamZero 从原有单路 Manipulation Action 预测迁移到 Mobile
Manipulation 的双路 Action Plan：

```text
Base tokens        → 未来 Base waypoint 序列
Manipulator tokens → 未来 EEF pose + hand configuration
```

固定路径：

```bash
REPO_ROOT=/mnt/yihao/codes/dreamzero
PYTHON_BIN=/mnt/yihao/envs/dreamzero/bin/python
RAW_ROOT=/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_opensource
FULL_ROOT=/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero/g1
SMOKE_ROOT=/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_smoke_v2/g1
RUN_DIR=/mnt/yihao/codes/dreamzero/work_dirs/mobilemanibench_g1_5tasks_wan22_5b_baseline_evalfix
VGGT_DATA_ROOT=/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_g1_5tasks/g1
VGGT_CHECKPOINT=/mnt/yihao/codes/ReconDrive/checkpoints/model.pt
VGGT_RUN_DIR=/mnt/yihao/codes/dreamzero/work_dirs/mobilemanibench_5tasks_vggt_evalfix
```

后续命令均在远程服务器执行：

```bash
cd /mnt/yihao/codes/dreamzero
```

## 2. Step 1：原始数据转换

### 2.1 本次实际启动过的全量转换命令

```bash
/mnt/yihao/envs/dreamzero/bin/python \
  scripts/data/convert_mobilemanibench_to_gear.py convert \
  --input-root /mnt/yihao/datasets/MobileManiBench/MobileManipVLA_opensource \
  --output-root /mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero \
  --embodiments g1 xhand \
  --max-episodes-per-embodiment 0 \
  --waypoint-offsets 1,4,8,12,16,24 \
  --control-fps 30 \
  --link-mode hardlink \
  --validate
```

该进程已经完成 G1 写入：

```text
137710 episodes
24550656 frames
```

之后进入逐 episode、逐视频的穷举 `ffprobe` validation。由于全量校验需要扫描
6 路 MP4，耗时过长，因此在 G1 写入完成后手动 `Ctrl+C` 取消。XHand 没有在
本次 baseline 中继续转换。

### 2.2 从头复现 G1 时推荐的转换命令

对全量数据不要在转换阶段增加 `--validate`：

```bash
/mnt/yihao/envs/dreamzero/bin/python \
  scripts/data/convert_mobilemanibench_to_gear.py convert \
  --input-root /mnt/yihao/datasets/MobileManiBench/MobileManipVLA_opensource \
  --output-root /mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero \
  --embodiments g1 \
  --max-episodes-per-embodiment 0 \
  --waypoint-offsets 1,4,8,12,16,24 \
  --control-fps 30 \
  --link-mode hardlink
```

注意：转换器拒绝覆盖已经存在的 output root。当前全量 G1 已经转换完成，不能
也不需要重复运行该命令。

转换后的核心监督字段包括：

```text
action.plan.base_waypoints  [6, 4]
action.plan.manipulator     [6, 10] for G1
action.plan.valid           [6]
```

其中：

```text
Base = [relative_x, relative_y, sin(relative_yaw), cos(relative_yaw)]
Manipulator = [eef_xyz, eef_rotation6d, hand_configuration]
```

6 个未来 offset 为：

```text
[1, 4, 8, 12, 16, 24] control frames @ 30 Hz
```

## 3. Step 2：生成 Train/Validation 划分

官方数据中的 `scene_infos.room_infos.split` 和路径 `train_0` 对全部
137710 个 episode 都是 train，因此本实验创建自定义划分：

```bash
/mnt/yihao/envs/dreamzero/bin/python \
  scripts/data/prepare_mobilemanibench_splits.py \
  --dataset-root /mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero/g1 \
  --validation-fraction 0.05 \
  --seed 42 \
  --group-by trajectory
```

划分策略：

```text
按 39 个 task 分层
按 trajectories/traj_* 整组划分
同一 trajectory 下的 episode 不会跨 Train/Validation
```

实际结果：

| Split | Trajectory groups | Episodes | Frames / samples |
|---|---:|---:|---:|
| Train | 13118 | 130799 | 23313897 |
| Validation | 693 | 6911 | 1236759 |
| Total | 13811 | 137710 | 24550656 |

生成文件：

```text
/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero/g1/meta/plan_splits.json
```

## 4. Step 3：生成 Train-only Action Plan normalization

normalization 只能使用 Train split，不能使用 Validation 数据：

```bash
/mnt/yihao/envs/dreamzero/bin/python \
  scripts/data/prepare_mobilemanibench_plan_metadata.py \
  --dataset-root /mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero/g1 \
  --split train \
  --quantile-sample-size 2000000 \
  --seed 42 \
  --force
```

检查结果：

```bash
/mnt/yihao/envs/dreamzero/bin/python -c \
'import json; p="/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero/g1/meta/plan_stats.json"; d=json.load(open(p)); print("fit_split:", d["fit_split"]); print("counts:", d["counts"]); print("geometry:", d["geometry_qa"])'
```

本次实际结果：

```text
fit_split: train
valid_waypoints: 131381545
all_waypoint_slots: 139883382
all_finite: True
base_yaw_sincos_max_unit_norm_error: 3.659741198980271e-08
rotation6d_max_row_unit_norm_error: 5.134268521445051e-08
rotation6d_max_abs_row_dot: 8.467148404633917e-08
```

## 5. Step 4：可选 Smoke 数据与链路测试

如果从原始数据重新建立两 episode smoke 数据：

```bash
/mnt/yihao/envs/dreamzero/bin/python \
  scripts/data/convert_mobilemanibench_to_gear.py convert \
  --input-root /mnt/yihao/datasets/MobileManiBench/MobileManipVLA_opensource \
  --output-root /mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_smoke_v2 \
  --embodiments g1 \
  --max-episodes-per-embodiment 2 \
  --waypoint-offsets 1,4,8,12,16,24 \
  --control-fps 30 \
  --link-mode hardlink \
  --validate
```

为两 episode smoke 数据创建一条 Train、一条 Validation：

```bash
/mnt/yihao/envs/dreamzero/bin/python \
  scripts/data/prepare_mobilemanibench_splits.py \
  --dataset-root /mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_smoke_v2/g1 \
  --validation-fraction 0.5 \
  --seed 42 \
  --group-by episode
```

生成 smoke 的 Train-only normalization：

```bash
/mnt/yihao/envs/dreamzero/bin/python \
  scripts/data/prepare_mobilemanibench_plan_metadata.py \
  --dataset-root /mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_smoke_v2/g1 \
  --split train \
  --quantile-sample-size 2000000 \
  --seed 42 \
  --force
```

运行 WAN2.2-5B 50-step 工程链路测试：

```bash
MOBILEMANIBENCH_DATA_ROOT=/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_smoke_v2/g1 \
OUTPUT_DIR=/mnt/yihao/codes/dreamzero/work_dirs/mobilemanibench_dual_plan_g1_wan22_5b_smoke_50step \
MAX_STEPS=50 \
SAVE_STEPS=50 \
bash scripts/train/mobilemanibench_plan_training_wan22_5b.sh
```

Smoke 只用于确认模型加载、forward/backward、双路 loss 和 checkpoint 保存，不用于
评价最终泛化性能。

## 6. Step 5：全量训练 Preflight

```bash
PREFLIGHT_ONLY=1 \
bash scripts/train/mobilemanibench_plan_training_wan22_5b.sh
```

必须以如下输出结束：

```text
Preflight checks passed; training was not started.
```

Preflight 会检查：

```text
数据 metadata 和 plan_splits.json
plan_stats.json 是否仅由 train split 生成
WAN2.2-TI2V-5B DiT/T5/VAE
Wan2.1 CLIP image encoder
tokenizer
训练参数基本约束
```

## 7. Step 6：启动本次全量训练

本次实际训练命令：

```bash
cd /mnt/yihao/codes/dreamzero

MAX_STEPS=200000 \
bash scripts/train/mobilemanibench_plan_training_wan22_5b.sh
```

已从实际运行进程核对的参数：

| 参数 | 数值 |
|---|---:|
| Backbone | `Wan2.2-TI2V-5B` |
| GPU | 8 |
| Per-device train batch | 32 |
| Global batch | 256 |
| Gradient accumulation | 1 |
| MAX_STEPS | 200000 |
| 约等效 epoch | 2.20 |
| Learning rate | `1e-5` |
| Scheduler | cosine |
| Warmup ratio | `0.05` |
| Weight decay | `1e-5` |
| Precision | BF16 + TF32 |
| Train architecture | LoRA |
| `save_lora_only` | `true` |
| Save interval | 5000 optimizer steps |
| Save total limit | 10 |
| Online eval interval | 5000 optimizer steps |
| Online eval samples | 1024 |
| Seed | 42 |

训练输出：

```text
/mnt/yihao/codes/dreamzero/work_dirs/mobilemanibench_dual_plan_g1_wan22_5b_baseline
```

训练日志：

```text
/mnt/yihao/codes/dreamzero/work_dirs/mobilemanibench_dual_plan_g1_wan22_5b_baseline/loss_log.jsonl
```

使用相同命令重新启动时，框架会检查该 output directory 下的最新 checkpoint 并自动
恢复，不需要额外设置 resume 开关。

## 8. Step 7：训练期间验证

训练脚本已经启用：

```text
do_eval=true
eval_strategy=steps
eval_steps=5000
max_eval_samples=1024
```

因此每 5000 optimizer steps 会在 Validation split 的固定 1024 个 anchor 上计算
`eval_loss`。完整 Validation split 仍保留 6911 episodes / 1236759 anchors。

训练期间建议重点记录：

```text
loss / eval_loss
dynamics_loss
action_loss
base_flow_loss
manipulator_flow_loss
learning_rate
```

在线 `eval_loss` 是 flow-matching 目标，不替代离线轨迹指标。

## 9. Step 8：Checkpoint 离线轨迹验证

### 9.1 先检查 Validation split 解析

```bash
DATA_ROOT=/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero/g1 \
SPLIT=val \
INSPECT_ONLY=1 \
bash scripts/eval/mobilemanibench_plan_eval.sh
```

输出中的 split source 应为：

```text
meta/plan_splits.json
```

### 9.2 固定 1024 样本离线验证

将 checkpoint 路径替换为实际要验证的 checkpoint：

```bash
CHECKPOINT=/mnt/yihao/codes/dreamzero/work_dirs/mobilemanibench_dual_plan_g1_wan22_5b_baseline/checkpoint-50000 \
DATA_ROOT=/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero/g1 \
SPLIT=val \
MAX_SAMPLES=1024 \
SAMPLE_STRIDE=1 \
NUM_INFERENCE_STEPS=16 \
NUM_GPUS=2 \
EVAL_GPUS=0,1 \
bash scripts/eval/mobilemanibench_plan_eval.sh
```

默认结果目录：

```text
<checkpoint>/mobile_plan_eval_val/
```

为了可比较性，不同 checkpoint 必须保持：

```text
SPLIT=val
MAX_SAMPLES=1024
SAMPLE_STRIDE=1
NUM_INFERENCE_STEPS=16
相同 SEED
```

推荐依次比较：

```text
checkpoint-50000
checkpoint-100000
checkpoint-150000
checkpoint-200000
```

离线核心指标：

```text
Base waypoint ADE/FDE
EEF position error
EEF orientation error
Hand joint error
Base/Manipulator 相对位姿一致性
轨迹平滑度与 normalization saturation
```

全量 1236759 个 validation anchors 的生成式推理成本非常高，第一版 baseline 应先用
固定 1024 样本比较 checkpoint，再根据结果扩大样本数。

五任务子集的当前 validation split 实际为 56,376 个 anchors。训练脚本中的
`MAX_EVAL_SAMPLES=1024` 会用等间隔索引选择固定 1,024 个样本，因此训练内验证是
确定性抽样验证，不是全量验证，也不会在每次 eval 重新随机抽样。

DreamZero `BaseTrainer` 已适配 batch-dict 验证接口，并把同一步的执行顺序调整为：

```text
先保存 checkpoint -> 再执行 validation
```

因此 validation 失败时仍可从该 step 的 checkpoint 自动恢复。

## 10. VGGT 2D/3D Tokenizer 启动命令

VGGT tokenizer 是独立的第一阶段训练，不会在这一阶段替换正在训练的
WAN2.2-5B WAM。当前固定接口为：

```text
video       [B,33,V,3,160,320]
z_2d_video  [B,V,48,9,10,20]
z_3d_video  [B,9,384,256]

2D decode: 9 -> 33 RGB frames
3D decode: 9 -> 33 B0-forward metric grids [4,12,8]
          -> 33-frame PointMap video
```

### 10.1 启动前检查

脚本默认使用五任务数据集和 ReconDrive VGGT checkpoint：

```bash
cd /mnt/yihao/codes/dreamzero

PREFLIGHT_ONLY=1 \
bash scripts/train/mobilemanibench_vggt_training.sh
```

必须以如下输出结束：

```text
video_contract=33x160x320 -> 9x10x20 -> 33x160x320
temporal_layout=frame0 + 8 chunks of 4 frames (shared by 2D/3D)
metric_grid=B0-forward x[0,3] y[-2,2] z[-0.5,2], 4x12x8=384 tokens
geometry_fusion=2-layer, 2-level, 8-head deformable cross-attention
dino=frozen, no LoRA, no_grad chunks of 4 images
aggregator=rank-8 LoRA + activation checkpointing
Preflight checks passed; training was not started.
```

### 10.2 生产规模训练前的 2-step GPU smoke

这一步用于确认完整 VGGT checkpoint、33 帧双视角输入和 backward 能在单张 A100
上运行。使用独立输出目录，避免污染正式训练的自动恢复目录：

```bash
cd /mnt/yihao/codes/dreamzero

CUDA_VISIBLE_DEVICES=0 \
NUM_GPUS=1 \
MAX_STEPS=2 \
SAVE_STEPS=2 \
EVAL_STEPS=100000 \
REPORT_TO=none \
OUTPUT_DIR=/mnt/yihao/codes/dreamzero/work_dirs/mobilemanibench_5tasks_vggt_smoke_2step \
bash scripts/train/mobilemanibench_vggt_training.sh
```

如果这一步出现 OOM，不能直接启动 8 卡 DDP；DDP 不会降低每张 GPU 上的模型和
单样本显存，需要先调整 activation/checkpointing 或模型执行策略。

### 10.3 启动 8 卡、10000-step VGGT tokenizer 训练

脚本已经写入默认数据、checkpoint、batch size、训练步数和保存间隔，因此正式启动
只需要：

```bash
cd /mnt/yihao/codes/dreamzero

bash scripts/train/mobilemanibench_vggt_training.sh
```

等价的关键默认参数为：

| 参数 | 数值 |
|---|---:|
| Dataset | 五任务 G1 |
| VGGT checkpoint | `/mnt/yihao/codes/ReconDrive/checkpoints/model.pt` |
| GPU | 8 |
| Per-device batch | 1 |
| Gradient accumulation | 1 |
| Global batch | 8 |
| Max steps | 30000 |
| 新建 2D/3D heads learning rate | `5e-5` |
| VGGT aggregator LoRA learning rate | `2e-5` |
| Scheduler | cosine with minimum LR |
| Warmup ratio | `0.01`（300 steps） |
| Minimum LR rate | `0.2` |
| Save interval | 5000 |
| Eval interval | 5000 |
| 训练内 validation | 从 24197 个 held-out clips 均匀固定抽取 256 个 |
| DINOv2 | 完全冻结、无 LoRA、4-image chunk、no-grad |
| VGGT frame/global | 冻结原始权重 + rank-8 LoRA + activation checkpointing |
| 3D grid | B0 前方 `[4,12,8]`，384 tokens |
| 2D-to-3D | 2-layer、2-level、8-head deformable cross-attention |
| 2D/3D 新增模块 | 完整训练 |

正式输出目录：

```text
/mnt/yihao/codes/dreamzero/work_dirs/mobilemanibench_5tasks_vggt_evalfix
```

相同命令重启时，`groot/vla/experiment/vggt_3d_wam.py` 会自动检查该目录中的最新
`checkpoint-*` 并恢复训练，不需要手动添加 resume 参数。

VGGT loss 日志持续追加在：

```text
/mnt/yihao/codes/dreamzero/work_dirs/mobilemanibench_5tasks_vggt_evalfix/loss_log.jsonl
```

其中包含总 loss、通用 learning rate、`backbone_learning_rate`、
`heads_learning_rate`、grad norm，以及 RGB、KL、PointMap、ray surface 等分支的
10-step 滑动平均。

旧的 576-token 静态采样 checkpoint 与新的 384-token deformable encoder 参数形状
不同；旧 identity-camera 或无 quality multiplier 的 optimizer state 也不应与新监督
混合。必须使用上述新输出目录从 VGGT backbone checkpoint 开始新训练。

不要在当前 WAN2.2 action baseline 占满 8 张 GPU 时同时启动 VGGT 训练。

### 10.4 VGGT checkpoint 验证

以 `checkpoint-2000` 为例：

```bash
cd /mnt/yihao/codes/dreamzero

CHECKPOINT=/mnt/yihao/codes/dreamzero/work_dirs/mobilemanibench_5tasks_vggt/checkpoint-2000 \
bash scripts/eval/mobilemanibench_vggt_validate.sh
```

默认验证 100 个样本，并将指标写入：

```text
<checkpoint>/validation_metrics.json
```

## 11. 最终执行顺序摘要

```text
1. convert_mobilemanibench_to_gear.py convert
2. prepare_mobilemanibench_splits.py
3. prepare_mobilemanibench_plan_metadata.py --split train
4. 可选：smoke 50-step
5. PREFLIGHT_ONLY=1
6. MAX_STEPS=200000 启动全量训练
7. 每 5000 step 自动计算 eval_loss 并保存 checkpoint
8. 在 split=val 的固定 1024 样本上运行离线轨迹评估
9. 比较多个 checkpoint，选择验证指标最优模型
10. VGGT：先 preflight，再运行单卡 2-step 显存 smoke
11. VGGT：启动 8 卡 10000-step tokenizer 训练
12. 使用 `mobilemanibench_vggt_validate.sh` 验证 VGGT checkpoint
```
