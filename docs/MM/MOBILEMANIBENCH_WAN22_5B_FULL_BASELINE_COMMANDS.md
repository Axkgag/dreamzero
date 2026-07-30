# MobileManiBench → DreamZero WAN2.2-5B 训练与验证命令

> 当前脚本默认使用五任务数据集、10,000 steps、`clean_prior +
> physical_consistency`；无 prior baseline 通过环境变量显式选择。
> 下文保留全量转换/划分记录，同时给出当前 five-task 训练入口。
> 实现状态和关键张量合同见 [README.md](./README.md)。

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
RUN_DIR=/mnt/yihao/codes/dreamzero/work_dirs/mobilemanibench_g1_5tasks_wan22_5b_baseline
VGGT_DATA_ROOT=/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_g1_5tasks/g1
VGGT_CHECKPOINT=/mnt/yihao/codes/ReconDrive/checkpoints/model.pt
VGGT_RUN_DIR=/mnt/yihao/codes/dreamzero/work_dirs/mobilemanibench_5tasks_vggt_v2_savefix
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
MOBILE_PLAN_ARCHITECTURE=dual_plan \
MOBILE_PLAN_LOSS_PROFILE=flow_only \
MAX_STEPS=50 \
SAVE_STEPS=50 \
bash scripts/train/mobilemanibench_plan_training_wan22_5b.sh
```

Smoke 只用于确认模型加载、forward/backward、双路 loss 和 checkpoint 保存，不用于
评价最终泛化性能。

## 6. Step 5：训练 Preflight

当前 sparse prior + physical consistency：

```bash
MOBILE_PLAN_ARCHITECTURE=clean_prior \
MOBILE_PLAN_LOSS_PROFILE=physical_consistency \
PREFLIGHT_ONLY=1 \
  bash scripts/train/mobilemanibench_plan_training_wan22_5b.sh
```

无 prior 双路 baseline：

```bash
MOBILE_PLAN_ARCHITECTURE=dual_plan \
MOBILE_PLAN_LOSS_PROFILE=flow_only \
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

## 7. Step 6：启动训练

当前脚本默认运行五任务 sparse clean-prior + physical-consistency 训练：

```bash
cd /mnt/yihao/codes/dreamzero

MOBILE_PLAN_ARCHITECTURE=clean_prior \
MOBILE_PLAN_LOSS_PROFILE=physical_consistency \
  bash scripts/train/mobilemanibench_plan_training_wan22_5b.sh
```

这等价于脚本当前默认选择。实际模型内部为 `3 clean prior + 12 noisy flow = 15`
registers；clean prior 配置为 `[8,16,24]`、Base+EEF heads、future-base EEF target。
若启动摘要仍打印旧的 `6 clean Base Prior / 18 internal` 文本，应以 resolved Hydra
config 和模型实际 layout 为准。

无 prior、flow-only baseline 必须显式选择，避免与 clean-prior checkpoint 混用：

```bash
MOBILE_PLAN_ARCHITECTURE=dual_plan \
MOBILE_PLAN_LOSS_PROFILE=flow_only \
OUTPUT_DIR=/mnt/yihao/codes/dreamzero/work_dirs/mobilemanibench_g1_5tasks_wan22_5b_baseline \
  bash scripts/train/mobilemanibench_plan_training_wan22_5b.sh
```

当前默认参数：

| 参数 | 数值 |
|---|---:|
| Backbone | `Wan2.2-TI2V-5B` |
| GPU | 1（可用 `NUM_GPUS` 覆盖） |
| Per-device train batch | 32 |
| Global batch | `NUM_GPUS × 32`，默认 32 |
| Gradient accumulation | 1 |
| MAX_STEPS | 10000 |
| Learning rate | `1e-5` |
| Scheduler | `cosine_with_min_lr`，最低倍率 `0.1` |
| Warmup ratio | `0.05` |
| Weight decay | `1e-5` |
| Precision | BF16 + TF32 |
| Train architecture | LoRA |
| `save_lora_only` | `true` |
| Save interval | 2000 optimizer steps |
| Save total limit | 10 |
| Online eval interval | 2000 optimizer steps |
| Online eval samples | 1024 |
| Seed | 42 |

当前 clean-prior 默认训练输出：

```text
/mnt/yihao/codes/dreamzero/work_dirs/mobilemanibench_g1_5tasks_wan22_5b_clean_prior_physical_consistency
```

训练日志：

```text
<OUTPUT_DIR>/loss_log.jsonl
```

使用相同命令重新启动时，框架会检查该 output directory 下的最新 checkpoint 并自动
恢复，不需要额外设置 resume 开关。

需要运行 full G1 时显式覆盖数据、输出目录和步数，避免与五任务 checkpoint 混用：

```bash
MOBILEMANIBENCH_DATA_ROOT=/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero/g1 \
OUTPUT_DIR=/mnt/yihao/codes/dreamzero/work_dirs/mobilemanibench_g1_full_wan22_5b_clean_prior_physical_consistency \
MOBILE_PLAN_ARCHITECTURE=clean_prior \
MOBILE_PLAN_LOSS_PROFILE=physical_consistency \
MAX_STEPS=200000 \
SAVE_STEPS=5000 \
EVAL_STEPS=5000 \
bash scripts/train/mobilemanibench_plan_training_wan22_5b.sh
```

脚本启动摘要目前打印 `scheduler=cosine`，但真正传给 Hydra 的配置是
`cosine_with_min_lr + min_lr_rate=0.1`；应以 resolved config 为准。

### 7.1 配置 sparse prior 目标

在
`groot/vla/configs/model/dreamzero/action_head/mobile_plan_flow_matching_clean_prior.yaml`
中修改：

```yaml
prior:
  time_offsets: [8, 16, 24]
  predict_base: true
  predict_eef: true
  eef_frame: future_base
```

约束与含义：

- `time_offsets` 必须是 `[1,4,8,12,16,24]` 的严格递增子集，其长度决定 prior token 数；
- `predict_base/predict_eef` 可分别形成 Base-only、EEF-only 或 Base+EEF；
- `eef_frame=future_base` 的 target 在训练时由 clean action 动态转换，不需要重建数据；
- EEF prior 只预测 `xyz+rotation6d`，不预测 hand；
- flow 输出仍是实际规划结果，prior 输出只用于 condition、辅助 loss 与诊断。

同一配置文件还提供 Base、EEF、joint composition 三项 prior loss 的 weight、
start step、ramp steps 和 gradient target ratio。当前初始总权重分别为 `0.1/0.1/0.05`。
修改这些权重后应重新运行 preflight，并用 calibration/gradient log 检查 auxiliary
loss 是否压过 flow 主目标。

## 8. Step 7：训练期间验证

训练脚本已经启用：

```text
do_eval=true
eval_strategy=steps
eval_steps=2000
max_eval_samples=1024
```

因此默认每 2000 optimizer steps 会在 five-task Validation split 的固定 1024 个
anchor 上计算 `eval_loss`。five-task 完整 Validation 为 286 episodes / 56,376
anchors；full G1 为 6,911 / 1,236,759。

训练期间建议重点记录：

```text
loss / eval_loss
dynamics_loss
action_loss
base_flow_loss
manipulator_flow_loss
base_xy_loss / base_yaw_loss
eef_position_loss / eef_rotation_loss / hand_loss
base_eef_consistency_loss
base_prior_loss / eef_prior_loss / joint_prior_consistency_loss
learning_rate
```

在线 `eval_loss` 是 flow-matching 目标，不替代离线轨迹指标。

## 9. Step 8：Checkpoint 离线轨迹验证

### 9.1 先检查 Validation split 解析

```bash
DATA_ROOT=/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_g1_5tasks/g1 \
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
CHECKPOINT=/mnt/yihao/codes/dreamzero/work_dirs/mobilemanibench_g1_5tasks_wan22_5b_baseline/checkpoint-10000 \
DATA_ROOT=/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_g1_5tasks/g1 \
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
checkpoint-2000
checkpoint-4000
checkpoint-6000
checkpoint-8000
checkpoint-10000
```

离线核心指标：

```text
Base waypoint ADE/FDE
EEF position error
EEF orientation error
Hand joint error
Base/Manipulator 相对位姿一致性
```

`mobilemanibench_plan_eval.sh` 不会自动计算 normalization saturation 或生成高低误差
轨迹图。需要对生成的 `predictions.npz` 另行运行：

```bash
/mnt/yihao/envs/dreamzero/bin/python \
  scripts/eval/analyze_mobilemanibench_plan_predictions.py \
  --predictions <checkpoint>/mobile_plan_eval_val/predictions.npz \
  --plan-stats /mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_g1_5tasks/g1/meta/plan_stats.json \
  --output-dir <checkpoint>/mobile_plan_eval_val/analysis
```

完整 validation 的生成式推理成本很高。five-task 与 full G1 都应先用固定 1024
样本比较 checkpoint，再根据结果扩大样本数。

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
z_3d_video  [B,9,768,256]

2D decode: 9 -> 33 RGB frames
3D decode: 9 -> 33 B0-forward metric grids [8,12,8]
          -> 33-frame 80x160 PointMap video
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
temporal_window=4 (Wan-aligned source-frame chunks)
metric_grid=B0-forward x[0,3] y[-2,2] z[-0.5,2], 8x12x8=768 tokens
video_losses=Charbonnier + LPIPS + SSIM + spatial-gradient + temporal-difference
pointmap_decoder=40x80 ray rendering -> learned 80x160 refinement
geometry_fusion=2-layer, 2-level, 8-head deformable cross-attention
dino=frozen, no LoRA, no_grad chunks of 4 images
aggregator=rank-8 LoRA + activation checkpointing
Preflight checks passed; training was not started.
```

上述 `Wan-aligned` 是启动脚本当前打印文本，只表示窗口宽度为4。实际 aggregator
windows 为 `[0:4],[4:8]...`，temporal codec 为 `frame0+[1:5],[5:9]...`，边界并未
严格对齐。

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

### 10.3 启动 8 卡、30000-step VGGT tokenizer 训练

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
| 3D grid | B0 前方 `[8,12,8]`，768 tokens |
| 2D-to-3D | 2-layer、2-level、8-head deformable cross-attention |
| 2D/3D 新增模块 | 完整训练 |

正式输出目录：

```text
/mnt/yihao/codes/dreamzero/work_dirs/mobilemanibench_5tasks_vggt_v2_savefix
```

相同命令重启时，`groot/vla/experiment/vggt_3d_wam.py` 会自动检查该目录中的最新
`checkpoint-*` 并恢复训练，不需要手动添加 resume 参数。

VGGT loss 日志持续追加在：

```text
/mnt/yihao/codes/dreamzero/work_dirs/mobilemanibench_5tasks_vggt_v2_savefix/loss_log.jsonl
```

其中包含总 loss、通用 learning rate、`backbone_learning_rate`、
`heads_learning_rate`、grad norm，以及 RGB、KL、PointMap、ray surface 等分支的
10-step 滑动平均。

旧的 384/576-token checkpoint 与当前 768-token V2 结构参数形状不同，不能执行
Trainer resume。若要复用其中仍匹配的 trainable 参数，使用 `INIT_CHECKPOINT`：

```bash
INIT_CHECKPOINT=/absolute/path/to/old/checkpoint-N \
OUTPUT_DIR=/mnt/yihao/codes/dreamzero/work_dirs/mobilemanibench_5tasks_vggt_v2_savefix \
bash scripts/train/mobilemanibench_vggt_training.sh
```

matching-only 不恢复 optimizer/global step；空 `INIT_CHECKPOINT` 会完全跳过。

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
6. 显式选择 `clean_prior/dual_plan` 与 `physical_consistency/flow_only` 后启动；
   当前默认是 five-task 10000-step clean-prior + physical-consistency
7. 默认每 2000 step 先保存 checkpoint，再计算 eval_loss
8. 在 split=val 的固定 1024 样本上运行离线轨迹评估
9. 比较多个 checkpoint，选择验证指标最优模型
10. VGGT：先 preflight，再运行单卡 2-step 显存 smoke
11. VGGT：启动 8 卡 30000-step tokenizer 训练
12. 使用 `mobilemanibench_vggt_validate.sh` 验证 VGGT checkpoint
```
