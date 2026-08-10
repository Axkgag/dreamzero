#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
DREAMZERO_ENV=${DREAMZERO_ENV:-"/mnt/yihao/envs/dreamzero"}
TORCHRUN_BIN="$DREAMZERO_ENV/bin/torchrun"

export PATH="$DREAMZERO_ENV/bin:$PATH"
export HYDRA_FULL_ERROR=1
export NO_ALBUMENTATIONS_UPDATE=1
cd "$REPO_ROOT"

DEFAULT_DATA_ROOT="/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_g1_5tasks/g1"
MOBILEMANIBENCH_DATA_ROOT=${MOBILEMANIBENCH_DATA_ROOT:-"$DEFAULT_DATA_ROOT"}
PLAN_CONFIG="$REPO_ROOT/groot/vla/configs/data/dreamzero/mobilemanibench_multiblock_plan.yaml"
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/work_dirs/mobilemanibench_g1_5tasks_wan22_5b_multiblock_k2_wp2"}
ACTION_HEAD_CONFIG=${ACTION_HEAD_CONFIG:-mobile_plan_multiblock_clean_prior_physical_consistency_wan22}
NUM_ACTION_PER_BLOCK=${NUM_ACTION_PER_BLOCK:-}
NUM_GPUS=${NUM_GPUS:-4}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-32}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}
MAX_STEPS=${MAX_STEPS:-10000}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
SAVE_STEPS=${SAVE_STEPS:-2000}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-10}
EVAL_STEPS=${EVAL_STEPS:-2000}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-1024}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"/mnt/yihao/codes/checkpoints/Wan2.2-TI2V-5B"}
IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-"/mnt/yihao/codes/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"/mnt/yihao/codes/checkpoints/umt5-xxl"}
WANDB_MODE=${WANDB_MODE:-offline}
REPORT_TO=${REPORT_TO:-wandb}
WANDB_PROJECT=${WANDB_PROJECT:-dreamzero-mobile-plan}
export WANDB_MODE

if [ ! -x "$TORCHRUN_BIN" ]; then
  echo "ERROR: torchrun is missing: $TORCHRUN_BIN" >&2
  exit 1
fi
for required_file in \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/info.json" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/extensions.json" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/plan_splits.json" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/robot_schema.json"; do
  if [ ! -f "$required_file" ]; then
    echo "ERROR: required multiblock dataset file is missing: $required_file" >&2
    exit 1
  fi
done

"$DREAMZERO_ENV/bin/python" \
  scripts/data/prepare_mobilemanibench_plan_metadata.py \
  --dataset-root "$MOBILEMANIBENCH_DATA_ROOT" \
  --split train \
  --dynamic-multiblock \
  --plan-config "$PLAN_CONFIG" \
  --reuse-existing

PLAN_STATS_PATH=$("$DREAMZERO_ENV/bin/python" -c '
import sys, yaml
from groot.vla.utils.mobile_plan_spec import dynamic_block_plan_stats_path
root, config_path = sys.argv[1:]
config = yaml.safe_load(open(config_path))
print(dynamic_block_plan_stats_path(
    root, config["block_anchor_offsets"], config["plan_local_offsets"]
))
' "$MOBILEMANIBENCH_DATA_ROOT" "$PLAN_CONFIG")

PLAN_SHAPE=$("$DREAMZERO_ENV/bin/python" -c '
import json, sys, yaml
from groot.vla.utils.mobile_plan_spec import block_plan_spec_hash
root, config_path, stats_path = sys.argv[1:]
config = yaml.safe_load(open(config_path))
anchors = config["block_anchor_offsets"]
local = config["plan_local_offsets"]
block_stride = 8
expected_anchors = [index * block_stride for index in range(len(anchors))]
assert anchors == expected_anchors, (
    f"block_anchor_offsets must follow video block boundaries {expected_anchors}, "
    f"got {anchors}"
)
assert max(local) <= block_stride, (
    f"local waypoint offsets must remain inside one {block_stride}-frame block"
)
stats = json.load(open(stats_path))
assert stats["fit_split"] == "train"
assert stats["label_source"] == "dynamic_world_trajectory"
assert stats["label_spec_hash"] == block_plan_spec_hash(anchors, local)
assert stats["block_anchor_offsets"] == anchors
assert stats["plan_time_offsets"] == local
assert stats["coordinate_frame"] == "each_block_anchor_base"
extensions = json.load(open(root + "/meta/extensions.json"))
assert float(extensions["time"]["control_fps"]) == float(config["control_fps"])
assert config["num_plan_blocks"] == len(anchors)
assert config["plan_horizon"] == len(local)
assert config["plan_waypoints_per_block"] == len(local)
assert config["action_horizon"] == 2 * len(local)
global_offsets = [anchor + offset for anchor in anchors for offset in local]
assert config["num_frames"] > max(global_offsets)
print(len(anchors), len(local), config["num_frames"], config["action_horizon"])
' "$MOBILEMANIBENCH_DATA_ROOT" "$PLAN_CONFIG" "$PLAN_STATS_PATH"
)
read -r NUM_PLAN_BLOCKS PLAN_WAYPOINTS NUM_VIDEO_FRAMES PLAN_ACTION_HORIZON <<< "$PLAN_SHAPE"
if [ -z "$NUM_ACTION_PER_BLOCK" ]; then
  if [[ "$ACTION_HEAD_CONFIG" == *clean_prior* ]]; then
    NUM_ACTION_PER_BLOCK=$((2 * PLAN_WAYPOINTS + 1))
  else
    NUM_ACTION_PER_BLOCK=$((2 * PLAN_WAYPOINTS))
  fi
fi

echo "MobileManiBench multiblock WAM configuration:"
echo "  branch=$(git branch --show-current)"
echo "  data_root=$MOBILEMANIBENCH_DATA_ROOT"
echo "  output_dir=$OUTPUT_DIR"
echo "  plan_stats=$PLAN_STATS_PATH"
echo "  action_head=$ACTION_HEAD_CONFIG"
echo "  plan_blocks=$NUM_PLAN_BLOCKS"
echo "  waypoints_per_block=$PLAN_WAYPOINTS"
echo "  video_frames=$NUM_VIDEO_FRAMES"
echo "  flow_tokens_per_block=$PLAN_ACTION_HORIZON"
echo "  internal_action_registers_per_block=$NUM_ACTION_PER_BLOCK"
echo "  global_batch=$((NUM_GPUS * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  echo "Preflight checks passed; training was not started."
  exit 0
fi

"$TORCHRUN_BIN" --nproc_per_node "$NUM_GPUS" --standalone \
  groot/vla/experiment/experiment.py \
  report_to="$REPORT_TO" \
  data=dreamzero/mobilemanibench_multiblock_plan \
  mobilemanibench_data_root="$MOBILEMANIBENCH_DATA_ROOT" \
  mobilemanibench_plan_stats_path="$PLAN_STATS_PATH" \
  wandb_project="$WANDB_PROJECT" \
  train_architecture=lora \
  num_views=2 \
  model=dreamzero/vla \
  model/dreamzero/action_head="$ACTION_HEAD_CONFIG" \
  model/dreamzero/transform=mobile_plan_multiblock_cotrain \
  num_frame_per_block=2 \
  num_action_per_block="$NUM_ACTION_PER_BLOCK" \
  num_state_per_block=1 \
  max_chunk_size=4 \
  frame_seqlen=50 \
  seed=42 \
  training_args.learning_rate="$LEARNING_RATE" \
  training_args.lr_scheduler_type=cosine_with_min_lr \
  +training_args.lr_scheduler_kwargs.min_lr_rate=0.1 \
  training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
  training_args.warmup_ratio=0.05 \
  training_args.gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
  output_dir="$OUTPUT_DIR" \
  per_device_train_batch_size="$PER_DEVICE_BATCH_SIZE" \
  max_steps="$MAX_STEPS" \
  weight_decay=1e-5 \
  save_strategy=steps \
  save_steps="$SAVE_STEPS" \
  save_total_limit="$SAVE_TOTAL_LIMIT" \
  logging_steps=100 \
  do_eval=true \
  eval_strategy=steps \
  eval_steps="$EVAL_STEPS" \
  per_device_eval_batch_size=1 \
  max_eval_samples="$MAX_EVAL_SAMPLES" \
  upload_checkpoints=false \
  bf16=true \
  tf32=true \
  eval_bf16=true \
  dataloader_pin_memory=false \
  dataloader_num_workers=1 \
  image_resolution_width=320 \
  image_resolution_height=160 \
  save_lora_only=true \
  dit_version="$WAN22_CKPT_DIR" \
  text_encoder_pretrained_path="$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
  image_encoder_pretrained_path="$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  vae_pretrained_path="$WAN22_CKPT_DIR/Wan2.2_VAE.pth" \
  tokenizer_path="$TOKENIZER_DIR" \
  pretrained_model_path=null \
  "$@"
