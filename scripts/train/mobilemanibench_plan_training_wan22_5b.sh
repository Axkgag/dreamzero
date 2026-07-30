#!/bin/bash
# Full-data MobileManiBench dual-plan LoRA baseline on Wan2.2-TI2V-5B.

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
DEFAULT_OUTPUT_DIR="$REPO_ROOT/work_dirs/mobilemanibench_g1_5tasks_wan22_5b_baseline"
DEFAULT_NUM_GPUS=8
PER_DEVICE_BATCH_SIZE=32
DEFAULT_MAX_STEPS=10000
DEFAULT_SAVE_STEPS=2000
DEFAULT_LOGGING_STEPS=100
DEFAULT_EVAL_STEPS=2000
DEFAULT_LEARNING_RATE="1e-5"
DEFAULT_WARMUP_RATIO="0.05"
DEFAULT_WEIGHT_DECAY="1e-5"

MOBILEMANIBENCH_DATA_ROOT=${MOBILEMANIBENCH_DATA_ROOT:-"$DEFAULT_DATA_ROOT"}
OUTPUT_DIR=${OUTPUT_DIR:-"$DEFAULT_OUTPUT_DIR"}
WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"/mnt/yihao/codes/checkpoints/Wan2.2-TI2V-5B"}
IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-"/mnt/yihao/codes/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"/mnt/yihao/codes/checkpoints/umt5-xxl"}
WANDB_MODE=${WANDB_MODE:-offline}
REPORT_TO=${REPORT_TO:-wandb}
WANDB_PROJECT=${WANDB_PROJECT:-dreamzero-mobile-plan}
NUM_GPUS=${NUM_GPUS:-"$DEFAULT_NUM_GPUS"}
MAX_STEPS=${MAX_STEPS:-"$DEFAULT_MAX_STEPS"}
SAVE_STEPS=${SAVE_STEPS:-"$DEFAULT_SAVE_STEPS"}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-10}
LOGGING_STEPS=${LOGGING_STEPS:-"$DEFAULT_LOGGING_STEPS"}
DO_EVAL=${DO_EVAL:-true}
EVAL_STEPS=${EVAL_STEPS:-"$DEFAULT_EVAL_STEPS"}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-1024}
LEARNING_RATE=${LEARNING_RATE:-"$DEFAULT_LEARNING_RATE"}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
WARMUP_RATIO=${WARMUP_RATIO:-"$DEFAULT_WARMUP_RATIO"}
WEIGHT_DECAY=${WEIGHT_DECAY:-"$DEFAULT_WEIGHT_DECAY"}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
export WANDB_MODE

if [ ! -x "$TORCHRUN_BIN" ]; then
  echo "ERROR: torchrun is missing or not executable: $TORCHRUN_BIN" >&2
  exit 1
fi
if [ "$SAVE_TOTAL_LIMIT" -lt 5 ]; then
  echo "ERROR: SAVE_TOTAL_LIMIT must be >= 5 for standardized evaluation." >&2
  exit 1
fi

for required_file in \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/info.json" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/modality.json" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/embodiment.json" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/stats.json" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/plan_stats.json" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/plan_splits.json" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/robot_schema.json" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/extensions.json" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/tasks.jsonl" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/episodes.jsonl"; do
  if [ ! -f "$required_file" ]; then
    echo "ERROR: required dataset file is missing: $required_file" >&2
    exit 1
  fi
done

PLAN_STATS_FIT_SPLIT=$(
  "$DREAMZERO_ENV/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["fit_split"])' \
    "$MOBILEMANIBENCH_DATA_ROOT/meta/plan_stats.json"
)
if [ "$PLAN_STATS_FIT_SPLIT" != "train" ]; then
  echo "ERROR: plan_stats.json must be fit on the train split only; got: $PLAN_STATS_FIT_SPLIT" >&2
  echo "Regenerate it with prepare_mobilemanibench_plan_metadata.py --split train --force." >&2
  exit 1
fi

for required_file in \
  "$WAN22_CKPT_DIR/diffusion_pytorch_model.safetensors.index.json" \
  "$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
  "$WAN22_CKPT_DIR/Wan2.2_VAE.pth" \
  "$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"; do
  if [ ! -f "$required_file" ]; then
    echo "ERROR: required checkpoint file is missing: $required_file" >&2
    exit 1
  fi
done
if [ ! -d "$TOKENIZER_DIR" ]; then
  echo "ERROR: tokenizer directory is missing: $TOKENIZER_DIR" >&2
  exit 1
fi

echo "MobileManiBench Wan2.2-5B dual-plan baseline:"
echo "  data_root=$MOBILEMANIBENCH_DATA_ROOT"
echo "  output_dir=$OUTPUT_DIR"
echo "  backbone=$WAN22_CKPT_DIR"
echo "  token_layout=6 base + 6 manipulator"
echo "  plan_offsets=1,4,8,12,16,24"
echo "  num_gpus=$NUM_GPUS"
echo "  per_device_batch_size=$PER_DEVICE_BATCH_SIZE"
echo "  global_batch_size=$((NUM_GPUS * PER_DEVICE_BATCH_SIZE))"
echo "  max_steps=$MAX_STEPS"
echo "  learning_rate=$LEARNING_RATE"
echo "  scheduler=cosine"
echo "  warmup_ratio=$WARMUP_RATIO"
echo "  save_steps=$SAVE_STEPS"
echo "  do_eval=$DO_EVAL"
echo "  eval_steps=$EVAL_STEPS"
echo "  max_eval_samples=$MAX_EVAL_SAMPLES"
echo "  wandb=$REPORT_TO (mode=$WANDB_MODE)"

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  echo "Preflight checks passed; training was not started."
  exit 0
fi

# Wan2.2-5B has no CLIP file, so the image encoder is loaded from Wan2.1.
# The raw Wan2.2 video weights initialize the 5B DiT; the dual-plan action
# encoder/decoder and type/time embeddings are trained for this baseline.
"$TORCHRUN_BIN" --nproc_per_node "$NUM_GPUS" --standalone \
  groot/vla/experiment/experiment.py \
  report_to="$REPORT_TO" \
  data=dreamzero/mobilemanibench_plan \
  mobilemanibench_data_root="$MOBILEMANIBENCH_DATA_ROOT" \
  wandb_project="$WANDB_PROJECT" \
  train_architecture=lora \
  num_frames=33 \
  action_horizon=12 \
  num_views=2 \
  model=dreamzero/vla \
  model/dreamzero/action_head=mobile_plan_flow_matching_wan22 \
  model/dreamzero/transform=mobile_plan_cotrain \
  num_frame_per_block=8 \
  num_action_per_block=12 \
  num_state_per_block=1 \
  seed=42 \
  training_args.learning_rate="$LEARNING_RATE" \
  training_args.lr_scheduler_type=cosine_with_min_lr \
  +training_args.lr_scheduler_kwargs.min_lr_rate=0.1 \
  training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
  training_args.warmup_ratio="$WARMUP_RATIO" \
  output_dir="$OUTPUT_DIR" \
  per_device_train_batch_size="$PER_DEVICE_BATCH_SIZE" \
  max_steps="$MAX_STEPS" \
  weight_decay="$WEIGHT_DECAY" \
  save_strategy=steps \
  save_steps="$SAVE_STEPS" \
  save_total_limit="$SAVE_TOTAL_LIMIT" \
  logging_steps="$LOGGING_STEPS" \
  do_eval="$DO_EVAL" \
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
  max_chunk_size=4 \
  frame_seqlen=50 \
  dit_version="$WAN22_CKPT_DIR" \
  text_encoder_pretrained_path="$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
  image_encoder_pretrained_path="$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  vae_pretrained_path="$WAN22_CKPT_DIR/Wan2.2_VAE.pth" \
  tokenizer_path="$TOKENIZER_DIR" \
  pretrained_model_path=null
