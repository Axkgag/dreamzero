#!/bin/bash
# MobileManiBench plan training with independently selectable architecture
# and physical-loss profile.

set -euo pipefail

# Resolve the repository and Python environment, so the script can be launched
# from any working directory without activating conda/venv first.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
DREAMZERO_ENV=${DREAMZERO_ENV:-"/mnt/yihao/envs/dreamzero"}
TORCHRUN_BIN="$DREAMZERO_ENV/bin/torchrun"
export PATH="$DREAMZERO_ENV/bin:$PATH"
export HYDRA_FULL_ERROR=1
export NO_ALBUMENTATIONS_UPDATE=1

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Defaults normally edited for a new experiment.
# The server currently contains only the 2-episode smoke conversion, so the
# safe default remains a G1 overfit run rather than pretending to be full-data
# training. Point DEFAULT_DATA_ROOT at a converted full G1/XHand root once it
# exists, then increase DEFAULT_MAX_STEPS/SAVE_STEPS as appropriate.
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT="/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_smoke_v2/g1"
DEFAULT_NUM_GPUS=8
DEFAULT_MAX_STEPS=5000
DEFAULT_SAVE_STEPS=500
DEFAULT_LOGGING_STEPS=1
DEFAULT_LEARNING_RATE="1e-5"
DEFAULT_PER_DEVICE_BATCH_SIZE=1
DEFAULT_WARMUP_RATIO="0.05"
DEFAULT_WEIGHT_DECAY="1e-5"

# Architecture options:
#   dual_plan   - 6 Base + 6 Manipulator noisy flow tokens (12 total).
#   clean_prior - 6 clean Base Prior + 12 noisy flow tokens (18 internal).
MOBILE_PLAN_ARCHITECTURE=${MOBILE_PLAN_ARCHITECTURE:-clean_prior}

# Loss-profile options:
#   flow_only            - flow objectives only; clean_prior still retains
#                          its direct Base Prior supervision.
#   physical_consistency - adds physical plan-component and Base/EEF
#                          consistency objectives.
MOBILE_PLAN_LOSS_PROFILE=${MOBILE_PLAN_LOSS_PROFILE:-physical_consistency}

case "$MOBILE_PLAN_ARCHITECTURE:$MOBILE_PLAN_LOSS_PROFILE" in
  dual_plan:flow_only)
    ACTION_HEAD_CONFIG=mobile_plan_flow_matching
    TOKEN_LAYOUT="6 Base + 6 Manipulator noisy tokens (12 total)"
    ;;
  dual_plan:physical_consistency)
    ACTION_HEAD_CONFIG=mobile_plan_flow_matching_physical_consistency
    TOKEN_LAYOUT="6 Base + 6 Manipulator noisy tokens (12 total)"
    ;;
  clean_prior:flow_only)
    ACTION_HEAD_CONFIG=mobile_plan_flow_matching_clean_prior
    TOKEN_LAYOUT="6 clean Base Prior + 12 noisy flow tokens (18 internal)"
    ;;
  clean_prior:physical_consistency)
    ACTION_HEAD_CONFIG=mobile_plan_flow_matching_clean_prior_physical_consistency
    TOKEN_LAYOUT="6 clean Base Prior + 12 noisy flow tokens (18 internal)"
    ;;
  *)
    echo "ERROR: unsupported MobileManiBench plan configuration:" >&2
    echo "  MOBILE_PLAN_ARCHITECTURE=$MOBILE_PLAN_ARCHITECTURE" >&2
    echo "  MOBILE_PLAN_LOSS_PROFILE=$MOBILE_PLAN_LOSS_PROFILE" >&2
    echo "Architecture options: dual_plan | clean_prior" >&2
    echo "Loss-profile options: flow_only | physical_consistency" >&2
    exit 2
    ;;
esac

RUN_ID=${RUN_ID:-"$(date +%Y%m%d_%H%M%S)"}
MOBILEMANIBENCH_DATA_ROOT=${MOBILEMANIBENCH_DATA_ROOT:-"$DEFAULT_DATA_ROOT"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/work_dirs/mobilemanibench_${MOBILE_PLAN_ARCHITECTURE}_${MOBILE_PLAN_LOSS_PROFILE}_${RUN_ID}"}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"/mnt/yihao/codes/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"/mnt/yihao/codes/checkpoints/umt5-xxl"}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-"/mnt/yihao/codes/checkpoints/DreamZero-AgiBot"}
WANDB_MODE=${WANDB_MODE:-offline}
REPORT_TO=${REPORT_TO:-wandb}
WANDB_PROJECT=${WANDB_PROJECT:-dreamzero-mobile-plan}
NUM_GPUS=${NUM_GPUS:-"$DEFAULT_NUM_GPUS"}
MAX_STEPS=${MAX_STEPS:-"$DEFAULT_MAX_STEPS"}
SAVE_STEPS=${SAVE_STEPS:-"$DEFAULT_SAVE_STEPS"}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}
LOGGING_STEPS=${LOGGING_STEPS:-"$DEFAULT_LOGGING_STEPS"}
LEARNING_RATE=${LEARNING_RATE:-"$DEFAULT_LEARNING_RATE"}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-"$DEFAULT_PER_DEVICE_BATCH_SIZE"}
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
  "$MOBILEMANIBENCH_DATA_ROOT/meta/robot_schema.json" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/extensions.json" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/tasks.jsonl" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/episodes.jsonl"; do
  if [ ! -f "$required_file" ]; then
    echo "ERROR: required dataset file is missing: $required_file" >&2
    exit 1
  fi
done

for required_dir in "$WAN_CKPT_DIR" "$TOKENIZER_DIR" "$PRETRAINED_MODEL_PATH"; do
  if [ ! -d "$required_dir" ]; then
    echo "ERROR: checkpoint directory is missing: $required_dir" >&2
    echo "This script does not download checkpoints automatically." >&2
    exit 1
  fi
done

echo "MobileManiBench plan configuration:"
echo "  data_root=$MOBILEMANIBENCH_DATA_ROOT"
echo "  output_dir=$OUTPUT_DIR"
echo "  architecture=$MOBILE_PLAN_ARCHITECTURE"
echo "  loss_profile=$MOBILE_PLAN_LOSS_PROFILE"
echo "  token_layout=$TOKEN_LAYOUT"
echo "  action_head_config=$ACTION_HEAD_CONFIG"
echo "  plan_offsets=1,4,8,12,16,24"
echo "  num_gpus=$NUM_GPUS"
echo "  max_steps=$MAX_STEPS"
echo "  learning_rate=$LEARNING_RATE"
echo "  per_device_batch_size=$PER_DEVICE_BATCH_SIZE"
echo "  wandb=$REPORT_TO (mode=$WANDB_MODE)"

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  echo "Preflight checks passed; training was not started."
  exit 0
fi

# The 33 RGB frames become 9 VAE frames: 1 condition + 8 future frames.
# One 8-frame video block corresponds to one 12-token dual-plan window.
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
  model/dreamzero/action_head="$ACTION_HEAD_CONFIG" \
  model/dreamzero/transform=mobile_plan_cotrain \
  num_frame_per_block=8 \
  num_action_per_block=12 \
  num_state_per_block=1 \
  seed=42 \
  training_args.learning_rate="$LEARNING_RATE" \
  training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
  save_steps="$SAVE_STEPS" \
  logging_steps="$LOGGING_STEPS" \
  training_args.warmup_ratio="$WARMUP_RATIO" \
  output_dir="$OUTPUT_DIR" \
  per_device_train_batch_size="$PER_DEVICE_BATCH_SIZE" \
  max_steps="$MAX_STEPS" \
  weight_decay="$WEIGHT_DECAY" \
  save_total_limit="$SAVE_TOTAL_LIMIT" \
  upload_checkpoints=false \
  bf16=true \
  tf32=true \
  eval_bf16=true \
  dataloader_pin_memory=false \
  dataloader_num_workers=1 \
  image_resolution_width=320 \
  image_resolution_height=176 \
  save_lora_only=true \
  max_chunk_size=4 \
  frame_seqlen=880 \
  save_strategy=steps \
  dit_version="$WAN_CKPT_DIR" \
  text_encoder_pretrained_path="$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
  image_encoder_pretrained_path="$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  vae_pretrained_path="$WAN_CKPT_DIR/Wan2.1_VAE.pth" \
  tokenizer_path="$TOKENIZER_DIR" \
  pretrained_model_path="$PRETRAINED_MODEL_PATH" \
  ++action_head_cfg.config.skip_component_loading=true \
  ++action_head_cfg.config.defer_lora_injection=true \
  "$@"
