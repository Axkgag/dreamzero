#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
DREAMZERO_ENV=${DREAMZERO_ENV:-"/mnt/yihao/envs/dreamzero"}
TORCHRUN_BIN="$DREAMZERO_ENV/bin/torchrun"

export PATH="$DREAMZERO_ENV/bin:$PATH"
export HYDRA_FULL_ERROR=1
export NO_ALBUMENTATIONS_UPDATE=1
export DS_ACCELERATOR=${DS_ACCELERATOR:-cuda}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/dreamzero-matplotlib}
export TORCH_HOME=${TORCH_HOME:-/mnt/yihao/.cache/torch}

DEFAULT_DATA_ROOT="/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_g1_5tasks/g1"
DEFAULT_VGGT_CHECKPOINT_PATH="/mnt/yihao/codes/ReconDrive/checkpoints/model.pt"
DEFAULT_INIT_CHECKPOINT=""
DEFAULT_NUM_GPUS=8
DEFAULT_MAX_STEPS=30000
DEFAULT_SAVE_STEPS=5000
DEFAULT_EVAL_STEPS=5000
DEFAULT_PER_DEVICE_BATCH_SIZE=1
DEFAULT_HEAD_LEARNING_RATE=5e-5
DEFAULT_BACKBONE_LEARNING_RATE=2e-5
DEFAULT_WARMUP_RATIO=0.01
DEFAULT_MIN_LR_RATE=0.2
DEFAULT_MAX_EVAL_SAMPLES=256

RUN_ID=${RUN_ID:-"$(date +%Y%m%d_%H%M%S)"}
MOBILEMANIBENCH_DATA_ROOT=${MOBILEMANIBENCH_DATA_ROOT:-"$DEFAULT_DATA_ROOT"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/work_dirs/mobilemanibench_5tasks_vggt_v2_savefix"}
VGGT_CHECKPOINT_PATH=${VGGT_CHECKPOINT_PATH:-"$DEFAULT_VGGT_CHECKPOINT_PATH"}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-"$DEFAULT_INIT_CHECKPOINT"}
INIT_RANDOM=${INIT_RANDOM:-0}
NUM_GPUS=${NUM_GPUS:-"$DEFAULT_NUM_GPUS"}
MAX_STEPS=${MAX_STEPS:-"$DEFAULT_MAX_STEPS"}
SAVE_STEPS=${SAVE_STEPS:-"$DEFAULT_SAVE_STEPS"}
EVAL_STEPS=${EVAL_STEPS:-"$DEFAULT_EVAL_STEPS"}
HEAD_LEARNING_RATE=${HEAD_LEARNING_RATE:-"$DEFAULT_HEAD_LEARNING_RATE"}
BACKBONE_LEARNING_RATE=${BACKBONE_LEARNING_RATE:-"$DEFAULT_BACKBONE_LEARNING_RATE"}
WARMUP_RATIO=${WARMUP_RATIO:-"$DEFAULT_WARMUP_RATIO"}
MIN_LR_RATE=${MIN_LR_RATE:-"$DEFAULT_MIN_LR_RATE"}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-"$DEFAULT_MAX_EVAL_SAMPLES"}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-"$DEFAULT_PER_DEVICE_BATCH_SIZE"}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
REPORT_TO=${REPORT_TO:-wandb}
WANDB_MODE=${WANDB_MODE:-offline}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
export WANDB_MODE

cd "$REPO_ROOT"

if [ ! -x "$TORCHRUN_BIN" ]; then
  echo "ERROR: torchrun is missing or not executable: $TORCHRUN_BIN" >&2
  exit 1
fi

if ! "$DREAMZERO_ENV/bin/python" -c "import lpips" >/dev/null 2>&1; then
  echo "ERROR: lpips==0.1.4 is required by the VGGT visual loss." >&2
  echo "Install it with: $DREAMZERO_ENV/bin/pip install --no-deps lpips==0.1.4" >&2
  exit 1
fi

for required_file in \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/info.json" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/modality.json" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/extensions.json" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/calibration.json"; do
  if [ ! -f "$required_file" ]; then
    echo "ERROR: required dataset file is missing: $required_file" >&2
    exit 1
  fi
done

if [ "$INIT_RANDOM" != "1" ] && [ ! -f "$VGGT_CHECKPOINT_PATH" ]; then
  echo "ERROR: set VGGT_CHECKPOINT_PATH to a local checkpoint, or INIT_RANDOM=1." >&2
  exit 1
fi

INIT_CHECKPOINT_OVERRIDE=()
INIT_CHECKPOINT_DISPLAY="none"
if [ -n "${INIT_CHECKPOINT//[[:space:]]/}" ]; then
  if [ -d "$INIT_CHECKPOINT" ]; then
    if [ ! -f "$INIT_CHECKPOINT/model.safetensors" ]; then
      echo "ERROR: model.safetensors is missing from INIT_CHECKPOINT: $INIT_CHECKPOINT" >&2
      exit 1
    fi
  elif [ ! -f "$INIT_CHECKPOINT" ]; then
    echo "ERROR: INIT_CHECKPOINT does not exist: $INIT_CHECKPOINT" >&2
    exit 1
  fi
  INIT_CHECKPOINT_OVERRIDE=("init_checkpoint=$INIT_CHECKPOINT")
  INIT_CHECKPOINT_DISPLAY="$INIT_CHECKPOINT"
fi

echo "VGGT encoder-decoder configuration:"
echo "  data_root=$MOBILEMANIBENCH_DATA_ROOT"
echo "  output_dir=$OUTPUT_DIR"
echo "  checkpoint=${VGGT_CHECKPOINT_PATH:-random initialization}"
echo "  matching_init=$INIT_CHECKPOINT_DISPLAY"
echo "  num_gpus=$NUM_GPUS"
echo "  video_contract=33x160x320 -> 9x10x20 -> 33x160x320"
echo "  temporal_layout=frame0 + 8 chunks of 4 frames (shared by 2D/3D)"
echo "  temporal_window=4 (Wan-aligned source-frame chunks)"
echo "  video_decoder=256->192->128->96->64, learned 2x upsampling"
echo "  video_losses=Charbonnier + 0.1 LPIPS + 0.2 SSIM + 0.1 spatial-gradient + 0.1 temporal-difference"
echo "  metric_grid=B0-forward x[0,3] y[-2,2] z[-0.5,2], 8x12x8=768 tokens"
echo "  pointmap_decoder=40x80 ray rendering -> learned 80x160 refinement"
echo "  geometry_fusion=2-layer, 2-level, 8-head deformable cross-attention"
echo "  camera_optical=OpenCV rays -> Isaac +X-forward pose frame"
echo "  geometry_weight=0.4 x quality_weight=0.25 (effective 0.1 after warmup)"
echo "  dino=frozen, no LoRA, no_grad chunks of 4 images"
echo "  aggregator=rank-8 LoRA + activation checkpointing"
echo "  per_device_batch_size=$PER_DEVICE_BATCH_SIZE"
echo "  max_steps=$MAX_STEPS"
echo "  head_learning_rate=$HEAD_LEARNING_RATE"
echo "  backbone_lora_learning_rate=$BACKBONE_LEARNING_RATE"
echo "  scheduler=cosine_with_min_lr (warmup_ratio=$WARMUP_RATIO, min_lr_rate=$MIN_LR_RATE)"
echo "  eval_every=$EVAL_STEPS steps, max_eval_samples=$MAX_EVAL_SAMPLES"

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  echo "Preflight checks passed; training was not started."
  exit 0
fi

CHECKPOINT_OVERRIDE="model.config.vggt_checkpoint_path=$VGGT_CHECKPOINT_PATH"
if [ "$INIT_RANDOM" = "1" ]; then
  CHECKPOINT_OVERRIDE="model.config.vggt_checkpoint_path=null"
fi

"$TORCHRUN_BIN" --standalone --nproc_per_node "$NUM_GPUS" \
  groot/vla/experiment/vggt_3d_wam.py \
  mobilemanibench_data_root="$MOBILEMANIBENCH_DATA_ROOT" \
  output_dir="$OUTPUT_DIR" \
  "$CHECKPOINT_OVERRIDE" \
  "${INIT_CHECKPOINT_OVERRIDE[@]}" \
  model.config.init_random="$([ "$INIT_RANDOM" = "1" ] && echo true || echo false)" \
  max_steps="$MAX_STEPS" \
  save_steps="$SAVE_STEPS" \
  eval_steps="$EVAL_STEPS" \
  learning_rate="$HEAD_LEARNING_RATE" \
  backbone_learning_rate="$BACKBONE_LEARNING_RATE" \
  warmup_ratio="$WARMUP_RATIO" \
  max_eval_samples="$MAX_EVAL_SAMPLES" \
  per_device_train_batch_size="$PER_DEVICE_BATCH_SIZE" \
  gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
  report_to="$REPORT_TO" \
  training_args.lr_scheduler_type=cosine_with_min_lr \
  ++training_args.lr_scheduler_kwargs.min_lr_rate="$MIN_LR_RATE"
