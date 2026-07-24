#!/bin/bash
# Offline evaluation launcher for MobileManiBench dual Base/Manipulator plans.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
DREAMZERO_ENV=${DREAMZERO_ENV:-"/mnt/yihao/envs/dreamzero"}
PYTHON_BIN="$DREAMZERO_ENV/bin/python"
TORCHRUN_BIN="$DREAMZERO_ENV/bin/torchrun"

export PATH="$DREAMZERO_ENV/bin:$PATH"
export NO_ALBUMENTATIONS_UPDATE=1

cd "$REPO_ROOT"

DEFAULT_RUN_DIR="$REPO_ROOT/work_dirs/mobilemanibench_dual_plan_g1_20260723_220248"
DEFAULT_DATA_ROOT="/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_smoke_v2/g1"

RUN_DIR=${RUN_DIR:-"$DEFAULT_RUN_DIR"}
DATA_ROOT=${DATA_ROOT:-"$DEFAULT_DATA_ROOT"}
SPLIT=${SPLIT:-train}
NUM_GPUS=${NUM_GPUS:-2}
EVAL_GPUS=${EVAL_GPUS:-"0,1"}
MAX_SAMPLES=${MAX_SAMPLES:-0}
SAMPLE_STRIDE=${SAMPLE_STRIDE:-1}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-16}
SEED=${SEED:-1140}
OUTPUT_DIR=${OUTPUT_DIR:-""}
INSPECT_ONLY=${INSPECT_ONLY:-0}

if [ ! -x "$PYTHON_BIN" ] || [ ! -x "$TORCHRUN_BIN" ]; then
  echo "ERROR: DreamZero Python environment is incomplete: $DREAMZERO_ENV" >&2
  exit 1
fi
if [ ! -d "$DATA_ROOT" ]; then
  echo "ERROR: dataset root does not exist: $DATA_ROOT" >&2
  exit 1
fi

if [ "$INSPECT_ONLY" = "1" ]; then
  exec "$PYTHON_BIN" scripts/eval/evaluate_mobilemanibench_plan.py \
    --dataset-root "$DATA_ROOT" \
    --split "$SPLIT" \
    --inspect-only
fi

if [ "$NUM_GPUS" -ne 1 ] && [ "$NUM_GPUS" -ne 2 ]; then
  echo "ERROR: NUM_GPUS must be 1 or 2 for DreamZero inference." >&2
  exit 1
fi

if [ -z "${CHECKPOINT:-}" ]; then
  if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: run directory does not exist: $RUN_DIR" >&2
    exit 1
  fi
  CHECKPOINT="$(
    find "$RUN_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -print \
      | sort -V \
      | tail -n 1
  )"
fi
if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT/model.safetensors" ]; then
  echo "ERROR: no usable checkpoint with model.safetensors was found." >&2
  echo "Set CHECKPOINT=/absolute/path/to/checkpoint-N explicitly." >&2
  exit 1
fi

ARGS=(
  --checkpoint "$CHECKPOINT"
  --dataset-root "$DATA_ROOT"
  --split "$SPLIT"
  --max-samples "$MAX_SAMPLES"
  --sample-stride "$SAMPLE_STRIDE"
  --seed "$SEED"
  --num-inference-steps "$NUM_INFERENCE_STEPS"
)
if [ -n "$OUTPUT_DIR" ]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi

echo "MobileManiBench dual-plan evaluation:"
echo "  checkpoint=$CHECKPOINT"
echo "  data_root=$DATA_ROOT"
echo "  split=$SPLIT"
echo "  eval_gpus=$EVAL_GPUS"
echo "  max_samples=$MAX_SAMPLES"
echo "  sample_stride=$SAMPLE_STRIDE"
echo "  num_inference_steps=$NUM_INFERENCE_STEPS"
echo "  output_dir=${OUTPUT_DIR:-$CHECKPOINT/mobile_plan_eval_$SPLIT}"

export CUDA_VISIBLE_DEVICES="$EVAL_GPUS"
exec "$TORCHRUN_BIN" \
  --standalone \
  --nproc_per_node="$NUM_GPUS" \
  scripts/eval/evaluate_mobilemanibench_plan.py \
  "${ARGS[@]}"
