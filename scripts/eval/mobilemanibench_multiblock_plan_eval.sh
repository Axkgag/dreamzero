#!/bin/bash
# Offline evaluator for MobileManiBench multiblock Base/Manipulator plans.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
DREAMZERO_ENV=${DREAMZERO_ENV:-"/mnt/yihao/envs/dreamzero"}
PYTHON_BIN="$DREAMZERO_ENV/bin/python"
TORCHRUN_BIN="$DREAMZERO_ENV/bin/torchrun"

export PATH="$DREAMZERO_ENV/bin:$PATH"
export NO_ALBUMENTATIONS_UPDATE=1
cd "$REPO_ROOT"

DEFAULT_RUN_DIR="$REPO_ROOT/work_dirs/mobilemanibench_g1_5tasks_wan22_5b_multiblock_k2_wp2"
DEFAULT_DATA_ROOT="/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_g1_5tasks/g1"

RUN_DIR=${RUN_DIR:-"$DEFAULT_RUN_DIR"}
DATA_ROOT=${DATA_ROOT:-"$DEFAULT_DATA_ROOT"}
SPLIT=${SPLIT:-val}
MODE=${MODE:-gt_history_cached}
NUM_GPUS=${NUM_GPUS:-2}
EVAL_GPUS=${EVAL_GPUS:-"2,3"}
MAX_SAMPLES=${MAX_SAMPLES:-1024}
SAMPLE_STRIDE=${SAMPLE_STRIDE:-1}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-16}
NUM_ROLLOUT_BLOCKS=${NUM_ROLLOUT_BLOCKS:-4}
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

ARGS=(
  --dataset-root "$DATA_ROOT"
  --split "$SPLIT"
  --mode "$MODE"
  --max-samples "$MAX_SAMPLES"
  --sample-stride "$SAMPLE_STRIDE"
  --seed "$SEED"
  --num-inference-steps "$NUM_INFERENCE_STEPS"
  --num-rollout-blocks "$NUM_ROLLOUT_BLOCKS"
)
if [ -n "$OUTPUT_DIR" ]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi

if [ "$INSPECT_ONLY" = "1" ]; then
  if [ -n "${CHECKPOINT:-}" ]; then
    ARGS+=(--checkpoint "$CHECKPOINT")
  fi
  exec "$PYTHON_BIN" scripts/eval/evaluate_mobilemanibench_multiblock_plan.py \
    "${ARGS[@]}" --inspect-only
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
if [ -z "$CHECKPOINT" ] || {
  [ ! -f "$CHECKPOINT/model.safetensors" ] &&
  [ ! -f "$CHECKPOINT/model.safetensors.index.json" ];
}; then
  echo "ERROR: no usable checkpoint was found." >&2
  echo "Set CHECKPOINT=/absolute/path/to/checkpoint-N explicitly." >&2
  exit 1
fi
ARGS+=(--checkpoint "$CHECKPOINT")

echo "MobileManiBench multiblock plan evaluation:"
echo "  checkpoint=$CHECKPOINT"
echo "  data_root=$DATA_ROOT"
echo "  split=$SPLIT"
echo "  mode=$MODE"
echo "  eval_gpus=$EVAL_GPUS"
echo "  max_root_windows=$MAX_SAMPLES"
echo "  sample_stride=$SAMPLE_STRIDE"
echo "  rollout_blocks=$NUM_ROLLOUT_BLOCKS"
echo "  num_inference_steps=$NUM_INFERENCE_STEPS"
echo "  output_dir=${OUTPUT_DIR:-$CHECKPOINT/mobile_multiblock_plan_eval_${SPLIT}_${MODE}}"

export CUDA_VISIBLE_DEVICES="$EVAL_GPUS"
exec "$TORCHRUN_BIN" \
  --standalone \
  --nproc_per_node="$NUM_GPUS" \
  scripts/eval/evaluate_mobilemanibench_multiblock_plan.py \
  "${ARGS[@]}"
