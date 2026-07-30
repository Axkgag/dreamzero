#!/bin/bash
# Fixed-diffusion-timestep action-flow diagnostic for MobileManiBench.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
DREAMZERO_ENV=${DREAMZERO_ENV:-"/mnt/yihao/envs/dreamzero"}
TORCHRUN_BIN="$DREAMZERO_ENV/bin/torchrun"

export PATH="$DREAMZERO_ENV/bin:$PATH"
export NO_ALBUMENTATIONS_UPDATE=1

cd "$REPO_ROOT"

DEFAULT_CHECKPOINT="$REPO_ROOT/work_dirs/mobilemanibench_dual_plan_g1_single_anchor_ep001_frame0038/checkpoint-500"
DEFAULT_DATA_ROOT="/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_smoke_v2_single_ep001_frame0038/g1"

CHECKPOINT=${CHECKPOINT:-"$DEFAULT_CHECKPOINT"}
DATA_ROOT=${DATA_ROOT:-"$DEFAULT_DATA_ROOT"}
EPISODE_INDEX=${EPISODE_INDEX:-1}
FRAME_INDEX=${FRAME_INDEX:-38}
NUM_GPUS=${NUM_GPUS:-2}
EVAL_GPUS=${EVAL_GPUS:-"0,1"}
TIMESTEPS=${TIMESTEPS:-"50 100 250 500 750 900"}
SEEDS=${SEEDS:-"1140 1141 1142 1143 1144"}
OUTPUT_DIR=${OUTPUT_DIR:-""}

if [ ! -x "$TORCHRUN_BIN" ]; then
  echo "ERROR: torchrun not found: $TORCHRUN_BIN" >&2
  exit 1
fi
if [ ! -f "$CHECKPOINT/model.safetensors" ]; then
  echo "ERROR: checkpoint model.safetensors not found: $CHECKPOINT" >&2
  exit 1
fi
if [ ! -d "$DATA_ROOT" ]; then
  echo "ERROR: dataset root does not exist: $DATA_ROOT" >&2
  exit 1
fi
if [ "$NUM_GPUS" -ne 1 ] && [ "$NUM_GPUS" -ne 2 ]; then
  echo "ERROR: NUM_GPUS must be 1 or 2." >&2
  exit 1
fi

read -r -a TIMESTEP_ARGS <<< "$TIMESTEPS"
read -r -a SEED_ARGS <<< "$SEEDS"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --dataset-root "$DATA_ROOT"
  --episode-index "$EPISODE_INDEX"
  --frame-index "$FRAME_INDEX"
  --timesteps "${TIMESTEP_ARGS[@]}"
  --seeds "${SEED_ARGS[@]}"
)
if [ -n "$OUTPUT_DIR" ]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi

echo "MobileManiBench fixed-timestep flow diagnostic:"
echo "  checkpoint=$CHECKPOINT"
echo "  data_root=$DATA_ROOT"
echo "  anchor=episode_${EPISODE_INDEX}/frame_${FRAME_INDEX}"
echo "  timesteps=$TIMESTEPS"
echo "  seeds=$SEEDS"
echo "  eval_gpus=$EVAL_GPUS"
echo "  output_dir=${OUTPUT_DIR:-$CHECKPOINT/mobile_plan_fixed_timestep_flow_ep$(printf '%03d' "$EPISODE_INDEX")_frame$(printf '%04d' "$FRAME_INDEX")}"

export CUDA_VISIBLE_DEVICES="$EVAL_GPUS"
exec "$TORCHRUN_BIN" \
  --standalone \
  --nproc_per_node="$NUM_GPUS" \
  scripts/eval/evaluate_mobilemanibench_fixed_timestep_flow.py \
  "${ARGS[@]}"
