#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
DREAMZERO_ENV=${DREAMZERO_ENV:-"/mnt/yihao/envs/dreamzero"}
PYTHON_BIN="$DREAMZERO_ENV/bin/python"

export PATH="$DREAMZERO_ENV/bin:$PATH"
export NO_ALBUMENTATIONS_UPDATE=1
export DS_ACCELERATOR=${DS_ACCELERATOR:-cuda}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/dreamzero-matplotlib}

MOBILEMANIBENCH_DATA_ROOT=${MOBILEMANIBENCH_DATA_ROOT:-"/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_smoke_v2/g1"}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to a trained VGGT checkpoint directory}
OUTPUT=${OUTPUT:-"$CHECKPOINT/validation_metrics.json"}
VISUALIZATION_ROOT=${VISUALIZATION_ROOT:-"$CHECKPOINT"}
MAX_SAMPLES=${MAX_SAMPLES:-100}
MAX_VISUALIZATIONS=${MAX_VISUALIZATIONS:-4}
BATCH_SIZE=${BATCH_SIZE:-1}
SPLIT=${SPLIT:-val}
VIDEO_DELTA_INDICES=${VIDEO_DELTA_INDICES:-"0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32"}

cd "$REPO_ROOT"
"$PYTHON_BIN" scripts/eval/validate_vggt_3d_wam.py \
  --dataset-root "$MOBILEMANIBENCH_DATA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --visualization-root "$VISUALIZATION_ROOT" \
  --max-visualizations "$MAX_VISUALIZATIONS" \
  --max-samples "$MAX_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --split "$SPLIT" \
  --video-delta-indices "$VIDEO_DELTA_INDICES"
