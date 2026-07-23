#!/bin/bash
# DreamZero MobileManiBench G1/XHand baseline training entry point.
#
# This script never downloads checkpoints. Point MOBILEMANIBENCH_DATA_ROOT at
# exactly one converted robot root (g1 or xhand), then provide existing model
# assets through WAN_CKPT_DIR, TOKENIZER_DIR and PRETRAINED_MODEL_PATH.

set -euo pipefail
export HYDRA_FULL_ERROR=1

export WAN_CKPT_DIR=/mnt/yihao/codes/checkpoints/Wan2.1-I2V-14B-480P
export TOKENIZER_DIR=/mnt/yihao/codes/checkpoints/umt5-xxl
export PRETRAINED_MODEL_PATH=/mnt/yihao/codes/checkpoints/DreamZero-AgiBot

export WANDB_MODE=offline

MOBILEMANIBENCH_DATA_ROOT=${MOBILEMANIBENCH_DATA_ROOT:-"/mnt/yihao/datasets/MobileManiBench/MobileManipVLA_dreamzero_smoke_v2/g1"}
OUTPUT_DIR=${OUTPUT_DIR:-"./work_dirs/dreamzero_mobilemanibench_overfit"}

if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-1}

WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-"./checkpoints/DreamZero-AgiBot"}

for required_file in \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/info.json" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/modality.json" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/embodiment.json" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/stats.json" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/tasks.jsonl" \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/episodes.jsonl"; do
  if [ ! -f "$required_file" ]; then
    echo "ERROR: required dataset file is missing: $required_file" >&2
    exit 1
  fi
done

if ! grep -Eq '"embodiment_tag"[[:space:]]*:[[:space:]]*"xdof"' \
  "$MOBILEMANIBENCH_DATA_ROOT/meta/embodiment.json"; then
  echo "ERROR: expected embodiment_tag=xdof in meta/embodiment.json" >&2
  exit 1
fi

for required_dir in "$WAN_CKPT_DIR" "$TOKENIZER_DIR" "$PRETRAINED_MODEL_PATH"; do
  if [ ! -d "$required_dir" ]; then
    echo "ERROR: checkpoint directory is missing: $required_dir" >&2
    echo "This script does not download checkpoints automatically." >&2
    exit 1
  fi
done

torchrun --nproc_per_node "$NUM_GPUS" --standalone \
  groot/vla/experiment/experiment.py \
  report_to=wandb \
  data=dreamzero/mobilemanibench_relative \
  mobilemanibench_data_root="$MOBILEMANIBENCH_DATA_ROOT" \
  wandb_project=dreamzero \
  train_architecture=lora \
  num_frames=33 \
  action_horizon=24 \
  num_views=2 \
  model=dreamzero/vla \
  model/dreamzero/action_head=wan_flow_matching_action_tf \
  model/dreamzero/transform=dreamzero_cotrain \
  num_frame_per_block=2 \
  num_action_per_block=24 \
  num_state_per_block=1 \
  seed=42 \
  training_args.learning_rate=1e-5 \
  training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
  save_steps=100 \
  training_args.warmup_ratio=0.05 \
  output_dir="$OUTPUT_DIR" \
  per_device_train_batch_size=1 \
  max_steps=1000 \
  weight_decay=1e-5 \
  save_total_limit=5 \
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
  ++action_head_cfg.config.defer_lora_injection=true
