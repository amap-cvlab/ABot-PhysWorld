#!/bin/bash
# ============================================================
# ABot-PhysWorld - Full-Parameter SFT Training Script
# ============================================================
# This script runs full-parameter SFT training on Wan2.1-I2V-14B-480P
# using DeepSpeed ZeRO-2 via Accelerate.
#
# Prerequisites:
#   1. Install dependencies (see training/README.md)
#   2. Prepare dataset in JSONL format (see training/assets/demo_train.jsonl)
#   3. Ensure sufficient GPU VRAM (>= 60GB per GPU recommended)
#
# Usage:
#   # Default 8-GPU training
#   bash run_train.sh
#
#   # With encoded cache
#   ENCODED_CACHE_DIR=./cache bash run_train.sh
#
#   # Custom dataset
#   DATASET_BASE_PATH=/data/my_dataset \
#   DATASET_METADATA_PATH=/data/my_dataset/metadata.jsonl \
#   bash run_train.sh
# ============================================================

set -e

# ==================== User Configuration ====================
# Number of GPUs (must match accelerate config num_processes)
NUM_GPUS=${NUM_GPUS:-8}

# Dataset paths
DATASET_BASE_PATH=${DATASET_BASE_PATH:-"./data"}
DATASET_METADATA_PATH=${DATASET_METADATA_PATH:-"./data/metadata.jsonl"}

# Output directory for checkpoints
OUTPUT_PATH=${OUTPUT_PATH:-"./outputs/sft_training"}

# Training hyperparameters
LEARNING_RATE=${LEARNING_RATE:-1e-5}
NUM_EPOCHS=${NUM_EPOCHS:-1}
SAVE_STEPS=${SAVE_STEPS:-200}

# Video resolution and frame count
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-832}
NUM_FRAMES=${NUM_FRAMES:-81}

# Encoded cache directory (optional, leave empty to disable)
ENCODED_CACHE_DIR=${ENCODED_CACHE_DIR:-""}
# =============================================================

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCELERATE_CONFIG="${SCRIPT_DIR}/accelerate_config_zero2.yaml"

# Update num_processes in accelerate config
if [ -f "$ACCELERATE_CONFIG" ]; then
    sed -i "s/num_processes:.*/num_processes: ${NUM_GPUS}/" "$ACCELERATE_CONFIG"
fi

# Build training arguments
TRAIN_ARGS="--dataset_base_path=${DATASET_BASE_PATH} \
--dataset_metadata_path=${DATASET_METADATA_PATH} \
--height=${HEIGHT} \
--width=${WIDTH} \
--num_frames=${NUM_FRAMES} \
--uniform_sampling=True \
--dataset_repeat=1 \
--model_id_with_origin_paths=Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-480P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-480P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
--learning_rate=${LEARNING_RATE} \
--num_epochs=${NUM_EPOCHS} \
--save_steps=${SAVE_STEPS} \
--remove_prefix_in_ckpt=pipe.dit. \
--trainable_models=dit \
--output_path=${OUTPUT_PATH} \
--extra_inputs=input_image \
--data_file_keys=video"

# Add encoded cache arguments if specified
if [ -n "$ENCODED_CACHE_DIR" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --save_encoded_cache --encoded_cache_dir=${ENCODED_CACHE_DIR}"
fi

# Print configuration
echo "============================================================"
echo "ABot-PhysWorld - Full-Parameter SFT Training"
echo "============================================================"
echo "  GPUs:            ${NUM_GPUS}"
echo "  Dataset:         ${DATASET_BASE_PATH}"
echo "  Metadata:        ${DATASET_METADATA_PATH}"
echo "  Output:          ${OUTPUT_PATH}"
echo "  Learning Rate:   ${LEARNING_RATE}"
echo "  Epochs:          ${NUM_EPOCHS}"
echo "  Save Steps:      ${SAVE_STEPS}"
echo "  Resolution:      ${WIDTH}x${HEIGHT}, ${NUM_FRAMES} frames"
if [ -n "$ENCODED_CACHE_DIR" ]; then
    echo "  Cache Dir:       ${ENCODED_CACHE_DIR}"
fi
echo "  Training Mode:   Full SFT (non-LoRA)"
echo "  Optimizer:       DeepSpeed ZeRO-2"
echo "============================================================"

# Launch training
accelerate launch --config_file="${ACCELERATE_CONFIG}" \
    "${SCRIPT_DIR}/train.py" \
    ${TRAIN_ARGS}
