#!/bin/bash
# ============================================================
# ABot-PhysWorld - Resume Training with Encoded Cache
# ============================================================
# This script resumes training from a previously saved checkpoint.
# It uses pre-computed encoded features (VAE, text encoder) from cache
# to speed up training and reduce VRAM usage.
#
# Key differences from run_train.sh:
#   - Skips VAE, text encoder, image encoder loading
#   - Uses --encoded_cache_dir (no --save_encoded_cache)
#   - Loads DIT weights from checkpoint
#   - Does NOT save new cache (only uses existing cache)
#
# Prerequisites:
#   1. Encoded cache directory from run_train.sh
#   2. A saved DIT checkpoint from previous training
#
# Usage:
#   # Resume with cache (skip encoders, use pre-computed features)
#   RESUME_CHECKPOINT=./outputs/sft_training/step-200.safetensors \
#   ENCODED_CACHE_DIR=./cache \
#   bash run_train_resume.sh
#
#   # Custom epochs
#   RESUME_CHECKPOINT=./outputs/sft_training/step-200.safetensors \
#   ENCODED_CACHE_DIR=./cache \
#   NUM_EPOCHS=5 \
#   bash run_train_resume.sh
# ============================================================

set -e

# ==================== User Configuration ====================
# Number of GPUs (must match accelerate config num_processes)
NUM_GPUS=${NUM_GPUS:-8}

# Resume checkpoint path (REQUIRED for cache-based training)
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-""}

# Resume from step (only used when continuing training with cache)
# Typically set to the step number where checkpoint was saved
RESUME_FROM_STEP=${RESUME_FROM_STEP:-0}

# Dataset paths
DATASET_BASE_PATH=${DATASET_BASE_PATH:-"./data"}
DATASET_METADATA_PATH=${DATASET_METADATA_PATH:-"./data/metadata.jsonl"}

# Encoded cache directory (REQUIRED for cache-based training)
# This should be the cache directory from run_train.sh
ENCODED_CACHE_DIR=${ENCODED_CACHE_DIR:-""}

# Output directory for new checkpoints
OUTPUT_PATH=${OUTPUT_PATH:-"./outputs/sft_training_resumed"}

# Log directory (optional)
LOG_DIR=${LOG_DIR:-""}

# Training hyperparameters
LEARNING_RATE=${LEARNING_RATE:-1e-5}
NUM_EPOCHS=${NUM_EPOCHS:-5}
SAVE_STEPS=${SAVE_STEPS:-200}

# Video resolution and frame count
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-832}
NUM_FRAMES=${NUM_FRAMES:-81}
# =============================================================

# Validate required parameters
if [ -z "$RESUME_CHECKPOINT" ]; then
    echo "ERROR: RESUME_CHECKPOINT is required for cache-based training."
    echo "Usage: RESUME_CHECKPOINT=/path/to/step-N.safetensors \\"
    echo "       ENCODED_CACHE_DIR=/path/to/cache \\"
    echo "       bash run_train_resume.sh"
    exit 1
fi

if [ ! -f "$RESUME_CHECKPOINT" ]; then
    echo "ERROR: Checkpoint file not found: ${RESUME_CHECKPOINT}"
    exit 1
fi

if [ -z "$ENCODED_CACHE_DIR" ]; then
    echo "WARNING: ENCODED_CACHE_DIR not set. Cache-based training will be disabled."
    echo "         Please provide cache directory from initial training:"
    echo "         ENCODED_CACHE_DIR=/path/to/cache bash run_train_resume.sh"
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCELERATE_CONFIG="${SCRIPT_DIR}/accelerate_config_zero2.yaml"

# Update num_processes in accelerate config
if [ -f "$ACCELERATE_CONFIG" ]; then
    sed -i "s/num_processes:.*/num_processes: ${NUM_GPUS}/" "$ACCELERATE_CONFIG"
fi

# Build training arguments for cache-based training
# Key: Skip VAE and encoders since we're using pre-computed cached features
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
--data_file_keys=video \
--skip_vae \
--skip_text_encoder \
--skip_image_encoder \
--resume_from_step=${RESUME_FROM_STEP} \
--dit_checkpoint=${RESUME_CHECKPOINT}"

# Add log directory if specified
if [ -n "$LOG_DIR" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --log_dir=${LOG_DIR}"
fi

# Use encoded cache (load pre-computed features, don't save)
if [ -n "$ENCODED_CACHE_DIR" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --encoded_cache_dir=${ENCODED_CACHE_DIR}"
fi

# Print configuration
echo "============================================================"
echo "ABot-PhysWorld - Resume Training with Encoded Cache"
echo "============================================================"
echo "  GPUs:                ${NUM_GPUS}"
echo "  DIT Checkpoint:      ${RESUME_CHECKPOINT}"
echo "  Resume From Step:    ${RESUME_FROM_STEP}"
echo "  Dataset:             ${DATASET_BASE_PATH}"
echo "  Metadata:            ${DATASET_METADATA_PATH}"
echo "  Encoded Cache Dir:   ${ENCODED_CACHE_DIR}"
echo "  Output:              ${OUTPUT_PATH}"
if [ -n "$LOG_DIR" ]; then
    echo "  Log Dir:             ${LOG_DIR}"
fi
echo "  Learning Rate:       ${LEARNING_RATE}"
echo "  Epochs:              ${NUM_EPOCHS}"
echo "  Save Steps:          ${SAVE_STEPS}"
echo "  Resolution:          ${WIDTH}x${HEIGHT}, ${NUM_FRAMES} frames"
echo ""
echo "  Training Strategy:"
echo "    ✓ Load DIT weights from checkpoint"
echo "    ✓ Use pre-computed cached features"
echo "    ✓ Skip VAE, TextEncoder, ImageEncoder loading"
echo "    ✓ Faster training with less VRAM"
echo "    ✗ NOT saving new cache (use existing cache)"
echo ""
echo "  Training Mode:       Full SFT (non-LoRA)"
echo "  Optimizer:           DeepSpeed ZeRO-2"
echo "============================================================"

# Launch training
accelerate launch --config_file="${ACCELERATE_CONFIG}" \
    "${SCRIPT_DIR}/train.py" \
    ${TRAIN_ARGS}
