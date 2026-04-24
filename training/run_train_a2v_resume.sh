#!/bin/bash
# ============================================================
# ABot-PhysWorld - Resume A2V VACE Training
# ============================================================
# This script resumes A2V VACE training from a previously saved
# checkpoint. It supports both cache-based and full training modes.
#
# Prerequisites:
#   1. A saved VACE checkpoint from previous training
#   2. A DiT checkpoint (from SFT training)
#   3. (Optional) Encoded cache directory from initial training
#
# Usage:
#   # Resume from step 800
#   DIT_CHECKPOINT=/path/to/dit_checkpoint.safetensors \
#   RESUME_FROM_STEP=800 \
#   bash run_train_a2v_resume.sh
#
#   # Resume with encoded cache
#   DIT_CHECKPOINT=/path/to/dit.safetensors \
#   RESUME_FROM_STEP=800 \
#   ENCODED_CACHE_DIR=./encoded_cache \
#   bash run_train_a2v_resume.sh
# ============================================================

set -e

# ==================== User Configuration ====================
# Number of GPUs (must match accelerate config num_processes)
NUM_GPUS=${NUM_GPUS:-8}

# DiT checkpoint (REQUIRED - from SFT training)
DIT_CHECKPOINT=${DIT_CHECKPOINT:-""}

# Resume from step (REQUIRED)
RESUME_FROM_STEP=${RESUME_FROM_STEP:-0}

# Dataset paths
DATASET_BASE_PATH=${DATASET_BASE_PATH:-"./data"}
DATASET_METADATA_PATH=${DATASET_METADATA_PATH:-"./data/metadata.jsonl"}

# Encoded cache directory (optional)
ENCODED_CACHE_DIR=${ENCODED_CACHE_DIR:-""}

# Output directory (should match initial training)
OUTPUT_PATH=${OUTPUT_PATH:-"./outputs/a2v_training"}

# Training hyperparameters
LEARNING_RATE=${LEARNING_RATE:-5e-6}
NUM_EPOCHS=${NUM_EPOCHS:-5}
SAVE_STEPS=${SAVE_STEPS:-200}

# Video resolution and frame count
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-640}
CHUNK_NUM_FRAMES=${CHUNK_NUM_FRAMES:-121}

# Temporal sampling
MIN_STRIDE=${MIN_STRIDE:-6}
MAX_STRIDE=${MAX_STRIDE:-6}

# Action condition settings
ACTION_CONDITION_ENABLED=${ACTION_CONDITION_ENABLED:-"true"}
ACTION_CONDITION_CHANNELS=${ACTION_CONDITION_CHANNELS:-9}
DISABLE_TEXT_CONDITION=${DISABLE_TEXT_CONDITION:-"true"}

# VACE settings
INIT_VACE_FROM_DIT=${INIT_VACE_FROM_DIT:-"true"}
INIT_VACE_FROM_DIT_VACE_IN_DIM=${INIT_VACE_FROM_DIT_VACE_IN_DIM:-96}
TRAINABLE_MODELS=${TRAINABLE_MODELS:-"vace"}
REMOVE_PREFIX_IN_CKPT=${REMOVE_PREFIX_IN_CKPT:-"pipe.vace."}
# =============================================================

# Validate required parameters
if [ -z "$DIT_CHECKPOINT" ]; then
    echo "ERROR: DIT_CHECKPOINT is required for A2V resume training."
    echo "Usage: DIT_CHECKPOINT=/path/to/dit.safetensors \\"
    echo "       RESUME_FROM_STEP=800 \\"
    echo "       bash run_train_a2v_resume.sh"
    exit 1
fi

if [ ! -f "$DIT_CHECKPOINT" ]; then
    echo "ERROR: DiT checkpoint file not found: ${DIT_CHECKPOINT}"
    exit 1
fi

# Check VACE checkpoint exists
VACE_CKPT="${OUTPUT_PATH}/step-${RESUME_FROM_STEP}.safetensors"
if [ "$RESUME_FROM_STEP" -gt 0 ] && [ ! -f "$VACE_CKPT" ]; then
    echo "ERROR: VACE checkpoint not found: ${VACE_CKPT}"
    echo "  Make sure OUTPUT_PATH and RESUME_FROM_STEP are correct."
    exit 1
fi

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
--num_frames=${CHUNK_NUM_FRAMES} \
--uniform_sampling=True \
--dataset_repeat=1 \
--model_id_with_origin_paths=Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-480P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-480P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
--learning_rate=${LEARNING_RATE} \
--num_epochs=${NUM_EPOCHS} \
--save_steps=${SAVE_STEPS} \
--remove_prefix_in_ckpt=${REMOVE_PREFIX_IN_CKPT} \
--trainable_models=${TRAINABLE_MODELS} \
--output_path=${OUTPUT_PATH} \
--extra_inputs=input_image \
--data_file_keys=video \
--action_condition_enabled=${ACTION_CONDITION_ENABLED} \
--action_condition_channels=${ACTION_CONDITION_CHANNELS} \
--disable_text_condition=${DISABLE_TEXT_CONDITION} \
--init_vace_from_dit=${INIT_VACE_FROM_DIT} \
--init_vace_from_dit_vace_in_dim=${INIT_VACE_FROM_DIT_VACE_IN_DIM} \
--chunk_num_frames=${CHUNK_NUM_FRAMES} \
--min_stride=${MIN_STRIDE} \
--max_stride=${MAX_STRIDE} \
--dataset_video_resize_mode=stretch \
--dit_checkpoint=${DIT_CHECKPOINT} \
--resume_from_step=${RESUME_FROM_STEP}"

# Add encoded cache arguments if specified
if [ -n "$ENCODED_CACHE_DIR" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --encoded_cache_dir=${ENCODED_CACHE_DIR} --skip_vae --skip_text_encoder --skip_image_encoder"
fi

# Print configuration
echo "============================================================"
echo "ABot-PhysWorld - Resume A2V VACE Training"
echo "============================================================"
echo "  GPUs:                  ${NUM_GPUS}"
echo "  DiT Checkpoint:        ${DIT_CHECKPOINT}"
echo "  VACE Checkpoint:       ${VACE_CKPT}"
echo "  Resume From Step:      ${RESUME_FROM_STEP}"
echo "  Dataset:               ${DATASET_BASE_PATH}"
echo "  Metadata:              ${DATASET_METADATA_PATH}"
echo "  Output:                ${OUTPUT_PATH}"
echo "  Learning Rate:         ${LEARNING_RATE}"
echo "  Epochs:                ${NUM_EPOCHS}"
echo "  Resolution:            ${WIDTH}x${HEIGHT}"
echo "  Chunk Frames:          ${CHUNK_NUM_FRAMES}"
if [ -n "$ENCODED_CACHE_DIR" ]; then
    echo "  Encoded Cache Dir:     ${ENCODED_CACHE_DIR}"
    echo "  Cache Mode:            Skip VAE/TextEncoder/ImageEncoder"
fi
echo "  Optimizer:             DeepSpeed ZeRO-2"
echo "============================================================"

# Launch training
accelerate launch --config_file="${ACCELERATE_CONFIG}" \
    "${SCRIPT_DIR}/train_a2v.py" \
    ${TRAIN_ARGS}
