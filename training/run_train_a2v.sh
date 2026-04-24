#!/bin/bash
# ============================================================
# ABot-PhysWorld - A2V VACE Training Script
# ============================================================
# This script runs A2V (Action-to-Video) VACE training on
# Wan2.1-I2V-14B-480P using DeepSpeed ZeRO-2 via Accelerate.
#
# The VACE module is initialized from pre-trained DiT weights
# and trained to inject action trajectory conditions into the
# video generation process.
#
# Prerequisites:
#   1. Install dependencies (see training/README_A2V.md)
#   2. Prepare dataset in JSONL format with action_path fields
#   3. Ensure sufficient GPU VRAM (>= 60GB per GPU recommended)
#   4. Provide a pre-trained DiT checkpoint (from SFT training)
#
# Usage:
#   # Default 8-GPU training
#   DIT_CHECKPOINT=/path/to/dit_checkpoint.safetensors bash run_train_a2v.sh
#
#   # Custom dataset and output
#   DATASET_BASE_PATH=/data/agibot \
#   DATASET_METADATA_PATH=/data/agibot/metadata.jsonl \
#   DIT_CHECKPOINT=/path/to/dit.safetensors \
#   bash run_train_a2v.sh
# ============================================================

set -e

# ==================== User Configuration ====================
# Number of GPUs (must match accelerate config num_processes)
NUM_GPUS=${NUM_GPUS:-8}

# Dataset paths
DATASET_BASE_PATH=${DATASET_BASE_PATH:-"./data"}
DATASET_METADATA_PATH=${DATASET_METADATA_PATH:-"./data/metadata.jsonl"}

# DiT checkpoint (from SFT training, required for VACE initialization)
DIT_CHECKPOINT=${DIT_CHECKPOINT:-""}

# Output directory for checkpoints
OUTPUT_PATH=${OUTPUT_PATH:-"./outputs/a2v_training"}

# Training hyperparameters
LEARNING_RATE=${LEARNING_RATE:-5e-6}
NUM_EPOCHS=${NUM_EPOCHS:-1}
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

# VACE initialization
INIT_VACE_FROM_DIT=${INIT_VACE_FROM_DIT:-"true"}
INIT_VACE_FROM_DIT_VACE_IN_DIM=${INIT_VACE_FROM_DIT_VACE_IN_DIM:-96}
TRAINABLE_MODELS=${TRAINABLE_MODELS:-"vace"}
REMOVE_PREFIX_IN_CKPT=${REMOVE_PREFIX_IN_CKPT:-"pipe.vace."}

# Encoded cache directory (optional, leave empty to disable)
ENCODED_CACHE_DIR=${ENCODED_CACHE_DIR:-""}
# =============================================================

# Validate required parameters
if [ -z "$DIT_CHECKPOINT" ]; then
    echo "WARNING: DIT_CHECKPOINT is not set."
    echo "  VACE will be initialized from base Wan2.1 I2V weights."
    echo "  For best results, provide a DiT checkpoint from SFT training:"
    echo "  DIT_CHECKPOINT=/path/to/dit_checkpoint.safetensors bash run_train_a2v.sh"
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
--dataset_video_resize_mode=stretch"

# Add DiT checkpoint if specified
if [ -n "$DIT_CHECKPOINT" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --dit_checkpoint=${DIT_CHECKPOINT}"
fi

# Add encoded cache arguments if specified
if [ -n "$ENCODED_CACHE_DIR" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --save_encoded_cache --encoded_cache_dir=${ENCODED_CACHE_DIR}"
fi

# Print configuration
echo "============================================================"
echo "ABot-PhysWorld - A2V VACE Training"
echo "============================================================"
echo "  GPUs:                  ${NUM_GPUS}"
echo "  Dataset:               ${DATASET_BASE_PATH}"
echo "  Metadata:              ${DATASET_METADATA_PATH}"
echo "  DiT Checkpoint:        ${DIT_CHECKPOINT:-'(not set - using base weights)'}"
echo "  Output:                ${OUTPUT_PATH}"
echo "  Learning Rate:         ${LEARNING_RATE}"
echo "  Epochs:                ${NUM_EPOCHS}"
echo "  Save Steps:            ${SAVE_STEPS}"
echo "  Resolution:            ${WIDTH}x${HEIGHT}"
echo "  Chunk Frames:          ${CHUNK_NUM_FRAMES}"
echo "  Stride:                ${MIN_STRIDE}-${MAX_STRIDE}"
echo "  Action Condition:      ${ACTION_CONDITION_ENABLED}"
echo "  Text Condition:        $([ "${DISABLE_TEXT_CONDITION}" = "true" ] && echo "disabled" || echo "enabled")"
echo "  VACE Init from DiT:   ${INIT_VACE_FROM_DIT}"
echo "  VACE In Dim:           ${INIT_VACE_FROM_DIT_VACE_IN_DIM}"
echo "  Trainable Models:      ${TRAINABLE_MODELS}"
if [ -n "$ENCODED_CACHE_DIR" ]; then
    echo "  Cache Dir:             ${ENCODED_CACHE_DIR}"
fi
echo "  Optimizer:             DeepSpeed ZeRO-2"
echo "============================================================"

# Launch training
accelerate launch --config_file="${ACCELERATE_CONFIG}" \
    "${SCRIPT_DIR}/train_a2v.py" \
    ${TRAIN_ARGS}
