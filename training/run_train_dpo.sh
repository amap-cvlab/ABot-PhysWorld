#!/bin/bash
# ============================================================
# ABot-PhysWorld - DPO LoRA Training Script
# ============================================================
# Train a LoRA adapter on the DiT backbone using Direct Preference
# Optimization (DPO) to align video generation with physical
# plausibility preferences.
#
# Prerequisites:
#   1. Pre-processed DPO cache from preprocess_dpo.py
#   2. SFT DiT checkpoint (e.g., dit_checkpoint.safetensors)
#   3. CUDA GPU with >= 60GB VRAM
#
# Usage:
#   # Default training
#   DIT_CHECKPOINT=/path/to/dit_checkpoint.safetensors \
#   DPO_CACHE_DIR=/path/to/dpo_cache \
#   bash run_train_dpo.sh
#
#   # Custom hyperparameters
#   DIT_CHECKPOINT=/path/to/dit.safetensors \
#   DPO_CACHE_DIR=/path/to/cache \
#   LEARNING_RATE=5e-7 \
#   BETA_DPO=10000 \
#   bash run_train_dpo.sh
# ============================================================

set -e

# ==================== User Configuration ====================
# DPO cache directory (REQUIRED - from preprocess_dpo.py)
DPO_CACHE_DIR=${DPO_CACHE_DIR:-""}

# DiT checkpoint (REQUIRED - from SFT training)
DIT_CHECKPOINT=${DIT_CHECKPOINT:-""}

# Output directory
OUTPUT_PATH=${OUTPUT_PATH:-"./outputs/dpo_training"}

# Training hyperparameters
LEARNING_RATE=${LEARNING_RATE:-1e-6}
MAX_EPOCHS=${MAX_EPOCHS:-100}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-500}
BATCH_SIZE=${BATCH_SIZE:-1}
ACCUMULATE_GRAD_BATCHES=${ACCUMULATE_GRAD_BATCHES:-1}

# LoRA configuration
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"q,k,v,o,ffn.0,ffn.2"}

# DPO configuration
BETA_DPO=${BETA_DPO:-5000}
WARMUP_STEPS=${WARMUP_STEPS:-10}

# Model configuration
MODEL_ID_WITH_ORIGIN_PATHS=${MODEL_ID_WITH_ORIGIN_PATHS:-"Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-480P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-480P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"}

# Cache mode: skip loading VAE/T5/CLIP to save VRAM
SKIP_VAE=${SKIP_VAE:-"true"}
SKIP_TEXT_ENCODER=${SKIP_TEXT_ENCODER:-"true"}
SKIP_IMAGE_ENCODER=${SKIP_IMAGE_ENCODER:-"true"}

# Gradient checkpointing
USE_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING:-"true"}

# DataLoader
NUM_WORKERS=${NUM_WORKERS:-4}
# =============================================================

# Validate required parameters
if [ -z "$DPO_CACHE_DIR" ]; then
    echo "ERROR: DPO_CACHE_DIR is required."
    echo "  Generate DPO cache first: bash run_preprocess_dpo.sh"
    echo "  Then: DPO_CACHE_DIR=/path/to/cache DIT_CHECKPOINT=/path/to/dit.safetensors bash run_train_dpo.sh"
    exit 1
fi

if [ ! -d "$DPO_CACHE_DIR" ]; then
    echo "ERROR: DPO cache directory not found: ${DPO_CACHE_DIR}"
    exit 1
fi

if [ -z "$DIT_CHECKPOINT" ]; then
    echo "ERROR: DIT_CHECKPOINT is required."
    echo "  Usage: DIT_CHECKPOINT=/path/to/dit_checkpoint.safetensors bash run_train_dpo.sh"
    exit 1
fi

if [ ! -f "$DIT_CHECKPOINT" ]; then
    echo "ERROR: DiT checkpoint file not found: ${DIT_CHECKPOINT}"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build training arguments
TRAIN_ARGS="--cache_dir=${DPO_CACHE_DIR} \
--dit_checkpoint=${DIT_CHECKPOINT} \
--output_path=${OUTPUT_PATH} \
--model_id_with_origin_paths=${MODEL_ID_WITH_ORIGIN_PATHS} \
--learning_rate=${LEARNING_RATE} \
--max_epochs=${MAX_EPOCHS} \
--steps_per_epoch=${STEPS_PER_EPOCH} \
--batch_size=${BATCH_SIZE} \
--accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES} \
--lora_rank=${LORA_RANK} \
--lora_alpha=${LORA_ALPHA} \
--lora_target_modules=${LORA_TARGET_MODULES} \
--beta_dpo=${BETA_DPO} \
--warmup_steps=${WARMUP_STEPS} \
--num_workers=${NUM_WORKERS}"

# Add optional flags
if [ "${USE_GRADIENT_CHECKPOINTING}" = "true" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --use_gradient_checkpointing"
fi

if [ "${SKIP_VAE}" = "true" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --skip_vae"
fi

if [ "${SKIP_TEXT_ENCODER}" = "true" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --skip_text_encoder"
fi

if [ "${SKIP_IMAGE_ENCODER}" = "true" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --skip_image_encoder"
fi

# Print configuration
echo "============================================================"
echo "ABot-PhysWorld - DPO LoRA Training"
echo "============================================================"
echo "  DPO Cache Dir:       ${DPO_CACHE_DIR}"
echo "  DiT Checkpoint:      ${DIT_CHECKPOINT}"
echo "  Output:              ${OUTPUT_PATH}"
echo "  Learning Rate:       ${LEARNING_RATE}"
echo "  Max Epochs:          ${MAX_EPOCHS}"
echo "  Steps/Epoch:         ${STEPS_PER_EPOCH}"
echo "  Batch Size:          ${BATCH_SIZE}"
echo "  Grad Accumulation:   ${ACCUMULATE_GRAD_BATCHES}"
echo "  LoRA Rank:           ${LORA_RANK}"
echo "  LoRA Alpha:          ${LORA_ALPHA}"
echo "  LoRA Targets:        ${LORA_TARGET_MODULES}"
echo "  Beta DPO:            ${BETA_DPO}"
echo "  Warmup Steps:        ${WARMUP_STEPS}"
echo "  Grad Checkpointing:  ${USE_GRADIENT_CHECKPOINTING}"
echo "  Skip VAE/T5/CLIP:    ${SKIP_VAE}/${SKIP_TEXT_ENCODER}/${SKIP_IMAGE_ENCODER}"
echo "============================================================"

# Launch training
python "${SCRIPT_DIR}/train_dpo.py" ${TRAIN_ARGS}
