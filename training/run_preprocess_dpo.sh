#!/bin/bash
# ============================================================
# ABot-PhysWorld - DPO Data Preprocessing Script
# ============================================================
# Pre-process DPO preference data (winner/loser video pairs)
# into .pth cache files for fast training data loading.
#
# This must be run before train_dpo.py.
#
# Prerequisites:
#   1. JSONL file with winner_video, loser_video, prompt fields
#   2. Pre-trained Wan2.1-I2V-14B-480P model weights
#   3. CUDA GPU for VAE/T5/CLIP encoding
#
# Usage:
#   DPO_JSONL=/path/to/dpo_pairs.jsonl \
#   CACHE_DIR=/path/to/output_cache \
#   bash run_preprocess_dpo.sh
# ============================================================

set -e

# ==================== User Configuration ====================
# Input JSONL file (REQUIRED)
DPO_JSONL=${DPO_JSONL:-""}

# Output cache directory (REQUIRED)
CACHE_DIR=${CACHE_DIR:-""}

# Model root directory (where Wan-AI models are stored)
MODEL_ROOT=${MODEL_ROOT:-""}

# Video parameters
NUM_FRAMES=${NUM_FRAMES:-81}
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-832}

# Tiled VAE (for large resolutions)
TILED=${TILED:-""}

# Max samples (0 = all)
MAX_SAMPLES=${MAX_SAMPLES:-0}

# DataLoader workers
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}

# Output path for trainer logs
OUTPUT_PATH=${OUTPUT_PATH:-"./"}
# =============================================================

# Validate required parameters
if [ -z "$DPO_JSONL" ]; then
    echo "ERROR: DPO_JSONL is required."
    echo "  Usage: DPO_JSONL=/path/to/pairs.jsonl CACHE_DIR=/path/to/cache bash run_preprocess_dpo.sh"
    exit 1
fi

if [ ! -f "$DPO_JSONL" ]; then
    echo "ERROR: JSONL file not found: ${DPO_JSONL}"
    exit 1
fi

if [ -z "$CACHE_DIR" ]; then
    echo "ERROR: CACHE_DIR is required."
    echo "  Usage: DPO_JSONL=/path/to/pairs.jsonl CACHE_DIR=/path/to/cache bash run_preprocess_dpo.sh"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build arguments
PREPROCESS_ARGS="--dpo_jsonl=${DPO_JSONL} \
--cache_dir=${CACHE_DIR} \
--num_frames=${NUM_FRAMES} \
--height=${HEIGHT} \
--width=${WIDTH} \
--max_samples=${MAX_SAMPLES} \
--output_path=${OUTPUT_PATH} \
--dataloader_num_workers=${DATALOADER_NUM_WORKERS}"

# Add model root if specified
if [ -n "$MODEL_ROOT" ]; then
    PREPROCESS_ARGS="${PREPROCESS_ARGS} --model_root=${MODEL_ROOT}"
fi

# Add tiled flag if specified
if [ -n "$TILED" ]; then
    PREPROCESS_ARGS="${PREPROCESS_ARGS} --tiled"
fi

# Print configuration
echo "============================================================"
echo "ABot-PhysWorld - DPO Data Preprocessing"
echo "============================================================"
echo "  Input JSONL:         ${DPO_JSONL}"
echo "  Output Cache Dir:    ${CACHE_DIR}"
echo "  Model Root:          ${MODEL_ROOT:-'(default)'}"
echo "  Video:               ${WIDTH}x${HEIGHT}, ${NUM_FRAMES} frames"
echo "  Max Samples:         ${MAX_SAMPLES} (0=all)"
echo "============================================================"

# Run preprocessing
python "${SCRIPT_DIR}/preprocess_dpo.py" ${PREPROCESS_ARGS}
