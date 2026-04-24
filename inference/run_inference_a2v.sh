#!/bin/bash
# ============================================================
# ABot-PhysWorld - A2V (Action-to-Video) VACE Inference
# ============================================================
# Single-GPU inference for action-conditioned video generation.
#
# Prerequisites:
#   1. Install dependencies (see inference/README_A2V.md)
#   2. Prepare JSONL input file with action_path fields
#   3. Ensure CUDA GPU with >= 60GB VRAM
#
# Usage:
#   # Default (auto-download checkpoint from ModelScope)
#   bash run_inference_a2v.sh
#
#   # Custom checkpoint paths
#   DIT_CHECKPOINT_PATH=/path/to/dit.safetensors \
#   VACE_CHECKPOINT_PATH=/path/to/vace.safetensors \
#   JSONL_PATH=./assets/demo_a2v.jsonl \
#   bash run_inference_a2v.sh
# ============================================================

set -e

# ==================== User Configuration ====================
# Input JSONL file
JSONL_PATH=${JSONL_PATH:-"./assets/demo_a2v.jsonl"}

# Output directory
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/a2v_results"}

# Checkpoint paths (leave empty for auto-download from ModelScope)
DIT_CHECKPOINT_PATH=${DIT_CHECKPOINT_PATH:-""}
VACE_CHECKPOINT_PATH=${VACE_CHECKPOINT_PATH:-""}
CHECKPOINT_CACHE_DIR=${CHECKPOINT_CACHE_DIR:-"./checkpoints"}

# Inference parameters
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-640}
NUM_FRAMES=${NUM_FRAMES:-81}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
CFG_SCALE=${CFG_SCALE:-5.0}
SEED=${SEED:-0}
VACE_IN_DIM=${VACE_IN_DIM:-96}

# A2V specific options
DISABLE_TEXT_CONDITION=${DISABLE_TEXT_CONDITION:-""}
OVERLAY_ACTION_CONDITION=${OVERLAY_ACTION_CONDITION:-""}
SAVE_FIRST_FRAMES=${SAVE_FIRST_FRAMES:-""}
NUM_SAMPLES=${NUM_SAMPLES:-""}
# =============================================================

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build inference arguments
INFER_ARGS="--jsonl_path=${JSONL_PATH} \
--output_dir=${OUTPUT_DIR} \
--height=${HEIGHT} \
--width=${WIDTH} \
--num_frames=${NUM_FRAMES} \
--num_inference_steps=${NUM_INFERENCE_STEPS} \
--cfg_scale=${CFG_SCALE} \
--seed=${SEED} \
--vace_in_dim=${VACE_IN_DIM} \
--checkpoint_cache_dir=${CHECKPOINT_CACHE_DIR}"

# Add checkpoint paths if specified
if [ -n "$DIT_CHECKPOINT_PATH" ]; then
    INFER_ARGS="${INFER_ARGS} --dit_checkpoint_path=${DIT_CHECKPOINT_PATH}"
fi

if [ -n "$VACE_CHECKPOINT_PATH" ]; then
    INFER_ARGS="${INFER_ARGS} --vace_checkpoint_path=${VACE_CHECKPOINT_PATH}"
fi

# Add optional flags
if [ -n "$DISABLE_TEXT_CONDITION" ]; then
    INFER_ARGS="${INFER_ARGS} --disable_text_condition"
fi

if [ -n "$OVERLAY_ACTION_CONDITION" ]; then
    INFER_ARGS="${INFER_ARGS} --overlay_action_condition"
fi

if [ -n "$SAVE_FIRST_FRAMES" ]; then
    INFER_ARGS="${INFER_ARGS} --save_first_frames"
fi

if [ -n "$NUM_SAMPLES" ]; then
    INFER_ARGS="${INFER_ARGS} --num_samples=${NUM_SAMPLES}"
fi

# Print configuration
echo "============================================================"
echo "ABot-PhysWorld - A2V VACE Inference"
echo "============================================================"
echo "  Input:               ${JSONL_PATH}"
echo "  Output:              ${OUTPUT_DIR}"
echo "  DiT Checkpoint:      ${DIT_CHECKPOINT_PATH:-'(auto-download)'}"
echo "  VACE Checkpoint:     ${VACE_CHECKPOINT_PATH:-'(auto-download)'}"
echo "  Resolution:          ${WIDTH}x${HEIGHT}"
echo "  Default Frames:      ${NUM_FRAMES}"
echo "  Inference Steps:     ${NUM_INFERENCE_STEPS}"
echo "  CFG Scale:           ${CFG_SCALE}"
echo "  Seed:                ${SEED}"
echo "  VACE In Dim:         ${VACE_IN_DIM}"
echo "============================================================"

# Run inference
python "${SCRIPT_DIR}/inference_a2v.py" ${INFER_ARGS}
