#!/bin/bash
# ============================================================
# Abot-PhysWorld - Local Inference Script
# ============================================================
# This script runs Abot-PhysWorld inference locally.
# The fine-tuned checkpoint will be auto-downloaded from ModelScope
# if not already present.
#
# Prerequisites:
#   1. Install DiffSynth-Studio:
#      git clone https://github.com/modelscope/DiffSynth-Studio.git
#      cd DiffSynth-Studio && pip install -e .
#
#   2. Install additional dependencies:
#      pip install modelscope imageio imageio-ffmpeg tqdm
#
# Usage:
#   # Single image inference
#   bash run_inference.sh --input_image path/to/image.jpg --prompt "your prompt"
#
#   # Batch inference from JSONL
#   bash run_inference.sh --jsonl_path path/to/data.jsonl
# ============================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run inference
python3 "${SCRIPT_DIR}/inference.py" "$@"
