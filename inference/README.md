# ABot-PhysWorld Inference - Detailed Guide

This document provides detailed instructions for running inference with ABot-PhysWorld.

For quick start, see the main [README.md](../README.md#-usage) in the project root.

---

## Environment Setup

### Step 1: Create Conda Environment (Recommended)

```bash
conda create -n abot-physworld python=3.10
conda activate abot-physworld
```

### Step 2: Install PyTorch

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install Project Dependencies

From the project root directory:
```bash
pip install -r requirements.txt
```

This installs all dependencies including DiffSynth-Studio modules and model download tools.

### Hardware Requirements

| Configuration | VRAM | Notes |
|---|---|---|
| **Recommended** | >= 60GB | Best performance, no tiling needed |
| **Minimum** | >= 24GB | Uses tiled VAE (enabled by default) |

---

## Quick Start

### Demo Inference

Run inference on 2 demo samples (Franka robot manipulation images):

```bash
python inference.py \
    --jsonl_path assets/demo.jsonl \
    --output_dir ./outputs/demo \
    --save_first_frames
```

The model checkpoint auto-downloads from ModelScope on first run.

### Single Image Inference

```bash
python inference.py \
    --input_image assets/franka_pick_0.jpg \
    --prompt "The robotic arm reaches down and picks up the yellow pear from the table, then places it onto the green plate." \
    --output_dir ./outputs/demo
```

---

## Inference Modes

### Mode 1: Single Image Input

Provide one image + one text prompt:

```bash
python inference.py \
    --input_image path/to/image.jpg \
    --prompt "robot arm picks up the red cube" \
    --output_dir ./outputs
```

### Mode 2: Batch Inference from JSONL

Each line in JSONL file:
```json
{"video": "path/to/image.jpg", "prompt": "description of robot action"}
```

Run batch inference:
```bash
python inference.py \
    --jsonl_path data.jsonl \
    --output_dir ./outputs \
    --num_samples 0  # 0 = process all
```

### Mode 3: Using Shell Script Wrapper

```bash
chmod +x run_inference.sh

# Demo
bash run_inference.sh --jsonl_path assets/demo.jsonl --output_dir ./outputs/demo

# Single image
bash run_inference.sh --input_image image.jpg --prompt "robot grasps object"

# Batch
bash run_inference.sh --jsonl_path data.jsonl --output_dir ./outputs
```

---

## Full Parameter Reference

```
python inference.py --help
```

| Parameter | Default | Description |
|---|---|---|
| `--input_image` | - | Path to input image (single-image mode) |
| `--jsonl_path` | - | Path to JSONL file (batch mode) |
| `--prompt` | - | Text prompt (required for single-image mode) |
| `--checkpoint_path` | auto-download | Path to fine-tuned checkpoint |
| `--cache_dir` | `./checkpoints` | Cache directory for downloaded weights |
| `--output_dir` | `./outputs` | Output directory |
| `--height` | 480 | Video height (pixels) |
| `--width` | 832 | Video width (pixels) |
| `--num_frames` | 81 | Number of frames (~5.4s at 15fps) |
| `--num_inference_steps` | 50 | Diffusion denoising steps (higher = better quality) |
| `--cfg_scale` | 5.0 | Classifier-free guidance scale |
| `--seed` | 0 | Random seed for reproducibility |
| `--no_tiled` | False | Disable tiled VAE (use if VRAM >= 80GB) |
| `--gpu_id` | 0 | GPU device index (for multi-GPU systems) |
| `--num_samples` | 0 (all) | Max samples to process in batch mode |
| `--save_first_frames` | False | Extract and save first frames as images |

---

## Model Weights

### Automatic Download (Recommended)

Model checkpoint is **auto-downloaded** from ModelScope on first inference run:
```bash
python inference.py --jsonl_path assets/demo.jsonl --output_dir ./outputs
```

### Manual Download

If you prefer to download manually:

**Option A: Using modelscope CLI**
```bash
pip install modelscope
modelscope download --model amap_cvlab/Abot-PhysWorld --local_dir ./checkpoints
```

**Option B: Web Download**
- Visit: https://www.modelscope.cn/models/amap_cvlab/Abot-PhysWorld/files
- Download: `abotpw_i2v_480p.safetensors`
- Place in: `./checkpoints/` or `./inference/checkpoints/`

Then specify the path:
```bash
python inference.py \
    --jsonl_path data.jsonl \
    --checkpoint_path ./checkpoints/abotpw_i2v_480p.safetensors \
    --output_dir ./outputs
```

The base model (Wan2.1-I2V-14B-480P) is also auto-downloaded by DiffSynth-Studio.

---

## Output Format

### Single-Image Mode
```
{output_dir}/{image_name}_generated.mp4
```

### Batch Mode
```
{output_dir}/{unique_id}_generated.mp4         # Generated video
{output_dir}/results.json                      # Processing results (JSON)
{output_dir}/frames/                           # (Optional) Extracted first frames
```

**Example results.json:**
```json
[
  {
    "index": 0,
    "video": "path/to/input.jpg",
    "prompt": "robot arm picks up the red cube",
    "output_video": "outputs/sample_0_generated.mp4",
    "status": "success"
  },
  {
    "index": 1,
    "video": "path/to/input2.jpg",
    "prompt": "robot places object on table",
    "output_video": "outputs/sample_1_generated.mp4",
    "status": "success"
  }
]
```

---

## Project Structure

```
ABot-PhysWorld/
├── README.md                           # Main project documentation
├── requirements.txt                    # Project dependencies
├── inference/                          # Inference module
│   ├── inference.py                   # Main inference script
│   ├── run_inference.sh               # Shell wrapper script
│   ├── README.md                      # This file
│   ├── diffsynth/                     # DiffSynth-Studio module
│   └── assets/
│       ├── demo.jsonl                 # Demo data (2 samples)
│       ├── franka_pick_0.jpg          # Demo image 1
│       └── franka_pick_1.jpg          # Demo image 2
├── EZS-Bench/                          # Evaluation benchmark
├── examples/                           # Qualitative result examples
└── tech_report/                        # Technical report
```

---

## Troubleshooting

### CUDA Out of Memory (OOM)

If you encounter VRAM issues:

1. **Tiled VAE is enabled by default** (for 24GB+ GPUs)
2. For 60GB+ VRAM, disable tiling:
   ```bash
   python inference.py --no_tiled --jsonl_path data.jsonl --output_dir ./outputs
   ```
3. Reduce resolution:
   ```bash
   python inference.py --height 384 --width 672 --jsonl_path data.jsonl --output_dir ./outputs
   ```

### Slow Inference

- Reduce `--num_inference_steps` (default: 50)
- Use `--seed` for reproducible results
- For GPU profiling, check `nvidia-smi`

### Model Download Fails

- Check internet connection
- Manually download from ModelScope (see section above)
- Ensure `pip install modelscope` is installed

---

## Acknowledgments

This project uses [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) for video generation pipeline. We thank the DiffSynth-Studio team for their excellent open-source framework.
