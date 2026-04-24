# ABot-PhysWorld Training

Full-parameter SFT training for Wan2.1-I2V-14B-480P on robot manipulation datasets.

For environment setup, see the main [README.md](../README.md).

---

## Data Preparation

### Dataset Format

```
dataset_root/
├── metadata.jsonl          # one JSON per line
├── episode_001/
│   └── observation.mp4
├── episode_002/
│   └── observation.mp4
└── ...
```

Each line in the JSONL file:

```json
{"video": "episode_001/observation.mp4", "prompt": "The robotic arm picks up the yellow pear and places it on the green plate."}
```

- `video`: path to video file, relative to `DATASET_BASE_PATH`
- `prompt`: text description of the robot manipulation action

Demo data is provided in [`assets/`](assets/). Place your `.mp4` files under `assets/episode_*/observation.mp4`, then run with `DATASET_BASE_PATH=./assets`.

### Video Requirements

- **Resolution**: 480×832 (height × width) recommended
- **Frame count**: 81 frames (~5.4s at 15fps)
- **Format**: MP4 or other formats supported by imageio

Videos are automatically resized and frame-sampled during training.

---

## Quick Start

### Stage 1: Full SFT (train + save encoded cache)

```bash
cd training

# Default 8-GPU training
bash run_train.sh

# With encoded cache (recommended for multi-epoch training)
ENCODED_CACHE_DIR=./encoded_cache \
DATASET_BASE_PATH=/path/to/dataset \
DATASET_METADATA_PATH=/path/to/dataset/metadata.jsonl \
bash run_train.sh
```

### Stage 2: Resume Training with Cache (fast)

After Stage 1, skip VAE/T5/CLIP loading and train directly from cached features:

```bash
RESUME_CHECKPOINT=./outputs/sft_training/step-800.safetensors \
ENCODED_CACHE_DIR=./encoded_cache \
DATASET_BASE_PATH=/path/to/dataset \
DATASET_METADATA_PATH=/path/to/dataset/metadata.jsonl \
NUM_EPOCHS=5 \
bash run_train_resume.sh
```

This skips encoder loading (~13.5GB VRAM saved) and re-encoding, making subsequent epochs ~5–10× faster.

---

## Training Modes

### Mode 1: Standard Full SFT

Encodes video frames on-the-fly through VAE, T5, and CLIP at each step:

```bash
bash run_train.sh
```

### Mode 2: Training with Encoded Cache

First pass saves encoded features to disk; subsequent passes load from cache, skipping encoding:

```bash
# First pass: train + save cache
ENCODED_CACHE_DIR=./encoded_cache bash run_train.sh

# Stage 2: use cached features (much faster)
RESUME_CHECKPOINT=./outputs/sft_training/step-800.safetensors \
ENCODED_CACHE_DIR=./encoded_cache \
bash run_train_resume.sh
```

### Mode 3: Real-time Text Encoding

When video data is unchanged but captions have been updated, reuse cached video features with fresh text encoding:

```bash
accelerate launch --config_file=accelerate_config_zero2.yaml \
    train.py \
    --dataset_base_path /path/to/dataset \
    --dataset_metadata_path /path/to/new_captions.jsonl \
    --encoded_cache_dir ./encoded_cache \
    --realtime_text_encode \
    --skip_vae \
    --skip_image_encoder \
    --trainable_models dit \
    --output_path ./outputs/recaptioned \
    --data_file_keys video
```

---

## Output

### Checkpoints

```
{output_path}/
├── step-200.safetensors    # DiT weights at step 200
├── step-400.safetensors
└── ...
```

Checkpoints contain only DiT weights (with `pipe.dit.` prefix stripped). To use for inference:

```bash
cd ../inference
python inference.py \
    --checkpoint_path ../training/outputs/sft_training/dit_checkpoint.safetensors \
    --jsonl_path assets/demo.jsonl \
    --output_dir ./outputs
```

### Encoded Cache

When `--save_encoded_cache` is enabled:

```
{encoded_cache_dir}/
├── {md5_hash}.pth          # per-video cached features
├── ...
└── cache_index.json        # video_path → cache_filename
```

Each `.pth` contains:
- `input_latents`: VAE-encoded video latents
- `context`: T5 text embeddings
- `y`: VAE-encoded input image latents
- `clip_feature`: CLIP image features
