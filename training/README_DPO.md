# DPO (Direct Preference Optimization) Training

Train a LoRA adapter on the DiT backbone using DPO to align video generation with physical plausibility preferences. Given pairs of winner (physically correct) and loser (physically incorrect) videos, the model learns to generate videos that better match physical laws.

## Overview

DPO adapts the language model alignment technique ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)) to video diffusion models. Instead of training a separate reward model, DPO directly optimizes the policy (video generation model) using preference pairs:

```
Winner Video (physically correct)   --+
                                      |--> DPO Loss --> LoRA Update
Loser Video (physically incorrect)  --+
```

The key insight is that the DPO loss can be expressed in terms of the MSE between the denoised prediction and the training target, making it compatible with standard diffusion training.

## Pipeline

The DPO training consists of two stages:

### Stage 1: Data Preprocessing

Encode winner/loser video pairs into cached tensors (VAE latents, T5/CLIP features):

```bash
cd training/

DPO_JSONL=/path/to/dpo_pairs.jsonl \
CACHE_DIR=/path/to/dpo_cache \
bash run_preprocess_dpo.sh
```

### Stage 2: DPO Training

Train a LoRA adapter using the cached data:

```bash
cd training/

DIT_CHECKPOINT=/path/to/dit_checkpoint.safetensors \
DPO_CACHE_DIR=/path/to/dpo_cache \
bash run_train_dpo.sh
```

## Data Format

### Input JSONL (for preprocessing)

Each line is a JSON object with winner/loser video paths and a text prompt:

```json
{
  "winner_video": "/data/dpo/episode_001_winner.mp4",
  "loser_video": "/data/dpo/episode_001_loser.mp4",
  "prompt": "A robot arm grasps a red cube on the table"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `winner_video` | string | Yes | Path to preferred (physically correct) video |
| `loser_video` | string | Yes | Path to non-preferred (physically incorrect) video |
| `prompt` | string | Yes | Text description of the scene |

### Cached Data (output of preprocessing)

Each `.pth` file contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `latents_w` | (C, T, H, W) | VAE-encoded winner video latents |
| `latents_l` | (C, T, H, W) | VAE-encoded loser video latents |
| `y_w` / `y_l` | (1, 20, 21, 60, 104) | Image conditioning features |
| `clip_fea_w` / `clip_fea_l` | (...) | CLIP image features |
| `prompt_emb` | dict | T5 text embeddings (with "context" key) |

## Key Parameters

### DPO Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEARNING_RATE` | `1e-6` | Learning rate for AdamW optimizer |
| `LORA_RANK` | `64` | LoRA rank (higher = more capacity) |
| `LORA_ALPHA` | `64` | LoRA scaling factor (usually = rank) |
| `BETA_DPO` | `5000` | DPO beta (controls preference strength) |
| `BATCH_SIZE` | `1` | Per-GPU batch size |
| `WARMUP_STEPS` | `10` | Steps with zero loss for stability |
| `STEPS_PER_EPOCH` | `500` | Training steps per epoch |
| `MAX_EPOCHS` | `100` | Maximum training epochs |

### Data Preprocessing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_FRAMES` | `81` | Frames per video (must match training data) |
| `HEIGHT` / `WIDTH` | `480` / `832` | Video resolution |
| `TILED` | `false` | Enable tiled VAE for large resolutions |

## Technical Details

### Per-Sample MSE Loss

The DPO loss uses per-sample MSE computation to correctly handle `batch_size > 1`:

```python
# Per-sample MSE: reduce over all dims except batch
reduce_dims = list(range(1, pred.ndim))
mse_w = (pred_w - target_w).pow(2).mean(dim=reduce_dims)  # shape: (batch,)
mse_l = (pred_l - target_l).pow(2).mean(dim=reduce_dims)  # shape: (batch,)

# DPO loss
inside = -0.5 * beta * ((mse_w - mse_l) - (ref_mse_w - ref_mse_l))
loss = -F.logsigmoid(inside).mean()
```

This ensures each sample contributes independently to the DPO gradient, unlike `F.mse_loss` which averages across the entire batch.

### Reference Model via disable_adapter

Instead of maintaining a separate reference model (which would double memory usage), we use PEFT's `disable_adapter` mechanism:

```python
# Current policy (LoRA enabled)
pred_w = model_fn(noisy_w, ...)

# Reference model (LoRA disabled, same base weights)
with pipe.dit.disable_adapter():
    ref_w = model_fn(noisy_w, ...)
```

This makes the reference model effectively free in terms of memory.

### Per-Sample Timestep Sampling

Each sample in a batch gets its own independent timestep:

```python
# Per-sample timestep: (batch_size,) instead of (1,)
timestep_ids = torch.randint(0, num_train_timesteps, (batch_size,))
```

This provides better gradient signal diversity within each batch.

### Cache Mode

When using pre-processed cache, the script skips loading VAE, T5, and CLIP encoders (only DiT is needed), saving significant VRAM:

```bash
SKIP_VAE=true SKIP_TEXT_ENCODER=true SKIP_IMAGE_ENCODER=true bash run_train_dpo.sh
```

This is the default behavior in `run_train_dpo.sh`.

## Output

Training checkpoints are saved by PyTorch Lightning's `ModelCheckpoint` callback to `OUTPUT_PATH/lightning_logs/`. Each checkpoint contains only the LoRA parameters (trainable weights).

## Dependencies

In addition to the base requirements:

```
lightning>=2.0.0   # PyTorch Lightning training framework
peft>=0.4.0        # Parameter-Efficient Fine-Tuning (LoRA)
```

## References

- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290) - Rafailov et al., 2023
- [Diffusion-DPO: Aligning Diffusion Models with Direct Preference Optimization](https://arxiv.org/abs/2311.12908) - Wallace et al., 2023
