# A2V (Action-to-Video) VACE Training

Train a VACE (Video Action Condition Encoding) module on top of the Wan2.1-I2V-14B DiT backbone for physically consistent robot manipulation video generation controlled by action trajectories.

The VACE module is initialized by cloning DiT attention blocks and zero-initializing extra parameters, then trained to inject action trajectory maps as parallel context into the diffusion process.

## Overview

```
Action Trajectory  -->  Trajectory Map (3ch RGB)  -->  VAE Encode  -->  VACE Context
                                                                            |
First Frame Image  -->  CLIP + VAE Encode  -------->  DiT Backbone  <-- residual hint
                                                            |
Text Prompt (optional) -->  T5 Encode  ------->  DiT Cross-Attention
                                                            |
                                                      Generated Video
```

## Data Format

Training data is specified via a JSONL file. Each line is a JSON object with the following fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `video` | string | Yes | Path to video file (mp4/avi) |
| `prompt` | string | No | Text description (ignored if `--disable_text_condition=true`) |
| `action_path` | string | No | Path to action file (.npy or .h5). Auto-inferred from video directory if absent |
| `intrinsic_path` | string | No | Path to camera intrinsic parameters (.json or .npy) |
| `extrinsic_path` | string | No | Path to camera extrinsic parameters (.json or .npy) |
| `original_size` | [int, int] | No | Original video resolution [height, width]. Defaults to [480, 640] |

### Example JSONL Entry

```json
{
  "video": "/data/agibot/episode_001/head_color/video.mp4",
  "prompt": "A robot arm grasps a red cube on the table",
  "action_path": "/data/agibot/episode_001/actions.npy",
  "intrinsic_path": "/data/agibot/episode_001/head_intrinsic_params.json",
  "extrinsic_path": "/data/agibot/episode_001/head_extrinsic_params_aligned.json",
  "original_size": [480, 640]
}
```

### Action File Formats

- **`.npy`**: NumPy array of shape `(T, C)` where T is the number of frames and C is the action dimension
- **`.h5`**: HDF5 file with end-effector pose data (loaded via `load_actions_with_quat`, producing quaternion-based 16-dim actions)

### Camera Parameter Auto-Discovery

If `intrinsic_path` and `extrinsic_path` are not provided, the script searches for:

1. `{video_dir}/head_intrinsic_params.json` + `{video_dir}/head_extrinsic_params_aligned.json`
2. `{video_dir}/../../parameters/camera/{cam_name}_intrinsic_params.json` + `{cam_name}_extrinsic_params_aligned.json`
3. Default 60-degree FOV with identity extrinsics (fallback)

## Quick Start

### First-Time Training

```bash
cd training/

# Minimal: provide dataset and DiT checkpoint
DIT_CHECKPOINT=/path/to/dit_checkpoint.safetensors \
DATASET_BASE_PATH=/path/to/dataset \
DATASET_METADATA_PATH=/path/to/metadata.jsonl \
bash run_train_a2v.sh
```

### Resume Training

```bash
cd training/

DIT_CHECKPOINT=/path/to/dit_checkpoint.safetensors \
RESUME_FROM_STEP=800 \
OUTPUT_PATH=./outputs/a2v_training \
bash run_train_a2v_resume.sh
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAINABLE_MODELS` | `vace` | Which modules to train (vace, dit, or both) |
| `ACTION_CONDITION_ENABLED` | `true` | Enable action trajectory injection |
| `DISABLE_TEXT_CONDITION` | `true` | Disable text prompt (pure action control) |
| `INIT_VACE_FROM_DIT` | `true` | Initialize VACE from DiT weights |
| `INIT_VACE_FROM_DIT_VACE_IN_DIM` | `96` | VACE input dimension (96 = 16ch latent * 6) |
| `CHUNK_NUM_FRAMES` | `121` | Max frames per training chunk |
| `MIN_STRIDE` / `MAX_STRIDE` | `6` / `6` | Temporal stride range for chunk sampling |
| `LEARNING_RATE` | `5e-6` | Learning rate |
| `HEIGHT` / `WIDTH` | `480` / `640` | Training resolution |

## VACE Initialization Strategy

The VACE module is created by cloning attention blocks from the pre-trained DiT, then zero-initializing the extra input projection and output projection parameters. This ensures:

1. **Stable training start**: VACE output is initially zero, so the DiT backbone produces the same output as before VACE injection
2. **Fast convergence**: The cloned attention weights already encode useful video generation priors
3. **No architectural modification to DiT**: The original DiT remains unchanged; VACE adds parallel context via residual connections

### Weight Loading Order (Critical)

When resuming or using a DiT checkpoint from SFT training:

```
1. Load base Wan2.1-I2V-14B model (pipe = WanVideoPipeline.from_pretrained(...))
2. Load DiT checkpoint (pipe.dit.load_state_dict(dit_ckpt, strict=False))
3. Create VACE from DiT (pipe.vace = create_vace_from_dit(pipe.dit, ...))
4. Load VACE checkpoint if resuming (pipe.vace.load_state_dict(vace_ckpt, strict=False))
```

> **Warning**: Steps 2 and 3 must not be reordered. The VACE module must be created from the DiT *after* loading the DiT checkpoint, so that it inherits the fine-tuned weights rather than the base model weights.

## Output

Checkpoints are saved to `OUTPUT_PATH` as `step-{N}.safetensors` containing only the VACE module weights (controlled by `REMOVE_PREFIX_IN_CKPT=pipe.vace.`).

## Dependencies

In addition to the base requirements:

```
h5py>=3.0.0       # For reading .h5 action files
scipy>=1.7.0       # For temporal interpolation
```
