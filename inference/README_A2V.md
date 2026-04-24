# A2V (Action-to-Video) VACE Inference

Generate physically consistent robot manipulation videos from an input image and action trajectory using the VACE (Video Action Condition Encoding) module.

## Overview

The A2V inference pipeline takes:
1. **First frame image** (extracted from video or provided directly)
2. **Action trajectory** (end-effector poses as .npy or .h5 files)
3. **Camera parameters** (intrinsic + extrinsic, optional)
4. **Text prompt** (optional, can be disabled for pure action control)

And generates a video where the robot executes the specified action trajectory starting from the given image.

## Quick Start

```bash
cd inference/

# Using default settings (auto-download checkpoints from ModelScope)
python inference_a2v.py \
    --jsonl_path ./assets/demo_a2v.jsonl \
    --output_dir ./outputs/a2v_results

# Or use the shell wrapper
bash run_inference_a2v.sh
```

### With Custom Checkpoints

```bash
python inference_a2v.py \
    --jsonl_path data.jsonl \
    --dit_checkpoint_path /path/to/dit_checkpoint.safetensors \
    --vace_checkpoint_path /path/to/vace_checkpoint.safetensors \
    --output_dir ./outputs
```

### With Trajectory Visualization

```bash
python inference_a2v.py \
    --jsonl_path data.jsonl \
    --output_dir ./outputs \
    --overlay_action_condition
```

This generates an additional `video_with_traj.mp4` for each sample with the action trajectory overlaid on the generated video at 50% opacity.

## Input Data Format

Input data is specified via a JSONL file. Each line is a JSON object:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `video` | string | Yes* | Path to source video (first frame is extracted) |
| `first_frame_image` / `image` | string | Yes* | Path to first frame image (alternative to video) |
| `prompt` | string | No | Text description |
| `action_path` | string | No | Path to action file (.npy or .h5) |
| `intrinsic_path` | string | No | Camera intrinsic parameters (.json or .npy) |
| `extrinsic_path` | string | No | Camera extrinsic parameters (.json or .npy) |
| `original_size` | [int, int] | No | Original video resolution [height, width] |

*At least one of `video` or `first_frame_image`/`image` must be provided.

### Example

```json
{
  "video": "/data/episode_001/head_color/video.mp4",
  "prompt": "A robot arm grasps a red cube",
  "action_path": "/data/episode_001/actions.npy",
  "intrinsic_path": "/data/episode_001/head_intrinsic_params.json",
  "extrinsic_path": "/data/episode_001/head_extrinsic_params_aligned.json",
  "original_size": [480, 640]
}
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--height` | 480 | Output video height |
| `--width` | 640 | Output video width |
| `--num_frames` | 81 | Default frame count (overridden by action length) |
| `--num_inference_steps` | 50 | Denoising steps |
| `--cfg_scale` | 5.0 | Classifier-free guidance scale |
| `--seed` | 0 | Random seed |
| `--vace_in_dim` | 96 | VACE input dimension (must match training) |
| `--disable_text_condition` | false | Disable text prompt (pure action control) |
| `--overlay_action_condition` | false | Generate trajectory overlay video |
| `--save_first_frames` | false | Save extracted first frames |
| `--tiled` | false | Use tiled VAE decoding (saves VRAM) |

## Adaptive Frame Count

When action data is available, the output video length automatically matches the action trajectory length. Specifically:

1. The action trajectory length is read from the action file
2. The frame count is rounded up to satisfy the `4n+1` constraint required by the model
3. After generation, padding frames are trimmed back to the original action length

This ensures the generated video precisely matches the input action trajectory.

## Model Loading Order

The checkpoint loading follows a strict order that must not be changed:

```
1. Load base Wan2.1-I2V-14B-480P model
2. Load DiT checkpoint (fine-tuned weights from SFT training)
3. Create VACE from DiT (clone attention blocks + zero-init extra params)
4. Load VACE checkpoint (trained VACE weights)
5. Enable VRAM management
```

> **Important**: If steps 2 and 3 are swapped, the VACE module will be initialized from base model weights instead of the fine-tuned DiT, leading to degraded generation quality.

## Output Structure

```
output_dir/
  episode_001_video/
    video.mp4                    # Generated video
    video_with_traj.mp4          # (optional) Trajectory overlay
    video_frames/
      frame_0000.png
      frame_0001.png
      ...
  episode_002_video/
    ...
  inference_results.json         # Batch results summary
```

## Dependencies

In addition to the base requirements:

```
h5py>=3.0.0       # For reading .h5 action files
scipy>=1.7.0       # For temporal interpolation in visualization
modelscope         # For automatic checkpoint download
```
