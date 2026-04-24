#!/usr/bin/env python3
"""
ABot-PhysWorld A2V (Action-to-Video) VACE Inference Script

Single-GPU inference for action-conditioned video generation using VACE.
Given an input image, action trajectory, and (optional) text prompt, generates
a physically consistent video of robot manipulation.

The script supports:
    - Automatic checkpoint download from ModelScope
    - Action trajectory visualization (overlay or side-by-side)
    - Batch inference from JSONL files
    - Per-sample adaptive frame count (aligned to action length)

Requirements:
    - DiffSynth-Studio (bundled in ./diffsynth/)
    - modelscope (for automatic checkpoint download)
    - h5py (for reading .h5 action files)
    - scipy (for temporal interpolation in visualization)
    - torch >= 2.0, CUDA GPU with >= 60GB VRAM (recommended)

Usage:
    # Batch inference from JSONL
    python inference_a2v.py \\
        --jsonl_path ./assets/demo_a2v.jsonl \\
        --output_dir ./outputs/a2v_results

    # With custom checkpoint paths
    python inference_a2v.py \\
        --jsonl_path data.jsonl \\
        --dit_checkpoint_path /path/to/dit_checkpoint.safetensors \\
        --vace_checkpoint_path /path/to/vace_checkpoint.safetensors \\
        --output_dir ./outputs

    # With action trajectory overlay visualization
    python inference_a2v.py \\
        --jsonl_path data.jsonl \\
        --output_dir ./outputs \\
        --overlay_action_condition
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import imageio
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new_vace import WanVideoPipeline, ModelConfig
from diffsynth.models.utils import load_state_dict
from diffsynth.utils.action_utils import (
    load_actions_with_quat,
    load_camera_params_from_json,
    load_camera_params_from_npy,
    simple_radius_gen_func,
    get_traj_maps,
)


# -- ModelScope auto-download configuration --
# Update these after uploading A2V checkpoints to ModelScope
MODELSCOPE_MODEL_ID = "amap_cvlab/Abot-PhysWorld"
DIT_CHECKPOINT_FILENAME = "dit_checkpoint.safetensors"  # placeholder: update after uploading A2V DiT weights
VACE_CHECKPOINT_FILENAME = "vace_checkpoint.safetensors"  # placeholder: update after uploading A2V VACE weights

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, overexposed, static, subtitle, painting, worst quality, low quality, "
    "ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, "
    "disfigured, static frame, cluttered background"
)


# =====================================================================
# Checkpoint download
# =====================================================================

def download_checkpoint(filename, cache_dir="./checkpoints"):
    """Download a checkpoint from ModelScope if not already cached."""
    from modelscope import snapshot_download

    checkpoint_path = os.path.join(cache_dir, filename)
    if os.path.exists(checkpoint_path):
        print("Checkpoint already exists: %s" % checkpoint_path)
        return checkpoint_path

    print("Downloading checkpoint from ModelScope: %s / %s ..." % (MODELSCOPE_MODEL_ID, filename))
    model_dir = snapshot_download(MODELSCOPE_MODEL_ID, cache_dir=cache_dir)
    downloaded_path = os.path.join(model_dir, filename)

    if os.path.exists(downloaded_path):
        print("Checkpoint downloaded to: %s" % downloaded_path)
        return downloaded_path

    for root, dirs, files in os.walk(model_dir):
        for fname in files:
            if fname == filename:
                found_path = os.path.join(root, fname)
                print("Checkpoint found at: %s" % found_path)
                return found_path

    raise FileNotFoundError(
        "Could not find %s after downloading from ModelScope. "
        "Please download manually from https://www.modelscope.cn/models/%s/files "
        "and specify the path via command-line arguments." % (filename, MODELSCOPE_MODEL_ID)
    )


# =====================================================================
# Video / image helpers
# =====================================================================

def extract_first_frame(media_path, output_image_path=None):
    """Extract the first frame from a video file, or load an image file directly."""
    media_path = Path(media_path)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    if media_path.suffix.lower() in image_extensions:
        image = Image.open(media_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
    else:
        reader = imageio.get_reader(str(media_path))
        frame = reader.get_data(0)
        reader.close()
        image = Image.fromarray(frame)

    if output_image_path:
        image.save(output_image_path)
    return image


def round_up_to_4n_plus_1(num_frames):
    """Round up frame count to the nearest value satisfying 4n+1."""
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1, got %d" % num_frames)
    return (num_frames + 2) // 4 * 4 + 1


# =====================================================================
# Action data loading (aligned with training module)
# =====================================================================

def _load_action_data(sample):
    """Load action data from file (.npy or .h5).

    Searches for action file from explicit path or infers from video path.
    Returns torch.FloatTensor of shape (T, C) or None if not found.
    """
    action_path = sample.get("action_path")
    video_path_str = sample.get("video", "")
    if action_path is None and video_path_str:
        video_dir = Path(video_path_str).parent
        for p in [
            video_dir / "actions.npy",
            video_dir / "action.npy",
            video_dir / "proprio_stats.h5",
            video_dir.parent / "proprio_stats" / video_dir.name / "proprio_stats.h5",
        ]:
            if p.exists():
                action_path = str(p)
                break
    if action_path is None:
        return None
    action_path_p = Path(action_path)
    if not action_path_p.exists():
        return None
    if action_path_p.suffix == ".npy":
        return torch.FloatTensor(np.load(str(action_path_p)))
    if action_path_p.suffix == ".h5":
        return load_actions_with_quat(str(action_path_p))
    return None


def _get_default_camera_params(num_frames, target_size):
    """Generate default camera parameters (60-degree FOV, identity extrinsics)."""
    h, w = target_size
    fx = fy = w / (2 * np.tan(np.radians(30)))
    intrinsic = torch.eye(3, dtype=torch.float32)
    intrinsic[0, 0], intrinsic[1, 1] = fx, fy
    intrinsic[0, 2], intrinsic[1, 2] = w / 2, h / 2
    extrinsics = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(num_frames, 1, 1)
    return intrinsic, extrinsics


def _load_camera_params_raw(sample, num_frames, frame_indices=None):
    """Load camera intrinsic and extrinsic parameters from JSON or NPY files."""
    extrinsic_path = sample.get("extrinsic_path")
    intrinsic_path = sample.get("intrinsic_path")
    video_path = sample.get("video", "")

    if video_path and (extrinsic_path is None or intrinsic_path is None):
        video_dir = Path(video_path).parent
        possible_intrinsic_json = video_dir / "head_intrinsic_params.json"
        possible_extrinsic_json = video_dir / "head_extrinsic_params_aligned.json"
        if possible_intrinsic_json.exists() and possible_extrinsic_json.exists():
            intrinsic_path = str(possible_intrinsic_json)
            extrinsic_path = str(possible_extrinsic_json)
        else:
            cam_name = video_dir.name.replace("_color", "")
            pe = video_dir.parent.parent / "parameters" / "camera" / ("%s_extrinsic_params_aligned.json" % cam_name)
            pi = video_dir.parent.parent / "parameters" / "camera" / ("%s_intrinsic_params.json" % cam_name)
            if pe.exists() and pi.exists():
                extrinsic_path, intrinsic_path = str(pe), str(pi)

    if extrinsic_path is None or intrinsic_path is None:
        return _get_default_camera_params(num_frames, (480, 640))

    extrinsic_p = Path(extrinsic_path)
    intrinsic_p = Path(intrinsic_path)
    if not extrinsic_p.exists() or not intrinsic_p.exists():
        return _get_default_camera_params(num_frames, (480, 640))

    if extrinsic_p.suffix == ".json":
        intrinsic, extrinsics = load_camera_params_from_json(
            str(intrinsic_p), str(extrinsic_p),
            frame_indices=frame_indices, original_size=None, target_size=None,
        )
    else:
        intrinsic, extrinsics = load_camera_params_from_npy(
            str(intrinsic_p), str(extrinsic_p),
            original_size=None, target_size=None,
        )

    if frame_indices is not None and len(frame_indices) > 0:
        n = extrinsics.shape[0]
        safe_indices = [min(max(i, 0), n - 1) for i in frame_indices]
        extrinsics = extrinsics[safe_indices]
    else:
        if extrinsics.shape[0] < num_frames:
            extrinsics = torch.cat(
                [extrinsics, extrinsics[-1:].repeat(num_frames - extrinsics.shape[0], 1, 1)],
                dim=0,
            )
        elif extrinsics.shape[0] > num_frames:
            extrinsics = extrinsics[:num_frames]
    return intrinsic, extrinsics


def _get_original_video_size(sample):
    """Infer original video resolution from sample metadata."""
    if "original_size" in sample:
        o = sample["original_size"]
        if isinstance(o, (list, tuple)) and len(o) == 2:
            return int(o[0]), int(o[1])
    return 480, 640


# =====================================================================
# Action condition generation (aligned with training module)
# =====================================================================

def _generate_action_condition(sample, target_size):
    """Generate action condition tensor (trajectory maps) for VACE injection.

    Returns (3, T, H, W) tensor in [-1, 1] range, or None if no action data.
    """
    actions = _load_action_data(sample)
    if actions is None:
        return None

    action_num_frames = actions.shape[0]
    frame_indices = sample.get("video_indices")
    if frame_indices is not None and len(frame_indices) > 0:
        safe_indices = [min(i, action_num_frames - 1) for i in frame_indices]
        actions = actions[safe_indices]
        num_frames = len(safe_indices)
    else:
        num_frames = sample.get("num_frames", action_num_frames)
        if action_num_frames >= num_frames:
            actions = actions[:num_frames]
        else:
            actions = torch.cat(
                [actions, actions[-1:].repeat(num_frames - action_num_frames, 1)], dim=0
            )

    original_size = _get_original_video_size(sample)
    intrinsic_orig, extrinsics = _load_camera_params_raw(sample, num_frames, frame_indices=frame_indices)

    # Scale intrinsic for resize
    orig_h, orig_w = original_size
    tgt_h, tgt_w = target_size
    h_scale = float(tgt_h) / float(orig_h)
    w_scale = float(tgt_w) / float(orig_w)
    intrinsic_adjusted = intrinsic_orig.clone()
    intrinsic_adjusted[0, 0] *= w_scale
    intrinsic_adjusted[0, 2] *= w_scale
    intrinsic_adjusted[1, 1] *= h_scale
    intrinsic_adjusted[1, 2] *= h_scale

    extrinsics_v = extrinsics.unsqueeze(0)
    intrinsic_adj_v = intrinsic_adjusted.unsqueeze(0)
    w2c = torch.linalg.inv(extrinsics_v)
    c2w = extrinsics_v

    trajs_orig = get_traj_maps(
        actions, w2c, c2w, intrinsic_adj_v, target_size,
        radius_gen_func=simple_radius_gen_func,
    )
    trajs_orig = (trajs_orig * 2 - 1).squeeze(1)  # (3, T, H, W) in [-1, 1]
    return trajs_orig


def _action_condition_to_vace_video_tensor(action_cond, num_frames, height, width):
    """Convert action condition (C,T,H,W) to (1,3,T,H,W) tensor for pipeline."""
    if action_cond.dim() == 4:
        rgb = action_cond[:3]
    else:
        rgb = action_cond[0, :3]
    C, T, H, W = rgb.shape
    if (T, H, W) != (num_frames, height, width):
        rgb = torch.nn.functional.interpolate(
            rgb.unsqueeze(0).view(1, 3, T, H, W),
            size=(num_frames, height, width),
            mode="trilinear", align_corners=False,
        ).squeeze(0)
    return rgb.unsqueeze(0)  # (1, 3, T, H, W)


# =====================================================================
# Overlay visualization
# =====================================================================

def _create_overlay_video(video_frames, action_cond, original_action_num_frames=None, overlay_mode=True):
    """Create visualization video with trajectory overlay or side-by-side view.

    Args:
        video_frames: List of PIL Images (generated video)
        action_cond: Tensor (3, T, H, W) in [-1, 1]
        original_action_num_frames: Original action frame count (for trimming)
        overlay_mode: True for 50% alpha overlay, False for vertical concatenation

    Returns:
        List of PIL Images for the combined video, or None on failure
    """
    if action_cond is None:
        return None

    trajs_normalized = (action_cond + 1) / 2  # [0, 1]
    trajs_normalized = torch.clamp(trajs_normalized, 0, 1)
    trajs_np = trajs_normalized.permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, 3)
    trajs_np = (trajs_np * 255).astype(np.uint8)

    ori_trajs = action_cond
    mask_np = (ori_trajs > 0).any(dim=0).to(torch.float32).cpu().numpy()  # (T, H, W)

    num_video_frames = len(video_frames)
    num_traj_frames = trajs_np.shape[0]

    # Trim to original action length if available
    if original_action_num_frames is not None:
        vis_frames = min(original_action_num_frames, num_video_frames, num_traj_frames)
        trajs_np = trajs_np[:vis_frames]
        mask_np = mask_np[:vis_frames] > 0.5
        num_traj_frames = trajs_np.shape[0]
        if num_video_frames != vis_frames:
            video_frames = video_frames[:vis_frames]
            num_video_frames = len(video_frames)
    elif num_traj_frames != num_video_frames:
        from scipy.ndimage import zoom
        zoom_factor = (num_video_frames / num_traj_frames, 1, 1, 1)
        trajs_np = zoom(trajs_np, zoom_factor, order=1)
        mask_np = zoom(mask_np, (num_video_frames / num_traj_frames, 1, 1), order=1) > 0.5
        num_traj_frames = trajs_np.shape[0]
    else:
        mask_np = mask_np > 0.5

    target_width = video_frames[0].width
    target_height = video_frames[0].height

    traj_frames = []
    mask_frames = []
    for i in range(min(num_traj_frames, num_video_frames)):
        traj_frames.append(Image.fromarray(trajs_np[i]))
        mask_frames.append(mask_np[i])

    combined_frames = []
    if overlay_mode:
        for i in range(len(video_frames)):
            traj_frame = traj_frames[i] if i < len(traj_frames) else traj_frames[-1]
            mask_frame = mask_frames[i] if i < len(mask_frames) else mask_frames[-1]
            if traj_frame.width != target_width or traj_frame.height != target_height:
                traj_frame = traj_frame.resize((target_width, target_height), Image.Resampling.LANCZOS)
            if mask_frame.shape != (target_height, target_width):
                mask_img = Image.fromarray(
                    (mask_frame.astype(np.uint8) * 255), mode="L"
                ).resize((target_width, target_height), Image.Resampling.NEAREST)
                mask_frame = np.array(mask_img) > 127
            video_np = np.array(video_frames[i]).astype(np.float32)
            traj_np_frame = np.array(traj_frame).astype(np.float32)
            overlay_np = video_np.copy()
            blended = video_np * 0.5 + traj_np_frame * 0.5
            overlay_np[mask_frame] = blended[mask_frame]
            combined_frames.append(Image.fromarray(overlay_np.astype(np.uint8)))
    else:
        for i in range(len(video_frames)):
            traj_frame = traj_frames[i] if i < len(traj_frames) else traj_frames[-1]
            if traj_frame.width != target_width or traj_frame.height != target_height:
                traj_frame = traj_frame.resize((target_width, target_height), Image.Resampling.LANCZOS)
            combined_frame = Image.new("RGB", (target_width, target_height * 2))
            combined_frame.paste(video_frames[i], (0, 0))
            combined_frame.paste(traj_frame, (0, target_height))
            combined_frames.append(combined_frame)

    return combined_frames


# =====================================================================
# Pipeline construction
# =====================================================================

def build_pipeline(dit_checkpoint_path, vace_checkpoint_path, vace_in_dim=96, device=None):
    """Build WanVideoPipeline with VACE module for A2V inference.

    Loading order (critical - must not be reordered):
        1. Load base Wan2.1-I2V-14B model
        2. Load DiT checkpoint (fine-tuned weights)
        3. Create VACE from DiT (clones attention blocks)
        4. Load VACE checkpoint (trained VACE weights)
        5. Enable VRAM management

    Args:
        dit_checkpoint_path: Path to DiT .safetensors checkpoint
        vace_checkpoint_path: Path to VACE .safetensors checkpoint
        vace_in_dim: VACE input dimension (default: 96)
        device: Target device (default: auto-detect)

    Returns:
        WanVideoPipeline with VACE module loaded
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[A2V] Building pipeline...")

    # Step 1: Load base model
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(
                model_id="Wan-AI/Wan2.1-I2V-14B-480P",
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.1-I2V-14B-480P",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.1-I2V-14B-480P",
                origin_file_pattern="Wan2.1_VAE.pth",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.1-I2V-14B-480P",
                origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                offload_device="cpu",
            ),
        ],
        init_vace_from_dit=False,
    )
    print("[A2V] Base model loaded")

    # Step 2: Load DiT checkpoint
    if dit_checkpoint_path and os.path.exists(dit_checkpoint_path):
        print("[A2V] Loading DiT checkpoint: %s" % dit_checkpoint_path)
        dit_state_dict = load_state_dict(dit_checkpoint_path)
        missing, unexpected = pipe.dit.load_state_dict(dit_state_dict, strict=False)
        print("[A2V] DiT loaded - missing: %d, unexpected: %d" % (len(missing), len(unexpected)))
    else:
        print("[A2V] No DiT checkpoint specified, using base Wan2.1 I2V weights")

    # Step 3: Create VACE from DiT
    pipe.vace = WanVideoPipeline.create_vace_from_dit(
        pipe.dit, vace_in_dim=vace_in_dim, zero_init_extra=True,
    )
    pipe.vace = pipe.vace.to(
        dtype=torch.bfloat16,
        device=next(pipe.dit.parameters()).device,
    )
    print("[A2V] VACE created from DiT (vace_in_dim=%d)" % vace_in_dim)

    # Step 4: Load VACE checkpoint
    if vace_checkpoint_path and os.path.exists(vace_checkpoint_path):
        print("[A2V] Loading VACE checkpoint: %s" % vace_checkpoint_path)
        vace_state_dict = load_state_dict(vace_checkpoint_path)
        missing, unexpected = pipe.vace.load_state_dict(vace_state_dict, strict=False)
        print("[A2V] VACE loaded - missing: %d, unexpected: %d" % (len(missing), len(unexpected)))
    else:
        print("[WARNING] No VACE checkpoint specified, using zero-initialized VACE")

    # Step 5: Enable VRAM management
    pipe.enable_vram_management()
    print("[A2V] Pipeline ready")

    return pipe


# =====================================================================
# Single sample processing
# =====================================================================

def process_sample(pipe, sample, sample_idx, output_dir, inference_args):
    """Process a single sample for A2V VACE inference.

    Args:
        pipe: WanVideoPipeline with VACE module
        sample: Dict with video/image path, prompt, action_path, etc.
        sample_idx: Index of the sample in the batch
        output_dir: Output directory
        inference_args: Dict of inference parameters

    Returns:
        Dict with inference results and status
    """
    video_path_str = sample.get("video", "")
    image_path_str = (
        sample.get("first_frame_image")
        or sample.get("image")
        or sample.get("image_path")
    )

    video_path = Path(video_path_str) if video_path_str else None
    image_path = Path(image_path_str) if image_path_str else None

    use_video = video_path is not None and video_path.exists()
    use_image = (not use_video) and image_path is not None and image_path.exists()

    prompt = sample.get("prompt", "")

    if not use_video and not use_image:
        err_src = video_path_str or image_path_str or "unknown"
        return {
            "index": sample_idx,
            "source": err_src,
            "prompt": prompt,
            "status": "error: no valid video or image input found",
        }

    # Extract or load first frame
    if use_video:
        path_parts = video_path.parts
        if len(path_parts) >= 2:
            unique_id = "%s_%s" % (path_parts[-2], video_path.stem)
        else:
            unique_id = "sample_%03d_%s" % (sample_idx, video_path.stem)

        first_frame_path = None
        if inference_args.get("save_first_frames"):
            first_frames_dir = Path(output_dir) / "first_frames"
            first_frames_dir.mkdir(exist_ok=True)
            first_frame_path = first_frames_dir / ("%s_first_frame.jpg" % unique_id)

        input_image = extract_first_frame(video_path, first_frame_path)
        input_source_str = str(video_path)
    else:
        path_parts = image_path.parts
        if len(path_parts) >= 2:
            unique_id = "%s_%s" % (path_parts[-2], image_path.stem)
        else:
            unique_id = "sample_%03d_%s" % (sample_idx, image_path.stem)

        input_image = Image.open(image_path).convert("RGB")
        input_source_str = str(image_path)

    # Resize input image to target resolution
    target_width = inference_args["width"]
    target_height = inference_args["height"]
    if input_image.size != (target_width, target_height):
        input_image = input_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    # Determine frame count from action data
    sample_for_cond = dict(sample)
    original_action_num_frames = None
    action_for_len = _load_action_data(sample_for_cond)

    if action_for_len is not None:
        original_action_num_frames = int(action_for_len.shape[0])
        num_frames = round_up_to_4n_plus_1(original_action_num_frames)
        sample_for_cond["num_frames"] = num_frames
        sample_for_cond["video_indices"] = list(range(num_frames))
        if num_frames != original_action_num_frames:
            print("[A2V] Action frames rounded: %d -> %d (4n+1 alignment)"
                  % (original_action_num_frames, num_frames))
    else:
        num_frames = inference_args["num_frames"]
        sample_for_cond["num_frames"] = num_frames
        sample_for_cond["video_indices"] = list(range(num_frames))

    # Generate VACE action condition
    target_size = (target_height, target_width)
    action_cond = _generate_action_condition(sample_for_cond, target_size)
    vace_video = None
    has_action = False

    if action_cond is not None:
        vace_video = _action_condition_to_vace_video_tensor(
            action_cond, num_frames, target_height, target_width,
        )
        has_action = True
        print("[A2V] Action condition generated: action_cond=%s, vace_video=%s, num_frames=%d"
              % (tuple(action_cond.shape), tuple(vace_video.shape), num_frames))
    else:
        print("[A2V] No action condition, falling back to pure TI2V inference")

    # Text condition control
    disable_text_condition = inference_args.get("disable_text_condition", False)
    if disable_text_condition:
        prompt = ""
        negative_prompt = ""
    else:
        negative_prompt = inference_args.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)

    # Run pipeline
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image=input_image,
        height=target_height,
        width=target_width,
        num_frames=num_frames,
        num_inference_steps=inference_args["num_inference_steps"],
        cfg_scale=inference_args["cfg_scale"],
        seed=inference_args["seed"],
        tiled=inference_args.get("tiled", False),
        vace_video=vace_video,
    )

    # Convert to PIL list
    video_frames = []
    for frame in video:
        if isinstance(frame, Image.Image):
            video_frames.append(frame)
        else:
            video_frames.append(Image.fromarray(np.array(frame)))

    # Trim padding frames to match original action length
    if original_action_num_frames is not None:
        keep_n = min(original_action_num_frames, len(video_frames))
        if keep_n != len(video_frames):
            print("[A2V] Trimming video frames: %d -> %d" % (len(video_frames), keep_n))
            video_frames = video_frames[:keep_n]

    # Save output
    case_dir = Path(output_dir) / unique_id
    case_dir.mkdir(parents=True, exist_ok=True)

    # Save video
    video_mp4_path = case_dir / "video.mp4"
    save_video(video_frames, video_mp4_path, fps=15, quality=5)

    # Save individual frames
    frames_dir = case_dir / "video_frames"
    frames_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(video_frames):
        frame.save(frames_dir / ("frame_%04d.png" % i))

    # Generate trajectory visualization if requested
    if has_action and action_cond is not None and inference_args.get("overlay_action_condition", False):
        overlay_mode = True  # 50% alpha overlay
        combined_frames = _create_overlay_video(
            video_frames, action_cond, original_action_num_frames, overlay_mode,
        )
        if combined_frames:
            save_video(combined_frames, case_dir / "video_with_traj.mp4", fps=15, quality=5)
            print("[A2V] Trajectory overlay video saved")

    return {
        "index": sample_idx,
        "source": input_source_str,
        "prompt": prompt,
        "output": str(case_dir),
        "status": "success",
        "has_action": has_action,
        "action_frames": original_action_num_frames,
        "output_frames": len(video_frames),
    }


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ABot-PhysWorld A2V (Action-to-Video) VACE Inference"
    )

    # Data
    parser.add_argument("--jsonl_path", type=str, required=True,
                        help="Path to JSONL file with inference data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for generated videos")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to process (default: all)")

    # Checkpoints
    parser.add_argument("--dit_checkpoint_path", type=str, default="",
                        help="Path to DiT .safetensors checkpoint (auto-download if empty)")
    parser.add_argument("--vace_checkpoint_path", type=str, default="",
                        help="Path to VACE .safetensors checkpoint (auto-download if empty)")
    parser.add_argument("--checkpoint_cache_dir", type=str, default="./checkpoints",
                        help="Directory for caching downloaded checkpoints")

    # Inference parameters
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Default frame count (overridden by action length when available)")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tiled", action="store_true", help="Use tiled VAE decoding")

    # A2V specific
    parser.add_argument("--vace_in_dim", type=int, default=96,
                        help="VACE input dimension (must match training config)")
    parser.add_argument("--disable_text_condition", action="store_true",
                        help="Disable text condition (pure action control)")
    parser.add_argument("--overlay_action_condition", action="store_true",
                        help="Generate trajectory overlay visualization video")
    parser.add_argument("--save_first_frames", action="store_true",
                        help="Save extracted first frames to output directory")

    args = parser.parse_args()

    # Resolve checkpoint paths
    dit_checkpoint_path = args.dit_checkpoint_path
    vace_checkpoint_path = args.vace_checkpoint_path

    if not dit_checkpoint_path or not os.path.exists(dit_checkpoint_path):
        print("[A2V] DiT checkpoint not found locally, attempting ModelScope download...")
        try:
            dit_checkpoint_path = download_checkpoint(
                DIT_CHECKPOINT_FILENAME, cache_dir=args.checkpoint_cache_dir,
            )
        except Exception as e:
            print("[WARNING] Failed to download DiT checkpoint: %s" % e)
            print("[WARNING] Proceeding with base Wan2.1 I2V weights")
            dit_checkpoint_path = ""

    if not vace_checkpoint_path or not os.path.exists(vace_checkpoint_path):
        print("[A2V] VACE checkpoint not found locally, attempting ModelScope download...")
        try:
            vace_checkpoint_path = download_checkpoint(
                VACE_CHECKPOINT_FILENAME, cache_dir=args.checkpoint_cache_dir,
            )
        except Exception as e:
            print("[ERROR] Failed to download VACE checkpoint: %s" % e)
            print("[ERROR] A VACE checkpoint is required for A2V inference.")
            print("  Please specify --vace_checkpoint_path or upload checkpoint to ModelScope.")
            sys.exit(1)

    # Build pipeline
    pipe = build_pipeline(
        dit_checkpoint_path=dit_checkpoint_path,
        vace_checkpoint_path=vace_checkpoint_path,
        vace_in_dim=args.vace_in_dim,
    )

    # Load data
    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError("JSONL file not found: %s" % jsonl_path)

    samples = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if args.num_samples is not None:
        samples = samples[:args.num_samples]

    print("[A2V] Loaded %d samples from %s" % (len(samples), jsonl_path))

    # Prepare output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inference_args = {
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "tiled": args.tiled,
        "disable_text_condition": args.disable_text_condition,
        "overlay_action_condition": args.overlay_action_condition,
        "save_first_frames": args.save_first_frames,
    }

    # Process samples
    results = []
    for idx, sample in enumerate(samples):
        print("\n[A2V] Processing sample %d/%d ..." % (idx + 1, len(samples)))
        try:
            result = process_sample(pipe, sample, idx, str(output_dir), inference_args)
            results.append(result)
            print("[A2V] Sample %d: %s" % (idx, result["status"]))
        except Exception as e:
            import traceback
            error_msg = "%s\n%s" % (str(e), traceback.format_exc())
            results.append({
                "index": idx,
                "source": sample.get("video", "unknown"),
                "prompt": sample.get("prompt", ""),
                "status": "error: %s" % error_msg,
            })
            print("[A2V] Sample %d failed: %s" % (idx, str(e)))

    # Summary
    success_count = sum(1 for r in results if r["status"] == "success")
    action_count = sum(1 for r in results if r.get("has_action", False))
    print("\n" + "=" * 60)
    print("[A2V] Inference complete!")
    print("  Total: %d, Success: %d, With Action: %d"
          % (len(results), success_count, action_count))
    print("=" * 60)

    # Save results
    result_file = output_dir / "inference_results.json"
    with result_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("[A2V] Results saved to: %s" % result_file)


if __name__ == "__main__":
    main()
