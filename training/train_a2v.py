#!/usr/bin/env python3
"""
ABot-PhysWorld A2V (Action-to-Video) VACE Training Script

Action-conditioned video generation training using VACE (Video Action Condition Encoding).
This script fine-tunes a VACE module on top of the DiT backbone for physically consistent
robot manipulation video generation controlled by action trajectories.

The action condition is injected via parallel context blocks (VACE) that residually
inject spatial action maps into cloned DiT blocks, preserving physical priors while
supporting cross-embodiment control.

Features:
    - VACE module training with action trajectory injection
    - Action trajectory map generation from end-effector poses and camera parameters
    - VACE initialization from pre-trained DiT weights
    - Chunk-based temporal sampling for long video training
    - Multiple temporal sampling strategies (uniform, stride-based)
    - Encoded feature caching (save/load VAE, T5, CLIP encodings)
    - Resume training from checkpoint

Usage:
    # First-time A2V VACE training
    bash run_train_a2v.sh

    # Resume from checkpoint
    bash run_train_a2v_resume.sh

    # Direct launch
    accelerate launch --config_file=accelerate_config_zero2.yaml \\
        train_a2v.py \\
        --dataset_base_path /path/to/dataset \\
        --dataset_metadata_path /path/to/metadata.jsonl \\
        --trainable_models vace \\
        --action_condition_enabled true \\
        --output_path ./outputs/a2v_training

Requirements:
    - DiffSynth-Studio (bundled in ../inference/diffsynth/)
    - accelerate, deepspeed
    - h5py (for reading .h5 action files)
    - scipy (for temporal interpolation)
    - torch >= 2.0, CUDA GPU with >= 60GB VRAM (recommended)
"""

import sys
import os
import json
import random
import numpy as np
import torch
import imageio
from pathlib import Path
from einops import rearrange
from PIL import Image

# Add inference directory to path for importing bundled diffsynth module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'inference'))

from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new_vace import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import (
    DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
)
from diffsynth.trainers.unified_dataset import (
    UnifiedDataset, LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
)
from diffsynth.utils.action_utils import (
    parse_h5,
    load_actions_with_quat,
    load_camera_params_from_json,
    load_camera_params_from_npy,
    simple_radius_gen_func,
    get_vace_traj_maps_with_scaled_intrinsic,
    adjust_intrinsic_for_resize_and_crop,
    adjust_intrinsic_for_resize_stretch,
    get_traj_maps,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanActionVaceTrainingModule(DiffusionTrainingModule):
    """Training module for A2V VACE action-conditioned video generation.

    Action conditions are injected via vace_context into VaceWanModel, adding
    hints at specified DiT layers without expanding the DiT input channels.
    """

    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        audio_processor_config=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=32,
        lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        timestep_sample_mode="uniform",
        timestep_strata_high_frac=0.25,
        timestep_strata_mid_frac=0.45,
        timestep_strata_low_frac=0.30,
        timestep_strata_high_weight=0.15,
        timestep_strata_mid_weight=0.5,
        timestep_strata_low_weight=0.35,
        timestep_logit_mu=0.35,
        timestep_logit_sigma=0.85,
        action_condition_enabled=True,
        action_condition_channels=9,
        action_space="eef",
        action_temporal_mode="interpolate",
        disable_text_condition=False,
        text_dropout_rate=0.0,
        save_encoded_cache=False,
        encoded_cache_dir=None,
        skip_vae=False,
        skip_text_encoder=False,
        skip_image_encoder=False,
        realtime_text_encode=False,
        visualize_condition_steps=0,
        visualize_output_dir=None,
        chunk_num_frames=41,
        min_stride=1,
        max_stride=1,
        init_vace_from_dit=True,
        init_vace_from_dit_vace_in_dim=None,
        vace_layers_step=None,
        chunk_uniform_sampling=False,
        video_resize_mode="stretch",
    ):
        super().__init__()

        self.vace_layers_step = vace_layers_step
        self.chunk_num_frames = chunk_num_frames
        self.min_stride = max(int(min_stride), 1)
        self.max_stride = max(int(max_stride), self.min_stride)
        self.chunk_uniform_sampling = chunk_uniform_sampling
        self.video_resize_mode = (video_resize_mode or "stretch").lower().strip()
        if self.video_resize_mode not in ("crop", "stretch"):
            raise ValueError(
                "video_resize_mode must be 'crop' or 'stretch', got: %s" % video_resize_mode
            )
        if self.video_resize_mode == "stretch":
            print("[A2V] resize_mode=stretch: video frames are bilinearly stretched to training resolution (no center_crop)")

        self.action_condition_enabled = action_condition_enabled
        self.action_condition_channels = action_condition_channels
        self.action_space = action_space
        self.action_temporal_mode = action_temporal_mode

        if self.action_temporal_mode == "channel_concat":
            self.action_condition_channels_actual = self.action_condition_channels * 4  # 36
        else:
            self.action_condition_channels_actual = self.action_condition_channels  # 9

        self.disable_text_condition = disable_text_condition
        self.text_dropout_rate = min(max(float(text_dropout_rate), 0.0), 1.0)
        self.init_vace_from_dit = init_vace_from_dit
        self.init_vace_from_dit_vace_in_dim = (
            init_vace_from_dit_vace_in_dim or self.action_condition_channels_actual
        )

        if realtime_text_encode and skip_text_encoder:
            print("=" * 60)
            print("[WARNING] realtime_text_encode=True but skip_text_encoder=True, "
                  "auto-disabling skip_text_encoder")
            print("=" * 60)
            skip_text_encoder = False

        model_configs = self.parse_model_configs(
            model_paths, model_id_with_origin_paths, enable_fp8_training=False
        )

        if skip_vae or skip_text_encoder or skip_image_encoder:
            filtered_configs = []
            skipped_models = []
            for config in model_configs:
                pattern = config.origin_file_pattern if hasattr(config, 'origin_file_pattern') else ""
                if skip_vae and "VAE" in pattern:
                    skipped_models.append("VAE (%s)" % pattern)
                    continue
                if skip_text_encoder and ("t5" in pattern.lower() or "text" in pattern.lower()):
                    skipped_models.append("TextEncoder (%s)" % pattern)
                    continue
                if skip_image_encoder and ("clip" in pattern.lower() or "image" in pattern.lower()):
                    skipped_models.append("ImageEncoder (%s)" % pattern)
                    continue
                filtered_configs.append(config)
            if skipped_models:
                print("[A2V cache mode] Skipping model loading: %s" % skipped_models)
            model_configs = filtered_configs

        if audio_processor_config is not None:
            audio_processor_config = ModelConfig(
                model_id=audio_processor_config.split(":")[0],
                origin_file_pattern=audio_processor_config.split(":")[1]
            )
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            audio_processor_config=audio_processor_config,
            init_vace_from_dit=False,
        )

        # Initialize VACE from DiT if action condition is enabled
        if (self.action_condition_enabled and self.init_vace_from_dit
                and self.pipe.dit is not None and self.pipe.vace is None):
            vace_in_dim = self.init_vace_from_dit_vace_in_dim
            self.pipe.vace = WanVideoPipeline.create_vace_from_dit(
                self.pipe.dit,
                vace_in_dim=vace_in_dim,
                zero_init_extra=True,
                vace_layers_step=self.vace_layers_step,
            )
            self.pipe.vace = self.pipe.vace.to(
                dtype=torch.bfloat16,
                device=next(self.pipe.dit.parameters()).device,
            )
            print("=" * 60)
            print("[VACE init from DiT] Created and zero-initialized extra parameters "
                  "(vace_in_dim=%d, vace_layers_step=%s)" % (vace_in_dim, self.vace_layers_step))
            print("=" * 60)
            if hasattr(self.pipe.vace, "vace_layers"):
                print("[VACE] vace_layers=%s, num_vace_layers=%d"
                      % (self.pipe.vace.vace_layers, len(self.pipe.vace.vace_layers)))

        # Set up trainable models
        trainable_list = (trainable_models or "dit").split(",")
        if self.action_condition_enabled and "vace" not in trainable_list:
            trainable_list.append("vace")
        self.switch_pipe_to_training_mode(
            self.pipe, ",".join(trainable_list),
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.timestep_sample_mode = (timestep_sample_mode or "uniform").lower().strip()
        self.timestep_strata_high_frac = float(timestep_strata_high_frac)
        self.timestep_strata_mid_frac = float(timestep_strata_mid_frac)
        self.timestep_strata_low_frac = float(timestep_strata_low_frac)
        self.timestep_strata_high_weight = float(timestep_strata_high_weight)
        self.timestep_strata_mid_weight = float(timestep_strata_mid_weight)
        self.timestep_strata_low_weight = float(timestep_strata_low_weight)
        self.timestep_logit_mu = float(timestep_logit_mu)
        self.timestep_logit_sigma = float(timestep_logit_sigma)
        if self.timestep_sample_mode != "uniform":
            print("=" * 60)
            print("[Training timestep sampling] mode=%s" % self.timestep_sample_mode)
            if self.timestep_sample_mode == "stratified":
                print("   strata fracs (high/mid/low index): %.3f / %.3f / %.3f"
                      % (self.timestep_strata_high_frac, self.timestep_strata_mid_frac, self.timestep_strata_low_frac))
                print("   sample weights: %.3f / %.3f / %.3f"
                      % (self.timestep_strata_high_weight, self.timestep_strata_mid_weight, self.timestep_strata_low_weight))
            elif self.timestep_sample_mode == "logit_normal":
                print("   logit_normal mu=%.4f sigma=%.4f" % (self.timestep_logit_mu, self.timestep_logit_sigma))
            print("=" * 60)

        self.save_encoded_cache = save_encoded_cache
        self.encoded_cache_dir = encoded_cache_dir
        if self.save_encoded_cache and self.encoded_cache_dir:
            Path(self.encoded_cache_dir).mkdir(parents=True, exist_ok=True)
        self.skip_vae = skip_vae
        self.skip_text_encoder = skip_text_encoder
        self.skip_image_encoder = skip_image_encoder
        self.realtime_text_encode = realtime_text_encode

        self.visualize_condition_steps = visualize_condition_steps
        self.visualize_output_dir = visualize_output_dir
        self.global_step = 0
        self._sampling_log_step = 0
        if self.visualize_condition_steps > 0:
            if self.visualize_output_dir is None:
                self.visualize_output_dir = "./visualize_conditions"
            Path(self.visualize_output_dir).mkdir(parents=True, exist_ok=True)

        if self.action_condition_enabled:
            print("=" * 60)
            print("[A2V VACE training mode] Enabled:")
            print("   - Condition method: VACE mid-layer hint injection (no DiT channel expansion)")
            print("   - vace_video=(1,3,T,H,W) tensor, vace_context=96ch(VAE latent+mask)")
            print("   - action_temporal_mode: %s" % self.action_temporal_mode)
            print("=" * 60)
        if self.disable_text_condition:
            print("[A2V] Text condition disabled: all prompts will be forced to empty")
        elif self.text_dropout_rate > 0:
            print("[A2V] Text dropout enabled: text_dropout_rate=%.3f" % self.text_dropout_rate)

    def _timestep_sampling_inputs(self):
        """Return timestep sampling configuration as a dict for pipeline."""
        return {
            "timestep_sample_mode": self.timestep_sample_mode,
            "timestep_strata_high_frac": self.timestep_strata_high_frac,
            "timestep_strata_mid_frac": self.timestep_strata_mid_frac,
            "timestep_strata_low_frac": self.timestep_strata_low_frac,
            "timestep_strata_high_weight": self.timestep_strata_high_weight,
            "timestep_strata_mid_weight": self.timestep_strata_mid_weight,
            "timestep_strata_low_weight": self.timestep_strata_low_weight,
            "timestep_logit_mu": self.timestep_logit_mu,
            "timestep_logit_sigma": self.timestep_logit_sigma,
        }

    def _get_prompt_with_dropout(self, data):
        """Get prompt with optional text dropout for training."""
        if self.disable_text_condition:
            return ""
        prompt = data.get("prompt", "")
        if self.text_dropout_rate > 0 and random.random() < self.text_dropout_rate:
            return ""
        return prompt

    # =====================================================================
    # Action data loading
    # =====================================================================

    def _load_action_data(self, data):
        """Load action data from file (.npy or .h5).

        Attempts to find action file from explicit path or by inferring from video path.
        Returns torch.FloatTensor of shape (T, C) or None if not found.
        """
        action_path = data.get("action_path")
        if action_path is None:
            video_path = data.get("video_path", "")
            if video_path:
                video_dir = Path(video_path).parent
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
        action_path = Path(action_path)
        if not action_path.exists():
            return None
        if action_path.suffix == ".npy":
            return torch.FloatTensor(np.load(str(action_path)))
        if action_path.suffix == ".h5":
            return load_actions_with_quat(str(action_path))
        return None

    def _load_camera_params_raw(self, data, num_frames, frame_indices=None):
        """Load camera intrinsic and extrinsic parameters from JSON or NPY files.

        Falls back to default parameters if files are not found.
        """
        extrinsic_path = data.get("extrinsic_path")
        intrinsic_path = data.get("intrinsic_path")
        video_path = data.get("video_path", "")

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
            return self._get_default_camera_params(num_frames, (480, 640))

        extrinsic_path = Path(extrinsic_path)
        intrinsic_path = Path(intrinsic_path)
        if not extrinsic_path.exists() or not intrinsic_path.exists():
            return self._get_default_camera_params(num_frames, (480, 640))

        if extrinsic_path.suffix == ".json":
            intrinsic, extrinsics = load_camera_params_from_json(
                str(intrinsic_path), str(extrinsic_path),
                frame_indices=frame_indices, original_size=None, target_size=None,
            )
        else:
            intrinsic, extrinsics = load_camera_params_from_npy(
                str(intrinsic_path), str(extrinsic_path),
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

    def _get_default_camera_params(self, num_frames, target_size):
        """Generate default camera parameters (60-degree FOV, identity extrinsics)."""
        h, w = target_size
        fx = fy = w / (2 * np.tan(np.radians(30)))
        intrinsic = torch.eye(3, dtype=torch.float32)
        intrinsic[0, 0], intrinsic[1, 1] = fx, fy
        intrinsic[0, 2], intrinsic[1, 2] = w / 2, h / 2
        extrinsics = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(num_frames, 1, 1)
        return intrinsic, extrinsics

    def _get_original_video_size(self, data):
        """Infer original video resolution from data."""
        if "original_size" in data:
            o = data["original_size"]
            if isinstance(o, (list, tuple)) and len(o) == 2:
                return tuple(o)
        video_frames = data.get("video", [])
        if video_frames and hasattr(video_frames[0], 'size'):
            w, h = video_frames[0].size
            return (h, w)
        return (480, 640)

    # =====================================================================
    # Action condition generation
    # =====================================================================

    def _crop_and_resize_action_condition(self, action_cond, original_size, target_size):
        """Resize and crop action condition tensor to match video preprocessing."""
        orig_h, orig_w = original_size
        tgt_h, tgt_w = target_size
        if orig_h == tgt_h and orig_w == tgt_w:
            return action_cond
        C, t, h, w = action_cond.shape

        if self.video_resize_mode == "stretch":
            action_cond_reshaped = action_cond.reshape(C * t, 1, h, w)
            out = torch.nn.functional.interpolate(
                action_cond_reshaped, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False
            )
            return out.reshape(C, t, tgt_h, tgt_w)

        scale = max(tgt_w / orig_w, tgt_h / orig_h)
        resized_h, resized_w = round(orig_h * scale), round(orig_w * scale)
        ac = action_cond.reshape(C * t, 1, h, w)
        ac = torch.nn.functional.interpolate(
            ac, size=(resized_h, resized_w), mode='bilinear', align_corners=False
        )
        ac = ac.reshape(C, t, resized_h, resized_w)
        crop_top = (resized_h - tgt_h) // 2
        crop_left = (resized_w - tgt_w) // 2
        return ac[:, :, crop_top:crop_top + tgt_h, crop_left:crop_left + tgt_w]

    def _generate_action_condition(self, data, target_size):
        """Generate action condition tensor (trajectory maps) for VACE injection.

        Uses get_vace_traj_maps_with_scaled_intrinsic for stretch mode, which handles
        intrinsic scaling internally. Returns (C, T, H, W) tensor or None.
        """
        if not self.action_condition_enabled:
            return None

        actions = self._load_action_data(data)
        if actions is None:
            return None

        action_num_frames = actions.shape[0]
        frame_indices = data.get("video_indices")

        if frame_indices is not None and len(frame_indices) > 0:
            safe_indices = [min(i, action_num_frames - 1) for i in frame_indices]
            actions = actions[safe_indices]
            num_frames = len(safe_indices)
        else:
            num_frames = len(data["video"])
            if action_num_frames >= num_frames:
                actions = actions[:num_frames]
            else:
                actions = torch.cat(
                    [actions, actions[-1:].repeat(num_frames - action_num_frames, 1)], dim=0
                )

        original_size = self._get_original_video_size(data)
        intrinsic_orig, extrinsics = self._load_camera_params_raw(
            data, num_frames, frame_indices=frame_indices
        )
        extrinsics_v = extrinsics.unsqueeze(0)
        intrinsic_orig_v = intrinsic_orig.unsqueeze(0)
        w2c = torch.linalg.inv(extrinsics_v)
        c2w = extrinsics_v

        trajs_resized = get_vace_traj_maps_with_scaled_intrinsic(
            actions, w2c, c2w, intrinsic_orig_v,
            original_size, target_size,
            radius_gen_func=simple_radius_gen_func,
        ).squeeze(1)

        return trajs_resized

    def _action_condition_to_vace_video_tensor(self, action_cond, num_frames, height, width):
        """Convert action condition (C,T,H,W) to (1,3,T,H,W) tensor for pipeline.

        Extracts first 3 channels (trajectory RGB) and wraps as batch tensor.
        Range is [-1, 1], passed directly to pipeline without PIL conversion.
        """
        if action_cond.dim() == 4:
            rgb = action_cond[:3]
        else:
            rgb = action_cond[0, :3]
        C, T, H, W = rgb.shape
        assert (T, H, W) == (num_frames, height, width), \
            "action_cond size mismatch: got (%d,%d,%d), expected (%d,%d,%d)" % (T, H, W, num_frames, height, width)
        if (T, H, W) != (num_frames, height, width):
            rgb = rgb.unsqueeze(0)
            rgb = torch.nn.functional.interpolate(
                rgb.view(1, 3, T, H, W),
                size=(num_frames, height, width),
                mode="trilinear", align_corners=False,
            ).squeeze(0)
        return rgb.unsqueeze(0)

    # =====================================================================
    # Visualization helpers
    # =====================================================================

    def _visualize_condition_overlay(self, data, action_cond):
        """Save condition overlay video for debugging (trajectory map on top of video)."""
        if self.visualize_output_dir is None:
            return
        video_frames = data.get("video", [])
        if not video_frames:
            return
        video_path = data.get("video_path", "") or (
            getattr(video_frames[0], "filename", None) or ""
        )
        unique_id = Path(video_path).parent.name if video_path else "unknown_%d" % self.global_step
        video_height, video_width = video_frames[0].size[1], video_frames[0].size[0]
        num_video_frames = len(video_frames)

        action_cond_np = action_cond.cpu().numpy()
        _, num_action_frames, action_h, action_w = action_cond_np.shape
        traj_map = action_cond_np[:3]

        overlay_frames = []
        for frame_idx in range(num_video_frames):
            gt_frame = np.array(video_frames[frame_idx])
            action_frame_idx = (
                min(int(frame_idx * num_action_frames / num_video_frames), num_action_frames - 1)
                if num_action_frames != num_video_frames else frame_idx
            )
            traj_frame = np.transpose(traj_map[:, action_frame_idx], (1, 2, 0))
            tmin, tmax = traj_frame.min(), traj_frame.max()
            traj_frame = (traj_frame - tmin) / (tmax - tmin) if tmax > tmin else np.zeros_like(traj_frame)
            traj_frame = (traj_frame * 255).astype(np.uint8)
            if traj_frame.shape[0] != video_height or traj_frame.shape[1] != video_width:
                traj_frame = np.array(
                    Image.fromarray(traj_frame).resize((video_width, video_height), Image.BILINEAR)
                )
            overlay_frames.append(
                (gt_frame.astype(np.float32) * 0.5 + traj_frame.astype(np.float32) * 0.5).astype(np.uint8)
            )

        out_path = Path(self.visualize_output_dir) / ("step_%06d_%s_overlay.mp4" % (self.global_step, unique_id))
        writer = imageio.get_writer(str(out_path), fps=24, codec='libx264', quality=8)
        for f in overlay_frames:
            writer.append_data(f)
        writer.close()
        print("[A2V] Condition overlay saved: %s" % out_path)

    # =====================================================================
    # Forward pass
    # =====================================================================

    def forward_preprocess(self, data):
        """Preprocess data for training: encode video, generate action condition, etc."""
        cache_file = None
        video_path = data.get("video_path") or (
            data.get("video") and (
                getattr(data["video"][0], "filename", None) if isinstance(data["video"], list) else None
            )
        )
        if not video_path and data.get("video"):
            it = data["video"][0] if isinstance(data["video"], list) else data["video"]
            video_path = getattr(it, "filename", None) or (it if isinstance(it, str) else None)

        if self.encoded_cache_dir and video_path:
            import hashlib
            cache_file = Path(self.encoded_cache_dir) / (
                "%s.pth" % hashlib.md5(str(video_path).encode()).hexdigest()
            )

        if not self.save_encoded_cache and self.encoded_cache_dir:
            if not video_path or not cache_file or not cache_file.exists():
                return None

        # ---- Load from cache ----
        if cache_file and cache_file.exists() and not self.save_encoded_cache:
            cached_data = torch.load(cache_file, map_location='cpu')
            inputs_shared = {}
            inputs_posi = {}

            if "input_latents" in cached_data:
                inputs_shared["input_latents"] = cached_data["input_latents"].to(
                    device=self.pipe.device, dtype=self.pipe.torch_dtype
                )
            if "y" in cached_data:
                inputs_shared["y"] = cached_data["y"].to(
                    device=self.pipe.device, dtype=self.pipe.torch_dtype
                )

            if self.realtime_text_encode:
                prompt = self._get_prompt_with_dropout(data)
                text_inputs_shared = {"cfg_scale": 1}
                text_inputs_posi = {"prompt": prompt}
                text_inputs_nega = {}
                for unit in self.pipe.units:
                    if "text" in unit.__class__.__name__.lower() or "t5" in unit.__class__.__name__.lower():
                        text_inputs_shared, text_inputs_posi, text_inputs_nega = self.pipe.unit_runner(
                            unit, self.pipe, text_inputs_shared, text_inputs_posi, text_inputs_nega
                        )
                if "context" in text_inputs_posi:
                    inputs_posi["context"] = text_inputs_posi["context"]
            else:
                if "context" in cached_data:
                    inputs_posi["context"] = cached_data["context"].to(
                        device=self.pipe.device, dtype=self.pipe.torch_dtype
                    )

            if "clip_feature" in cached_data:
                inputs_shared["clip_feature"] = cached_data["clip_feature"].to(
                    device=self.pipe.device, dtype=self.pipe.torch_dtype
                )

            if "action_condition" in cached_data:
                inputs_shared["action_condition"] = cached_data["action_condition"].to(
                    device=self.pipe.device, dtype=self.pipe.torch_dtype
                )
            elif self.action_condition_enabled and "input_latents" in inputs_shared:
                latent_shape = cached_data["input_latents"].shape
                target_size = (latent_shape[-2] * 8, latent_shape[-1] * 8)
                action_cond = self._generate_action_condition(data, target_size)
                if action_cond is not None:
                    inputs_shared["action_condition"] = action_cond

            if "input_latents" in inputs_shared:
                noise = self.pipe.generate_noise(
                    inputs_shared["input_latents"].shape, seed=None,
                    device=self.pipe.device, torch_dtype=self.pipe.torch_dtype,
                )
                inputs_shared["noise"] = inputs_shared["latents"] = noise

            inputs_shared["use_gradient_checkpointing"] = self.use_gradient_checkpointing
            inputs_shared["use_gradient_checkpointing_offload"] = self.use_gradient_checkpointing_offload
            inputs_shared["max_timestep_boundary"] = self.max_timestep_boundary
            inputs_shared["min_timestep_boundary"] = self.min_timestep_boundary
            inputs_shared.update(self._timestep_sampling_inputs())
            inputs_shared["cfg_scale"] = 1
            inputs_shared["cfg_merge"] = False
            inputs_shared["vace_scale"] = 1

            if "vace_context" in cached_data:
                inputs_shared["vace_context"] = cached_data["vace_context"].to(
                    device=self.pipe.device, dtype=self.pipe.torch_dtype
                )

            return inputs_shared

        # ---- Online preprocessing ----
        all_frames = data["video"]
        total_frames = len(all_frames)
        max_chunk_len = self.chunk_num_frames
        cap = min(max_chunk_len, total_frames)
        target_len = ((cap - 1) // 4 + 1) * 4 + 1 if cap > 1 else 1
        if target_len > cap:
            target_len = cap
        target_len = max(1, int(target_len))

        if self.chunk_uniform_sampling:
            # Uniform temporal sampling: evenly space frames across the entire video
            if total_frames <= target_len:
                aligned_len = ((total_frames - 1) // 4) * 4 + 1 if total_frames > 1 else 1
                aligned_len = max(1, min(aligned_len, total_frames))
                if aligned_len >= total_frames:
                    frame_indices = list(range(total_frames))
                else:
                    step = (total_frames - 1) / (aligned_len - 1) if aligned_len > 1 else 0
                    frame_indices = [int(round(i * step)) for i in range(aligned_len)]
                    frame_indices[-1] = total_frames - 1
            else:
                step = (total_frames - 1) / (target_len - 1) if target_len > 1 else 0
                frame_indices = [int(round(i * step)) for i in range(target_len)]
                frame_indices[-1] = total_frames - 1
        else:
            # Stride-based chunk sampling (default)
            min_s, max_s = self.min_stride, self.max_stride
            stride = random.randint(min_s, max_s) if max_s > min_s else min_s
            needed_span = stride * (target_len - 1) + 1

            if total_frames < needed_span:
                max_feasible_stride = (total_frames - 1) // max(target_len - 1, 1)
                allowed_max_stride = min(max_s, max_feasible_stride)
                stride = allowed_max_stride if allowed_max_stride >= min_s else min_s
                if stride == min_s and total_frames < stride * (target_len - 1) + 1:
                    feasible_len = (total_frames - 1) // stride + 1
                    target_len = ((feasible_len - 1) // 4) * 4 + 1 if feasible_len > 1 else 1
                needed_span = stride * (target_len - 1) + 1

            max_start = max(0, total_frames - needed_span)
            start_idx = random.randint(0, max_start) if max_start > 0 else 0
            frame_indices = [start_idx + k * stride for k in range(target_len)]

        data["video_indices"] = frame_indices
        data["video"] = [all_frames[i] for i in frame_indices]

        # Sampling info logging
        self._sampling_log_step += 1
        should_print = (self._sampling_log_step == 1) or (self._sampling_log_step % 50 == 0)
        if should_print:
            actual_frames = len(frame_indices)
            sampling_mode = "uniform" if self.chunk_uniform_sampling else "stride-based"
            if sampling_mode == "stride-based" and len(frame_indices) >= 2:
                actual_stride = frame_indices[1] - frame_indices[0]
                actual_fps = 30.0 / actual_stride if actual_stride > 0 else 0
            elif total_frames > 1 and actual_frames > 1:
                time_span = frame_indices[-1] - frame_indices[0]
                actual_fps = (actual_frames - 1) / time_span * 30.0 if time_span > 0 else 0
            else:
                actual_fps = 0
            print("[Step %d] Sampling: total_frames=%d, sampled_frames=%d, mode=%s, effective_fps=%.2f"
                  % (self.global_step, total_frames, actual_frames, sampling_mode, actual_fps))

        inputs_posi = {"prompt": self._get_prompt_with_dropout(data)}
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1, "tiled": False, "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False, "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            **self._timestep_sampling_inputs(),
        }

        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input in ("reference_image", "vace_reference_image"):
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data.get(extra_input)

        # Generate action condition and convert to VACE video tensor
        target_size = (data["video"][0].size[1], data["video"][0].size[0])
        action_cond = self._generate_action_condition(data, target_size)
        if action_cond is not None:
            inputs_shared["vace_video"] = self._action_condition_to_vace_video_tensor(
                action_cond,
                inputs_shared["num_frames"],
                inputs_shared["height"],
                inputs_shared["width"],
            )

        # Run pipeline units (VAE encoding, text encoding, etc.)
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, _ = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, {}
            )

        # Save encoded cache if requested
        if self.save_encoded_cache and cache_file:
            cache_data = {
                "prompt": data.get("prompt"),
                "input_latents": inputs_shared.get("input_latents", inputs_shared.get("latents")).cpu(),
                "context": inputs_posi.get("context", inputs_posi.get("prompt_emb")).cpu(),
            }
            if "y" in inputs_shared:
                cache_data["y"] = inputs_shared["y"].cpu()
            if "clip_feature" in inputs_shared:
                cache_data["clip_feature"] = inputs_shared["clip_feature"].cpu()
            if "vace_context" in inputs_shared:
                cache_data["vace_context"] = inputs_shared["vace_context"].cpu()
            torch.save(cache_data, cache_file)

        return {**inputs_shared, **inputs_posi}

    def forward(self, data, inputs=None):
        """Compute training loss."""
        if inputs is None:
            inputs = self.forward_preprocess(data)
        if inputs is None:
            return torch.tensor(0.0, device=self.pipe.device, requires_grad=True)

        self.global_step += 1

        # Visualize condition overlay periodically
        if (self.visualize_condition_steps > 0
                and self.global_step % self.visualize_condition_steps == 0
                and inputs.get("vace_video") is not None):
            vv = inputs["vace_video"]
            if isinstance(vv, torch.Tensor):
                traj_tensor = (vv[0] if vv.dim() == 5 else vv).cpu()
            else:
                traj_tensor = torch.stack(
                    [torch.from_numpy(np.array(im)).permute(2, 0, 1) for im in vv]
                ).float() / 255.0 * 2 - 1
                traj_tensor = traj_tensor.permute(1, 0, 2, 3)
            self._visualize_condition_overlay(data, traj_tensor)

        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        return self.pipe.training_loss(**models, **inputs)


# =====================================================================
# Argument parser
# =====================================================================

def action_vace_parser():
    """Create argument parser with A2V VACE-specific arguments."""
    parser = wan_parser()

    # Action condition arguments
    parser.add_argument("--action_condition_enabled", type=str, default="true",
                        help="Enable action condition injection (default: true)")
    parser.add_argument("--action_condition_channels", type=int, default=9,
                        help="Number of action condition channels (default: 9)")
    parser.add_argument("--action_space", type=str, default="eef", choices=["eef", "joint"],
                        help="Action space: eef (end-effector) or joint")
    parser.add_argument("--action_temporal_mode", type=str, default="interpolate",
                        choices=["interpolate", "channel_concat"],
                        help="Action temporal processing: interpolate (9ch) or channel_concat (36ch)")
    parser.add_argument("--disable_text_condition", type=str, default="false",
                        help="Disable text condition (pure action control)")
    parser.add_argument("--text_dropout_rate", type=float, default=0.0,
                        help="Text dropout rate for training (0.0 = no dropout)")

    # Visualization
    parser.add_argument("--visualize_condition_steps", type=int, default=0,
                        help="Visualize action condition overlay every N steps (0 = disabled)")
    parser.add_argument("--visualize_output_dir", type=str, default=None,
                        help="Directory for condition visualization outputs")

    # Chunk-based temporal sampling
    parser.add_argument("--chunk_num_frames", type=int, default=41,
                        help="Maximum frames per training chunk")
    parser.add_argument("--min_stride", type=int, default=1,
                        help="Minimum temporal stride for chunk sampling")
    parser.add_argument("--max_stride", type=int, default=1,
                        help="Maximum temporal stride for chunk sampling")
    parser.add_argument("--chunk_uniform_sampling", type=str, default="false",
                        help="Use uniform temporal sampling across entire video (default: false)")

    # VACE initialization
    parser.add_argument("--init_vace_from_dit", type=str, default="true",
                        help="Initialize VACE module from DiT weights (default: true)")
    parser.add_argument("--init_vace_from_dit_vace_in_dim", type=int, default=None,
                        help="VACE input dimension for DiT initialization (default: auto)")
    parser.add_argument("--vace_layers_step", type=int, default=None,
                        help="VACE layers step for selective layer initialization")

    # Timestep sampling strategies
    parser.add_argument("--timestep_sample_mode", type=str, default="uniform",
                        choices=["uniform", "stratified", "logit_normal"],
                        help="Timestep sampling strategy")
    parser.add_argument("--timestep_strata_high_frac", type=float, default=0.25)
    parser.add_argument("--timestep_strata_mid_frac", type=float, default=0.45)
    parser.add_argument("--timestep_strata_low_frac", type=float, default=0.30)
    parser.add_argument("--timestep_strata_high_weight", type=float, default=0.15)
    parser.add_argument("--timestep_strata_mid_weight", type=float, default=0.5)
    parser.add_argument("--timestep_strata_low_weight", type=float, default=0.35)
    parser.add_argument("--timestep_logit_mu", type=float, default=0.35)
    parser.add_argument("--timestep_logit_sigma", type=float, default=0.85)

    # Video preprocessing
    parser.add_argument("--dataset_video_resize_mode", type=str, default="stretch",
                        choices=["crop", "stretch"],
                        help="Video resize mode: crop (aspect-ratio preserving + center_crop) "
                             "or stretch (bilinear stretch to target resolution)")

    return parser


# =====================================================================
# Main entry point
# =====================================================================

if __name__ == "__main__":
    parser = action_vace_parser()
    args = parser.parse_args()

    action_condition_enabled = args.action_condition_enabled.lower() == "true"
    disable_text_condition = args.disable_text_condition.lower() == "true"
    init_vace_from_dit = getattr(args, "init_vace_from_dit", "true").lower() == "true"
    chunk_num_frames = getattr(args, "chunk_num_frames", args.num_frames)

    # Build dataset
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=-1,
            time_division_factor=4,
            time_division_remainder=1,
            uniform_sampling=False,
            resize_mode=getattr(args, "dataset_video_resize_mode", "stretch"),
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(
                chunk_num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)
            ),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
        },
    )

    # Build training module
    model = WanActionVaceTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        audio_processor_config=args.audio_processor_config,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        timestep_sample_mode=getattr(args, "timestep_sample_mode", "uniform"),
        timestep_strata_high_frac=getattr(args, "timestep_strata_high_frac", 0.25),
        timestep_strata_mid_frac=getattr(args, "timestep_strata_mid_frac", 0.45),
        timestep_strata_low_frac=getattr(args, "timestep_strata_low_frac", 0.30),
        timestep_strata_high_weight=getattr(args, "timestep_strata_high_weight", 0.15),
        timestep_strata_mid_weight=getattr(args, "timestep_strata_mid_weight", 0.5),
        timestep_strata_low_weight=getattr(args, "timestep_strata_low_weight", 0.35),
        timestep_logit_mu=getattr(args, "timestep_logit_mu", 0.35),
        timestep_logit_sigma=getattr(args, "timestep_logit_sigma", 0.85),
        action_condition_enabled=action_condition_enabled,
        action_condition_channels=args.action_condition_channels,
        action_space=args.action_space,
        action_temporal_mode=getattr(args, "action_temporal_mode", "interpolate"),
        disable_text_condition=disable_text_condition,
        text_dropout_rate=getattr(args, "text_dropout_rate", 0.0),
        save_encoded_cache=getattr(args, "save_encoded_cache", False),
        encoded_cache_dir=getattr(args, "encoded_cache_dir", None),
        skip_vae=getattr(args, "skip_vae", False),
        skip_text_encoder=getattr(args, "skip_text_encoder", False),
        skip_image_encoder=getattr(args, "skip_image_encoder", False),
        realtime_text_encode=getattr(args, "realtime_text_encode", False),
        visualize_condition_steps=getattr(args, "visualize_condition_steps", 0),
        visualize_output_dir=getattr(args, "visualize_output_dir", None) or os.path.join(
            args.output_path, "visualize_conditions"
        ),
        chunk_num_frames=chunk_num_frames,
        min_stride=getattr(args, "min_stride", 1),
        max_stride=getattr(args, "max_stride", 1),
        init_vace_from_dit=init_vace_from_dit,
        init_vace_from_dit_vace_in_dim=getattr(args, "init_vace_from_dit_vace_in_dim", None),
        vace_layers_step=getattr(args, "vace_layers_step", None),
        chunk_uniform_sampling=getattr(args, "chunk_uniform_sampling", "false").lower() == "true",
        video_resize_mode=getattr(args, "dataset_video_resize_mode", "stretch"),
    )

    # Load DiT checkpoint if specified (for VACE training on top of SFT DiT)
    dit_checkpoint = getattr(args, "dit_checkpoint", None)
    if dit_checkpoint and os.path.exists(dit_checkpoint) and hasattr(model.pipe, "dit") and model.pipe.dit is not None:
        print("[A2V] Loading DiT checkpoint: %s" % dit_checkpoint)
        dit_state_dict = load_state_dict(dit_checkpoint)
        model.pipe.dit.load_state_dict(dit_state_dict, strict=False)
        print("[A2V] DiT checkpoint loaded")
    elif dit_checkpoint:
        print("[WARNING] DiT checkpoint not found: %s" % dit_checkpoint)

    # Resume from VACE checkpoint
    resume_from_step = getattr(args, "resume_from_step", 0)
    ckpt_path = os.path.join(args.output_path, "step-%d.safetensors" % resume_from_step)
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        resume_from_step=resume_from_step,
    )

    if os.path.exists(ckpt_path):
        ckpt_sd = load_state_dict(ckpt_path)
        prefix = (args.remove_prefix_in_ckpt or "").strip()
        if "vace" in prefix and getattr(model.pipe, "vace", None) is not None:
            model.pipe.vace.load_state_dict(ckpt_sd, strict=False)
            print("[A2V] Resumed VACE weights from %s" % ckpt_path)
        elif "dit" in prefix and getattr(model.pipe, "dit", None) is not None:
            model.pipe.dit.load_state_dict(ckpt_sd, strict=False)
            print("[A2V] Resumed DiT weights from %s" % ckpt_path)
        else:
            if getattr(model.pipe, "vace", None) is not None:
                model.pipe.vace.load_state_dict(ckpt_sd, strict=False)
                print("[A2V] Resumed trainable (VACE) weights from %s" % ckpt_path)
            elif getattr(model.pipe, "dit", None) is not None:
                model.pipe.dit.load_state_dict(ckpt_sd, strict=False)
                print("[A2V] Resumed trainable (DiT) weights from %s" % ckpt_path)
    elif resume_from_step > 0:
        print("[WARNING] Resume checkpoint not found: %s (training will continue with current weights)" % ckpt_path)

    # Launch training
    launch_training_task(dataset, model, model_logger, args=args)
