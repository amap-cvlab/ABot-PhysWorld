#!/usr/bin/env python3
"""
ABot-PhysWorld Training Script

Full-parameter SFT training for Wan2.1-I2V-14B-480P using DiffSynth-Studio.
This script fine-tunes the DiT model for physically consistent robot manipulation
video generation.

Features:
    - Full-parameter SFT (non-LoRA) or LoRA training
    - DeepSpeed ZeRO-2 distributed training via Accelerate
    - Encoded feature caching (save/load VAE, T5, CLIP encodings)
    - Real-time text encoding with cached video features
    - Resume training from checkpoint

Usage:
    # Full SFT training (single machine, 8 GPUs)
    bash run_train.sh

    # Resume from checkpoint
    bash run_train_resume.sh

    # Direct launch
    accelerate launch --config_file=accelerate_config_zero2.yaml \\
        train.py \\
        --dataset_base_path /path/to/dataset \\
        --dataset_metadata_path /path/to/metadata.jsonl \\
        --trainable_models dit \\
        --output_path ./outputs/sft_training

Requirements:
    - DiffSynth-Studio (bundled in ../inference/diffsynth/)
    - accelerate, deepspeed
    - torch >= 2.0, CUDA GPU with >= 60GB VRAM (recommended)
"""

import sys
import os

# Add inference directory to path for importing bundled diffsynth module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'inference'))

import torch
import json
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import (
    DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
)
from diffsynth.trainers.unified_dataset import (
    UnifiedDataset, LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    """Training module for Wan2.1-I2V-14B video generation model."""

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
        save_encoded_cache=False,
        encoded_cache_dir=None,
        skip_vae=False,
        skip_text_encoder=False,
        skip_image_encoder=False,
        realtime_text_encode=False,
    ):
        super().__init__()

        # Real-time text encoding mode: must keep text encoder loaded
        if realtime_text_encode and skip_text_encoder:
            print("=" * 60)
            print("Warning: realtime_text_encode=True but skip_text_encoder=True")
            print("  Real-time text encoding requires text encoder, disabling skip_text_encoder")
            print("=" * 60)
            skip_text_encoder = False

        # Parse model configurations
        model_configs = self.parse_model_configs(
            model_paths, model_id_with_origin_paths, enable_fp8_training=False
        )

        # Filter out models that should be skipped (for cache-based training)
        if skip_vae or skip_text_encoder or skip_image_encoder:
            filtered_configs = []
            skipped_models = []

            for config in model_configs:
                pattern = config.origin_file_pattern if hasattr(config, 'origin_file_pattern') else ""

                if skip_vae and "VAE" in pattern:
                    skipped_models.append(f"VAE ({pattern})")
                    continue
                if skip_text_encoder and ("t5" in pattern.lower() or "text" in pattern.lower()):
                    skipped_models.append(f"TextEncoder ({pattern})")
                    continue
                if skip_image_encoder and ("clip" in pattern.lower() or "image" in pattern.lower()):
                    skipped_models.append(f"ImageEncoder ({pattern})")
                    continue

                filtered_configs.append(config)

            if skipped_models:
                print("=" * 60)
                print("[Cache Training Mode] Skipping the following models to save VRAM:")
                for model in skipped_models:
                    print(f"   - {model}")
                print("=" * 60)

            model_configs = filtered_configs

        # Load models
        if audio_processor_config is not None:
            audio_processor_config = ModelConfig(
                model_id=audio_processor_config.split(":")[0],
                origin_file_pattern=audio_processor_config.split(":")[1],
            )
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            audio_processor_config=audio_processor_config,
        )

        # Set up training mode
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        # Store configurations
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

        # Encoded cache configuration
        self.save_encoded_cache = save_encoded_cache
        self.encoded_cache_dir = encoded_cache_dir
        self.cache_index = {}
        if self.save_encoded_cache and self.encoded_cache_dir:
            from pathlib import Path
            Path(self.encoded_cache_dir).mkdir(parents=True, exist_ok=True)
            print(f"Encoded cache mode enabled, save directory: {self.encoded_cache_dir}")
            self.cache_index_file = Path(self.encoded_cache_dir) / "cache_index.json"

        self.skip_vae = skip_vae
        self.skip_text_encoder = skip_text_encoder
        self.skip_image_encoder = skip_image_encoder
        self.realtime_text_encode = realtime_text_encode

        if self.realtime_text_encode:
            print("=" * 60)
            print("[Real-time Text Encoding Mode] Enabled:")
            print("   - Using cached video features (input_latents, y, clip_feature)")
            print("   - Reading new prompts from JSONL and encoding text in real-time")
            print("   - Use case: video unchanged, but captions re-annotated")
            print("=" * 60)

    def forward_preprocess(self, data):
        """Preprocess data: load from cache or encode from raw data."""
        cache_file = None
        if self.encoded_cache_dir:
            from pathlib import Path

            # Determine video path for cache key
            if "video_path" in data:
                video_path = data["video_path"]
            elif "video" in data:
                video_item = data["video"][0] if isinstance(data["video"], list) else data["video"]
                if hasattr(video_item, 'filename') and video_item.filename:
                    video_path = str(video_item.filename)
                elif isinstance(video_item, str):
                    video_path = video_item
                else:
                    video_path = None
            else:
                video_path = None

            if video_path:
                # Use MD5 hash of video path as cache filename for uniqueness
                import hashlib
                cache_key = hashlib.md5(video_path.encode()).hexdigest()
                cache_file = Path(self.encoded_cache_dir) / f"{cache_key}.pth"

            # Load from cache if available and not in save mode
            if cache_file and cache_file.exists() and not self.save_encoded_cache:
                try:
                    cached_data = torch.load(cache_file, map_location='cpu')
                except (EOFError, RuntimeError, Exception) as error:
                    error_type = type(error).__name__
                    print(f"    [Cache] Warning: corrupted cache file, skipping: {cache_file.name}")
                    print(f"              Error: {error_type}: {error}")

                    if not hasattr(self, '_skipped_cache_count'):
                        self._skipped_cache_count = 0
                    self._skipped_cache_count += 1
                    if self._skipped_cache_count % 10 == 0:
                        print(f"    [Cache] Skipped {self._skipped_cache_count} corrupted cache files so far")

                    return None

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

                # Real-time text encoding: encode new prompt instead of using cached context
                if self.realtime_text_encode:
                    prompt = data.get("prompt", "")
                    text_inputs_shared = {"cfg_scale": 1}
                    text_inputs_posi = {"prompt": prompt}
                    text_inputs_nega = {}

                    for unit in self.pipe.units:
                        unit_name = unit.__class__.__name__.lower()
                        if "text" in unit_name or "t5" in unit_name or "prompt" in unit_name:
                            text_inputs_shared, text_inputs_posi, text_inputs_nega = self.pipe.unit_runner(
                                unit, self.pipe, text_inputs_shared, text_inputs_posi, text_inputs_nega
                            )

                    if "context" in text_inputs_posi:
                        inputs_posi["context"] = text_inputs_posi["context"]

                    if hasattr(self, '_realtime_encode_count'):
                        self._realtime_encode_count += 1
                        if self._realtime_encode_count % 100 == 0:
                            print(f"    [RealtimeText] Encoded {self._realtime_encode_count} texts so far")
                    else:
                        self._realtime_encode_count = 1
                        print(f"    [RealtimeText] Starting real-time text encoding (using cached video features)...")
                else:
                    if "context" in cached_data:
                        inputs_posi["context"] = cached_data["context"].to(
                            device=self.pipe.device, dtype=self.pipe.torch_dtype
                        )

                if "clip_feature" in cached_data:
                    inputs_shared["clip_feature"] = cached_data["clip_feature"].to(
                        device=self.pipe.device, dtype=self.pipe.torch_dtype
                    )

                # Generate noise (consistent with source logic)
                if "input_latents" in inputs_shared:
                    noise_shape = inputs_shared["input_latents"].shape
                    noise = self.pipe.generate_noise(
                        noise_shape,
                        seed=None,
                        device=self.pipe.device,
                        torch_dtype=self.pipe.torch_dtype,
                    )
                    inputs_shared["noise"] = noise
                    inputs_shared["latents"] = noise

                # Add required training parameters
                inputs_shared["use_gradient_checkpointing"] = self.use_gradient_checkpointing
                inputs_shared["use_gradient_checkpointing_offload"] = self.use_gradient_checkpointing_offload
                inputs_shared["max_timestep_boundary"] = self.max_timestep_boundary
                inputs_shared["min_timestep_boundary"] = self.min_timestep_boundary
                inputs_shared["cfg_scale"] = 1
                inputs_shared["cfg_merge"] = False
                inputs_shared["vace_scale"] = 1

                if hasattr(self, '_cache_load_count'):
                    self._cache_load_count += 1
                    if self._cache_load_count % 100 == 0:
                        print(f"    [Cache] Loaded {self._cache_load_count} cached features")
                else:
                    self._cache_load_count = 1
                    if self.realtime_text_encode:
                        print(f"    [Cache] Using encoded cache (video features) + real-time text encoding...")
                    else:
                        print(f"    [Cache] Using encoded cache...")

                return {**inputs_shared, **inputs_posi}

        # ========== Raw data pipeline (real-time encoding) ==========
        if "video" not in data or not isinstance(data.get("video"), list):
            print(f"Warning: video frames not loaded and cache miss, skipping sample "
                  f"(video_path={data.get('video_path', 'unknown')})")
            return None

        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}

        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }

        # Extra inputs (e.g., input_image for I2V)
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input in ("reference_image", "vace_reference_image"):
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]

        # Run pipeline encoding units
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )

        # Save encoded features to cache
        if self.save_encoded_cache and cache_file:
            cache_data = {
                "prompt": data["prompt"],
                "input_latents": inputs_shared.get("input_latents", inputs_shared.get("latents")).cpu(),
                "context": inputs_posi.get("context", inputs_posi.get("prompt_emb")).cpu(),
            }
            if "y" in inputs_shared:
                cache_data["y"] = inputs_shared["y"].cpu()
            if "clip_feature" in inputs_shared:
                cache_data["clip_feature"] = inputs_shared["clip_feature"].cpu()

            torch.save(cache_data, cache_file)

            if video_path:
                self.cache_index[video_path] = cache_file.name

            if hasattr(self, '_cache_save_count'):
                self._cache_save_count += 1
                if self._cache_save_count % 100 == 0:
                    print(f"    [Cache] Saved {self._cache_save_count} encoded features")
                if self._cache_save_count % 1000 == 0:
                    try:
                        self._save_cache_index()
                    except Exception as error:
                        print(f"    [Cache] Warning: index save failed (training unaffected): {error}")
            else:
                self._cache_save_count = 1
                print(f"    [Cache] Saving encoded features to: {self.encoded_cache_dir}")

        return {**inputs_shared, **inputs_posi}

    def _save_cache_index(self):
        """Save cache index to JSON file with multi-process concurrency support."""
        if not (hasattr(self, 'cache_index_file') and self.cache_index):
            return

        import fcntl
        import time
        from pathlib import Path

        cache_index_file = Path(self.cache_index_file)

        try:
            with open(cache_index_file, 'a+') as file_handle:
                try:
                    fcntl.flock(file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except IOError:
                    time.sleep(0.1)
                    fcntl.flock(file_handle, fcntl.LOCK_EX)

                file_handle.seek(0)
                existing_content = file_handle.read()
                existing_index = {}
                if existing_content.strip():
                    try:
                        existing_index = json.loads(existing_content)
                    except json.JSONDecodeError:
                        existing_index = {}

                merged_index = {**existing_index, **self.cache_index}

                file_handle.seek(0)
                file_handle.truncate()
                json.dump(merged_index, file_handle, indent=2, ensure_ascii=False)
                file_handle.flush()

                fcntl.flock(file_handle, fcntl.LOCK_UN)

                print(f"    [Cache] Index saved: {cache_index_file} "
                      f"(new: {len(self.cache_index)}, total: {len(merged_index)})")
        except Exception as error:
            print(f"    [Cache] Warning: file lock unavailable ({error}), using simple mode")
            existing_index = {}
            if cache_index_file.exists():
                try:
                    with open(cache_index_file, 'r') as file_handle:
                        existing_index = json.load(file_handle)
                except (json.JSONDecodeError, IOError):
                    existing_index = {}

            merged_index = {**existing_index, **self.cache_index}
            with open(cache_index_file, 'w') as file_handle:
                json.dump(merged_index, file_handle, indent=2, ensure_ascii=False)
            print(f"    [Cache] Index saved: {cache_index_file} "
                  f"(new: {len(self.cache_index)}, total: {len(merged_index)})")

    def forward(self, data, inputs=None):
        """Compute training loss."""
        if inputs is None:
            inputs = self.forward_preprocess(data)

        # Skip corrupted cache samples
        if inputs is None:
            return torch.tensor(0.0, device=self.pipe.device, dtype=self.pipe.torch_dtype, requires_grad=False)

        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()

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
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
            uniform_sampling=getattr(args, 'uniform_sampling', 'True').lower() == 'true',
        ),
        special_operator_map={
            "animate_face_video": (
                ToAbsolutePath(args.dataset_base_path)
                >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16))
            ),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
        },
        encoded_cache_dir=getattr(args, 'encoded_cache_dir', None),
        save_encoded_cache=getattr(args, 'save_encoded_cache', False),
    )

    model = WanTrainingModule(
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
        save_encoded_cache=getattr(args, 'save_encoded_cache', False),
        encoded_cache_dir=getattr(args, 'encoded_cache_dir', None),
        skip_vae=getattr(args, 'skip_vae', False),
        skip_text_encoder=getattr(args, 'skip_text_encoder', False),
        skip_image_encoder=getattr(args, 'skip_image_encoder', False),
        realtime_text_encode=getattr(args, 'realtime_text_encode', False),
    )

    # Load DIT checkpoint for resume training
    dit_checkpoint = getattr(args, 'dit_checkpoint', None)
    if dit_checkpoint is not None and os.path.exists(dit_checkpoint):
        print("=" * 60)
        print(f"Loading DIT checkpoint from: {dit_checkpoint}")
        dit_state_dict = load_state_dict(dit_checkpoint)

        if hasattr(model.pipe, 'dit') and model.pipe.dit is not None:
            missing_keys, unexpected_keys = model.pipe.dit.load_state_dict(dit_state_dict, strict=False)
            print("DIT checkpoint loaded successfully")
            if len(missing_keys) > 0:
                print(f"   Missing keys: {len(missing_keys)}")
            if len(unexpected_keys) > 0:
                print(f"   Unexpected keys: {len(unexpected_keys)}")
        else:
            print("Warning: DIT model not found in pipeline, checkpoint not loaded")
        print("=" * 60)
    elif dit_checkpoint is not None:
        print(f"Warning: DIT checkpoint not found: {dit_checkpoint}")

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        resume_from_step=getattr(args, 'resume_from_step', 0),
    )

    launch_training_task(dataset, model, model_logger, args=args)

    # Save final cache index after training
    if hasattr(model, '_save_cache_index'):
        try:
            model._save_cache_index()
            print("\nCache index saved successfully")
        except Exception as error:
            print(f"\nWarning: cache index save failed (cache files are intact): {error}")
