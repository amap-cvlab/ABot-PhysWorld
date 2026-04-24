#!/usr/bin/env python3
"""
ABot-PhysWorld DPO Data Preprocessing Script

Pre-process DPO preference data (winner/loser video pairs) into .pth cache files
containing VAE-encoded latents, T5 text embeddings, and CLIP image features.

This preprocessing step is required before running train_dpo.py, as it:
    - Encodes videos into VAE latent space
    - Encodes text prompts via T5
    - Encodes first frames via CLIP
    - Saves everything as .pth for fast training data loading

The cached tensors include both winner and loser video encodings with the same
noise and timestep, ready for DPO training.

Features:
    - JSONL input format with winner/loser video paths and prompts
    - Tiled VAE encoding support for large resolutions
    - Automatic skip for already-processed samples
    - PyTorch Lightning test loop for distributed processing

Usage:
    python preprocess_dpo.py \\
        --dpo_jsonl /path/to/dpo_pairs.jsonl \\
        --cache_dir /path/to/output_cache

    # Or use the shell wrapper
    bash run_preprocess_dpo.sh

Requirements:
    - DiffSynth-Studio (bundled in ../inference/diffsynth/)
    - lightning >= 2.0.0
    - Pre-trained Wan2.1-I2V-14B-480P model weights
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add inference directory to path for importing bundled diffsynth module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'inference'))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import lightning as pl
from torchvision.transforms import v2
from einops import rearrange
from PIL import Image
import torchvision
import imageio

from diffsynth import WanVideoPipeline, ModelManager


class DPOTI2VJsonlDataset(torch.utils.data.Dataset):
    """Dataset for DPO preference pairs from JSONL file.

    Each JSONL line should contain:
        - winner_video: path to the preferred video
        - loser_video: path to the non-preferred video
        - prompt: text description

    Videos are loaded, center-cropped, resized, and normalized to [-1, 1].
    First frames are extracted for TI2V conditioning.
    """

    def __init__(self, jsonl_path, num_frames=81, height=480, width=832, max_samples=0):
        self.rows = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

        if max_samples > 0:
            self.rows = self.rows[:max_samples]

        print("[DPO preprocess] Loaded %d pairs from %s" % (len(self.rows), jsonl_path))

        self.num_frames = num_frames
        self.height = height
        self.width = width

        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        return image

    def load_frames_from_video(self, file_path):
        """Load frames from video file using imageio."""
        try:
            reader = imageio.get_reader(file_path)
            meta = reader.get_meta_data()
            fps = meta.get("fps") or meta.get("fps0", 25)

            if reader.count_frames() < self.num_frames:
                reader.close()
                return None, fps

            frames = []
            for frame_id in range(self.num_frames):
                frame = reader.get_data(frame_id)
                frame = Image.fromarray(frame)
                frame = self.crop_and_resize(frame)
                frame = self.frame_process(frame)
                frames.append(frame)
            reader.close()

            frames = torch.stack(frames, dim=0)
            frames = rearrange(frames, "T C H W -> C T H W")
            return frames, fps
        except Exception as e:
            print("[DPO preprocess] Error loading %s: %s" % (file_path, e))
            return None, 25

    def __getitem__(self, index):
        try:
            r = self.rows[index]
            text = r["prompt"]
            winner_path = r["winner_video"]
            loser_path = r["loser_video"]

            winner_frames, fps_w = self.load_frames_from_video(winner_path)
            loser_frames, fps_l = self.load_frames_from_video(loser_path)

            if winner_frames is None or loser_frames is None:
                print("[DPO preprocess] Skip index %d: video load failed" % index)
                return self.__getitem__((index + 1) % len(self))

            # TI2V: extract first frame for image conditioning
            winner_first_frame = winner_frames[:, :1, :, :]  # [C, 1, H, W]
            loser_first_frame = loser_frames[:, :1, :, :]

            return {
                "text": text,
                "winner_video": winner_frames,
                "loser_video": loser_frames,
                "winner_first_frame": winner_first_frame,
                "loser_first_frame": loser_first_frame,
                "winner_path": winner_path,
                "loser_path": loser_path,
                "index": index,
                "fps": fps_w,
            }
        except Exception as e:
            print("[DPO preprocess] Error in __getitem__(%d): %s" % (index, e))
            return self.__getitem__((index + 1) % len(self))

    def __len__(self):
        return len(self.rows)


class LightningModelForDataProcessTI2V(pl.LightningModule):
    """Lightning module for DPO TI2V cache generation.

    Uses ModelManager + WanVideoPipeline.from_model_manager to load models,
    then encodes video/text/image data and saves as .pth cache files.

    Only implements test_step (no training), run via trainer.test().
    """

    def __init__(
        self,
        cache_dir,
        model_id_with_origin_paths,
        model_root=None,
        height=480,
        width=832,
        num_frames=81,
        tiled=False,
        tile_size=(34, 34),
        tile_stride=(18, 16),
    ):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_frames = num_frames
        self.height = height
        self.width = width

        print("[DPO preprocess] Initializing ModelManager and WanVideoPipeline...")
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")

        model_paths = self._parse_model_paths(model_id_with_origin_paths, model_root)
        model_manager.load_models(model_paths)

        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        self.tiler_kwargs = {
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }

        print("[DPO preprocess] Pipeline initialized")

    def _parse_model_paths(self, model_id_with_origin_paths, model_root=None):
        """Parse model_id_with_origin_paths into file paths for ModelManager.

        Args:
            model_id_with_origin_paths: Comma-separated "model_id:pattern" strings
            model_root: Root directory for model files (default: auto-detect)

        Returns:
            List of file paths for ModelManager.load_models()
        """
        import glob

        if model_root is None:
            model_root = os.environ.get("DIFFSYNTH_MODEL_PATH", "./models")

        parts = model_id_with_origin_paths.split(",")
        parsed_paths = []

        for part in parts:
            if ":" in part:
                model_id, pattern = part.split(":", 1)
                base_path = os.path.join(model_root, model_id)

                if "*" in pattern:
                    matched = sorted(glob.glob(os.path.join(base_path, pattern)))
                    if matched:
                        parsed_paths.append(matched)
                        print("[DPO preprocess] Found %d DiT files: %s/%s"
                              % (len(matched), base_path, pattern))
                    else:
                        print("[WARNING] No files matched %s/%s" % (base_path, pattern))
                else:
                    full_path = os.path.join(base_path, pattern)
                    if os.path.exists(full_path):
                        parsed_paths.append(full_path)
                        print("[DPO preprocess] Found: %s" % pattern)
                    else:
                        print("[WARNING] File not found: %s" % full_path)
            else:
                print("[WARNING] Invalid format: %s" % part)

        return parsed_paths

    def test_step(self, batch, batch_idx):
        """Encode one DPO pair and save as .pth cache.

        Steps:
            1. Encode text prompt (T5)
            2. Encode winner/loser videos (VAE)
            3. Encode winner/loser first frames (CLIP + VAE)
            4. Generate noise and timestep
            5. Save all tensors to .pth file
        """
        text = batch["text"][0]
        winner_video = batch["winner_video"]
        loser_video = batch["loser_video"]
        winner_first_frame = batch["winner_first_frame"]
        loser_first_frame = batch["loser_first_frame"]
        winner_path = batch["winner_path"][0]
        loser_path = batch["loser_path"][0]
        index = batch["index"][0].item()

        winner_name = Path(winner_path).stem
        loser_name = Path(loser_path).stem
        save_name = "%s_%s.tensors.pth" % (winner_name, loser_name)
        save_path = self.cache_dir / save_name

        if save_path.exists():
            print("[DPO preprocess] %s exists, skipping" % save_path)
            return

        self.pipe.device = self.device

        winner_video = winner_video.to(device=self.device, dtype=torch.bfloat16)
        loser_video = loser_video.to(device=self.device, dtype=torch.bfloat16)

        with torch.no_grad():
            # 1. Encode text prompt
            prompt_emb = self.pipe.encode_prompt(text)

            # 2. Encode winner/loser videos
            latents_w = self.pipe.encode_video(winner_video, **self.tiler_kwargs)[0]
            latents_l = self.pipe.encode_video(loser_video, **self.tiler_kwargs)[0]

            latents_w = latents_w.to(self.device)
            latents_l = latents_l.to(self.device)

            # 3. Encode first frames (CLIP features + VAE conditioning)
            def encode_first_frame(first_frame_tensor):
                """Encode first frame for TI2V conditioning."""
                first_frame = first_frame_tensor[0].squeeze(1)  # [C, H, W]
                first_frame_np = first_frame.permute(1, 2, 0).cpu().numpy()
                first_frame_np = ((first_frame_np * 0.5 + 0.5) * 255).clip(0, 255).astype("uint8")
                first_frame_pil = Image.fromarray(first_frame_np)

                image_emb = self.pipe.encode_image(
                    first_frame_pil,
                    end_image=None,
                    num_frames=self.num_frames,
                    height=self.height,
                    width=self.width,
                    **self.tiler_kwargs,
                )

                if "clip_feature" in image_emb:
                    clip_fea = image_emb["clip_feature"]
                elif "clip_fea" in image_emb:
                    clip_fea = image_emb["clip_fea"]
                else:
                    raise KeyError(
                        "Neither 'clip_feature' nor 'clip_fea' found in image_emb. Keys: %s"
                        % list(image_emb.keys())
                    )

                y = image_emb["y"]
                return clip_fea, y

            clip_fea_w, y_w = encode_first_frame(winner_first_frame)
            clip_fea_l, y_l = encode_first_frame(loser_first_frame)

            # 4. Generate noise and timestep
            timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
            timestep = self.pipe.scheduler.timesteps[timestep_id].to(self.device)
            noise = torch.randn_like(latents_l).to(self.device)

            # 5. Compute noisy latents and training targets
            noisy_latents_w = self.pipe.scheduler.add_noise(
                latents_w.unsqueeze(0), noise, timestep
            )
            training_target_w = self.pipe.scheduler.training_target(
                latents_w.unsqueeze(0), noise, timestep
            )
            noisy_latents_l = self.pipe.scheduler.add_noise(
                latents_l.unsqueeze(0), noise, timestep
            )
            training_target_l = self.pipe.scheduler.training_target(
                latents_l.unsqueeze(0), noise, timestep
            )

            # 6. Save cache
            data = {
                "latents_w": latents_w.cpu(),
                "latents_l": latents_l.cpu(),
                "noise": noise.cpu(),
                "timestep": timestep.cpu(),
                "prompt_emb": {k: v.cpu() for k, v in prompt_emb.items()},
                "clip_fea_w": clip_fea_w.cpu(),
                "clip_fea_l": clip_fea_l.cpu(),
                "y_w": y_w.cpu(),
                "y_l": y_l.cpu(),
            }

            torch.save(data, save_path)
            print("[DPO preprocess] Saved: %s" % save_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ABot-PhysWorld DPO Data Preprocessing"
    )
    parser.add_argument("--task", type=str, default="data_process",
                        choices=["data_process"])
    parser.add_argument("--dpo_jsonl", type=str, required=True,
                        help="Path to DPO JSONL file (winner_video, loser_video, prompt)")
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Output directory for .pth cache files")
    parser.add_argument("--model_id_with_origin_paths", type=str,
                        default="Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors,"
                                "Wan-AI/Wan2.1-I2V-14B-480P:models_t5_umt5-xxl-enc-bf16.pth,"
                                "Wan-AI/Wan2.1-I2V-14B-480P:Wan2.1_VAE.pth,"
                                "Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                        help="Model paths in format: model_id:pattern,...")
    parser.add_argument("--model_root", type=str, default=None,
                        help="Root directory for model files (default: DIFFSYNTH_MODEL_PATH env or ./models)")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--tiled", default=False, action="store_true",
                        help="Enable tiled VAE encoding")
    parser.add_argument("--tile_size_height", type=int, default=34)
    parser.add_argument("--tile_size_width", type=int, default=34)
    parser.add_argument("--tile_stride_height", type=int, default=18)
    parser.add_argument("--tile_stride_width", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to process (0 = all)")
    parser.add_argument("--output_path", type=str, default="./",
                        help="Output path for trainer logs")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    return parser.parse_args()


def data_process(args):
    """Run DPO data preprocessing via PyTorch Lightning test loop."""

    def custom_collate(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    dataset = DPOTI2VJsonlDataset(
        jsonl_path=args.dpo_jsonl,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        max_samples=args.max_samples,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        collate_fn=custom_collate,
    )

    model = LightningModelForDataProcessTI2V(
        cache_dir=args.cache_dir,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        model_root=args.model_root,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )

    trainer.test(model, dataloader)


if __name__ == "__main__":
    args = parse_args()
    if args.task == "data_process":
        data_process(args)
