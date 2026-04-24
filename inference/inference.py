#!/usr/bin/env python3
"""
Abot-PhysWorld Inference Script

Image-to-Video (I2V) inference using fine-tuned Wan2.1-I2V-14B-480P model.
This script loads the Abot-PhysWorld checkpoint from ModelScope and generates
physically plausible robot manipulation videos from input images and text prompts.

Requirements:
    - DiffSynth-Studio (https://github.com/modelscope/DiffSynth-Studio)
    - modelscope (for downloading model weights)
    - torch >= 2.0
    - CUDA GPU with >= 24GB VRAM (recommended >= 60GB for best performance)

Usage:
    # Single image inference
    python inference.py --input_image path/to/image.jpg --prompt "robot arm picks up the red cube"

    # Batch inference from JSONL file
    python inference.py --jsonl_path path/to/data.jsonl --output_dir ./outputs

    # Specify a local checkpoint path (skip auto-download)
    python inference.py --input_image image.jpg --prompt "..." --checkpoint_path ./abotpw_i2v_480p.safetensors
"""

import torch
import argparse
import json
import imageio
import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.models.utils import load_state_dict


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

MODELSCOPE_MODEL_ID = "amap_cvlab/Abot-PhysWorld"
CHECKPOINT_FILENAME = "abotpw_i2v_480p.safetensors"


def download_checkpoint(cache_dir="./checkpoints"):
    """Download the Abot-PhysWorld checkpoint from ModelScope if not already cached."""
    from modelscope import snapshot_download

    checkpoint_path = os.path.join(cache_dir, CHECKPOINT_FILENAME)
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint already exists: {checkpoint_path}")
        return checkpoint_path

    print(f"Downloading checkpoint from ModelScope: {MODELSCOPE_MODEL_ID} ...")
    model_dir = snapshot_download(MODELSCOPE_MODEL_ID, cache_dir=cache_dir)
    downloaded_path = os.path.join(model_dir, CHECKPOINT_FILENAME)

    if os.path.exists(downloaded_path):
        print(f"Checkpoint downloaded to: {downloaded_path}")
        return downloaded_path

    # Try to find the file in the downloaded directory
    for root, dirs, files in os.walk(model_dir):
        for filename in files:
            if filename == CHECKPOINT_FILENAME:
                found_path = os.path.join(root, filename)
                print(f"Checkpoint found at: {found_path}")
                return found_path

    raise FileNotFoundError(
        f"Could not find {CHECKPOINT_FILENAME} after downloading from ModelScope. "
        f"Please download manually from https://www.modelscope.cn/models/{MODELSCOPE_MODEL_ID}/files "
        f"and specify --checkpoint_path."
    )


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


def load_pipeline(device="cuda"):
    """Load the Wan2.1-I2V-14B-480P base pipeline."""
    print("Loading Wan2.1-I2V-14B-480P base model...")
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
    )
    return pipe


def load_checkpoint(pipe, checkpoint_path):
    """Load the Abot-PhysWorld fine-tuned checkpoint into the pipeline."""
    print(f"Loading Abot-PhysWorld checkpoint: {checkpoint_path}")
    checkpoint_state_dict = load_state_dict(checkpoint_path)
    print(f"  Checkpoint contains {len(checkpoint_state_dict)} parameter keys")

    missing_keys, unexpected_keys = pipe.dit.load_state_dict(
        checkpoint_state_dict, strict=False
    )
    print(f"  Loaded - Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

    if unexpected_keys:
        print(f"  (Unexpected keys are normal for fine-tuned checkpoints)")


def generate_video(
    pipe,
    input_image,
    prompt,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    height=480,
    width=832,
    num_frames=81,
    num_inference_steps=50,
    cfg_scale=5.0,
    seed=0,
    tiled=True,
):
    """Generate a video from an input image and text prompt."""
    if input_image.size != (width, height):
        input_image = input_image.resize((width, height), Image.Resampling.LANCZOS)

    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image=input_image,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        seed=seed,
        tiled=tiled,
    )
    return video


def run_single_inference(args):
    """Run inference on a single image."""
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cuda"

    # Download or locate checkpoint
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        checkpoint_path = args.checkpoint_path
    else:
        checkpoint_path = download_checkpoint(cache_dir=args.cache_dir)

    # Load pipeline and checkpoint
    pipe = load_pipeline(device=device)
    load_checkpoint(pipe, checkpoint_path)
    pipe.enable_vram_management()

    # Load input image
    input_image = extract_first_frame(args.input_image)

    # Generate video
    print(f"\nGenerating video...")
    print(f"  Prompt: {args.prompt}")
    print(f"  Resolution: {args.width}x{args.height}, Frames: {args.num_frames}")

    video = generate_video(
        pipe,
        input_image=input_image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        tiled=args.tiled,
    )

    # Save output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_stem = Path(args.input_image).stem
    output_path = output_dir / f"{input_stem}_generated.mp4"
    save_video(video, str(output_path), fps=15, quality=5)
    print(f"\nVideo saved to: {output_path}")

    # Cleanup
    del pipe, video
    torch.cuda.empty_cache()


def run_batch_inference(args):
    """Run batch inference from a JSONL file."""
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cuda"

    # Download or locate checkpoint
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        checkpoint_path = args.checkpoint_path
    else:
        checkpoint_path = download_checkpoint(cache_dir=args.cache_dir)

    # Read JSONL data
    samples = []
    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if args.num_samples > 0:
        samples = samples[: args.num_samples]

    print(f"Loaded {len(samples)} samples from {args.jsonl_path}")

    # Load pipeline and checkpoint
    pipe = load_pipeline(device=device)
    load_checkpoint(pipe, checkpoint_path)
    pipe.enable_vram_management()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    first_frames_dir = output_dir / "first_frames"
    if args.save_first_frames:
        first_frames_dir.mkdir(exist_ok=True)

    # Run inference
    results = []
    for sample_idx, sample in enumerate(tqdm(samples, desc="Inference")):
        video_path = Path(sample["video"])
        prompt = sample["prompt"]

        if not video_path.exists():
            print(f"  [SKIP] Video not found: {video_path}")
            results.append({
                "index": sample_idx,
                "video": str(video_path),
                "prompt": prompt,
                "status": "error: file not found",
            })
            continue

        # Build unique output name
        path_parts = video_path.parts
        if len(path_parts) >= 2:
            unique_id = f"{path_parts[-2]}_{video_path.stem}"
        else:
            unique_id = f"sample_{sample_idx:04d}_{video_path.stem}"

        try:
            # Extract first frame
            first_frame_path = None
            if args.save_first_frames:
                first_frame_path = str(first_frames_dir / f"{unique_id}_first_frame.jpg")

            input_image = extract_first_frame(str(video_path), first_frame_path)

            # Generate video
            video = generate_video(
                pipe,
                input_image=input_image,
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                seed=args.seed,
                tiled=args.tiled,
            )

            # Save video
            output_video_path = output_dir / f"{unique_id}_generated.mp4"
            save_video(video, str(output_video_path), fps=15, quality=5)

            results.append({
                "index": sample_idx,
                "video": str(video_path),
                "prompt": prompt,
                "output_video": str(output_video_path),
                "status": "success",
            })

            del video
            torch.cuda.empty_cache()

        except Exception as exc:
            import traceback
            traceback.print_exc()
            results.append({
                "index": sample_idx,
                "video": str(video_path),
                "prompt": prompt,
                "status": f"error: {exc}",
            })

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"\nDone! {success_count}/{len(results)} samples succeeded.")
    print(f"Results saved to: {results_path}")

    # Cleanup
    del pipe
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Abot-PhysWorld: Image-to-Video inference for physically plausible robot manipulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image inference
  python inference.py --input_image demo.jpg --prompt "robot arm picks up the red cube"

  # Batch inference from JSONL
  python inference.py --jsonl_path data.jsonl --output_dir ./outputs

  # Use a local checkpoint (skip download)
  python inference.py --input_image demo.jpg --prompt "..." --checkpoint_path ./abotpw_i2v_480p.safetensors

  # Adjust generation parameters
  python inference.py --input_image demo.jpg --prompt "..." --num_frames 41 --seed 42
        """,
    )

    # Input mode (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_image", type=str,
        help="Path to a single input image (or video, first frame will be extracted)",
    )
    input_group.add_argument(
        "--jsonl_path", type=str,
        help="Path to a JSONL file for batch inference. Each line should have 'video' and 'prompt' fields",
    )

    # Prompt (required for single-image mode)
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt (required for single-image mode)")

    # Model
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="Path to the fine-tuned checkpoint (abotpw_i2v_480p.safetensors). If not provided, will auto-download from ModelScope",
    )
    parser.add_argument(
        "--cache_dir", type=str, default="./checkpoints",
        help="Directory to cache downloaded model weights (default: ./checkpoints)",
    )

    # Generation parameters
    parser.add_argument("--height", type=int, default=480, help="Video height (default: 480)")
    parser.add_argument("--width", type=int, default=832, help="Video width (default: 832)")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate (default: 81)")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps (default: 50)")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="Classifier-free guidance scale (default: 5.0)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt for generation",
    )
    parser.add_argument("--no_tiled", action="store_true", help="Disable tiled VAE (uses more VRAM)")

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory (default: ./outputs)")
    parser.add_argument("--num_samples", type=int, default=0, help="Max samples for batch mode (0 = all, default: 0)")
    parser.add_argument("--save_first_frames", action="store_true", help="Save extracted first frames as images")

    # Hardware
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID (default: 0)")

    args = parser.parse_args()
    args.tiled = not args.no_tiled

    # Validate arguments
    if args.input_image and not args.prompt:
        parser.error("--prompt is required when using --input_image")

    if args.input_image:
        if not os.path.exists(args.input_image):
            parser.error(f"Input image not found: {args.input_image}")
        run_single_inference(args)
    else:
        if not os.path.exists(args.jsonl_path):
            parser.error(f"JSONL file not found: {args.jsonl_path}")
        run_batch_inference(args)


if __name__ == "__main__":
    main()
