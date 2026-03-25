#!/usr/bin/env python3
"""
EZS-Bench: Embodied Zero-Shot Benchmark for Physically Consistent Video Generation

This script evaluates video generation methods on two complementary dimensions:

1. **Video Quality Metrics** (8 metrics via PAI-Bench):
   - aesthetic_quality, imaging_quality, motion_smoothness
   - background_consistency, subject_consistency, overall_consistency
   - i2v_background, i2v_subject

2. **Domain Score** (Robot Fidelity via VQA):
   - Robot Score: evaluates robot morphology and motion plausibility
   - Computed via VQA with a large vision-language model (default: Qwen2.5-VL-72B-Instruct)

Prerequisites:
    pip install -e .   (from the EZS-Bench root directory)

Usage:
    # Run both evaluations (requires torchrun for multi-GPU video quality metrics)
    torchrun --standalone --nproc_per_node=4 evaluate_ezsbench.py \
        --method_name "YourMethod" \
        --method_dir /path/to/generated_videos \
        --prompt_file /path/to/prompts.jsonl \
        --vqa_questions_file /path/to/vqa_questions.jsonl \
        --output_dir ./results

    # Run only Domain Score (no torchrun needed)
    python evaluate_ezsbench.py \
        --method_name "YourMethod" \
        --method_dir /path/to/generated_videos \
        --prompt_file /path/to/prompts.jsonl \
        --vqa_questions_file /path/to/vqa_questions.jsonl \
        --skip_video_quality \
        --output_dir ./results

    # Run only video quality metrics
    torchrun --standalone --nproc_per_node=4 evaluate_ezsbench.py \
        --method_name "YourMethod" \
        --method_dir /path/to/generated_videos \
        --prompt_file /path/to/prompts.jsonl \
        --skip_domain_score \
        --output_dir ./results

    # Evaluate multiple methods at once
    torchrun --standalone --nproc_per_node=4 evaluate_ezsbench.py \
        --methods_config methods.json \
        --prompt_file /path/to/prompts.jsonl \
        --vqa_questions_file /path/to/vqa_questions.jsonl \
        --output_dir ./results

Input File Formats:
    See README.md for detailed format specifications.
"""

import json
import os
import re
import shutil
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import torch

# PAI-Bench imports (install from https://github.com/SHI-Labs/physical-ai-bench)
from pbench import PBench
from pbench.distributed import dist_init, get_rank, print0

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent

VIDEO_QUALITY_METRICS = [
    "aesthetic_quality",
    "background_consistency",
    "imaging_quality",
    "motion_smoothness",
    "overall_consistency",
    "subject_consistency",
    "i2v_background",
    "i2v_subject",
]

# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_prompt_data(prompt_file: Path) -> Tuple[Dict[str, Dict], List[Tuple[str, Dict]]]:
    """Load prompt data from a JSONL file.

    Each line should be a JSON object with at least:
        - "video_id": unique identifier for the sample
        - "prompt": text description of the expected video content

    Optional fields:
        - "image_path": path to the conditioning image (for I2V evaluation)

    Returns:
        A tuple of (dict keyed by video_id, ordered list of (video_id, info) pairs).
    """
    dataset = {}
    ordered = []

    with open(prompt_file, "r") as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            video_id = data["video_id"]
            info = {
                "prompt": data.get("prompt", ""),
                "prompt_en": data.get("prompt_en", data.get("prompt", "")),
                "image_path": data.get("image_path", ""),
            }
            dataset[video_id] = info
            ordered.append((video_id, info))

    return dataset, ordered

def load_vqa_questions(questions_file: Path) -> Dict[str, Dict]:
    """Load VQA question sets from a JSONL file.

    Each line should be a JSON object with:
        - "video_id": unique identifier matching the prompt data
        - "questions": list of question dicts, each containing:
            - "uid": unique question identifier
            - "question": the question text
            - "answer": correct answer key (e.g., "A")
            - "index2ans": mapping from answer keys to text (e.g., {"A": "yes", "B": "no"})
            - "task": category label (e.g., "robot")

    Returns:
        Dict mapping video_id to question data.
    """
    questions_dict = {}

    with open(questions_file, "r") as file_handle:
        for line in file_handle:
            data = json.loads(line.strip())
            video_id = data["video_id"]
            questions = data.get("questions", [])
            if not questions:
                continue
            questions_dict[video_id] = {
                "video_id": video_id,
                "questions": questions,
            }

    return questions_dict

def load_combined_data(
    data_file: Path,
) -> Tuple[Dict[str, Dict], List[Tuple[str, Dict]], Dict[str, Dict]]:
    """Load a combined JSONL file containing prompts, images, and VQA questions.

    Each line should be a JSON object with:
        - "video": path to the conditioning image (used as image_path;
          the filename stem is used as video_id)
        - "prompt": text description of the expected video content
        - "question": list of VQA question dicts

    Returns:
        A tuple of (prompt_data dict, ordered prompt list, questions dict).
    """
    prompt_dataset = {}
    prompt_ordered = []
    questions_dict = {}

    with open(data_file, "r") as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            video_path = data.get("video", "")
            video_id = Path(video_path).stem
            prompt_text = data.get("prompt", "")
            questions = data.get("question", [])

            info = {
                "prompt": prompt_text,
                "prompt_en": prompt_text,
                "image_path": video_path,
            }
            prompt_dataset[video_id] = info
            prompt_ordered.append((video_id, info))

            if questions:
                questions_dict[video_id] = {
                    "video_id": video_id,
                    "questions": questions,
                }

    return prompt_dataset, prompt_ordered, questions_dict


def discover_generated_videos(method_dir: Path, prompt_data: Dict[str, Dict]) -> List[Dict]:
    """Discover generated video files in a method's output directory.

    Supports two naming conventions:
        1. {video_id}.mp4  — direct match by video_id
        2. Any .mp4 files  — matched to prompts in order (fallback)

    Args:
        method_dir: Directory containing generated video files.
        prompt_data: Dict of prompt data keyed by video_id.

    Returns:
        List of entry dicts with video_id, prompt, and video_path.
    """
    entries = []

    if not method_dir.exists():
        print0(f"  [WARNING] Method directory does not exist: {method_dir}")
        return entries

    # Strategy 1: Try to match by video_id in filename
    for video_id, info in prompt_data.items():
        candidate = method_dir / f"{video_id}.mp4"
        if candidate.exists():
            entries.append({
                "video_id": video_id,
                "prompt": info["prompt"],
                "prompt_en": info.get("prompt_en", info["prompt"]),
                "image_path": info.get("image_path", ""),
                "video_path": str(candidate),
            })

    if entries:
        return entries

    # Strategy 2: Fallback — match all .mp4 files in sorted order to prompts
    video_files = sorted(method_dir.rglob("*.mp4"))
    prompt_list = list(prompt_data.items())

    for index, video_file in enumerate(video_files):
        if index >= len(prompt_list):
            break
        video_id, info = prompt_list[index]
        entries.append({
            "video_id": video_id,
            "prompt": info["prompt"],
            "prompt_en": info.get("prompt_en", info["prompt"]),
            "image_path": info.get("image_path", ""),
            "video_path": str(video_file),
        })

    return entries


# ──────────────────────────────────────────────────────────────────────────────
# Video Quality Metrics (8 Metrics via PAI-Bench)
# ──────────────────────────────────────────────────────────────────────────────

def _copy_or_link(source: Path, destination: Path):
    """Copy a file, falling back to symlink if copy fails."""
    if destination.exists():
        return
    try:
        shutil.copy2(source, destination)
    except (PermissionError, OSError):
        try:
            destination.symlink_to(source.resolve())
        except OSError:
            with open(source, "rb") as src, open(destination, "wb") as dst:
                dst.write(src.read())


def _prepare_condition_image(
    entry: Dict, video_id: str, image_dir: Path, video_path: Path
) -> Optional[str]:
    """Prepare a conditioning image for I2V metrics.

    If an image_path is provided in the entry, copies it. Otherwise, extracts
    the first frame from the video as a fallback.

    Returns:
        Path string to the prepared image, or None.
    """
    input_path = entry.get("image_path", "")
    if input_path:
        source_image = Path(input_path)
        if source_image.exists():
            extension = source_image.suffix.lower()
            if extension not in (".jpg", ".jpeg", ".png"):
                extension = ".jpg"
            destination_image = image_dir / f"{video_id}{extension}"
            _copy_or_link(source_image, destination_image)
            if destination_image.exists():
                return str(destination_image)

    # Fallback: extract first frame from video
    try:
        import cv2
        capture = cv2.VideoCapture(str(video_path))
        success, frame = capture.read()
        capture.release()
        if success:
            destination_image = image_dir / f"{video_id}.jpg"
            cv2.imwrite(str(destination_image), frame)
            return str(destination_image)
    except Exception:
        pass

    return None


def prepare_video_quality_eval_dir(
    entries: List[Dict], output_base: Path, method_subdir: str
) -> Tuple[Path, Path, Optional[Path]]:
    """Prepare the directory structure required by the 8-metric evaluator.

    Copies (or symlinks) generated videos and conditioning images into a
    standardized layout, and writes a prompt JSON file.

    Returns:
        Tuple of (video_dir, prompt_file, custom_image_dir_or_none).
    """
    eval_dir = output_base / method_subdir / "video_quality"
    eval_dir.mkdir(parents=True, exist_ok=True)

    video_dir = eval_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    image_dir = eval_dir / "condition_images"
    image_dir.mkdir(exist_ok=True)

    prompt_data = []
    has_any_image = False
    for entry in entries:
        video_id = entry["video_id"]
        source_video = Path(entry["video_path"])
        destination_video = video_dir / f"{video_id}.mp4"

        if not source_video.exists():
            continue

        _copy_or_link(source_video, destination_video)

        custom_image_path = _prepare_condition_image(
            entry, video_id, image_dir, destination_video
        )
        if custom_image_path:
            has_any_image = True

        item = {
            "video_id": video_id,
            "prompt": entry.get("prompt_en", entry["prompt"]),
            "prompt_en": entry.get("prompt_en", entry["prompt"]),
        }
        if custom_image_path:
            item["custom_image_path"] = custom_image_path
        prompt_data.append(item)

    prompt_file = eval_dir / "prompts.json"
    with open(prompt_file, "w") as file_handle:
        json.dump(prompt_data, file_handle, indent=2, ensure_ascii=False)

    return video_dir, prompt_file, (image_dir if has_any_image else None)


def run_video_quality_evaluation(
    video_dir: Path,
    prompt_file: Path,
    output_dir: Path,
    method_name: str,
    custom_image_folder: Optional[Path] = None,
) -> Dict[str, float]:
    """Run the 8-metric video quality evaluation using PBench in-process.

    This calls PBench.evaluate() directly within the current distributed
    process group (initialized by torchrun).

    Args:
        video_dir: Directory containing the prepared video files.
        prompt_file: Path to the prompt JSON file.
        output_dir: Directory to write evaluation results.
        method_name: Display name of the method being evaluated.
        custom_image_folder: Optional path to conditioning images.

    Returns:
        Dict mapping metric names to their scores.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir = output_dir / "evaluation_results"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    print0(f"  Running video quality evaluation: {method_name}")

    device = torch.device("cuda")
    full_json_dir = str(SCRIPT_DIR / "pbench" / "VBench_full_info.json")

    # PBench accepts a dummy full_json_dir when using custom_input mode;
    # it is not actually read in that mode.
    evaluator = PBench(device, full_json_dir, str(eval_output_dir))

    # Build prompt_list as a dict: {video_filename: prompt_dict}
    with open(prompt_file, "r") as file_handle:
        prompt_data = json.load(file_handle)

    video_path_to_prompt = {}
    for item in prompt_data:
        video_id = item["video_id"]
        video_filename = f"{video_id}.mp4"
        video_path_to_prompt[video_filename] = item

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    evaluator.evaluate(
        videos_path=str(video_dir),
        name=f"results_{current_time}",
        prompt_list=video_path_to_prompt,
        dimension_list=VIDEO_QUALITY_METRICS,
        local=True,
        read_frame=False,
        mode="custom_input",
        custom_image_folder=str(custom_image_folder) if custom_image_folder else None,
        enable_missing_videos=True,
    )

    print0(f"  Video quality evaluation complete: {method_name}")

    # Load results
    return _load_video_quality_results(eval_output_dir)


def _load_video_quality_results(eval_output_dir: Path) -> Dict[str, float]:
    """Load evaluation results from the PBench evaluator output."""
    if not eval_output_dir or not eval_output_dir.exists():
        return {}

    result_files = list(eval_output_dir.glob("results_*_eval_results.json"))
    if not result_files:
        return {}

    latest_file = max(result_files, key=lambda path: path.stat().st_mtime)
    with open(latest_file, "r") as file_handle:
        data = json.load(file_handle)

    raw_results = data.get("results", data)
    results = {}
    for metric in VIDEO_QUALITY_METRICS:
        value = raw_results.get(metric)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            results[metric] = float(value)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
            results[metric] = float(value[0])

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Domain Score (Robot Score + Physics Score via VQA)
# ──────────────────────────────────────────────────────────────────────────────

def prepare_domain_score_dir(
    questions_dict: Dict,
    entries: List[Dict],
    output_base: Path,
    method_subdir: str,
) -> Tuple[Path, Path, Path]:
    """Prepare the directory structure for Domain Score evaluation.

    Creates VQA question files (one per video) and links generated videos
    into a standardized layout expected by compute_vqa_accuracy().

    Returns:
        Tuple of (video_dir, vqa_questions_dir, prompt_file) paths.
    """
    eval_dir = output_base / method_subdir / "domain_score"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Write individual VQA question files (one per video)
    vqa_dir = eval_dir / "vqa_questions"
    vqa_dir.mkdir(exist_ok=True)

    question_count = 0
    for video_id, data in questions_dict.items():
        questions = data["questions"]
        if not questions:
            continue
        question_file = vqa_dir / f"robot_{video_id}.json"
        with open(question_file, "w") as file_handle:
            json.dump(questions, file_handle, indent=2, ensure_ascii=False)
        question_count += 1

    print0(f"  Created {question_count} VQA question files")

    # Link generated videos with the expected naming convention
    video_output_dir = eval_dir / "videos"
    video_output_dir.mkdir(exist_ok=True)

    entry_ids = {entry["video_id"] for entry in entries}
    linked_count = 0
    for entry in entries:
        video_id = entry["video_id"]
        if video_id not in questions_dict:
            continue

        source_video = Path(entry["video_path"])
        if not source_video.exists():
            continue

        destination = video_output_dir / f"robot_{video_id}.mp4"
        if destination.exists():
            destination.unlink()

        try:
            destination.symlink_to(source_video.resolve())
        except OSError:
            shutil.copy2(source_video, destination)

        linked_count += 1

    print0(f"  Matched {linked_count} videos with VQA questions")

    # Write prompt file for VQA evaluator
    prompt_data = []
    for video_id in questions_dict:
        if video_id not in entry_ids:
            continue
        for entry in entries:
            if entry["video_id"] == video_id:
                prompt_data.append({
                    "video_id": f"robot_{video_id}",
                    "prompt": entry.get("prompt_en", entry["prompt"]),
                    "prompt_en": entry.get("prompt_en", entry["prompt"]),
                })
                break

    prompt_file = eval_dir / "prompts_for_vqa.json"
    with open(prompt_file, "w") as file_handle:
        json.dump(prompt_data, file_handle, indent=2, ensure_ascii=False)

    return video_output_dir, vqa_dir, prompt_file


def run_domain_score_evaluation(
    video_dir: Path,
    vqa_questions_dir: Path,
    prompt_file: Path,
    output_dir: Path,
    method_name: str,
    shared_evaluator: Any,
    model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct",
    tensor_parallel_size: int = 4,
    batch_size: int = 32,
    gpu_memory_utilization: float = 0.75,
) -> Dict:
    """Run Domain Score evaluation using a VLM-based VQA evaluator.

    Evaluates generated videos against physics and robot-specific checklists
    using a shared Qwen2.5-VL evaluator instance.

    Returns:
        Summary dict with overall_accuracy, robot_score, and physics_score.
    """
    if not vqa_questions_dir.exists():
        print0(f"  [ERROR] VQA questions directory not found: {vqa_questions_dir}")
        return {}

    if shared_evaluator is None:
        print0("  [ERROR] Shared evaluator is None")
        return {}

    try:
        from pbench.vqa_evaluation import compute_vqa_accuracy

        print0(f"  Running Domain Score evaluation: {method_name}")
        print0(f"  Video directory: {video_dir}")
        print0(f"  VQA questions directory: {vqa_questions_dir}")

        overall_accuracy, detailed_results, category_scores = compute_vqa_accuracy(
            vqa_questions_dir=str(vqa_questions_dir),
            video_dir=str(video_dir),
            prompt_file=str(prompt_file),
            model_name=model_name,
            device="cuda",
            tensor_parallel_size=tensor_parallel_size,
            enable_missing_videos=True,
            batch_size=batch_size,
            gpu_memory_utilization=gpu_memory_utilization,
            shared_evaluator=shared_evaluator,
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "method": method_name,
            "overall_accuracy": overall_accuracy,
            "total_videos": len(detailed_results),
            "model_name": model_name,
            "category_scores": category_scores,
            "robot_score": category_scores.get("robot_score", 0.0),
        }

        with open(output_dir / "domain_scores.json", "w") as file_handle:
            json.dump(summary, file_handle, indent=2)

        with open(output_dir / "domain_detailed_results.json", "w") as file_handle:
            json.dump(detailed_results, file_handle, indent=2)

        print0(f"    Overall Accuracy: {overall_accuracy:.4f}")
        print0(f"    Robot Score:      {summary['robot_score']:.4f}")

        return summary

    except Exception as error:
        print0(f"  [ERROR] Domain Score evaluation failed: {error}")
        import traceback
        traceback.print_exc()
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def compute_mean_video_quality(results: Dict[str, float]) -> Dict[str, float]:
    """Add the mean of all 8 video quality metrics to the results dict."""
    values = [results[metric] for metric in VIDEO_QUALITY_METRICS if metric in results]
    if values:
        results = dict(results)
        results["mean_8_metrics"] = sum(values) / len(values)
    return results


def generate_summary_report(all_results: Dict[str, Dict[str, Any]], output_dir: Path):
    """Generate JSON and Markdown summary reports comparing all methods."""
    report_file = output_dir / "ezsbench_summary.json"
    with open(report_file, "w") as file_handle:
        json.dump(all_results, file_handle, indent=2, ensure_ascii=False)

    # Build Markdown comparison table
    columns = VIDEO_QUALITY_METRICS + [
        "mean_8_metrics", "robot_score", "overall_accuracy",
    ]
    display_names = {
        "aesthetic_quality": "Aesthetic",
        "background_consistency": "BG Consist.",
        "imaging_quality": "Imaging",
        "motion_smoothness": "Motion",
        "overall_consistency": "Overall Consist.",
        "subject_consistency": "Subject Consist.",
        "i2v_background": "I2V BG",
        "i2v_subject": "I2V Subject",
        "mean_8_metrics": "Mean (8 metrics)",
        "robot_score": "Robot Score",
        "overall_accuracy": "Domain Accuracy",
    }

    # Find best values for bolding
    best_values = {}
    for metric in columns:
        values = [
            all_results[method].get(metric)
            for method in all_results
            if all_results[method].get(metric) is not None
        ]
        if values:
            best_values[metric] = max(values)

    markdown_file = output_dir / "ezsbench_summary.md"
    with open(markdown_file, "w") as file_handle:
        file_handle.write("## EZS-Bench Evaluation Results\n\n")
        header = "| Method | " + " | ".join(display_names.get(m, m) for m in columns) + " |\n"
        separator = "|" + "---|" * (len(columns) + 1) + "\n"
        file_handle.write(header)
        file_handle.write(separator)

        for method_name, results in all_results.items():
            row = [method_name]
            for metric in columns:
                value = results.get(metric)
                if value is not None:
                    is_best = abs(value - best_values.get(metric, 0)) < 1e-4
                    formatted = f"**{value:.4f}**" if is_best else f"{value:.4f}"
                    row.append(formatted)
                else:
                    row.append("N/A")
            file_handle.write("| " + " | ".join(row) + " |\n")

    print0("\n" + "=" * 60)
    print0("Summary reports generated:")
    print0(f"  JSON: {report_file}")
    print0(f"  Markdown: {markdown_file}")
    print0("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="EZS-Bench: Evaluate video generation methods on physical consistency and video quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data inputs
    parser.add_argument(
        "--data_file", type=str, default=None,
        help="Path to a combined JSONL file with video, prompt, and question fields. "
             "When provided, --prompt_file and --vqa_questions_file are not needed.",
    )
    parser.add_argument(
        "--prompt_file", type=str, default=None,
        help="Path to the prompt JSONL file (one JSON object per line with video_id and prompt). "
             "Not needed if --data_file is provided.",
    )
    parser.add_argument(
        "--vqa_questions_file", type=str, default=None,
        help="Path to the VQA questions JSONL file for Domain Score evaluation. "
             "Not needed if --data_file is provided.",
    )

    # Single method
    parser.add_argument(
        "--method_name", type=str, default=None,
        help="Display name of the method to evaluate.",
    )
    parser.add_argument(
        "--method_dir", type=str, default=None,
        help="Directory containing generated videos for the method.",
    )

    # Multiple methods
    parser.add_argument(
        "--methods_config", type=str, default=None,
        help=(
            "Path to a JSON file listing multiple methods. "
            'Format: [{"name": "MethodA", "dir": "/path/to/videos"}, ...]'
        ),
    )

    # Output
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results.")

    # Evaluation control
    parser.add_argument("--skip_video_quality", action="store_true", help="Skip the 8-metric video quality evaluation.")
    parser.add_argument("--skip_domain_score", action="store_true", help="Skip the Domain Score (VQA) evaluation.")

    # Domain Score model parameters
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2.5-VL-72B-Instruct", help="VLM model for Domain Score.")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Tensor parallel size for VLM inference.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for VLM inference.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75, help="GPU memory utilization for vLLM.")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Initialize distributed environment (required by PBench for video quality metrics).
    # When launched with torchrun, this sets up the process group.
    # When launched with plain python, this is a no-op single-process fallback.
    dist_init()

    # Resolve output directory
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = SCRIPT_DIR / "results" / f"ezsbench_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)

    # Build method list
    methods = []
    if args.methods_config:
        with open(args.methods_config, "r") as file_handle:
            methods = [(item["name"], item["dir"]) for item in json.load(file_handle)]
    elif args.method_name and args.method_dir:
        methods = [(args.method_name, args.method_dir)]
    else:
        print0("[ERROR] Provide either --method_name/--method_dir or --methods_config.")
        return

    print0("=" * 70)
    print0("EZS-Bench: Embodied Zero-Shot Benchmark Evaluation")
    print0("=" * 70)
    print0(f"Output directory: {output_base}")
    print0(f"Video quality evaluation: {'SKIP' if args.skip_video_quality else 'ON'}")
    print0(f"Domain Score evaluation:  {'SKIP' if args.skip_domain_score else 'ON'}")
    print0(f"Methods to evaluate: {len(methods)}")

    # ── Load data ─────────────────────────────────────────────────────────
    print0("\n[Step 1] Loading data...")
    questions_dict = {}

    if args.data_file:
        prompt_data, prompt_ordered, questions_dict = load_combined_data(Path(args.data_file))
        print0(f"  Loaded {len(prompt_data)} prompts and {len(questions_dict)} VQA question sets from combined file")
    elif args.prompt_file:
        prompt_data, prompt_ordered = load_prompt_data(Path(args.prompt_file))
        print0(f"  Loaded {len(prompt_data)} prompts")
        if args.vqa_questions_file and not args.skip_domain_score:
            questions_dict = load_vqa_questions(Path(args.vqa_questions_file))
            print0(f"  Loaded {len(questions_dict)} VQA question sets")
    else:
        print0("[ERROR] Provide either --data_file or --prompt_file.")
        return

    # ── Discover videos ───────────────────────────────────────────────────
    print0("\n[Step 2] Discovering generated videos...")
    all_method_results = {}
    method_entries = {}
    method_subdirs = {}

    for method_name, method_dir in methods:
        method_dir_path = Path(method_dir)
        entries = discover_generated_videos(method_dir_path, prompt_data)

        if not entries:
            print0(f"  [WARNING] {method_name}: no matching videos found, skipping")
            continue

        print0(f"  {method_name}: found {len(entries)} videos")
        method_entries[method_name] = entries
        method_subdirs[method_name] = re.sub(r"[^\w\-]", "_", method_name)
        all_method_results[method_name] = {}

    if not method_entries:
        print0("[ERROR] No videos found for any method.")
        return

    # ── Video Quality Evaluation (requires torchrun) ──────────────────────
    if not args.skip_video_quality:
        print0("\n[Step 3] Running video quality evaluation...")
        for method_name, entries in method_entries.items():
            subdir = method_subdirs[method_name]

            # Prepare directory structure
            video_dir, prompt_file, custom_image_dir = prepare_video_quality_eval_dir(
                entries, output_base, subdir
            )
            eval_dir = output_base / subdir / "video_quality"

            # Run evaluation in-process via PBench
            results = run_video_quality_evaluation(
                video_dir, prompt_file, eval_dir, method_name,
                custom_image_folder=custom_image_dir,
            )
            if results:
                results = compute_mean_video_quality(results)
                all_method_results[method_name].update(results)
                print0(f"  {method_name}: 8 metrics loaded")
            else:
                print0(f"  [WARNING] {method_name}: no results found")
            time.sleep(2)

    # ── Domain Score Evaluation ───────────────────────────────────────────
    if not args.skip_domain_score and questions_dict:
        print0("\n[Step 4] Loading VLM evaluator (one-time initialization)...")
        shared_evaluator = None
        try:
            from pbench.vqa_evaluation import create_and_load_qwen_evaluator
            shared_evaluator = create_and_load_qwen_evaluator(
                model_name=args.vlm_model,
                device="cuda",
                tensor_parallel_size=args.tensor_parallel_size,
                batch_size=args.batch_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            print0("  VLM evaluator loaded successfully")
        except Exception as error:
            print0(f"  [ERROR] Failed to load VLM evaluator: {error}")
            import traceback
            traceback.print_exc()
            return

        print0("\n[Step 5] Running Domain Score evaluation...")
        for method_name, entries in method_entries.items():
            subdir = method_subdirs[method_name]

            # Prepare directory structure
            video_dir, vqa_dir, prompt_file = prepare_domain_score_dir(
                questions_dict, entries, output_base, subdir
            )
            eval_dir = output_base / subdir / "domain_score"

            print0(f"\n  [{method_name}]")
            summary = run_domain_score_evaluation(
                video_dir, vqa_dir, prompt_file, eval_dir,
                method_name=method_name,
                shared_evaluator=shared_evaluator,
                model_name=args.vlm_model,
                tensor_parallel_size=args.tensor_parallel_size,
                batch_size=args.batch_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )

            if summary:
                all_method_results[method_name].update({
                    "robot_score": summary.get("robot_score", 0.0),
                    "overall_accuracy": summary.get("overall_accuracy", 0.0),
                })

    # ── Generate Report ───────────────────────────────────────────────────
    print0("\n[Step 6] Generating summary report...")
    if all_method_results:
        generate_summary_report(all_method_results, output_base)

        print0("\nFinal Results:")
        for method_name, results in all_method_results.items():
            print0(f"\n  {method_name}:")
            all_metrics = VIDEO_QUALITY_METRICS + [
                "mean_8_metrics", "robot_score", "overall_accuracy",
            ]
            for metric in all_metrics:
                if metric in results:
                    print0(f"    {metric}: {results[metric]:.4f}")
    else:
        print0("[WARNING] No results to summarize.")


if __name__ == "__main__":
    main()
