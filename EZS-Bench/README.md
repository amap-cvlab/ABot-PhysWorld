# EZS-Bench: Embodied Zero-Shot Benchmark for Physically Consistent Video Generation

EZS-Bench is a comprehensive evaluation benchmark for assessing video generation methods in embodied (robotic) scenarios. It measures both **video quality** and **robot fidelity** of generated videos.

## Evaluation Dimensions

### Video Quality Metrics (8 Metrics)

Evaluates the visual quality of generated videos across 8 dimensions:

| Metric | Description |
|---|---|
| aesthetic_quality | Overall aesthetic appeal of the video |
| imaging_quality | Low-level image quality (sharpness, noise, etc.) |
| motion_smoothness | Temporal smoothness of motion |
| background_consistency | Consistency of background across frames |
| subject_consistency | Consistency of the main subject across frames |
| overall_consistency | Overall temporal coherence |
| i2v_background | Background fidelity to the conditioning image (I2V) |
| i2v_subject | Subject fidelity to the conditioning image (I2V) |

### Domain Score (Robot Score)

Evaluates robot fidelity via VQA (Visual Question Answering) using a large vision-language model (default: Qwen2.5-VL-72B-Instruct):

- **Robot Score**: Evaluates robot morphology and motion plausibility

## Installation

### Requirements

- Python >= 3.10
- CUDA-compatible GPUs (recommended: 4x A100 80GB or equivalent)
- ~150 GB disk space for the VLM model weights (auto-downloaded on first run)

### Setup

Clone the repository and install:

    git clone <repo_url>
    cd EZS-Bench
    conda create -n ezsbench python=3.10 -y
    conda activate ezsbench
    pip install -e .

All evaluation code and dependencies are bundled in this package. No external repositories needed.

### Download Evaluation Data

Download the evaluation dataset (196 samples with conditioning images and VQA questions) from ModelScope:

    git lfs install
    git clone https://www.modelscope.cn/datasets/amap_cvlab/EZS-Bench_data.git

### VLM Model (Auto-Download)

The Domain Score evaluation uses **Qwen2.5-VL-72B-Instruct** as the default VLM judge. The model weights (~150 GB) are **automatically downloaded** on first run via HuggingFace. You can also specify a different model with `--vlm_model`.

## Input File Formats

### Data File (data.jsonl)

A JSONL file where each line is a JSON object containing the prompt, conditioning image, and VQA questions:

    {
      "video": "/path/to/conditioning_image.jpg",
      "prompt": "A robot arm picks up a red cube from the table...",
      "question": [
        {
          "question": "Does the robot arm grasp the red cube?",
          "index2ans": {"A": "yes", "B": "no"},
          "answer": "A",
          "uid": "scene_001_Q1",
          "split": "val",
          "task": "task:success:discrete:True"
        }
      ]
    }

- **video** (required): Path to the conditioning image. The filename stem is used as the video_id.
- **prompt** (required): Text description of the expected video content.
- **question** (required): List of VQA question dicts for Domain Score evaluation.

### Generated Videos

Place your generated videos in a directory. Two naming conventions are supported:

1. **By video_id** (recommended): Name each file {video_id}.mp4 (e.g., scene_001.mp4)
2. **By order**: If no video_id match is found, .mp4 files are matched to prompts in sorted order

### Methods Config (for batch evaluation)

To evaluate multiple methods at once, create a JSON file:

    [
      {"name": "MethodA", "dir": "/path/to/methodA/videos"},
      {"name": "MethodB", "dir": "/path/to/methodB/videos"}
    ]

## Usage

### Full Evaluation (Video Quality + Domain Score)

Requires torchrun for multi-GPU video quality evaluation:

    torchrun --standalone --nproc_per_node=4 evaluate_ezsbench.py \
        --data_file /path/to/data.jsonl \
        --method_name "YourMethod" \
        --method_dir /path/to/generated_videos \
        --output_dir ./results

### Video Quality Only

    torchrun --standalone --nproc_per_node=4 evaluate_ezsbench.py \
        --data_file /path/to/data.jsonl \
        --method_name "YourMethod" \
        --method_dir /path/to/generated_videos \
        --skip_domain_score \
        --output_dir ./results

### Domain Score Only

Does not require torchrun, can be launched with plain python:

    python evaluate_ezsbench.py \
        --data_file /path/to/data.jsonl \
        --method_name "YourMethod" \
        --method_dir /path/to/generated_videos \
        --skip_video_quality \
        --output_dir ./results

### Batch Evaluation (Multiple Methods)

    torchrun --standalone --nproc_per_node=4 evaluate_ezsbench.py \
        --data_file /path/to/data.jsonl \
        --methods_config methods.json \
        --output_dir ./results

### Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| --data_file | (required) | Path to the combined JSONL file with video, prompt, and question fields |
| --method_name | None | Display name of the method to evaluate |
| --method_dir | None | Directory containing generated videos |
| --methods_config | None | JSON file listing multiple methods for batch evaluation |
| --output_dir | ./results/timestamp | Output directory for results |
| --skip_video_quality | False | Skip the 8-metric video quality evaluation |
| --skip_domain_score | False | Skip the Domain Score (VQA) evaluation |
| --vlm_model | Qwen/Qwen2.5-VL-72B-Instruct | VLM model for Domain Score |
| --tensor_parallel_size | 4 | Tensor parallel size for VLM inference |
| --batch_size | 32 | Batch size for VLM inference |
| --gpu_memory_utilization | 0.75 | GPU memory utilization for vLLM |

## Output

Results are saved to the specified --output_dir:

    results/
    +-- ezsbench_summary.json          # All metrics for all methods (JSON)
    +-- ezsbench_summary.md            # Comparison table (Markdown)
    +-- YourMethod/
        +-- video_quality/
        |   +-- videos/                 # Prepared video files
        |   +-- prompts.json            # Prepared prompt file
        |   +-- evaluation_results/     # Per-metric results
        +-- domain_score/
            +-- videos/                 # Linked video files
            +-- vqa_questions/          # Per-video VQA question files
            +-- domain_scores.json      # Robot Score
            +-- domain_detailed_results.json

## Citation

If you use EZS-Bench in your research, please cite:

```bibtex
@article{chen2026abotphysworld,
  title={ABot-PhysWorld: Interactive World Foundation Model for Robotic Manipulation with Physics Alignment},
  author={Yuzhi Chen, Ronghan Chen, Dongjie Huo, Yandan Yang, Dekang Qi, Haoyun Liu, Tong Lin, Shuang Zeng, Junjin Xiao, Xinyuan Chang, Feng Xiong, Xing Wei, Zhiheng Ma, Mu Xu},
  journal={arXiv preprint arXiv:2603.23376},
  year={2026}
}
```

## License

This project is licensed under the MIT License.
