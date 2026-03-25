import os
import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import logging

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
import requests
from requests.auth import HTTPBasicAuth
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(verbose=True)

# 设置 HuggingFace 缓存目录到项目目录，避免使用系统 cache
CUR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE_DIR = os.path.join(CUR_DIR, 'checkpoints', 'huggingface')
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# 设置环境变量，确保 HuggingFace 使用项目目录而不是系统 cache
os.environ.setdefault('HF_HOME', HF_CACHE_DIR)
os.environ.setdefault('TRANSFORMERS_CACHE', HF_CACHE_DIR)
os.environ.setdefault('HF_DATASETS_CACHE', HF_CACHE_DIR)

from .distributed import (
    get_world_size,
    get_rank,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)

from .utils import load_json
from .prompts.evaluation_prompt import (
    system_template_binary_v0,
    begin_user_template_binary_v0,
    user_template_binary_v0,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_multimodal_input(messages, processor, cached_vision_info=None):
    """
    Prepare multimodal input for vLLM following the pattern from qwen2p5_vl_instruct_vllm.py

    Args:
        messages: List of message dictionaries containing text and media content
        processor: AutoProcessor instance for chat template formatting
        cached_vision_info: Optional cached (image_inputs, video_inputs, video_kwargs) tuple to reuse

    Returns:
        tuple: (formatted_text, mm_data_dict) for vLLM input
    """
    # Apply chat template to format the conversation
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if cached_vision_info is not None:
        image_inputs, video_inputs, video_kwargs = cached_vision_info
    else:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

    # Format multi-modal data for vLLM
    mm_data = {}
    if image_inputs:
        mm_data["image"] = image_inputs
    if video_inputs:
        mm_data["video"] = video_inputs

    return {"prompt": text, "multi_modal_data": mm_data, "mm_processor_kwargs": video_kwargs}

class QwenVLEvaluator:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-72B-Instruct", device="cuda", tensor_parallel_size=8, batch_size=16, gpu_memory_utilization=0.55):
        """Initialize Qwen2.5-72B-VL model for VQA evaluation using vLLM"""
        self.device = device
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.batch_size = batch_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.sampling_params = None

    def load_model(self):
        """Load the specified VL model using vLLM"""
        logger.info(f"Loading model {self.model_name} using vLLM")

        # 如果model_name是HuggingFace路径，尝试使用本地路径
        model_path = self.model_name
        if "/" in model_path and not os.path.isabs(model_path):
            # 尝试从本地缓存目录加载（直接使用模型名称作为目录名）
            # 例如: Qwen/Qwen2.5-VL-72B-Instruct -> Qwen2.5-VL-72B-Instruct
            model_dir_name = model_path.split("/")[-1]  # 获取最后一部分
            local_model_path = os.path.join(HF_CACHE_DIR, model_dir_name)
            if os.path.exists(local_model_path) and os.path.isdir(local_model_path):
                # 检查是否有config.json文件，确认是完整的模型目录
                config_file = os.path.join(local_model_path, "config.json")
                if os.path.exists(config_file):
                    model_path = local_model_path
                    logger.info(f"Using local model path: {model_path}")
                else:
                    logger.warning(f"Local model directory exists but missing config.json, using model name")
            else:
                logger.info(f"Local model path not found: {local_model_path}, will use model name with offline mode")

        # Configure vLLM parameters for optimal performance
        # vLLM 会使用 HuggingFace 的缓存机制，通过环境变量 HF_HOME 和 TRANSFORMERS_CACHE 控制
        # 注意：Qwen2.5-VL-72B-Instruct不是MoE模型，不需要enable_expert_parallel
        model_kwargs = {
            "model": model_path,
            "trust_remote_code": True,
            "max_model_len": 65536,
            "limit_mm_per_prompt": {"video": 1},
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "download_dir": HF_CACHE_DIR,  # 指定下载目录
        }
        # 只有MoE模型才需要enable_expert_parallel，Qwen2.5-VL-72B-Instruct不是MoE模型
        # enable_expert_parallel 已移除
        
        # 如果设置了离线模式，确保vLLM也使用本地文件
        if os.environ.get('HF_HUB_OFFLINE') == '1':
            logger.info("Using offline mode - will only use local files")

        self.model = LLM(**model_kwargs)
        logger.info(f"Successfully loaded model: {self.model_name} using vLLM")

        # Initialize processor and tokenizer following qwen2p5_vl_instruct_vllm.py pattern
        # 使用 cache_dir 参数确保下载到项目目录，不使用软链接
        # 如果使用本地路径，processor和tokenizer也应该使用本地路径
        processor_model_name = model_path if os.path.isabs(model_path) or not "/" in model_path else self.model_name
        local_files_only = os.environ.get('HF_HUB_OFFLINE') == '1'
        
        self.processor = AutoProcessor.from_pretrained(
            processor_model_name,
            min_pixels=4 * 28 * 28,
            max_pixels=768 * 28 * 28,
            cache_dir=HF_CACHE_DIR,
            local_files_only=local_files_only
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            processor_model_name,
            cache_dir=HF_CACHE_DIR,
            local_files_only=local_files_only
        )
        self.tokenizer.model_max_length = 65536  # Match max_model_len
        self.processor.tokenizer = self.tokenizer

        # Set up sampling parameters for consistent generation
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1024,
            stop_token_ids=None,
            top_p=1.0,
            top_k=-1,
        )

    def extract_video_frames(self, video_path, max_frames=16):
        """Extract frames from video for VQA (kept for compatibility but not needed for vLLM)"""
        # This method is kept for compatibility but vLLM handles video processing directly
        # We'll use OpenCV as fallback
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames uniformly
        if frame_count > max_frames:
            indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
        else:
            indices = range(frame_count)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)

        cap.release()
        return frames

    def answer_questions_batch(self, video_path, qa_data, cached_vision_info=None, batch_size=16):
        """Answer all questions for a video in batch using vLLM with proper input preprocessing"""
        if self.model is None:
            self.load_model()

        if not qa_data:
            return []

        # Use imaginaire4 prompt templates for consistent formatting
        system_prompt = system_template_binary_v0
        begin_user_prompt = begin_user_template_binary_v0

        all_responses = []

        # Process questions in batches to avoid GPU memory issues
        for i in range(0, len(qa_data), batch_size):
            batch_qa = qa_data[i:i + batch_size]

            # Prepare batch inputs
            batch_inputs = []
            for qa in batch_qa:
                # Format the question using imaginaire4 template
                question_prompt = user_template_binary_v0(qa, is_reasoning=False)

                # Combine text prompts
                combined_text_prompt = f"{begin_user_prompt}\n\n{question_prompt}"

                # Create messages in the format expected by the processor
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_path,
                                "fps": 3.0,
                                "max_pixels": 768 * 28 * 28,
                            },
                            {"type": "text", "text": combined_text_prompt},
                        ],
                    }
                ]

                # Use the standardized preprocessing function
                single_input = prepare_multimodal_input(messages, self.processor, cached_vision_info)
                batch_inputs.append(single_input)

            # Generate responses for this batch
            outputs = self.model.generate(batch_inputs, sampling_params=self.sampling_params, use_tqdm=False)
            batch_responses = [output.outputs[0].text.strip() for output in outputs]
            all_responses.extend(batch_responses)

        return all_responses

    def evaluate_video_qa(self, video_path, qa_data):
        """Evaluate all questions for a single video using batch processing"""
        # Verify video file exists
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return []

        if not qa_data:
            return []

        # Process video once to get cached vision info
        # Create a dummy message to extract vision info
        dummy_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": 3.0,
                        "max_pixels": 768 * 28 * 28,
                    },
                ],
            }
        ]
        cached_vision_info = process_vision_info(dummy_messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        # logger.info(f"Processed vision info for video: {video_path}")

        # Get all model answers in batch
        model_answers = self.answer_questions_batch(video_path, qa_data, cached_vision_info, batch_size=self.batch_size)

        # Process results
        results = []
        for i, qa in enumerate(qa_data):
            question = qa['question']
            correct_answer = qa['answer']
            model_answer = model_answers[i]

            # Simple accuracy check - check if the model answer contains the correct choice
            is_correct = self.check_answer_accuracy(model_answer, correct_answer, qa['index2ans'])

            result = {
                'uid': qa['uid'],
                'question': question,
                'correct_answer': correct_answer,
                'model_answer': model_answer,
                'is_correct': is_correct,
                'task': qa['task']
            }
            results.append(result)

        return results

    def evaluate_multi_seed_video_qa(self, video_info_list, qa_data):
        """Evaluate all questions for multiple seed versions of the same video"""
        if not video_info_list:
            logger.warning("No video files found")
            return []

        # Collect results from all seed versions
        # Each seed is a different video file, so each needs to be processed separately
        all_seed_results = []
        for video_info in video_info_list:
            video_path = video_info["path"]
            seed = video_info["seed"]

            logger.info(f"Evaluating seed {seed}: {video_path}")
            # The evaluate_video_qa method now processes the video once and reuses vision info
            seed_results = self.evaluate_video_qa(video_path, qa_data)

            # Add seed information to each result
            for result in seed_results:
                result["seed"] = seed

            all_seed_results.extend(seed_results)

        # Aggregate results across seeds for each question
        return self.aggregate_multi_seed_results(all_seed_results)

    def aggregate_multi_seed_results(self, all_seed_results):
        """Aggregate results from multiple seeds using majority vote and average accuracy"""
        # Group results by question uid
        question_groups = {}
        for result in all_seed_results:
            uid = result['uid']
            if uid not in question_groups:
                question_groups[uid] = []
            question_groups[uid].append(result)

        aggregated_results = []
        for uid, seed_results in question_groups.items():
            if not seed_results:
                continue

            # Use the first result as template
            template = seed_results[0]

            # Calculate accuracy across seeds
            correct_count = sum(1 for r in seed_results if r['is_correct'])
            total_seeds = len(seed_results)
            accuracy = correct_count / total_seeds if total_seeds > 0 else 0.0

            # Collect all model answers for reference
            all_answers = [r['model_answer'] for r in seed_results]

            # Create aggregated result using average accuracy instead of majority vote
            aggregated_result = {
                'uid': uid,
                'question': template['question'],
                'correct_answer': template['correct_answer'],
                'model_answers': all_answers,  # Keep all answers for debugging
                'accuracy': accuracy,
                'total_seeds': total_seeds,
                'task': template['task']
            }

            aggregated_results.append(aggregated_result)

        return aggregated_results

    def check_answer_accuracy(self, model_answer, correct_answer, index2ans):
        """Check if model answer is correct"""
        model_answer = model_answer.lower().strip()

        # Get the correct answer text
        correct_answer_text = index2ans[correct_answer].lower()

        # Check if model answer contains "yes" or "no" for binary questions
        if correct_answer_text == "yes":
            return "yes" in model_answer and "no" not in model_answer
        elif correct_answer_text == "no":
            return "no" in model_answer and "yes" not in model_answer
        else:
            # For other types of answers, check if the correct answer is in model response
            return correct_answer_text in model_answer


class OpenAIEvaluator:
    def __init__(self, model_name="gpt-4o", max_frames_num=8, max_retries=5, timeout=10, max_size_in_mb=20, reasoning_effort="medium"):
        """Initialize OpenAI-compatible model for VQA evaluation"""
        self.model_name = model_name
        self.max_frames_num = max_frames_num
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_size_in_mb = max_size_in_mb
        self.reasoning_effort = reasoning_effort
        self.use_auth_api = model_name.endswith("-auth")

        if self.use_auth_api:
            self.base_model_name = model_name[:-5]
        else:
            self.base_model_name = model_name

        self.client = None

    def get_client_api_key(self, api_service: str) -> str:
        """Request a new API access token from authentication server"""
        client_id = os.getenv("AUTH_SERVER_CLIENT_ID")
        client_secret = os.getenv("AUTH_SERVER_CLIENT_SECRET")
        url = os.getenv("AUTH_SERVER_TOKEN_URL")

        if not all([client_id, client_secret, url]):
            raise ValueError("AUTH_SERVER_CLIENT_ID, AUTH_SERVER_CLIENT_SECRET, and AUTH_SERVER_TOKEN_URL must be set")

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        scope_map = {"oai": "azureopenai-readwrite"}
        data = {"grant_type": "client_credentials", "scope": scope_map[api_service]}

        response = requests.post(url, headers=headers, data=data, auth=HTTPBasicAuth(client_id, client_secret))
        if response.status_code != 200:
            raise RuntimeError("Error: Could not generate a Bearer API token, please try again")

        return response.json()["access_token"]

    def refresh_token_and_client(self):
        """Refresh the API token and recreate the client"""
        if not self.use_auth_api:
            logger.warning("Token refresh requested but use_auth_api is False")
            return

        logger.info("Refreshing API token...")
        api_key = self.get_client_api_key("oai")
        base_url = os.getenv("OPENAI_API_BASE")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={"dataClassification": "sensitive", "dataSource": "internet"}
        )
        logger.info("API token refreshed successfully")

    def load_model(self):
        """Initialize the OpenAI client"""
        logger.info(f"Initializing OpenAI client for model: {self.base_model_name}")

        if self.use_auth_api:
            api_key = self.get_client_api_key("oai")
            base_url = os.getenv("OPENAI_API_BASE")
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers={"dataClassification": "sensitive", "dataSource": "internet"}
            )
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY must be set for standard models")
            self.client = OpenAI(api_key=api_key)

        logger.info(f"Successfully initialized client for: {self.base_model_name}")

    def encode_image(self, image):
        """Encode image to base64 string"""
        max_size = self.max_size_in_mb * 1024 * 1024
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()

        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()

        while len(byte_data) > max_size and img.size[0] > 100 and img.size[1] > 100:
            new_size = (int(img.size[0] * 0.75), int(img.size[1] * 0.75))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()

        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    def encode_video(self, video_path, max_num_frames):
        """Encode video frames to base64 strings using qwen_vl_utils"""
        if max_num_frames == 0:
            return []

        try:
            from decord import VideoReader, cpu
        except ImportError:
            logger.error("decord is required for video processing")
            return []

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        nframes = max(min(max_num_frames, total_frame_num), 2)

        video_element = {"type": "video", "video": video_path, "nframes": nframes}
        message = [{"role": "user", "content": [video_element]}]
        _, video_inputs = process_vision_info([message])

        if video_inputs is None or len(video_inputs) == 0:
            return []

        video_tensor = video_inputs[0]
        actual_nframes = len(video_tensor)

        if max_num_frames == 1:
            video_tensor = video_inputs[0][:1]
            actual_nframes = 1

        base64_frames = []
        for idx in range(actual_nframes):
            if hasattr(video_tensor, "numpy"):
                frame = video_tensor[idx].numpy()
            elif hasattr(video_tensor[idx], "numpy"):
                frame = video_tensor[idx].numpy()
            else:
                frame = video_tensor[idx]

            if frame.shape[-1] != 3 and frame.shape[0] == 3:
                frame = np.transpose(frame, (1, 2, 0))

            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)

            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames

    def answer_questions_batch(self, video_path, qa_data, batch_size=16):
        """Answer all questions for a video using OpenAI API with threading"""
        if self.client is None:
            self.load_model()

        if not qa_data:
            return []

        system_prompt = system_template_binary_v0
        begin_user_prompt = begin_user_template_binary_v0

        encoded_frames = self.encode_video(video_path, self.max_frames_num)
        if not encoded_frames:
            logger.warning(f"Failed to encode video: {video_path}")
            return ["" for _ in qa_data]

        def process_single_question(qa, idx):
            """Process a single question with retries"""
            question_prompt = user_template_binary_v0(qa, is_reasoning=False)
            combined_text_prompt = f"{begin_user_prompt}\n\n{question_prompt}"

            user_content = []
            for frame in encoded_frames:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{frame}"}
                })
            user_content.append({"type": "text", "text": combined_text_prompt})

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            for attempt in range(self.max_retries):
                try:
                    api_params = {
                        "model": self.base_model_name,
                        "messages": messages,
                    }

                    if "o1" in self.base_model_name or "o3" in self.base_model_name or "gpt-5" in self.base_model_name:
                        api_params["max_completion_tokens"] = 1024
                        if self.reasoning_effort is not None:
                            api_params["reasoning_effort"] = self.reasoning_effort
                    else:
                        api_params["max_tokens"] = 1024
                        api_params["temperature"] = 0.0

                    response = self.client.chat.completions.create(**api_params)
                    response_text = response.choices[0].message.content

                    logger.info(f"[OpenAI VQA] Question: {qa['question']}")
                    logger.info(f"[OpenAI VQA] Correct Answer: {qa['index2ans'][qa['answer']]}")
                    logger.info(f"[OpenAI VQA] Model Response: {response_text}")

                    return response_text, idx

                except Exception as e:
                    error_msg = str(e)
                    logger.info(f"Question {idx} attempt {attempt + 1}/{self.max_retries} failed: {error_msg}")

                    if "content_filter" in error_msg.lower() or "responsibleaipolicyviolation" in error_msg.lower():
                        logger.warning(f"Question {idx} filtered by content policy, skipping retries")
                        return "[CONTENT_FILTERED]", idx

                    if "401" in error_msg and "token has expired" in error_msg.lower():
                        logger.info("Token expired, refreshing token and retrying...")
                        try:
                            self.refresh_token_and_client()
                            continue
                        except Exception as refresh_error:
                            logger.error(f"Failed to refresh token: {refresh_error}")

                    if attempt == self.max_retries - 1:
                        logger.error(f"All {self.max_retries} attempts failed for question {idx}")
                        return "", idx
                    else:
                        time.sleep(self.timeout)

            return "", idx

        all_responses = [""] * len(qa_data)

        max_workers = min(len(qa_data), 32)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_single_question, qa, i): i
                for i, qa in enumerate(qa_data)
            }

            for future in tqdm(as_completed(future_to_idx), total=len(qa_data),
                             desc=f"Processing questions for {os.path.basename(video_path)}",
                             disable=get_rank() > 0):
                response_text, idx = future.result()
                all_responses[idx] = response_text

        return all_responses

    def evaluate_video_qa(self, video_path, qa_data):
        """Evaluate all questions for a single video"""
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return []

        if not qa_data:
            return []

        model_answers = self.answer_questions_batch(video_path, qa_data)

        results = []
        for i, qa in enumerate(qa_data):
            question = qa['question']
            correct_answer = qa['answer']
            model_answer = model_answers[i]

            is_correct = self.check_answer_accuracy(model_answer, correct_answer, qa['index2ans'])

            result = {
                'uid': qa['uid'],
                'question': question,
                'correct_answer': correct_answer,
                'model_answer': model_answer,
                'is_correct': is_correct,
                'task': qa['task']
            }
            results.append(result)

        return results

    def aggregate_multi_seed_results(self, all_seed_results):
        """Aggregate results from multiple seeds using majority vote and average accuracy"""
        question_groups = {}
        for result in all_seed_results:
            uid = result['uid']
            if uid not in question_groups:
                question_groups[uid] = []
            question_groups[uid].append(result)

        aggregated_results = []
        for uid, seed_results in question_groups.items():
            if not seed_results:
                continue

            template = seed_results[0]

            correct_count = sum(1 for r in seed_results if r['is_correct'])
            total_seeds = len(seed_results)
            accuracy = correct_count / total_seeds if total_seeds > 0 else 0.0

            all_answers = [r['model_answer'] for r in seed_results]

            aggregated_result = {
                'uid': uid,
                'question': template['question'],
                'correct_answer': template['correct_answer'],
                'model_answers': all_answers,
                'accuracy': accuracy,
                'total_seeds': total_seeds,
                'task': template['task']
            }

            aggregated_results.append(aggregated_result)

        return aggregated_results

    def check_answer_accuracy(self, model_answer, correct_answer, index2ans):
        """Check if model answer is correct"""
        model_answer = model_answer.lower().strip()

        correct_answer_text = index2ans[correct_answer].lower()

        if correct_answer_text == "yes":
            return "yes" in model_answer and "no" not in model_answer
        elif correct_answer_text == "no":
            return "no" in model_answer and "yes" not in model_answer
        else:
            return correct_answer_text in model_answer


def create_and_load_qwen_evaluator(
    model_name="Qwen/Qwen2.5-VL-72B-Instruct",
    device="cuda",
    tensor_parallel_size=8,
    batch_size=16,
    gpu_memory_utilization=0.55,
    **kwargs
):
    """
    创建 Qwen VQA 评估器并只加载一次权重。
    用于多方法评测时复用同一模型，避免重复 load 导致 NCCL/僵尸进程卡住。
    调用方需保证与 compute_vqa_accuracy 使用相同的分布式环境（若用 torch.distributed 则需先 init）。
    Returns:
        evaluator: QwenVLEvaluator 实例，可直接传给 compute_vqa_accuracy(..., shared_evaluator=evaluator)
    """
    evaluator = QwenVLEvaluator(
        model_name=model_name,
        device=device,
        tensor_parallel_size=tensor_parallel_size,
        batch_size=batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if get_rank() == 0:
        evaluator.load_model()
        barrier()
    else:
        barrier()
        evaluator.load_model()
    logger.info("Shared Qwen evaluator loaded once (will be reused for all methods)")
    return evaluator


def find_all_video_files(video_id, video_dir):
    """Find all video files with different seeds for the given video_id"""
    # Try different suffixes that might match, including seed variations
    possible_suffixes = ["__0.mp4", "__1.mp4", "__2.mp4", "__3.mp4", "__4.mp4", ".mp4"]

    found_videos = []
    for suffix in possible_suffixes:
        video_path = os.path.join(video_dir, f"{video_id}{suffix}")
        if os.path.exists(video_path):
            # Extract seed from filename
            if "__" in suffix:
                seed = suffix.split("__")[1].split(".")[0]
            else:
                seed = "default"
            found_videos.append({"path": video_path, "seed": seed})

    return found_videos


def find_video_file(video_id, video_dir):
    """Find the first video file matching the video_id (for backward compatibility)"""
    videos = find_all_video_files(video_id, video_dir)
    return videos[0]["path"] if videos else None


def compute_vqa_accuracy(
    vqa_questions_dir,
    video_dir,
    prompt_file,
    model_name="Qwen/Qwen2.5-VL-72B-Instruct",
    device="cuda",
    tensor_parallel_size=8,
    enable_missing_videos=False,
    shared_evaluator=None,  # 新增：支持共享的评估器
    **kwargs
):
    """
    Compute VQA accuracy using either Qwen2.5-72B-VL or OpenAI-compatible models

    Refactored logic:
    1. Enumerate by video_id first
    2. For each video_id, enumerate all corresponding video_path/seed
    3. Call process_vision_info only once per video (for Qwen) or encode video once (for OpenAI)
    4. Batch call vllm/API on qa_pair dimension
    5. Record answers and report results

    Args:
        vqa_questions_dir: Directory containing VQA question files
        video_dir: Directory containing videos
        prompt_file: JSON file containing video metadata
        model_name: Name of the VQA model to use (e.g., 'Qwen/Qwen2.5-VL-72B-Instruct', 'gpt-4o', 'gpt-5-auth')
        device: Device to run the model on (for vLLM models)
        tensor_parallel_size: Number of tensor parallel processes for vLLM

    Returns:
        overall_accuracy: Overall accuracy across all questions
        detailed_results: Detailed results for each video
        category_scores: Category-specific accuracy scores
    """

    prompt_data = load_json(prompt_file)
    video_id_set = {item['video_id'] for item in prompt_data}

    is_openai_model = any(model_name.startswith(prefix) for prefix in ['gpt-', 'o1-', 'o3-'])

    if is_openai_model:
        logger.info(f"Using OpenAI-compatible evaluator for model: {model_name}")
        reasoning_effort = kwargs.get('reasoning_effort', 'medium')
        evaluator = OpenAIEvaluator(model_name=model_name, reasoning_effort=reasoning_effort)
        evaluator.load_model()
    else:
        logger.info(f"Using Qwen vLLM evaluator for model: {model_name}")
        
        # 如果提供了共享的评估器，直接使用；否则创建新实例
        if shared_evaluator is not None:
            logger.info("Using shared evaluator instance")
            evaluator = shared_evaluator
        else:
            # 创建新的评估器实例并加载模型
            batch_size = kwargs.get('batch_size', 16)
            gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.55)
            evaluator = QwenVLEvaluator(
                model_name=model_name, 
                device=device, 
                tensor_parallel_size=tensor_parallel_size,
                batch_size=batch_size,
                gpu_memory_utilization=gpu_memory_utilization
            )

            if get_rank() == 0:
                evaluator.load_model()
                barrier()
            else:
                barrier()
                evaluator.load_model()

    # Get all VQA files and map them to video_ids
    vqa_files = [f for f in os.listdir(vqa_questions_dir) if f.endswith('.json')]
    vqa_files = distribute_list_to_rank(vqa_files)

    # Create video_id to vqa_file mapping
    video_id_to_vqa_file = {}
    for vqa_file in vqa_files:
        video_id = vqa_file.replace('.json', '')
        video_id_to_vqa_file[video_id] = vqa_file

    all_results = []
    correct_count = 0
    total_count = 0

    # Initialize category-specific counters
    category_stats = {
        'av': {'correct': 0, 'total': 0},
        'common_sense': {'correct': 0, 'total': 0},
        'human': {'correct': 0, 'total': 0},
        'industry': {'correct': 0, 'total': 0},
        'misc': {'correct': 0, 'total': 0},
        'physics': {'correct': 0, 'total': 0},
        'robot': {'correct': 0, 'total': 0}
    }

    def get_category_from_filename(filename):
        """Extract category from filename"""
        if filename.startswith('av_'):
            return 'av'
        elif filename.startswith('common_sense_'):
            return 'common_sense'
        elif filename.startswith('human_'):
            return 'human'
        elif filename.startswith('industry_'):
            return 'industry'
        elif filename.startswith('misc_'):
            return 'misc'
        elif filename.startswith('physics_'):
            return 'physics'
        elif filename.startswith('robot_'):
            return 'robot'
        else:
            assert False, f"Unknown category: {filename}"

    # Main evaluation loop: enumerate by video_id first
    for video_id, vqa_file in tqdm(video_id_to_vqa_file.items(), disable=get_rank() > 0, desc="Processing video_ids"):
        category = get_category_from_filename(vqa_file)

        # Skip if video_id not in prompt data
        if video_id not in video_id_set:
            logger.warning(f"Video ID {video_id} not found in prompt file, skipping")
            continue

        # Load QA data for this video_id
        qa_file_path = os.path.join(vqa_questions_dir, vqa_file)
        qa_data = load_json(qa_file_path)

        if not qa_data:
            logger.warning(f"No QA data found for {video_id}")
            continue

        # Find all corresponding video files (different seeds) for this video_id
        video_info_list = find_all_video_files(video_id, video_dir)
        if not video_info_list:
            if enable_missing_videos:
                # Skip missing videos when flag is enabled
                continue
            else:
                # Error out when flag is disabled
                error_msg = f"No video files found for {video_id}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

        # logger.info(f"Processing video_id: {video_id} with {len(video_info_list)} seed versions")

        # Process each video_path/seed for this video_id
        all_seed_results = []
        for video_info in video_info_list:
            video_path = video_info["path"]
            seed = video_info["seed"]

            # logger.info(f"Evaluating video_id: {video_id}, seed: {seed}, path: {video_path}")

            # Use the optimized evaluate_video_qa method:
            # - Calls process_vision_info only once per video
            # - Batch processes all qa_pairs using vLLM
            seed_results = evaluator.evaluate_video_qa(video_path, qa_data)

            # Add seed information to each result
            for result in seed_results:
                result["seed"] = seed
                result["video_id"] = video_id

            all_seed_results.extend(seed_results)

        # Aggregate results across seeds for each question
        aggregated_results = evaluator.aggregate_multi_seed_results(all_seed_results)

        # Calculate video-level accuracy (average across questions within this video)
        video_accuracy = sum(r['accuracy'] for r in aggregated_results) / len(aggregated_results) if aggregated_results else 0.0

        # Update counters using video-level accuracy
        total_count += 1
        correct_count += video_accuracy
        if category in category_stats:
            category_stats[category]['total'] += 1
            category_stats[category]['correct'] += video_accuracy

        # Store results with multi-seed information
        video_paths = [v["path"] for v in video_info_list]
        seeds = [v["seed"] for v in video_info_list]
        all_results.append({
            'video_id': video_id,
            'video_paths': video_paths,  # List of all video paths
            'seeds': seeds,  # List of all seeds
            'category': category,
            'results': aggregated_results,  # Already aggregated across seeds
            'accuracy': video_accuracy,  # Use the already calculated video-level accuracy
            'total_seeds': len(video_info_list)
        })

    # Gather results from all ranks
    if get_world_size() > 1:
        all_results = gather_list_of_dict(all_results)

        # Recalculate totals from gathered results
        correct_count = 0
        total_count = 0
        # Reset category stats for recalculation
        for cat in category_stats:
            category_stats[cat] = {'correct': 0, 'total': 0}

        for video_result in all_results:
            category = video_result.get('category', 'unknown')
            video_accuracy = video_result.get('accuracy', 0.0)

            # Count each video once and use its overall accuracy
            total_count += 1
            correct_count += video_accuracy
            if category in category_stats:
                category_stats[category]['total'] += 1
                category_stats[category]['correct'] += video_accuracy

    # Calculate overall accuracy
    overall_accuracy = correct_count / total_count if total_count > 0 else 0.0

    # Calculate category-specific scores
    category_scores = {}
    for category, stats in category_stats.items():
        if stats['total'] > 0:
            category_scores[f"{category}_score"] = stats['correct'] / stats['total']
        else:
            category_scores[f"{category}_score"] = 0.0

    # Calculate total seeds processed
    total_seeds_processed = sum(result.get('total_seeds', 1) for result in all_results)

    logger.info(f"VQA Evaluation Complete: {correct_count}/{total_count} = {overall_accuracy:.4f}")
    logger.info(f"Total video_ids processed: {len(all_results)}")
    logger.info(f"Total seeds processed: {total_seeds_processed}")

    # Log category scores
    for category, score in category_scores.items():
        stats = category_stats[category.replace('_score', '')]
        logger.info(f"{category}: {stats['correct']}/{stats['total']} = {score:.4f}")

    return overall_accuracy, all_results, category_scores


if __name__ == "__main__":
    # Test the VQA evaluation
    vqa_questions_dir = "/lustre/fs1/portfolios/dir/projects/dir_cosmos_misc/users/fengzhez/repos/pai/predict/v1.0/vqa_clean"
    video_dir = "/lustre/fs1/portfolios/dir/projects/dir_cosmos_misc/users/fengzhez/evals/data/pai/organized-mini/andrewwan_20250818155604_Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-Note-T2V_high_sigma_loss_reweighted_I2W_long_iter-10000"
    prompt_file = "/lustre/fs1/portfolios/dir/projects/dir_cosmos_misc/users/fengzhez/repos/VBench/data/cosmos_predict2_bench_full_info_mini.json"

    accuracy, results, category_scores = compute_vqa_accuracy(
        vqa_questions_dir=vqa_questions_dir,
        video_dir=video_dir,
        prompt_file=prompt_file,
        model_name="Qwen/Qwen2.5-VL-72B-Instruct"
    )

    print(f"Overall VQA Accuracy: {accuracy:.4f}")
    print("Category Scores:")
    for score_name, score_value in category_scores.items():
        print(f"  {score_name}: {score_value:.4f}")
