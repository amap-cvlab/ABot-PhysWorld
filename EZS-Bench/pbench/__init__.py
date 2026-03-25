import os

from .utils import get_prompt_from_filename, init_submodules, save_json, load_json
from .utils_i2v import init_submodules as init_submodules_i2v
import importlib
from itertools import chain
from pathlib import Path

from .distributed import get_rank, print0


class PBench(object):
    def __init__(self, device, full_info_dir, output_path):
        self.device = device                        # cuda or cpu
        self.full_info_dir = full_info_dir          # full json file that VBench originally provides
        self.output_path = output_path              # output directory to save VBench results
        os.makedirs(self.output_path, exist_ok=True)

    def build_full_dimension_list(self, ):
        # Only include the 8 dimensions we need
        return ["aesthetic_quality", "background_consistency", "imaging_quality", "motion_smoothness", "overall_consistency", "subject_consistency", "i2v_background", "i2v_subject"]

    def check_dimension_requires_extra_info(self, dimension_list):
        # For pbench, we only support the 8 dimensions specified
        supported_dims = {"aesthetic_quality", "background_consistency", "imaging_quality", "motion_smoothness", "overall_consistency", "subject_consistency", "i2v_background", "i2v_subject"}
        unsupported_dims = set(dimension_list) - supported_dims

        assert len(unsupported_dims) == 0, f"dimensions : {unsupported_dims} not supported in pbench"

    def build_custom_image_dict(self, directory):
        image_dict = {}

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)

            if os.path.isfile(file_path):
                image_name, extension = os.path.splitext(filename)
                extension = extension.lower()

                if extension in ['.jpg', '.jpeg', '.png']:
                    image_dict[image_name] = file_path

        return image_dict


    def build_full_info_json(self, videos_path, name, dimension_list, prompt_list=[],  special_str='', verbose=False, custom_image_folder=None, mode='vbench_standard', enable_missing_videos=False, **kwargs):
        cur_full_info_list=[] # to save the prompt and video path info for the current dimensions

        # Track missing videos for reporting
        missing_videos_count = 0

        if mode=='custom_input':
            self.check_dimension_requires_extra_info(dimension_list)
            if custom_image_folder:
                custom_image_dict = self.build_custom_image_dict(custom_image_folder)

            if os.path.isfile(videos_path):
                # Check if single video file exists
                if not os.path.exists(videos_path):
                    if not enable_missing_videos:
                        raise FileNotFoundError(f"Video file not found: {videos_path}")
                    else:
                        print0(f"WARNING: Skipping missing video file: {videos_path}")
                        missing_videos_count += 1
                else:
                    if custom_image_folder is None:
                        cur_full_info_list = [{"prompt_en": get_prompt_from_filename(videos_path), "dimension": dimension_list, "video_list": [videos_path]}]
                    else:
                        cur_full_info_list = [{"prompt_en": get_prompt_from_filename(videos_path), "dimension": dimension_list, "video_list": [videos_path], "custom_image_path": custom_image_dict[get_prompt_from_filename(videos_path)]}]

                    if len(prompt_list) == 1:
                        cur_full_info_list[0]["prompt_en"] = prompt_list[0]
            else:
                video_names = os.listdir(videos_path)

                cur_full_info_list = []

                if custom_image_folder is None or len(prompt_list) > 0:
                    for filename in video_names:
                        postfix = Path(os.path.join(videos_path, filename)).suffix
                        if postfix.lower() not in ['.mp4', '.gif',]: #  '.jpg', '.png'
                            continue
                        video_path = os.path.join(videos_path, filename)
                        # Check if video exists
                        if not os.path.exists(video_path):
                            if not enable_missing_videos:
                                raise FileNotFoundError(f"Video file not found: {video_path}")
                            else:
                                print0(f"WARNING: Skipping missing video: {video_path}")
                                missing_videos_count += 1
                                continue
                        cur_full_info_list.append({
                            "prompt_en": get_prompt_from_filename(filename),
                            "dimension": dimension_list,
                            "video_list": [video_path]
                        })
                else:
                    for filename in video_names:
                        postfix = Path(os.path.join(videos_path, filename)).suffix
                        if postfix.lower() not in ['.mp4', '.gif']: #  '.jpg', '.png'
                            continue
                        video_path = os.path.join(videos_path, filename)
                        # Check if video exists
                        if not os.path.exists(video_path):
                            if not enable_missing_videos:
                                raise FileNotFoundError(f"Video file not found: {video_path}")
                            else:
                                print0(f"WARNING: Skipping missing video: {video_path}")
                                missing_videos_count += 1
                                continue

                        prompt_name = get_prompt_from_filename(filename)
                        # Check if custom image exists for this prompt
                        if prompt_name not in custom_image_dict:
                            if not enable_missing_videos:
                                raise FileNotFoundError(f"Custom image not found for prompt: {prompt_name}")
                            else:
                                print0(f"WARNING: Skipping video due to missing custom image for prompt: {prompt_name}")
                                missing_videos_count += 1
                                continue

                        cur_full_info_list.append({
                            "prompt_en": prompt_name,
                            "dimension": dimension_list,
                            "video_list": [video_path],
                            "custom_image_path": custom_image_dict[prompt_name]
                        })

                if len(prompt_list) > 0:
                    prompt_list = {os.path.join(videos_path, path): prompt_list[path] for path in prompt_list}

                    if not enable_missing_videos:
                        assert len(prompt_list) >= len(cur_full_info_list), """
                            Number of prompts should match with number of videos.\n
                            Got {len(prompt_list)=}, {len(cur_full_info_list)=}\n
                            To read the prompt from filename, delete --prompt_file and --prompt_list
                            """

                    all_video_path = [os.path.abspath(file) for file in list(chain.from_iterable(vid["video_list"] for vid in cur_full_info_list))]
                    backslash = "\n"
                    missing_prompts = set(all_video_path) - set([os.path.abspath(path_key) for path_key in prompt_list])
                    if missing_prompts:
                        if not enable_missing_videos:
                            assert len(missing_prompts) == 0, f"""
                            The prompts for the following videos are not found in the prompt file: \n
                            {backslash.join(missing_prompts)}
                            """
                        else:
                            print0(f"WARNING: {len(missing_prompts)} videos have no prompts in prompt file, skipping them")
                            # Filter out videos without prompts
                            cur_full_info_list = [vid for vid in cur_full_info_list
                                                 if os.path.abspath(vid["video_list"][0]) not in missing_prompts]
                            missing_videos_count += len(missing_prompts)

                    video_map = {}
                    for prompt_key in prompt_list:
                        video_map[os.path.abspath(prompt_key)] = prompt_list[prompt_key]

                    for video_info in cur_full_info_list:
                        prompt_dict = video_map[os.path.abspath(video_info["video_list"][0])]
                        # Handle both dict and string prompt formats
                        if isinstance(prompt_dict, dict):
                            # Extract prompt text from dict
                            video_info["prompt_en"] = prompt_dict.get("prompt_en") or prompt_dict.get("prompt") or str(prompt_dict)
                            # If the prompt dictionary has image information, add it to the video_info
                            if "custom_image_path" in prompt_dict:
                                # Use custom_image_path directly if provided
                                video_info["custom_image_path"] = prompt_dict["custom_image_path"]
                            elif "image_name" in prompt_dict:
                                if custom_image_folder:
                                    image_path = os.path.join(custom_image_folder, prompt_dict["image_name"])
                                    # Check if custom image exists
                                    if not os.path.exists(image_path):
                                        if not enable_missing_videos:
                                            raise FileNotFoundError(f"Custom image not found: {image_path}")
                                        else:
                                            print0(f"WARNING: Custom image not found: {image_path}, using filename as path")
                                    video_info["custom_image_path"] = image_path
                                else:
                                    video_info["custom_image_path"] = prompt_dict["image_name"]
                        else:
                            # If prompt_dict is a string, use it directly
                            video_info["prompt_en"] = prompt_dict

        elif mode=='vbench_category':
            self.check_dimension_requires_extra_info(dimension_list)
            CUR_DIR = os.path.dirname(os.path.abspath(__file__))
            category_supported = [ Path(category).stem for category in os.listdir(f'prompts/prompts_per_category') ]# TODO: probably need refactoring again
            if 'category' not in kwargs:
                category = category_supported
            else:
                category = kwargs['category']

            assert category is not None, "Please specify the category to be evaluated with --category"
            assert category in category_supported, f'''
            The following category is not supported, {category}.
            '''

            video_names = os.listdir(videos_path)
            postfix = Path(video_names[0]).suffix

            with open(f'{CUR_DIR}/prompts_per_category/{category}.txt', 'r') as f:
                video_prompts = [line.strip() for line in f.readlines()]

            for prompt in video_prompts:
                video_list = []
                for filename in video_names:
                    if (not Path(filename).stem.startswith(prompt)):
                        continue
                    postfix = Path(os.path.join(videos_path, filename)).suffix
                    if postfix.lower() not in ['.mp4', '.gif', '.jpg', '.png']:
                        continue
                    video_path = os.path.join(videos_path, filename)
                    # Check if video exists
                    if not os.path.exists(video_path):
                        if not enable_missing_videos:
                            raise FileNotFoundError(f"Video file not found: {video_path}")
                        else:
                            print0(f"WARNING: Skipping missing video: {video_path}")
                            missing_videos_count += 1
                            continue
                    video_list.append(video_path)

                cur_full_info_list.append({
                    "prompt_en": prompt,
                    "dimension": dimension_list,
                    "video_list": video_list
                })

        else:
            full_info_list = load_json(self.full_info_dir)
            video_names = os.listdir(videos_path)
            postfix = Path(video_names[0]).suffix
            for prompt_dict in full_info_list:
                # if the prompt belongs to any dimension we want to evaluate
                if set(dimension_list) & set(prompt_dict["dimension"]):
                    prompt = prompt_dict['prompt_en']
                    prompt_dict['video_list'] = []
                    for i in range(5): # video index for the same prompt
                        intended_video_name = f'{prompt}{special_str}-{str(i)}{postfix}'
                        if intended_video_name in video_names: # if the video exists
                            intended_video_path = os.path.join(videos_path, intended_video_name)
                            # Check if video file actually exists
                            if os.path.exists(intended_video_path):
                                prompt_dict['video_list'].append(intended_video_path)
                                if verbose:
                                    print0(f'Successfully found video: {intended_video_name}')
                            else:
                                if not enable_missing_videos:
                                    raise FileNotFoundError(f'Video file not found: {intended_video_path}')
                                else:
                                    print0(f'WARNING!!! Skipping missing video: {intended_video_name}')
                                    missing_videos_count += 1
                        else:
                            if not enable_missing_videos:
                                raise FileNotFoundError(f'Required video not found: {intended_video_name}')
                            else:
                                print0(f'WARNING!!! This required video is not found! Missing benchmark videos can lead to unfair evaluation result. The missing video is: {intended_video_name}')
                                missing_videos_count += 1
                    # Only add to the list if we have at least some videos (or all if enable_missing_videos is False)
                    if prompt_dict['video_list'] or not enable_missing_videos:
                        cur_full_info_list.append(prompt_dict)


        cur_full_info_path = os.path.join(self.output_path, name+'_full_info.json')
        save_json(cur_full_info_list, cur_full_info_path)
        print0(f'Evaluation meta data saved to {cur_full_info_path}')

        if missing_videos_count > 0:
            print0(f'WARNING: {missing_videos_count} videos were missing and skipped')

        return cur_full_info_path


    def evaluate(self, videos_path, name, prompt_list=[], dimension_list=None, local=False, read_frame=False, mode='vbench_standard', custom_image_folder=None, resolution="1-1", enable_missing_videos=False, **kwargs):
        results_dict = {}
        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()

        # Define which dimensions are i2v specific
        i2v_dims = ["i2v_subject", "i2v_background"]
        t2v_dims = [d for d in dimension_list if d not in i2v_dims]

        # Initialize submodules for both types
        submodules_dict = {}
        if t2v_dims:
            t2v_submodules = init_submodules(t2v_dims, local=local, read_frame=read_frame)
            submodules_dict.update(t2v_submodules)
        if any(d in i2v_dims for d in dimension_list):
            i2v_submodules = init_submodules_i2v([d for d in dimension_list if d in i2v_dims], local=local, read_frame=read_frame, resolution=resolution or "1-1")
            submodules_dict.update(i2v_submodules)

        cur_full_info_path = self.build_full_info_json(videos_path, name, dimension_list, prompt_list, mode=mode, custom_image_folder=custom_image_folder, enable_missing_videos=enable_missing_videos, **kwargs)
        summary_results = {}
        for dimension in dimension_list:
            try:
                if dimension in i2v_dims:
                    dimension_module = importlib.import_module(f'pbench.{dimension}')
                else:
                    dimension_module = importlib.import_module(f'pbench.{dimension}')
                evaluate_func = getattr(dimension_module, f'compute_{dimension}')
            except Exception as e:
                raise NotImplementedError(f'UnImplemented dimension {dimension}!, {e}')
            submodules_list = submodules_dict[dimension]
            print0(f'cur_full_info_path: {cur_full_info_path}') # TODO: to delete
            results = evaluate_func(cur_full_info_path, self.device, submodules_list, **kwargs)
            results_dict[dimension] = results
            print0(dimension, results[0])
            summary_results[dimension] = results[0]
        output_name = os.path.join(self.output_path, name+'_eval_results.json')
        summary_output_name = os.path.join(self.output_path, name+'_summary_results.json')
        print0(summary_results)
        if get_rank() == 0:
            save_json(results_dict, output_name)
            print0(f'Evaluation results saved to {output_name}')
            # Save summary results for quick viewing
            save_json(summary_results, summary_output_name)
            print0(f'Summary results saved to {summary_output_name}')
