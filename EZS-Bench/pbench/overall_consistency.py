import os
import json
import numpy as np
from pathlib import Path

import torch
import clip
from tqdm import tqdm
from pbench.utils import load_video, load_dimension_info, clip_transform, read_frames_decord_by_fps, CACHE_DIR
from pbench.third_party.ViCLIP.viclip import ViCLIP
from pbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer

from .distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)


def get_text_features(model, input_text, tokenizer, text_feature_dict={}):
    if input_text in text_feature_dict:
        return text_feature_dict[input_text]
    text_template= f"{input_text}"
    with torch.no_grad():
        text_features = model.encode_text(text_template).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_feature_dict[input_text] = text_features
    return text_features

def get_vid_features(model, input_frames):
    with torch.no_grad():
        clip_feat = model.encode_vision(input_frames,test=True).float()
        clip_feat /= clip_feat.norm(dim=-1, keepdim=True)
    return clip_feat

def get_predict_label(clip_feature, text_feats_tensor, top=5):
    label_probs = (100.0 * clip_feature @ text_feats_tensor.T).softmax(dim=-1)
    top_probs, top_labels = label_probs.cpu().topk(top, dim=-1)
    return top_probs, top_labels

def overall_consistency(clip_model, video_dict, tokenizer, device, sample="middle"):
    sim = []
    video_results = []
    image_transform = clip_transform(224)
    
    # DEBUG: 检查前3个视频的 prompt 是否正确传递
    if get_rank() == 0 and len(video_dict) > 0:
        print("\n[DEBUG] overall_consistency - 前3个视频的 prompt 检查:")
        for idx, info in enumerate(video_dict[:3]):
            prompt_data = info.get('prompt', '')
            if isinstance(prompt_data, dict):
                query = prompt_data.get('prompt_en') or prompt_data.get('prompt') or ''
            else:
                query = prompt_data
            video_path = info.get('video_list', [''])[0] if 'video_list' in info else ''
            print(f"  视频{idx+1}: {Path(video_path).name if video_path else 'N/A'}")
            print(f"    Prompt长度: {len(query)} 字符")
            print(f"    Prompt前80字符: {query[:80]}...")
    
    for info in tqdm(video_dict, disable=get_rank() > 0):
        # 兼容多种格式：
        # 1. info['prompt'] 是字符串（load_dimension_info 从标准格式返回）
        # 2. info['prompt']['prompt_en'] 是字典格式（某些情况下）
        # 3. info['prompt'] 是字典，包含 'prompt' 键（用户自定义格式）
        prompt_data = info['prompt']
        if isinstance(prompt_data, dict):
            # 如果是字典，尝试获取 prompt 文本
            query = prompt_data.get('prompt_en') or prompt_data.get('prompt') or ''
            if not query:
                # 如果都没有，尝试获取第一个字符串值
                for value in prompt_data.values():
                    if isinstance(value, str) and value:
                        query = value
                        break
        else:
            # 如果是字符串，直接使用
            query = prompt_data
        
        if not query:
            print(f"Warning: Could not extract prompt from {info}, skipping...")
            continue
        
        # DEBUG: 检查是否使用了文件名作为 prompt（长度很短）
        if get_rank() == 0 and len(query) < 50:
            video_path = info.get('video_list', [''])[0] if 'video_list' in info else ''
            print(f"\n⚠️  WARNING: 检测到极短 prompt (长度={len(query)}): '{query}'")
            print(f"    视频: {video_path}")
            print(f"    可能原因: prompt 未正确传递，退化为使用文件名")
        
        # WARNING: ViCLIP tokenizer 截断警告
        if get_rank() == 0 and len(query) > 300:
            video_path = info.get('video_list', [''])[0] if 'video_list' in info else ''
            est_tokens = len(query) / 4
            print(f"\n⚠️  WARNING: 检测到超长 prompt (长度={len(query)} 字符, 估算≈{est_tokens:.0f} tokens)")
            print(f"    视频: {Path(video_path).name if video_path else 'N/A'}")
            print(f"    ViCLIP tokenizer 上限为 77 tokens，超出部分将被截断")
            print(f"    这会导致丢失 {(est_tokens - 77) / est_tokens * 100:.1f}% 的文本内容")
        # text = clip.tokenize([query]).to(device)
        video_list = info['video_list']
        for video_path in video_list:
            cur_video = []
            with torch.no_grad():
                images = read_frames_decord_by_fps(video_path, num_frames=8, sample=sample)
                images = image_transform(images)
                images = images.to(device)
                clip_feat = get_vid_features(clip_model,images.unsqueeze(0))
                text_feat = get_text_features(clip_model, query, tokenizer)
                logit_per_text =  clip_feat @ text_feat.T
                score_per_video =  float(logit_per_text[0][0].cpu())
                sim.append(score_per_video)
                video_results.append({'video_path': video_path, 'video_results': score_per_video})
    avg_score = np.mean(sim)
    return avg_score, video_results

def compute_overall_consistency(json_dir, device, submodules_list, **kwargs):
    tokenizer = SimpleTokenizer(os.path.join(CACHE_DIR, "ViCLIP/bpe_simple_vocab_16e6.txt.gz"))
    viclip = ViCLIP(tokenizer= tokenizer, **submodules_list).to(device)
    _, video_dict = load_dimension_info(json_dir, dimension='overall_consistency', lang='en')
    video_dict = distribute_list_to_rank(video_dict)
    all_results, video_results = overall_consistency(viclip, video_dict, tokenizer, device)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results
