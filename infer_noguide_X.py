import argparse
import os
import random
from typing import Optional
import torch
import numpy as np
from core.inference.wan_noguide import generate_video
from core.finetune.datasets.utils import load_from_json_file, iproj_disp
from pathlib import Path
import re

def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_chunk_prompt(prompt, chunk_idx, num_chunks):
    if not isinstance(prompt, str):
        return prompt
        
    def split_into_sentences(text):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s.strip()]
        
    exo_part = ""
    ego_part = ""
    
    if "[Ego view]" in prompt:
        parts = prompt.split("[Ego view]")
        exo_part = parts[0].replace("[Exo view]", "").strip()
        ego_part = parts[1].strip()
    else:
        exo_part = prompt.replace("[Exo view]", "").strip()
        
    def parse_section(part_text):
        scene = ""
        action = ""
        action_match = re.search(r"(?:\*\*)?Action Analysis:(?:\*\*)?", part_text)
        scene_pattern = r"(?:\*\*)?Scene Overview:(?:\*\*)?"
        if action_match:
            scene_text = part_text[:action_match.start()]
            scene = re.sub(scene_pattern, "", scene_text, count=1).strip()
            action = part_text[action_match.end():].strip()
        elif re.search(scene_pattern, part_text):
            scene = re.sub(scene_pattern, "", part_text, count=1).strip()
        else:
            action = part_text
        return scene, action

    exo_scene, exo_action = parse_section(exo_part)
    ego_scene, ego_action = parse_section(ego_part)
    
    if not exo_action and not ego_action:
        return prompt
        
    exo_sentences = split_into_sentences(exo_action) if exo_action else []
    ego_sentences = split_into_sentences(ego_action) if ego_action else []
    
    def get_chunk_sentences(sentences, idx, total):
        S = len(sentences)
        if S == 0:
            return ""
        if total <= 1:
            return " ".join(sentences)
            
        c_i = idx * (S - 1) / (total - 1)
        w_s = max(2, int(round(S / total)) + 1)
        
        start = max(0, int(round(c_i - w_s / 2.0)))
        end = min(S, int(round(c_i + w_s / 2.0)))
        if end - start < w_s:
            if start == 0:
                end = min(S, w_s)
            elif end == S:
                start = max(0, S - w_s)
                
        chunk_sens = sentences[start:end]
        return " ".join(chunk_sens)

    chunk_exo_action = get_chunk_sentences(exo_sentences, chunk_idx, num_chunks)
    chunk_ego_action = get_chunk_sentences(ego_sentences, chunk_idx, num_chunks)
    
    reconstructed = ""
    if exo_scene or chunk_exo_action:
        reconstructed += "[Exo view]\n"
        if exo_scene:
            reconstructed += f"**Scene Overview:**\n{exo_scene}\n\n"
        if chunk_exo_action:
            reconstructed += f"**Action Analysis:**\n{chunk_exo_action}\n"
            
    if ego_scene or chunk_ego_action:
        if reconstructed:
            reconstructed += "\n"
        reconstructed += "[Ego view]\n"
        if ego_scene:
            reconstructed += f"**Scene Overview:**\n{ego_scene}\n\n"
        if chunk_ego_action:
            reconstructed += f"**Action Analysis:**\n{chunk_ego_action}\n"
            
    return reconstructed.strip()


def main(args):
    _set_seed(args.seed)

    meta_data_file = args.meta_data_file
    meta_data = load_from_json_file(meta_data_file)
    if isinstance(meta_data, dict) and 'test_datasets' in meta_data:
        meta_data = meta_data['test_datasets']

    prompts = []
    exo_videos = []
    ego_prior_videos = []
    take_names = []

    for i, meta in enumerate(meta_data):
        exo_path = meta['exo_path']
        exo_videos.append(exo_path)
        ego_prior_videos.append(meta.get('ego_prior_path', None))
        prompts.append(meta['prompt'])
        
        take_name = Path(exo_path).parent.name
        take_names.append(take_name)
    
    from transformers import CLIPVisionModel
    model_path = args.model_path
    transformer_path = os.path.join(model_path, 'transformer')
    lora_path = args.lora_path
    dtype = torch.bfloat16

    from core.finetune.models.wan_i2v.custom_transformer import WanTransformer3DModel_GGA as WanTransformer3DModel
    transformer = WanTransformer3DModel.from_pretrained(transformer_path, torch_dtype=dtype)

    image_encoder = CLIPVisionModel.from_pretrained(model_path, subfolder="image_encoder", torch_dtype=torch.float32)

    from core.finetune.models.wan_i2v.sft_trainer_noguide import WanWidthConcatImageToVideoPipelineNoGuide
    pipe = WanWidthConcatImageToVideoPipelineNoGuide.from_pretrained(
        model_path, image_encoder=image_encoder, transformer=transformer, torch_dtype=dtype
    )
    
    if lora_path:
        print('loading lora:', lora_path)
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora(components=["transformer"], lora_scale=1.0)

    pipe.to("cuda")

    os.makedirs(args.out, exist_ok=True)
    for i in range(len(prompts)):
        if i != args.idx and args.idx != -1:
            continue
        if args.start_idx != -1 and args.end_idx != -1:
            if i < args.start_idx or i >= args.end_idx:
                continue
        prompt, exo_video_path, ego_prior_video_path = prompts[i], exo_videos[i], ego_prior_videos[i]

        take_name = take_names[i]
        exo_video_path = str(exo_video_path)
        
        p_exo = Path(exo_video_path)
        if not p_exo.exists() and "datasets/" in exo_video_path:
            part = exo_video_path[exo_video_path.find("datasets/"):]
            p_exo = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), part))
        exo_video_path = str(p_exo)
        
        # Check if ego_prior_video_path exists
        if args.no_ego_prior:
            ego_prior_video_path = None
        elif ego_prior_video_path:
            p_ego = Path(str(ego_prior_video_path))
            if not p_ego.exists() and "datasets/" in str(ego_prior_video_path):
                part = str(ego_prior_video_path)[str(ego_prior_video_path).find("datasets/"):]
                p_ego = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), part))
            if p_ego.exists():
                ego_prior_video_path = str(p_ego)
            else:
                print(f"[NoGuide] Ego prior video path not found: {p_ego}. Running without ego prior.")
                ego_prior_video_path = None
        else:
            ego_prior_video_path = None

        print(f"[NoGuide] Resolved Exo Video Path: {exo_video_path}")
        print(f"[NoGuide] Ego Prior Video Path: {ego_prior_video_path}")

        from diffusers.utils import load_video
        full_exo_video = load_video(video=exo_video_path)
        if ego_prior_video_path is not None:
            full_ego_prior_video = load_video(video=ego_prior_video_path)
            total_frames = min(len(full_exo_video), len(full_ego_prior_video))
        else:
            full_ego_prior_video = None
            total_frames = len(full_exo_video)
        
        if args.max_frames > 0:
            total_frames = min(total_frames, args.max_frames)
        
        chunk_size = 49
        overlap = 8
        stride = chunk_size - overlap
        
        if total_frames <= chunk_size:
            num_chunks = 1
        else:
            num_chunks = (total_frames - overlap - 1) // stride + 1
            
        print(f"[NoGuide] Total frames: {total_frames}, chunks to generate: {num_chunks}")

        # Global ego noise generation for temporal noise consistency
        num_channels_latents = 16
        latent_height = 448 // 8  # 56
        ego_latent_width = 448 // 8  # 56
        num_latent_frames_total = (total_frames - 1) // 4 + 1
        
        generator_noise = torch.Generator(device="cpu").manual_seed(args.seed)
        global_ego_noise = torch.randn(
            (1, num_channels_latents, num_latent_frames_total, latent_height, ego_latent_width),
            generator=generator_noise,
            dtype=torch.float32
        )

        generated_ego_chunks = []
        chunk_start_indices = []

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * stride
            end_idx = start_idx + chunk_size
            if end_idx > total_frames:
                end_idx = total_frames
                start_idx = max(0, end_idx - chunk_size)
            
            chunk_start_indices.append(start_idx)
            
            start_idx_latent = start_idx // 4
            num_latent_frames_chunk = (chunk_size - 1) // 4 + 1
            chunk_ego_noise = global_ego_noise[:, :, start_idx_latent : start_idx_latent + num_latent_frames_chunk].clone().cuda()
                
            print(f"[NoGuide] Generating chunk {chunk_idx + 1}/{num_chunks} (frames {start_idx} to {end_idx})")
            
            exo_chunk = full_exo_video[start_idx:end_idx]
            if full_ego_prior_video is not None:
                ego_prior_chunk = list(full_ego_prior_video[start_idx:end_idx])
                if chunk_idx > 0:
                    prev_start_idx = chunk_start_indices[chunk_idx - 1]
                    overlap_index = start_idx - prev_start_idx
                    prev_ego_frame = generated_ego_chunks[chunk_idx - 1][overlap_index]
                    from PIL import Image
                    prev_frame_np = np.clip(prev_ego_frame * 255, 0, 255).astype(np.uint8)
                    ego_prior_chunk[0] = Image.fromarray(prev_frame_np)
            else:
                ego_prior_chunk = None

            temp_output_filename = f'{take_name}_{i}_chunk_{chunk_idx}_noguide.mp4'  
            temp_output_path = os.path.join(args.out, temp_output_filename)
            
            if args.idx == -1 or i == args.idx:
                print('\nProcessing item index:', i)
                if isinstance(prompt, list):
                    chunk_prompt = prompt[min(chunk_idx, len(prompt) - 1)]
                else:
                    chunk_prompt = get_chunk_prompt(prompt, chunk_idx, num_chunks)
                    
                video = generate_video(
                    prompt=chunk_prompt,
                    exo_video_path=exo_chunk,
                    ego_prior_video_path=ego_prior_chunk,
                    output_path=temp_output_path,
                    num_frames=chunk_size,
                    width=784+448,
                    height=448,
                    num_inference_steps=50,
                    guidance_scale=5.0,
                    fps=30,
                    num_videos_per_prompt=1,
                    seed=args.seed,
                    attention_GGA=None,
                    attention_mask_GGA=None,
                    point_vecs_per_frame=None,
                    cam_rays=None,
                    do_kv_cache=args.do_kv_cache,
                    cos_sim_scaling_factor=args.cos_sim_scaling_factor,
                    pipe=pipe,
                    mask_first_frame_ego=(chunk_idx > 0 and ego_prior_chunk is not None),
                    ego_latents=chunk_ego_noise,
                )
                
                generated_ego_chunks.append(video)

        # Alpha blending & concatenation
        if num_chunks == 1:
            final_ego_video = generated_ego_chunks[0]
        else:
            final_ego_video = generated_ego_chunks[0][:-overlap].copy()
            for k in range(1, num_chunks):
                prev_chunk = generated_ego_chunks[k-1]
                curr_chunk = generated_ego_chunks[k]
                
                c1_overlap = prev_chunk[-overlap:]
                c2_overlap = curr_chunk[:overlap]
                
                weights = np.linspace(0.0, 1.0, overlap)
                weights = weights[:, np.newaxis, np.newaxis, np.newaxis]
                
                blended_overlap = (1.0 - weights) * c1_overlap + weights * c2_overlap
                final_ego_video = np.concatenate([final_ego_video, blended_overlap], axis=0)
                
                if k == num_chunks - 1:
                    final_ego_video = np.concatenate([final_ego_video, curr_chunk[overlap:]], axis=0)
                else:
                    final_ego_video = np.concatenate([final_ego_video, curr_chunk[overlap:-overlap]], axis=0)

        import imageio
        output_filename = f'{take_name}_{i}_noguide_long.mp4'  
        output_path = os.path.join(args.out, output_filename)
        video_frames = np.clip(final_ego_video * 255, 0, 255).astype(np.uint8)
        imageio.mimsave(output_path, video_frames, fps=30, quality=9)
        print(f"★ [NoGuide] Saved long video to: {output_path}")

        for chunk_idx in range(num_chunks):
            temp_filename = f'{take_name}_{i}_chunk_{chunk_idx}_noguide.mp4'
            temp_path = os.path.join(args.out, temp_filename)
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using Wan without GGA/EgoPrior")
    parser.add_argument("--idx", type=int, default=-1)
    parser.add_argument("--model_path", type=str, default=None, help="The path of the SFT weights to be used")
    parser.add_argument("--out", type=str, default="./outputs_noguide", help="The path save generated video")
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=64, help="The rank of the LoRA weights to be used")
    parser.add_argument("--meta_data_file", type=str, default="./datasets/kitchen1/meta.json", help="meta data file")
    parser.add_argument("--use_GGA", action='store_true', help="whether to use GGA or not")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible inference")
    parser.add_argument("--do_kv_cache", action='store_true', help="whether to use key-value caching")
    parser.add_argument("--cos_sim_scaling_factor", type=float, default=1.0, help="scaling factor for cos similarity attention map")
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--in_the_wild", action='store_true', help="whether to use in-the-wild inference")
    parser.add_argument("--no_ego_prior", action='store_true', help="Force running without ego prior video even if path exists")
    parser.add_argument("--max_frames", type=int, default=49, help="Maximum number of frames to generate")

    args = parser.parse_args()

    main(args)
