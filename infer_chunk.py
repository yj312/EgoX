import argparse
import os
import random
from typing import Optional
import torch
import numpy as np
from core.inference.wan import generate_video

from core.finetune.datasets.utils import load_from_json_file, iproj_disp
from pathlib import Path

def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


import re

def get_chunk_prompt(prompt, chunk_idx, num_chunks):
    if not isinstance(prompt, str):
        return prompt
        
    # Helper to split text into sentences
    def split_into_sentences(text):
        # Split by period, question mark, or exclamation mark followed by space or newline
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s.strip()]
        
    # Parse Exo and Ego sections
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
        # Accept both the README's Markdown labels and plain caption labels.
        # Caption files should not need model-specific formatting rewrites.
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
    
    # If no action descriptions found, fallback to original prompt
    if not exo_action and not ego_action:
        return prompt
        
    # Split actions into sentences
    exo_sentences = split_into_sentences(exo_action) if exo_action else []
    ego_sentences = split_into_sentences(ego_action) if ego_action else []
    
    # Helper to get sliding window subset of sentences for current chunk
    def get_chunk_sentences(sentences, idx, total):
        S = len(sentences)
        if S == 0:
            return ""
        if total <= 1:
            return " ".join(sentences)
            
        # Calculate center sentence index for current chunk
        c_i = idx * (S - 1) / (total - 1)
        # Determine window size (at least 2 sentences, or proportional to chunk density)
        w_s = max(2, int(round(S / total)) + 1)
        
        start = max(0, int(round(c_i - w_s / 2.0)))
        end = min(S, int(round(c_i + w_s / 2.0)))
        # Adjust window to make sure we get at least some sentences
        if end - start < w_s:
            if start == 0:
                end = min(S, w_s)
            elif end == S:
                start = max(0, S - w_s)
                
        chunk_sens = sentences[start:end]
        return " ".join(chunk_sens)

    chunk_exo_action = get_chunk_sentences(exo_sentences, chunk_idx, num_chunks)
    chunk_ego_action = get_chunk_sentences(ego_sentences, chunk_idx, num_chunks)
    
    # Reconstruct the prompt
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


def _stitch_video_chunks(chunks, start_indices):
    """Blend chunks at their original frame positions, including a shifted last chunk."""
    final_video = chunks[0].copy()
    for chunk, start_idx in zip(chunks[1:], start_indices[1:]):
        actual_overlap = len(final_video) - start_idx
        if actual_overlap > 0:
            weights = np.linspace(0.0, 1.0, actual_overlap, dtype=final_video.dtype)
            weights = weights[:, np.newaxis, np.newaxis, np.newaxis]
            final_video[start_idx:] = (
                (1.0 - weights) * final_video[start_idx:]
                + weights * chunk[:actual_overlap]
            )
        final_video = np.concatenate([final_video, chunk[actual_overlap:]], axis=0)
    return final_video


def main(args):
    _set_seed(args.seed)

    meta_data_file = args.meta_data_file
    meta_data = load_from_json_file(meta_data_file)
    meta_data = meta_data['test_datasets']

    prompts = []
    exo_videos = []
    ego_prior_videos = []
    depth_map_paths = []
    camera_intrinsics = []
    camera_extrinsics = []
    ego_extrinsics = []
    ego_intrinsics = []
    take_names = []

    for i, meta in enumerate(meta_data):
        exo_path = meta['exo_path']
        exo_videos.append(exo_path)
        ego_prior_videos.append(meta['ego_prior_path'])
        prompts.append(meta['prompt'])
        
        take_name = Path(exo_path).parent.name
        depth_root = Path(exo_path).parents[2]
        depth_map_paths.append(depth_root / 'depth_maps' / take_name)
        
        camera_extrinsics.append(meta['camera_extrinsics'])
        camera_intrinsics.append(meta['camera_intrinsics'])
        ego_extrinsics.append(meta['ego_extrinsics'])
        ego_intrinsics.append(meta['ego_intrinsics'])
        take_names.append(take_name)
    
    assert len(prompts) == len(exo_videos) == len(ego_prior_videos)
    
    import shutil
    from transformers import CLIPVisionModel

    model_path=args.model_path
    transformer_path = os.path.join(model_path, 'transformer')
    lora_path=args.lora_path
    dtype=torch.bfloat16

    from core.finetune.models.wan_i2v.custom_transformer import WanTransformer3DModel_GGA as WanTransformer3DModel
    transformer = WanTransformer3DModel.from_pretrained(transformer_path, torch_dtype=dtype)

    image_encoder = CLIPVisionModel.from_pretrained(model_path, subfolder="image_encoder", torch_dtype=torch.float32)

    from core.finetune.models.wan_i2v.sft_trainer import WanWidthConcatImageToVideoPipeline
    pipe = WanWidthConcatImageToVideoPipeline.from_pretrained(model_path, image_encoder=image_encoder, transformer=transformer, torch_dtype=dtype)
    
    if lora_path:
        print('loading lora')
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora(components=["transformer"], lora_scale = 1.0 )

    pipe.to("cuda")

    os.makedirs(args.out, exist_ok=True)
    for i in range(len(prompts)):
        if i!=args.idx and args.idx!=-1:
            continue
        if args.start_idx != -1 and args.end_idx != 1:
            if i < args.start_idx or i >= args.end_idx :
                continue
        prompt, exo_video_path, ego_prior_video_path = prompts[i], exo_videos[i], ego_prior_videos[i]

        take_name=take_names[i]
        exo_video_path = str(exo_video_path)
        ego_prior_video_path = str(ego_prior_video_path)
        
        # Robust path resolution relative to current script directory
        p_exo = Path(exo_video_path)
        if not p_exo.exists() and "datasets/" in exo_video_path:
            part = exo_video_path[exo_video_path.find("datasets/"):]
            p_exo = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), part))
        exo_video_path = str(p_exo)
        
        p_ego = Path(ego_prior_video_path)
        if not p_ego.exists() and "datasets/" in ego_prior_video_path:
            part = ego_prior_video_path[ego_prior_video_path.find("datasets/"):]
            p_ego = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), part))
        ego_prior_video_path = str(p_ego)

        print(f"Resolved Exo Video Path: {exo_video_path}")
        print(f"Resolved Ego Prior Video Path: {ego_prior_video_path}")

        from diffusers.utils import load_video
        full_exo_video = load_video(video=exo_video_path)
        full_ego_prior_video = load_video(video=ego_prior_video_path)
        total_frames = min(len(full_exo_video), len(full_ego_prior_video), len(ego_extrinsics[i]))
        
        chunk_size = 49
        overlap = 8
        stride = chunk_size - overlap
        
        if total_frames <= chunk_size:
            num_chunks = 1
        else:
            num_chunks = (total_frames - overlap - 1) // stride + 1
            
        print(f"Total frames: {total_frames}, chunks to generate: {num_chunks}")

        if args.use_GGA:
            depth_map_path = depth_map_paths[i]
            
            # Robust depth map path resolution
            if not depth_map_path.exists():
                depth_map_str = str(depth_map_path)
                if "datasets/" in depth_map_str:
                    part = depth_map_str[depth_map_str.find("datasets/"):]
                    depth_map_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), part))
            print(f"Resolved Depth Map Path: {depth_map_path}")

            camera_intrinsic_full = camera_intrinsics[i]
            camera_extrinsic_full = camera_extrinsics[i]
            ego_extrinsic_full = ego_extrinsics[i]
            ego_intrinsic_full = ego_intrinsics[i]

            # Load all depth maps for the video
            depth_maps_all = []
            for depth_map_file in sorted(depth_map_path.glob("*.npy")):
                depth_map = np.load(depth_map_file)
                depth_maps_all.append(torch.from_numpy(depth_map).unsqueeze(0))
            depth_maps_all = torch.cat(depth_maps_all, dim=0) # [Total_F, H, W]

            ego_intrinsic_full = torch.tensor(ego_intrinsic_full)
            ego_extrinsic_full = torch.tensor(ego_extrinsic_full)
            camera_extrinsic_full = torch.tensor(camera_extrinsic_full)
            camera_intrinsic_full = torch.tensor(camera_intrinsic_full)

            if ego_extrinsic_full.shape[1] == 3 and ego_extrinsic_full.shape[2] == 4:
                ego_extrinsic_full = torch.cat([ego_extrinsic_full, torch.tensor([[[0, 0, 0, 1]]], dtype=ego_extrinsic_full.dtype).expand(ego_extrinsic_full.shape[0], -1, -1)], dim=1)
            if camera_extrinsic_full.shape == (3, 4):
                camera_extrinsic_full = torch.cat([torch.tensor(camera_extrinsic_full, dtype=ego_extrinsic_full.dtype), torch.tensor([[0, 0, 0, 1]], dtype=ego_extrinsic_full.dtype)], dim=0)

        # Generate global ego noise for the entire video to ensure temporal noise consistency across chunks
        num_channels_latents = 16
        latent_height = 448 // 8  # 56
        ego_latent_width = 448 // 8  # 56
        num_latent_frames_total = (total_frames - 1) // 4 + 1
        
        # We use a CPU generator for reproducibility across different GPU states
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
            
            # Slice global ego noise for the current chunk
            start_idx_latent = start_idx // 4
            num_latent_frames_chunk = (chunk_size - 1) // 4 + 1
            chunk_ego_noise = global_ego_noise[:, :, start_idx_latent : start_idx_latent + num_latent_frames_chunk].clone().cuda()
                
            print(f"Generating chunk {chunk_idx + 1}/{num_chunks} (frames {start_idx} to {end_idx})")
            
            exo_chunk = full_exo_video[start_idx:end_idx]
            ego_prior_chunk = list(full_ego_prior_video[start_idx:end_idx])
            
            if chunk_idx > 0:
                prev_start_idx = chunk_start_indices[chunk_idx - 1]
                overlap_index = start_idx - prev_start_idx
                # The generated frame contains Exo+Ego; only Ego belongs in the prior.
                prev_ego_frame = generated_ego_chunks[chunk_idx - 1][overlap_index, :, -448:, :]
                from PIL import Image
                prev_frame_np = np.clip(prev_ego_frame * 255, 0, 255).astype(np.uint8)
                ego_prior_chunk[0] = Image.fromarray(prev_frame_np)
                
            if args.use_GGA:
                depth_maps = depth_maps_all[start_idx:end_idx]
                ego_extrinsic = ego_extrinsic_full[start_idx:end_idx]
                camera_extrinsic = camera_extrinsic_full.clone()
                camera_intrinsic = camera_intrinsic_full.clone()
                ego_intrinsic = ego_intrinsic_full.clone()
                
                device = 'cpu'

                C, F_hardcoded, H, W = 16, 13, 56, 154 #! Hard coding
                exo_H, exo_W = H, W - H
                W = H

                num_frames = (F_hardcoded - 1) * 4 + 1
                ego_extrinsic = ego_extrinsic[:num_frames]
                actual_F = ego_extrinsic[::4].shape[0]

                scale = 1/8
                scaled_intrinsic = ego_intrinsic.clone()
                scaled_intrinsic[0, 0] *= scale  # fx' = fx * s
                scaled_intrinsic[1, 1] *= scale  # fy' = fy * s
                scaled_intrinsic[0, 2] *= scale  # cx' = cx * s
                scaled_intrinsic[1, 2] *= scale

                ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
                ones = torch.ones_like(xs)
                pixel_coords = torch.stack([xs, ys, ones], dim=-1).view(-1, 3).to(dtype=ego_intrinsic.dtype)  # (H*W, 3)

                pixel_coords_cv = pixel_coords[..., :2].cpu().numpy().reshape(-1, 1, 2).astype(np.float32)
                K = scaled_intrinsic.cpu().numpy().astype(np.float32)

                import cv2
                distortion_coeffs = np.array([[-0.02340373583137989,0.09388021379709244,-0.06088035926222801,0.0053304750472307205,0.003342868760228157,-0.0006356257363222539,0.0005087381578050554,-0.0004747129278257489,-0.0011330085108056664,-0.00025734835071489215,0.00009328465239377692,0.00009424977179151028]])
                D = distortion_coeffs.astype(np.float32) # ex: [k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4]
                normalized_points = cv2.undistortPoints(pixel_coords_cv, K, D, R=np.eye(3), P=np.eye(3))

                normalized_points = torch.from_numpy(normalized_points).squeeze(1).to(device) # (N, 2)

                ones = torch.ones_like(normalized_points[..., :1])
                cam_rays_fish = torch.cat([normalized_points, ones], dim=-1) # (N, 3)

                cam_rays = cam_rays_fish / torch.norm(cam_rays_fish, dim=-1, keepdim=True)

                cam_rays = cam_rays @ ego_extrinsic[::4, :3, :3]

                cam_rays = cam_rays.view(actual_F, H, W, 3)  # (B, H, W, 3)

                height, width = depth_maps.shape[1], depth_maps.shape[2]
                cx = width / 2.0
                cy = height / 2.0
                camera_intrinsic_scale_y = cy / camera_intrinsic[1,2]
                camera_intrinsic_scale_x = cx / camera_intrinsic[0,2]
                camera_intrinsic[0, 0] = camera_intrinsic[0, 0] * camera_intrinsic_scale_x
                camera_intrinsic[1, 1] = camera_intrinsic[1, 1] * camera_intrinsic_scale_y
                camera_intrinsic[0, 2] = cx
                camera_intrinsic[1, 2] = cy

                camera_intrinsic = np.array([camera_intrinsic[0, 0] , camera_intrinsic[1, 1], cx, cy])
                
                disp_v, disp_u = torch.meshgrid(
                    torch.arange(depth_maps.shape[1], device=device).float(),
                    torch.arange(depth_maps.shape[2],device=device).float(),
                    indexing="ij",
                )

                disp = torch.ones_like(disp_v)

                disp_cpu = disp.cpu()
                disp_u_cpu = disp_u.cpu()
                disp_v_cpu = disp_v.cpu()
                
                pts, _, _ = iproj_disp(torch.from_numpy(camera_intrinsic), disp_cpu, disp_u_cpu, disp_v_cpu)

                if isinstance(pts, torch.Tensor):
                    pts = pts.to(device)
                else:
                    pts = torch.from_numpy(pts).to(device).float()
                
                rays = pts[..., :3] 

                rays = rays / rays[..., 2:3]    
                rays = rays.unsqueeze(0).expand(depth_maps.size(0), -1, -1, -1)  # (F, H, W, 3)
                camera_extrinsics_c2w = torch.linalg.inv(camera_extrinsic)

                pcd_camera = rays * depth_maps.unsqueeze(-1)
                point_map = pcd_camera.to(dtype=camera_extrinsics_c2w.dtype)
                point_map = torch.tensor(point_map) #! [F, H, W, 3]

                p_f, p_h, p_w, p_p = point_map.shape
                point_map_world = point_map.reshape(-1, 3)

                camera_extrinsics_c2w = torch.linalg.inv(camera_extrinsic)
                ones_point = torch.ones(point_map_world.shape[0], 1, device=point_map_world.device)
                point_map_world = torch.cat([point_map_world, ones_point], dim=-1) #! [F, H*W, 4]
                point_map_world = (camera_extrinsics_c2w @ point_map_world.T).T[...,:3]
                point_map = point_map_world.reshape(p_f, p_h, p_w, 3).permute(0, 3, 1, 2)

                point_map = point_map[:, :, (point_map.shape[2] - 448)//2:(point_map.shape[2] + 448)//2, (point_map.shape[3] - 784)//2:(point_map.shape[3] + 784)//2] #! [F, 3, 448, 784]
                point_map = torch.nn.functional.interpolate(point_map, size=(exo_H, exo_W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1) #! [B, H, W, 3]

                ego_extrinsic_c2w = torch.linalg.inv(ego_extrinsic)

                cam_origins = ego_extrinsic_c2w[::4, :3, 3].unsqueeze(1).expand(-1, exo_H * exo_W, -1)  # (B, H*W, 3)
                cam_origins = cam_origins.view(actual_F, exo_H, exo_W, 3)  # (B, H, W, 3)

                if point_map.size(0) != ego_extrinsic_c2w.size(0):
                    min_size = min(point_map.size(0), ego_extrinsic_c2w.size(0))
                    point_map = point_map[:min_size]
                

                point_vecs_per_frame = []
                for j in range(cam_origins.size(0)):
                    point_vec = point_map[::4] - cam_origins[j].unsqueeze(0)  #! [B, H, W, 3]
                    point_vec = point_vec / torch.norm(point_vec, dim=-1, keepdim=True)  #! [B,H, W, 3]
                    point_vecs_per_frame.append(point_vec)
                point_vecs_per_frame = torch.stack(point_vecs_per_frame, dim=0)  #! [B, B, H, W, 3]

                point_vecs = point_map[::4] - cam_origins #! [B, H, W, 3]
                point_vecs = point_vecs / torch.norm(point_vecs, dim=-1, keepdim=True) #! [B, H, W, 3]

                cam_rays = torch.rot90(cam_rays, k=-1, dims=[1, 2])

                attn_maps = torch.cat((point_vecs, cam_rays), dim = 2)
                attn_masks = torch.cat((torch.ones_like(point_vecs), torch.zeros_like(cam_rays)), dim = 2)
            else:
                point_vecs_per_frame = None
                attn_maps = None
                attn_masks = None
                cam_rays = None

            temp_output_filename = f'{take_name}_{i}_chunk_{chunk_idx}.mp4'  
            temp_output_path = os.path.join(args.out, temp_output_filename)
            
            if args.idx==-1 or i==args.idx:
                print('\n', i)
                if isinstance(prompt, list):
                    chunk_prompt = prompt[min(chunk_idx, len(prompt) - 1)]
                    print(f"Using manual chunk-specific prompt for chunk {chunk_idx}: {chunk_prompt[:100]}...")
                else:
                    chunk_prompt = get_chunk_prompt(prompt, chunk_idx, num_chunks)
                    print(f"Automatically split prompt for chunk {chunk_idx}:\n{chunk_prompt}\n" + "="*40)
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
                    attention_GGA=attn_maps.unsqueeze(0) if attn_maps is not None else None,
                    attention_mask_GGA=attn_masks.unsqueeze(0) if attn_masks is not None else None,
                    point_vecs_per_frame = point_vecs_per_frame,
                    cam_rays = cam_rays,
                    do_kv_cache = args.do_kv_cache,
                    cos_sim_scaling_factor=args.cos_sim_scaling_factor,
                    pipe = pipe,
                    mask_first_frame_ego = (chunk_idx > 0),
                    ego_latents = chunk_ego_noise,
                )
                
                # Use full video chunk (exo+ego view)
                # video shape: [49, H, W_total, C]
                generated_ego_chunks.append(video)

        # 4. Perform alpha blending and concatenation of full view chunks
        final_ego_video = _stitch_video_chunks(generated_ego_chunks, chunk_start_indices)

        # 5. Save the final blended long video to disk
        import imageio
        output_filename = f'{take_name}_{i}_long.mp4'  
        output_path = os.path.join(args.out, output_filename)
        video_frames = np.clip(final_ego_video * 255, 0, 255).astype(np.uint8)
        imageio.mimsave(output_path, video_frames, fps=30, quality=9)
        print(f"★ Saved long video to: {output_path}")

        # 6. Clean up temporary chunk files
        for chunk_idx in range(num_chunks):
            temp_filename = f'{take_name}_{i}_chunk_{chunk_idx}.mp4'
            temp_path = os.path.join(args.out, temp_filename)
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    print(f"Removed temporary chunk file: {temp_path}")
                except Exception as e:
                    print(f"Failed to remove temporary chunk file {temp_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using Wan")
    parser.add_argument("--idx", type=int, default=-1)
    parser.add_argument("--model_path", type=str, default=None, help="The path of the SFT weights to be used")
    parser.add_argument("--out", type=str, default="/outputs", help="The path save generated video")
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=64, help="The rank of the LoRA weights to be used")
    parser.add_argument("--meta_data_file", type=str, default=None, help="meta data for GGA")
    parser.add_argument("--use_GGA", action='store_true', help="whether to use GGA or not")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible inference")
    parser.add_argument("--do_kv_cache", action='store_true', help="whether to use key-value caching")
    parser.add_argument("--cos_sim_scaling_factor", type=float, default=1.0, help="scaling factor for cos similarity attention map")
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--in_the_wild", action='store_true', help="whether to use in-the-wild inference")
    args = parser.parse_args()
    main(args)
