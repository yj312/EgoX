import logging
import os
from typing import Literal, Optional, Union, List
import numpy as np

import torch
from diffusers.utils import load_video

logging.basicConfig(level=logging.INFO)

def generate_video(
    prompt: str,
    num_frames: int = 49,
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_path: str = "./output.mp4",
    exo_video_path: Union[str, List] = "",
    ego_prior_video_path: Optional[Union[str, List]] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    num_videos_per_prompt: int = 1,
    seed: int = 42,
    fps: int = 30,
    attention_GGA: Optional[torch.Tensor] = None,
    attention_mask_GGA: Optional[torch.Tensor] = None,
    point_vecs_per_frame: Optional[torch.Tensor] = None,
    cam_rays: Optional[torch.Tensor] = None,
    cos_sim_scaling_factor: float = 1.0,
    do_kv_cache: bool = False,
    pipe = None,
    mask_first_frame_ego: bool = False,
    ego_latents: Optional[torch.Tensor] = None,
    save_ego_only: bool = True,
):
    """
    Generates a video based on the given prompt without requiring GGA or Ego Prior video.
    """

    if isinstance(exo_video_path, str):
        exo_video = load_video(video=exo_video_path)
    else:
        exo_video = exo_video_path

    cropped_exo_video = []
    for img in exo_video:
        w, h = img.size
        cropped_img = img.crop((0, 0, w, h))
        cropped_img = cropped_img.resize((784, 448))
        cropped_exo_video.append(cropped_img)
    exo_video = cropped_exo_video

    # Process ego_prior_video if provided and non-empty
    ego_prior_video = None
    if ego_prior_video_path:
        try:
            if isinstance(ego_prior_video_path, str) and os.path.exists(ego_prior_video_path):
                ego_prior_video = load_video(video=ego_prior_video_path)
            elif isinstance(ego_prior_video_path, list) and len(ego_prior_video_path) > 0:
                ego_prior_video = ego_prior_video_path
        except Exception as e:
            logging.warning(f"Failed to load ego_prior_video from {ego_prior_video_path}: {e}. Proceeding without ego_prior.")
            ego_prior_video = None

    if ego_prior_video is not None:
        cropped_ego_prior_video = []
        for img in ego_prior_video:
            w, h = img.size
            cropped_img = img.crop((0, 0, w, h))
            cropped_img = cropped_img.resize((448, 448))
            cropped_ego_prior_video.append(cropped_img)
        ego_prior_video = cropped_ego_prior_video

    exo_first_frame = exo_video[0] if exo_video else None
    exo_width, exo_height = exo_first_frame.size if exo_first_frame else (784, 448)
    ego_width = 448
    width = exo_width + ego_width
    height = exo_height

    video_generate = pipe(
        height=height,
        width=width,
        prompt=prompt,
        exo_video=exo_video,
        ego_prior_video=ego_prior_video, 
        num_videos_per_prompt=num_videos_per_prompt, 
        num_inference_steps=num_inference_steps,
        num_frames=num_frames, 
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
        attention_GGA=attention_GGA,
        attention_mask_GGA=attention_mask_GGA,
        point_vecs_per_frame=point_vecs_per_frame,
        cam_rays=cam_rays,
        cos_sim_scaling_factor=cos_sim_scaling_factor,
        do_kv_cache=do_kv_cache,
        mask_first_frame_ego=mask_first_frame_ego,
        ego_latents=ego_latents,
    ).frames[0]
    
    import imageio
    
    # Save the output video
    video_to_save = video_generate
    video_frames = np.clip(video_to_save * 255, 0, 255).astype(np.uint8)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    imageio.mimsave(output_path, video_frames, fps=fps, quality=9)
    print(f"Video saved to: {output_path}")

    return video_generate
