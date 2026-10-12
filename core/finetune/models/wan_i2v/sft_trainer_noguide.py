from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanImageToVideoPipeline,
)
from core.finetune.models.wan_i2v.custom_transformer import WanTransformer3DModel_GGA as WanTransformer3DModel
from diffusers.utils import (
    replace_example_docstring,
    logging,
    is_torch_xla_available,
)
from diffusers.image_processor import PipelineImageInput
from diffusers.callbacks import PipelineCallback, MultiPipelineCallbacks
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
import PIL

# Constants and conditional imports
logger = logging.get_logger(__name__)

# XLA support (conditional)
try:
    XLA_AVAILABLE = is_torch_xla_available()
    if XLA_AVAILABLE:
        import torch_xla.core.xla_model as xm  # type: ignore
    else:
        xm = None  # type: ignore
except ImportError:
    XLA_AVAILABLE = False
    xm = None  # type: ignore

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import WanImageToVideoPipeline

        >>> pipe = WanImageToVideoPipeline.from_pretrained("path/to/model")
        >>> images = load_images("path/to/images/")  # Load all frames
        >>> video = pipe(images=images, prompt="A beautiful sunset").frames[0]
        ```
"""

from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
import numpy as np
from numpy import dtype
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel
from typing_extensions import override

from core.finetune.schemas import Wan_Components as Components
from core.finetune.trainer import Trainer
from core.finetune.utils import unwrap_model

from ..utils import register

from diffusers import (
    WanImageToVideoPipeline,
)
from diffusers.utils.torch_utils import randn_tensor


class WanWidthConcatImageToVideoPipelineNoGuide(WanImageToVideoPipeline):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        image_encoder: CLIPVisionModel,
        image_processor: CLIPImageProcessor,
        transformer: WanTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__(tokenizer, text_encoder, image_encoder, image_processor, transformer, vae, scheduler)
        self._shared_latent_exo_randn = None

    @override
    def prepare_latents(
        self,
        ego_prior_video: Optional[torch.Tensor] = None,  # ego prior video for latent condition (optional)
        batch_size: int = 1,
        num_channels_latents: int = 16,
        height: int = 480,
        exo_width: int = 784,
        ego_width: int = 448, 
        num_frames: int = 49,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        last_images: Optional[torch.Tensor] = None,
        mask_first_frame_ego: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        # Calculate latent dimensions for each view type
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        exo_latent_width = exo_width // self.vae_scale_factor_spatial  
        ego_latent_width = ego_width // self.vae_scale_factor_spatial
        total_latent_width = exo_latent_width + ego_latent_width

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, total_latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # Use shared IC-LoRA latent if available
        if self._shared_latent_exo_randn is not None:
            latents = self._shared_latent_exo_randn.to(device=device, dtype=dtype)
            print(f"[NoGuide] Using shared IC-LoRA latent: latents_shape={latents.shape}")
        elif latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # Process ego prior video if provided, else use default shared latent
        video_condition = self._shared_latent_exo_randn.clone()
        latent_total_width = self._shared_latent_exo_randn.shape[-1]
        exo_latent_width = latent_total_width - ego_latent_width

        ego_prior_latents = None
        if ego_prior_video is not None:
            if ego_prior_video.dim() == 4:
                ego_prior_video = ego_prior_video.unsqueeze(0)  # Add batch dimension
            
            ego_prior_latents = self.encode_video(ego_prior_video.to(device=device, dtype=self.vae.dtype))
            video_condition[:, :, :, :, exo_latent_width:] = ego_prior_latents

        latent_condition = video_condition.to(dtype)

        # Create mask for exo and ego views
        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, total_latent_width)
        
        # Set exo_view mask (left part) to 1.0 (condition)
        mask_lat_size[:, :, :, :, :exo_latent_width] = 1.0
        # Set ego_view mask (right part) to 0.0 (target to generate)
        mask_lat_size[:, :, :, :, exo_latent_width:] = 0.0

        if mask_first_frame_ego and ego_prior_latents is not None:
            # Set first frame of ego_view to 1.0 if ego prior is provided
            mask_lat_size[:, :, 0, :, exo_latent_width:] = 1.0
            latents[:, :, 0, :, exo_latent_width:] = ego_prior_latents[:, :, 0, :, :].to(dtype=dtype)

        if last_images is None:
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
        else:
            mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal) # 1 -> 4
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2) # 48 -> 48 + 4
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, total_latent_width) # 1*52 -> 13*4
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        return latents, torch.concat([mask_lat_size, latent_condition], dim=1), (exo_latent_width, ego_latent_width)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video to latent space"""
        vae = self.vae
        video = video.to(vae.device, dtype=vae.dtype)
        
        if video.dim() == 5 and video.shape[1] != 3:
            video = video.permute(0, 2, 1, 3, 4)
        
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample()
        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(latent.device, latent.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
            latent.device, latent.dtype
        )
        latent = (latent - latents_mean) * latents_std
        return latent

    def _setup_ic_lora_latent(self, exo_video, generator, device, height, width, 
                            num_frames, batch_size=1, num_videos_per_prompt=1,
                            ego_latents: Optional[torch.Tensor] = None):
        """Setup shared latent for IC-LoRA (exo + random ego)"""
        if exo_video is None:
            return
            
        latent_height = height // self.vae_scale_factor_spatial
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        num_channels_latents = self.vae.config.z_dim
        
        ego_pixel_width = 448
        ego_latent_width = ego_pixel_width // self.vae_scale_factor_spatial
        
        batch_size = batch_size * num_videos_per_prompt
        
        if exo_video.dim() == 4:
            exo_video = exo_video.unsqueeze(0)
        exo_latents = self.encode_video(exo_video)
        exo_latents = exo_latents.repeat(batch_size, 1, 1, 1, 1)
        
        if ego_latents is None:
            ego_shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, ego_latent_width)
            ego_latents = torch.randn(ego_shape, generator=generator, dtype=exo_latents.dtype).to(device)
        else:
            ego_latents = ego_latents.to(device=device, dtype=exo_latents.dtype)
            ego_shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, ego_latent_width)
            assert ego_latents.shape == ego_shape, f"ego_latents shape {ego_latents.shape} does not match expected shape {ego_shape}"
        
        self._shared_latent_exo_randn = torch.cat([exo_latents, ego_latents], dim=-1)
        print(f"[NoGuide] Setup IC-LoRA latent: exo_shape={exo_latents.shape}, ego_shape={ego_latents.shape}, total_shape={self._shared_latent_exo_randn.shape}")

    @override
    def encode_image(
        self,
        exo_video: PipelineImageInput,
        device: Optional[torch.device] = None,
    ):
        """
        Override encode_image to use shared latent for image embedding
        Input: exo video only
        Return: CLIPEncode(decode(encode(exo video first frame)+randn_tensor))
        """
        device = device or self._execution_device
        
        # Use full latent for temporal consistency (not just first frame)
        full_latent = self._shared_latent_exo_randn  # [B, C, F, H, W]
        
        # Decode latent back to pixel space
        # Remove VAE normalization first
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(full_latent.device, full_latent.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            full_latent.device, full_latent.dtype
        )
        
        # Denormalize
        denorm_latent = full_latent / latents_std + latents_mean
        
        # Decode full video for temporal consistency, then extract first frame
        decoded_video = self.vae.decode(denorm_latent.to(self.vae.dtype), return_dict=False)[0]
        decoded_image = decoded_video[:, :, 0, :, :]  # Extract first frame [B, C, H, W]
        
        print(f"Decoded image shape: {decoded_image.shape} for CLIP encoding")
        
        # Convert to float32 for CLIP compatibility (BFloat16 not supported by transformers image processor)
        decoded_image = decoded_image.to(dtype=torch.float32)
        
        # Convert from [-1, 1] to [0, 1] range for CLIP processor and clamp values to [0, 1]
        decoded_image = (decoded_image + 1.0) / 2.0
        decoded_image = torch.clamp(decoded_image, 0.0, 1.0)
        
        # Use decoded image for CLIP encoding - processor handles all normalization
        image = self.image_processor(images=decoded_image, return_tensors="pt", do_rescale=False).to(device)
        image_embeds = self.image_encoder(**image, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    @override
    def check_inputs(
        self,
        prompt,
        negative_prompt,
        exo_video,
        ego_prior_video=None,
        height=448,
        width=1232,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if exo_video is None:
            raise ValueError("Provide `exo_video`. Cannot leave `exo_video` undefined.")
        if exo_video is not None and not isinstance(exo_video, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(f"`exo_video` has to be of type `torch.Tensor`, `PIL.Image.Image`, or `list` but is {type(exo_video)}")
        if ego_prior_video is not None and not isinstance(ego_prior_video, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(f"`ego_prior_video` has to be of type `torch.Tensor`, `PIL.Image.Image`, or `list` but is {type(ego_prior_video)}")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

    @override
    @torch.no_grad()
    def __call__(
        self,
        exo_video: PipelineImageInput,
        ego_prior_video: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 448,
        width: int = 784 + 448,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        attention_GGA: Optional[torch.Tensor] = None,
        attention_mask_GGA: Optional[torch.Tensor] = None,
        cam_rays: Optional[torch.Tensor] = None,
        point_vecs_per_frame: Optional[torch.Tensor] = None,
        cos_sim_scaling_factor: float = 1.0,
        do_kv_cache: bool = True,
        do_custom_test: bool = False,
        save_attn_dir_name: Optional[str] = None,
        mask_first_frame_ego: bool = False,
        ego_latents: Optional[torch.Tensor] = None,
    ):
        self.check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            exo_video=exo_video,
            ego_prior_video=ego_prior_video,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image_embeds=image_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._current_step = 0

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # Encode text prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # Encode image embedding
        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # Preprocess exo video
        exo_video = self.video_processor.preprocess_video(exo_video, height=height, width=784)
        if ego_prior_video is not None:
            ego_prior_video = self.video_processor.preprocess_video(ego_prior_video, height=height, width=448)

        # Setup IC-LoRA latent
        self._setup_ic_lora_latent(
            exo_video, generator, device, height, width, num_frames, 
            batch_size, num_videos_per_prompt, ego_latents=ego_latents
        )

        # Encode image
        if image_embeds is None:
            image_embeds = self.encode_image(exo_video, device)
        image_embeds = image_embeds.repeat(batch_size, 1, 1)
        image_embeds = image_embeds.to(transformer_dtype)

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latents
        num_channels_latents = self.vae.config.z_dim
        latents, condition, _ = self.prepare_latents(
            ego_prior_video=ego_prior_video,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            exo_width=784,
            ego_width=448,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
            mask_first_frame_ego=mask_first_frame_ego,
        )

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self._current_step = i
                latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    return_dict=False,
                    attention_GGA=attention_GGA,
                    attention_mask_GGA=attention_mask_GGA,
                    cam_rays=cam_rays,
                    point_vecs_per_frame=point_vecs_per_frame,
                    cos_sim_scaling_factor=cos_sim_scaling_factor,
                    do_kv_cache=do_kv_cache,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        return_dict=False,
                        attention_GGA=attention_GGA,
                        attention_mask_GGA=attention_mask_GGA,
                        cam_rays=cam_rays,
                        point_vecs_per_frame=point_vecs_per_frame,
                        cos_sim_scaling_factor=cos_sim_scaling_factor,
                        do_kv_cache=do_kv_cache,
                    )[0]
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                # Set exo region (left part) of noise_pred to 0 to prevent exo latents from being updated
                ego_latent_width = 448 // self.vae_scale_factor_spatial
                noise_pred[:, :, :, :, :-ego_latent_width] = 0.0

                if mask_first_frame_ego:
                    # Set the noise prediction of the first frame of the ego region to 0.0
                    noise_pred[:, :, 0, :, -ego_latent_width:] = 0.0

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # Decode latents
        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        self._shared_latent_exo_randn = None
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)

