import os
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Dict, Any

from transformers import CLIPVisionModel, AutoTokenizer, CLIPImageProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler, DDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import logging

from .ImageConditioning import ImageConditioningNet
from .TextConditioning import TextConditioningNet
from utils import import_model_class_from_model_name_or_path

# Import clip_loader from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from clip_loader import load_clip_model, get_clip_model_path

logger = logging.get_logger(__name__)


class PRISM(nn.Module):
    """
    PRISM: Packaged Restoration Intelligence and Synthesis Model
    
    A comprehensive model that packages all components needed for image restoration
    including diffusion models, vision encoders, and custom restoration networks.
    """
    
    def __init__(
        self,
        pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4",
        clip_path: str = "auto",
        device: Union[str, torch.device] = "cuda",
        num_cross_proj_layers: int = 2,
        clip_v_dim: int = 1024,
        used_clip_vision_layers: int = 24,
        used_clip_vision_global: bool = False,
        down_block_types: Optional[list] = None,
        block_out_channels: Optional[list] = None,
        load_weights_from_unet: bool = False,
    ):
        super().__init__()
        
        self.device = device
        self.used_clip_vision_layers = used_clip_vision_layers
        self.used_clip_vision_global = used_clip_vision_global
        
        # Initialize base diffusion components
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder="vae", 
            revision=None
        )
        
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder="unet", 
            revision=None
        )
        
        # Text encoder (for compatibility, though mainly using CLIP vision)
        text_encoder_cls = import_model_class_from_model_name_or_path(
            pretrained_model_name_or_path, None
        )
        self.text_encoder = text_encoder_cls.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder="text_encoder", 
            revision=None
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder="tokenizer", 
            revision=None, 
            use_fast=False
        )
        
        # CLIP Vision Model - use .pt format for faster loading
        actual_clip_path = get_clip_model_path() if clip_path == "auto" else clip_path
        self.clip_vision, self.clip_image_processor = load_clip_model(actual_clip_path)
        
        # Schedulers
        self.noise_scheduler_inference = UniPCMultistepScheduler.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder="scheduler"
        )
        self.noise_scheduler_training = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder="scheduler"
        )
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_image_processor = VaeImageProcessor(
            vae_scale_factor=vae_scale_factor, 
            do_convert_rgb=True, 
            do_normalize=True
        )
        
        # Custom restoration networks
        self.text_conditioning_net = TextConditioningNet(
            num_cross_proj_layers=num_cross_proj_layers,
            clip_v_dim=clip_v_dim
        )
        
        # Image Conditioning Net - initialize with proper configuration
        if down_block_types is not None or block_out_channels is not None:
            backup_unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_name_or_path, 
                subfolder="unet", 
                revision=None
            )
            if down_block_types is not None:
                backup_unet.config.down_block_types = down_block_types
            if block_out_channels is not None:
                backup_unet.config.block_out_channels = block_out_channels
            self.image_conditioning_net = ImageConditioningNet.from_unet(backup_unet, load_weights_from_unet=load_weights_from_unet)
        else:
            self.image_conditioning_net = ImageConditioningNet.from_unet(self.unet, load_weights_from_unet=load_weights_from_unet)
        
        # Move to device
        self.to(device)
        
    def load_checkpoint(self, image_conditioning_path: str, text_conditioning_path: str):
        """Load pretrained weights for Image Conditioning and Text Conditioning networks"""
        if os.path.exists(image_conditioning_path):
            logger.info(f"Loading Image Conditioning from: {image_conditioning_path}")
            self.image_conditioning_net = ImageConditioningNet.from_pretrained(image_conditioning_path)
            self.image_conditioning_net.to(self.device)
        
        if os.path.exists(text_conditioning_path):
            logger.info(f"Loading Text Conditioning from: {text_conditioning_path}")
            try:
                self.text_conditioning_net.load_state_dict(torch.load(text_conditioning_path)['model'], strict=True)
            except:
                # Handle DataParallel case
                self.text_conditioning_net = torch.nn.DataParallel(self.text_conditioning_net)
                self.text_conditioning_net.load_state_dict(torch.load(text_conditioning_path)['model'], strict=True)

    # Backward compatibility properties and methods
    @property
    def img_net(self):
        """Backward compatibility property for SCB network"""
        return self.image_conditioning_net
    
    @img_net.setter
    def img_net(self, value):
        """Backward compatibility setter for SCB network"""
        self.image_conditioning_net = value
    
    @property
    def txt_net(self):
        """Backward compatibility property for TPB network"""
        return self.text_conditioning_net
    
    @txt_net.setter
    def txt_net(self, value):
        """Backward compatibility setter for TPB network"""
        self.text_conditioning_net = value
    
    def set_training_mode(self, training: bool = True):
        """Set training mode for trainable components"""
        if training:
            self.image_conditioning_net.train()
            self.text_conditioning_net.train()
        else:
            self.image_conditioning_net.eval()
            self.text_conditioning_net.eval()
    
    def get_trainable_parameters(self):
        """Get parameters that should be trained"""
        params = []
        params.extend(list(self.image_conditioning_net.parameters()))
        params.extend(list(self.text_conditioning_net.parameters()))
        return params
    
    def encode_image_to_latent(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to VAE latent space"""
        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents
    
    def decode_latent_to_image(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents to image"""
        with torch.no_grad():
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        return image
    
    def get_clip_visual_embeddings(self, image) -> torch.Tensor:
        """Get CLIP visual embeddings and process through TPB"""
        with torch.no_grad():
            if isinstance(image, torch.Tensor):
                # If it's already a tensor, assume it's preprocessed for CLIP
                clip_visual_input = image.to(device=self.device)
            else:
                # If it's a PIL image, preprocess it
                clip_visual_input = self.clip_image_processor(
                    images=image, return_tensors="pt"
                ).pixel_values.to(device=self.device)
            
            clip_outputs = self.clip_vision(
                clip_visual_input, 
                output_attentions=True, 
                output_hidden_states=True
            )
        
        # Process through Text Conditioning
        prompt_embeds = self.text_conditioning_net(
            clip_vision_outputs=clip_outputs,
            use_global=self.used_clip_vision_global,
            layer_ids=self.used_clip_vision_layers,
        )
        
        return prompt_embeds
    
    def prepare_conditioning_image(self, image: torch.Tensor) -> torch.Tensor:
        """Prepare conditioning image for Image Conditioning"""
        with torch.no_grad():
            image_cond = self.vae.config.scaling_factor * torch.chunk(
                self.vae.quant_conv(self.vae.encoder(image)), 2, dim=1
            )[0]
        return image_cond
    
    def forward_training(
        self, 
        pixel_values: torch.Tensor,
        conditioning_pixel_values: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        weight_dtype: torch.dtype = torch.float32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training
        
        Args:
            pixel_values: Clean target images
            conditioning_pixel_values: Degraded input images  
            timesteps: Diffusion timesteps
            noise: Random noise
            weight_dtype: Data type for computations
            
        Returns:
            model_pred: Predicted noise/velocity
            target: Ground truth target
        """
        # Encode images to latents
        latents = self.encode_image_to_latent(pixel_values.to(dtype=weight_dtype))
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler_training.add_noise(latents, noise, timesteps)
        
        # Get visual prompt guidance through TPB
        # CLIP normalization constants (hardcoded to avoid import issues)
        image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device, dtype=weight_dtype)
        image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device, dtype=weight_dtype)
        
        normalized_pixel_values = (conditioning_pixel_values.to(dtype=weight_dtype) + 1.0) / 2.0
        normalized_pixel_values = torch.nn.functional.interpolate(
            normalized_pixel_values, size=(224, 224), mode="bilinear", align_corners=False
        )
        normalized_pixel_values = (normalized_pixel_values - image_mean) / image_std
        
        clip_visual_input = self.clip_vision(
            normalized_pixel_values, 
            output_attentions=True, 
            output_hidden_states=True
        )
        visual_prompt_guidance = self.text_conditioning_net(
            clip_visual_input,
            use_global=self.used_clip_vision_global,
            layer_ids=self.used_clip_vision_layers,
        )
        
        # Get Image Conditioning
        image_cond = self.prepare_conditioning_image(conditioning_pixel_values.to(dtype=weight_dtype))
        
        # Image Conditioning forward pass
        down_block_res_samples = self.image_conditioning_net(
            noisy_latents,
            timesteps,
            encoder_hidden_states=visual_prompt_guidance,
            cond_img=image_cond,
            return_dict=False,
        )
        
        # UNet prediction
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=visual_prompt_guidance,
            down_block_additional_residuals=down_block_res_samples,
        ).sample
        
        # Get target based on prediction type
        if self.noise_scheduler_training.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler_training.config.prediction_type == "v_prediction":
            target = self.noise_scheduler_training.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler_training.config.prediction_type}")
        
        return model_pred, target
    
    def forward_inference(
        self,
        image: torch.Tensor,
        num_inference_steps: int = 20,
        time_threshold: int = 960,
        inp_of_unet_is_random_noise: bool = False,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Forward pass for inference
        
        Args:
            image: Input degraded image tensor (preprocessed)
            num_inference_steps: Number of denoising steps
            time_threshold: Threshold for noise injection
            inp_of_unet_is_random_noise: Whether to use random noise as input
            generator: Random number generator
            
        Returns:
            Restored image tensor
        """
        with torch.no_grad():
            # For inference, we need to convert tensor to a format suitable for CLIP
            # The image tensor should be preprocessed already by vae_image_processor
            
            # Convert tensor to PIL-like format for CLIP processing
            # Assuming image is normalized to [-1, 1], convert to [0, 1]
            clip_input = (image + 1.0) / 2.0
            # Resize to CLIP's expected input size (224, 224)
            clip_input = torch.nn.functional.interpolate(
                clip_input, size=(224, 224), mode="bilinear", align_corners=False
            )
            
            # Normalize for CLIP (approximating OPENAI_CLIP_MEAN and OPENAI_CLIP_STD)
            clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device)
            clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)
            clip_input = (clip_input - clip_mean) / clip_std
            
            # Get CLIP embeddings
            clip_outputs = self.clip_vision(
                clip_input, 
                output_attentions=True, 
                output_hidden_states=True
            )
            
            prompt_embeds = self.text_conditioning_net(
                clip_vision_outputs=clip_outputs,
                use_global=self.used_clip_vision_global,
                layer_ids=self.used_clip_vision_layers,
            )
            
            # Prepare conditioning
            image_cond = self.prepare_conditioning_image(image)
            b, c, h, w = image_cond.size()
            
            # Set up scheduler
            self.noise_scheduler_inference.set_timesteps(num_inference_steps, device=self.device)
            timesteps = self.noise_scheduler_inference.timesteps.long()
            
            # Initialize latents
            # Ensure generator is on the correct device if provided
            if generator is not None:
                # Create a new generator on the correct device
                device_generator = torch.Generator(device=self.device)
                device_generator.manual_seed(generator.initial_seed())
            else:
                device_generator = None
            
            if inp_of_unet_is_random_noise:
                latents = torch.randn((1, 4, h, w), generator=device_generator, device=self.device)
            else:
                noise = torch.randn((1, 4, h, w), generator=device_generator, device=self.device)
                latents = None
            
            # Denoising loop
            for i, t in enumerate(timesteps):
                # Add noise if needed
                if t >= time_threshold and not inp_of_unet_is_random_noise:
                    latents = self.noise_scheduler_inference.add_noise(image_cond, noise, t)
                elif inp_of_unet_is_random_noise and latents is None:
                    latents = torch.randn((1, 4, h, w), generator=device_generator, device=self.device)
                
                # Image Conditioning forward
                down_block_res_samples = self.image_conditioning_net(
                    latents,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cond_img=image_cond,
                    return_dict=False,
                )
                
                # UNet prediction
                noise_pred = self.unet(
                    latents,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples,
                ).sample
                
                # Update latents
                latents = self.noise_scheduler_inference.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
            
            # Decode to image
            pred = self.decode_latent_to_image(latents)
            
        return pred
    
    def save_pretrained(self, save_directory: str):
        """Save the model components"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save Image Conditioning
        image_conditioning_path = os.path.join(save_directory, "image_conditioning")
        self.image_conditioning_net.save_pretrained(image_conditioning_path)
        
        # Save Text Conditioning
        text_conditioning_path = os.path.join(save_directory, "text_conditioning.pt")
        if hasattr(self.text_conditioning_net, 'module'):  # Handle DataParallel
            torch.save({'model': self.text_conditioning_net.module.state_dict()}, text_conditioning_path)
        else:
            torch.save({'model': self.text_conditioning_net.state_dict()}, text_conditioning_path)
        
        logger.info(f"PRISM model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        img_path: Optional[str] = None,
        txt_path: Optional[str] = None,
        **kwargs
    ):
        """Load a pretrained PRISM model"""
        # Create model instance
        model = cls(**kwargs)
        
        # Load checkpoints if provided
        if img_path is None:
            img_path = os.path.join(pretrained_model_path, "scb")
        if txt_path is None:
            txt_path = os.path.join(pretrained_model_path, "tpb.pt")
            
        if os.path.exists(img_path) and os.path.exists(txt_path):
            model.load_checkpoint(img_path, txt_path)
        
        return model