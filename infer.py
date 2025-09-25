import os
import argparse
import torch

from transformers import CLIPVisionModel, AutoTokenizer, CLIPImageProcessor
from diffusers.utils import load_image

from modules import PRISM
from utils import concat_imgs
from clip_loader import load_clip_model, get_clip_model_path


def load_prism_model(unified_checkpoint_path, distortion_type, device, clip_path="auto"):
    """
    Load PRISM model from the checkpoint file.
    
    Args:
        unified_checkpoint_path: Path to the unified checkpoint file
        distortion_type: Type of distortion to load models for
        device: Device to load models on
        clip_path: Path to CLIP model
    
    Returns:
        PRISM: Loaded PRISM model
    """
    import tempfile
    import json
    
    if not os.path.exists(unified_checkpoint_path):
        raise FileNotFoundError(f"Unified checkpoint not found: {unified_checkpoint_path}")
    
    print(f"Loading unified checkpoint from: {unified_checkpoint_path}")
    unified_checkpoint = torch.load(unified_checkpoint_path, map_location='cpu')
    
    if distortion_type not in unified_checkpoint['distortion_models']:
        available_types = list(unified_checkpoint['distortion_models'].keys())
        raise ValueError(f"Distortion type '{distortion_type}' not found in unified checkpoint. "
                        f"Available types: {available_types}")
    
    # Initialize PRISM model
    prism_model = PRISM(
        pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
        clip_path=clip_path,
        device=device
    )
    
    distortion_data = unified_checkpoint['distortion_models'][distortion_type]
    
    # Load image conditioning weights
    img_config = distortion_data['scb']['config']
    img_state_dict = distortion_data['scb']['state_dict']
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create temporary config and model files for image conditioning
        config_path = os.path.join(temp_dir, "config.json")
        model_path = os.path.join(temp_dir, "diffusion_pytorch_model.bin")
        
        # Save config and state dict
        with open(config_path, 'w') as f:
            json.dump(img_config, f, indent=2)
        torch.save(img_state_dict, model_path)
        
        # Load image conditioning network
        from modules import ImageConditioningNet
        prism_model.image_conditioning_net = ImageConditioningNet.from_pretrained(temp_dir).to(device)
    
    # Load text conditioning weights
    txt_state_dict = distortion_data['tpb']
    prism_model.text_conditioning_net.load_state_dict(txt_state_dict, strict=True)
    
    print(f"✓ Successfully loaded {distortion_type} models from unified checkpoint")
    
    return prism_model


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Diff-Plugin inference script with unified checkpoint support.")

    parser.add_argument("--pretrained_model_name_or_path", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--clip_path", default="auto", help="Path to CLIP model - use 'auto' for automatic selection")
    parser.add_argument("--inp_of_crossatt", type=str, default='clip', choices=['text', 'clip'])
    parser.add_argument("--inp_of_unet_is_random_noise", action="store_true", default=False, 
                       help="only set this to True for lowlight and highlight tasks")

    # Updated checkpoint arguments
    parser.add_argument("--unified_checkpoint_path", type=str, default="pre-trained/unified_checkpoint.pt",
                       help="Path to the unified checkpoint file")
    parser.add_argument("--distortion_type", type=str, required=True,
                       choices=['cloud_low', 'deblur', 'decloud', 'dehaze', 'demoire', 'derain', 'desnow',
                                'face', 'highlight', 'lowlight', 'unrefract', 'deblur_contrast_low',
                                'decloud_low', 'decolor', 'defocus', 'denoise', 'denoise_contrast_low',
                                'low_contrast_color', 'superresolve_denoise', 'unwarp_unrefract'],
                       help="Type of distortion to process")
    
    # Backward compatibility - if ckpt_dir is provided, use the old method
    parser.add_argument("--ckpt_dir", type=str, default="", required=False,
                       help="Legacy: directory containing individual checkpoints")

    parser.add_argument("--used_clip_vision_layers", type=int, default=24)
    parser.add_argument("--used_clip_vision_global", action="store_true", default=False)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--time_threshold", type=int, default=960, 
                       help='this is used when we set the initial noise as inp+noise')
    parser.add_argument("--save_root", default="temp_results/")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_path", type=str, required=True)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    # Auto-set inp_of_unet_is_random_noise for specific distortion types
    if args.distortion_type in ['lowlight', 'highlight']:
        args.inp_of_unet_is_random_noise = True
    
    return args


if __name__ == "__main__":

    args = parse_args()

    # step-1: settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_root, exist_ok=True)

    # step-2: Load PRISM model
    print(f"Processing {args.distortion_type} distortion...")
    
    # Determine CLIP path
    clip_path = get_clip_model_path() if args.clip_path == "auto" else args.clip_path
    
    # Check if using legacy individual checkpoints or unified checkpoint
    if args.ckpt_dir:
        # Legacy mode - load individual weights into PRISM
        print("Using legacy checkpoint loading...")
        SCBNet_path = os.path.join(args.ckpt_dir, "scb") 
        TPBNet_path = os.path.join(args.ckpt_dir, "tpb.pt")
        print(f'Loading SCB from: {SCBNet_path}, TPB from: {TPBNet_path}')
        
        # Initialize PRISM model
        prism_model = PRISM(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            clip_path=clip_path,
            device=device
        )
        
        # Load individual weights
        from modules import ImageConditioningNet
        prism_model.image_conditioning_net = ImageConditioningNet.from_pretrained(SCBNet_path).to(device)
        
        txt_state_dict = torch.load(TPBNet_path, map_location=device)
        if 'model' in txt_state_dict:
            txt_state_dict = txt_state_dict['model']
        prism_model.text_conditioning_net.load_state_dict(txt_state_dict, strict=True)
        
    else:
        # New unified checkpoint mode
        print("Using unified checkpoint loading...")
        prism_model = load_prism_model(
            args.unified_checkpoint_path, 
            args.distortion_type, 
            device,
            clip_path=clip_path
        )
    
    prism_model.eval()

    # Step-3: Run PRISM inference
    image = load_image(args.img_path)
    pil_image = image.copy()
    
    print("Running PRISM inference...")
    with torch.no_grad():
        # Set PRISM parameters for inference
        prism_model.used_clip_vision_global = args.used_clip_vision_global
        prism_model.used_clip_vision_layers = args.used_clip_vision_layers
        
        # Preprocess image for PRISM
        width, height = image.size
        if width < 512 or height < 512:
            if width < height:
                new_width = 512
                new_height = int((512 / width) * height)
            else:
                new_height = 512
                new_width = int((512 / height) * width)
            image = image.resize((new_width, new_height))
        
        # Preprocess image to tensor format expected by PRISM
        processed_image = prism_model.vae_image_processor.preprocess(
            image, height=image.size[1], width=image.size[0]
        ).to(device=prism_model.device)
        
        # Create generator for reproducible results
        generator = torch.Generator(device=prism_model.device)
        generator.manual_seed(args.seed)
        
        # Use PRISM's unified forward_inference method
        pred_tensor = prism_model.forward_inference(
            image=processed_image,
            num_inference_steps=args.num_inference_steps,
            time_threshold=args.time_threshold,
            inp_of_unet_is_random_noise=args.inp_of_unet_is_random_noise,
            generator=generator
        )
        
        # Post-process the result
        pred = prism_model.vae_image_processor.postprocess(pred_tensor, output_type='pil')[0]
    
    # Save result
    save_ = concat_imgs([pil_image.resize(pred.size), pred], target_size=pred.size, target_dim=1)
    output_filename = f"{args.distortion_type}_{os.path.basename(args.img_path)}"
    output_path = os.path.join(args.save_root, output_filename)
    save_.save(output_path)
    
    print(f'Processing complete!')
    print(f'  - Distortion type: {args.distortion_type}')
    print(f'  - Input: {args.img_path}')
    print(f'  - Output: {output_path}')