import os
import argparse
import torch

from transformers import CLIPVisionModel, AutoTokenizer, CLIPImageProcessor
from diffusers.utils import load_image

from modules import PRISM
from modules.utils import concat_imgs
from clip_loader import load_clip_model, get_clip_model_path
from data_generation.prompts import embed


def predict_distortion_from_image(image_path, checkpoint_path, class_names_file):
    """
    Use degradation encoder to predict distortion type from image.
    
    Args:
        image_path: Path to the image
        checkpoint_path: Path to degradation encoder checkpoint
        class_names_file: Path to class names JSON
        
    Returns:
        str: Predicted distortion type or None
    """
    import json
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms
    from weight_utils import load_classifier_weights
    
    print("Auto-detecting distortion type from image...")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the checkpoint
        checkpoint = load_classifier_weights(checkpoint_path, device=device)
        
        # Extract model info
        num_classes = None
        model_type = 'resnet50'
        
        if 'args' in checkpoint:
            args = checkpoint['args']
            model_type = args.get('model_type', 'resnet50')
            num_classes = args.get('num_classes', None)
        elif 'config' in checkpoint:
            config = checkpoint['config']
            model_type = config.get('model_type', 'resnet50')
            num_classes = config.get('num_classes', None)
        
        if num_classes is None:
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            for key in state_dict.keys():
                if 'fc.weight' in key or 'classifier.weight' in key:
                    num_classes = state_dict[key].shape[0]
                    break
        
        # Create model
        from torchvision.models import resnet50, resnet101, resnet18
        if model_type == 'resnet50':
            model = resnet50(weights=None)
        elif model_type == 'resnet101':
            model = resnet101(weights=None)
        elif model_type == 'resnet18':
            model = resnet18(weights=None)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if num_classes is not None:
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, num_classes)
        
        # Load weights
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()
        
        # Load class names
        class_names = None
        if class_names_file and os.path.exists(class_names_file):
            with open(class_names_file, 'r') as f:
                class_names = json.load(f)
                if isinstance(class_names, dict):
                    class_names = [class_names[str(i)] for i in sorted([int(k) for k in class_names.keys()])]
        elif 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
        
        if not class_names:
            # print("Warning: No class names found")
            return None
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        predicted_distortion = class_names[pred_class] if pred_class < len(class_names) else None
        
        if predicted_distortion:
            print(f"Detected distortion: {predicted_distortion}")
        
        return predicted_distortion
        
    except Exception as e:
        print(f"Error in auto-detection: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_prism_model(prism_checkpoint_path, distortion_type, device, clip_path="auto"):
    """
    Load PRISM model from the checkpoint file.
    
    Args:
        prism_checkpoint_path: Path to the PRISM weights file
        distortion_type: Type of distortion to load models for
        device: Device to load models on
        clip_path: Path to CLIP model
    
    Returns:
        PRISM: Loaded PRISM model
    """
    import tempfile
    import json
    
    if not os.path.exists(prism_checkpoint_path):
        raise FileNotFoundError(f"PRISM weights not found: {prism_checkpoint_path}")
    
    print(f"Loading PRISM weights from: {prism_checkpoint_path}")
    unified_checkpoint = torch.load(prism_checkpoint_path, map_location='cpu')
    
    if distortion_type not in unified_checkpoint['distortion_models']:
        available_types = list(unified_checkpoint['distortion_models'].keys())
        raise ValueError(f"Distortion type '{distortion_type}' not found in PRISM weights. "
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
    
    return prism_model


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Diff-Plugin inference script with PRISM weights support.")

    parser.add_argument("--pretrained_model_name_or_path", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--clip_path", default="auto", help="Path to CLIP model - use 'auto' for automatic selection")
    parser.add_argument("--inp_of_crossatt", type=str, default='clip', choices=['text', 'clip'])
    parser.add_argument("--inp_of_unet_is_random_noise", action="store_true", default=False, 
                       help="only set this to True for lowlight and highlight tasks")

    # Updated checkpoint arguments
    parser.add_argument("--prism_checkpoint_path", type=str, default="pre-trained/prism_model.pt",
                       help="Path to the PRISM weights file")
    parser.add_argument("--distortion_type", type=str, default=None,
                       help="Type of distortion to process (if not provided, will use prompt or auto-detect)")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Natural language prompt describing the distortion (alternative to --distortion_type)")
    parser.add_argument("--degradation_encoder_checkpoint", type=str, default=None,
                       help="Path to degradation encoder for auto-detection (used when prompt mapping fails)")
    parser.add_argument("--class_names_file", type=str, default="data_generation/class_names.json",
                       help="Path to class names JSON file for auto-detection")
    
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
    parser.add_argument("--save_comparison", action="store_true", default=False,
                       help="Save side-by-side comparison (original | restored)")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_path", type=str, required=True)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    return args


if __name__ == "__main__":

    args = parse_args()

    # step-1: settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_root, exist_ok=True)
    
    # Determine distortion type from prompt, explicit type, or auto-detection
    distortion_type = args.distortion_type
    
    if distortion_type is None:
        if args.prompt:
            # Try to map prompt to distortion type
            print(f"Input prompt: '{args.prompt}'")
            distortion_type = embed(args.prompt)
            
            if distortion_type is None:
                
                # Fall back to auto-detection
                if args.degradation_encoder_checkpoint is None:
                    args.degradation_encoder_checkpoint = "pre-trained/best_model.pt"
                
                if os.path.exists(args.degradation_encoder_checkpoint):
                    distortion_type = predict_distortion_from_image(
                        args.img_path,
                        args.degradation_encoder_checkpoint,
                        args.class_names_file
                    )
                
                if distortion_type is None:
                    print("\nError: Could not determine distortion type.")
                    print("Please either:")
                    print("  1. Provide --distortion_type explicitly")
                    print("  2. Use a supported prompt phrase")
                    exit(1)
            # else:
                # print(f"Mapped prompt to distortion type: {distortion_type}")
        else:
            # No prompt and no explicit type - try auto-detection
            if args.degradation_encoder_checkpoint is None:
                args.degradation_encoder_checkpoint = "pre-trained/best_model.pt"
            
            if os.path.exists(args.degradation_encoder_checkpoint):
                distortion_type = predict_distortion_from_image(
                    args.img_path,
                    args.degradation_encoder_checkpoint,
                    args.class_names_file
                )
            
            if distortion_type is None:
                print("\nError: No distortion type provided and auto-detection failed.")
                print("Please provide either --distortion_type or --prompt")
                exit(1)
    
    print(f"\nProcessing {distortion_type} distortion...")
    
    # Auto-set inp_of_unet_is_random_noise for specific distortion types
    if distortion_type in ['lowlight', 'highlight']:
        args.inp_of_unet_is_random_noise = True

    # step-2: Load PRISM model
    
    # Determine CLIP path
    clip_path = get_clip_model_path() if args.clip_path == "auto" else args.clip_path
    
    # Check if using legacy individual checkpoints or PRISM weights
    if args.ckpt_dir:
        # Legacy mode - load individual weights into PRISM
        print("Loading...")
        SCBNet_path = os.path.join(args.ckpt_dir, "scb") 
        TPBNet_path = os.path.join(args.ckpt_dir, "tpb.pt")
        
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
        prism_model = load_prism_model(
            args.prism_checkpoint_path, 
            distortion_type, 
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
    output_filename = f"{distortion_type}_{os.path.basename(args.img_path)}"
    output_path = os.path.join(args.save_root, output_filename)
    
    if args.save_comparison:
        # Save side-by-side comparison
        save_ = concat_imgs([pil_image.resize(pred.size), pred], target_size=pred.size, target_dim=1)
        save_.save(output_path)
    else:
        # Save only the restored image
        pred.save(output_path)
    
    print(f'Processing complete!')