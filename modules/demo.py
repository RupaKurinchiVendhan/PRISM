#!/usr/bin/env python3
"""
Natural Language Demo for Image Restoration
Simple wrapper that maps natural language prompts to distortion types and calls infer.py
If no prompt is given, uses degradation encoder to automatically detect distortions.
"""

import argparse
import subprocess
import sys
import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from data_generation.prompts import PROMPT_TO_DISTORTION, embed
from weight_utils import load_classifier_weights


def load_class_names(class_names_file):
    """Load class names from JSON file."""
    with open(class_names_file, 'r') as f:
        class_names = json.load(f)
        # Handle dict format
        if isinstance(class_names, dict):
            class_names = [class_names[str(i)] for i in sorted([int(k) for k in class_names.keys()])]
    return class_names


def predict_distortions_from_image(image_path, checkpoint_path, class_names_file=None):
    """
    Use the degradation encoder to predict distortions present in an image.
    Returns a predicted distortion type string.
    """
    print("No prompt provided. Using degradation encoder to detect distortions...")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the checkpoint using weight_utils (handles combined format automatically)
        checkpoint = load_classifier_weights(checkpoint_path, device=device)
        
        # Extract model architecture info
        num_classes = None
        model_type = 'resnet50'  # default
        
        if 'args' in checkpoint:
            args = checkpoint['args']
            model_type = args.get('model_type', 'resnet50')
            num_classes = args.get('num_classes', None)
        elif 'config' in checkpoint:
            config = checkpoint['config']
            model_type = config.get('model_type', 'resnet50')
            num_classes = config.get('num_classes', None)
        
        # If still None, infer from state dict
        if num_classes is None:
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            for key in state_dict.keys():
                if 'fc.weight' in key or 'classifier.weight' in key:
                    num_classes = state_dict[key].shape[0]
                    break
        
        # Create model
        if model_type.startswith('resnet'):
            from torchvision.models import resnet50, resnet101, resnet18
            if model_type == 'resnet50':
                model = resnet50(weights=None)
            elif model_type == 'resnet101':
                model = resnet101(weights=None)
            elif model_type == 'resnet18':
                model = resnet18(weights=None)
            
            # Replace final layer
            if num_classes is not None:
                in_features = model.fc.in_features
                model.fc = torch.nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()
        
        # Load class names
        class_names = None
        if class_names_file and os.path.exists(class_names_file):
            class_names = load_class_names(class_names_file)
        elif 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
        elif os.path.exists('data_generation/class_names.json'):
            class_names = load_class_names('data_generation/class_names.json')
        
        if not class_names:
            print("Warning: No class names found")
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
            print(f"Detected distortion: {predicted_distortion} (confidence: {confidence:.3f})")
            
            # Show top-3 predictions
            top_probs, top_indices = torch.topk(probs[0], min(3, len(class_names)))
            print("\nTop 3 predictions:")
            for prob, idx in zip(top_probs, top_indices):
                print(f"  {class_names[idx]}: {prob:.3f}")
        
        return predicted_distortion
        
    except Exception as e:
        print(f"Error predicting distortions: {e}")
        import traceback
        traceback.print_exc()
        return None



def main():
    parser = argparse.ArgumentParser(description="Natural Language Image Restoration Demo")
    
    # Required arguments
    parser.add_argument("--img_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, default=None, help="Natural language prompt describing what to fix (optional - will auto-detect if not provided)")
    
    # Optional arguments that get passed to infer.py
    parser.add_argument("--save_root", type=str, default="demo_results", help="Directory to save results")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prism_checkpoint_path", type=str, default="pre-trained/prism_model.pt", help="Path to unified checkpoint")
    parser.add_argument("--clip_path", type=str, default="pre-trained/ca_clip.pt", help="Path to CLIP model (can be combined weights)")
    parser.add_argument("--degradation_encoder_checkpoint", type=str, default=None, help="Path to degradation encoder checkpoint (for auto-detection, can be combined weights)")
    parser.add_argument("--class_names_file", type=str, default="data_generation/class_names.json", help="Path to class names JSON file")
    parser.add_argument("--combined_weights_path", type=str, default=None, help="Path to combined weights file (overrides clip_path and degradation_encoder_checkpoint)")
    
    args = parser.parse_args()
    
    # Handle combined weights path
    if args.combined_weights_path is not None:
        print(f"Using combined weights from: {args.combined_weights_path}")
        if os.path.exists(args.combined_weights_path):
            # Override both paths to use combined checkpoint
            args.clip_path = args.combined_weights_path
            if args.degradation_encoder_checkpoint is None:
                args.degradation_encoder_checkpoint = args.combined_weights_path
        else:
            print(f"Warning: Combined weights file not found at {args.combined_weights_path}")
    
    # Check for default combined weights if no explicit paths given
    default_combined = "pre-trained/combined_weights.pt"
    if (args.clip_path == "pre-trained/ca_clip.pt" and 
        args.degradation_encoder_checkpoint is None and
        os.path.exists(default_combined)):
        print(f"Found combined weights at {default_combined}, using it for both CA-CLIP and classifier")
        args.clip_path = default_combined
        args.degradation_encoder_checkpoint = default_combined
    
    print("Natural Language Image Restoration Demo")
    print("=" * 50)
    
    # Check if image exists
    if not os.path.exists(args.img_path):
        print(f"Error: Image not found at {args.img_path}")
        return 1
    
    distortion_type = None
    
    if args.prompt:
        # Map prompt to distortion type
        print(f"Input prompt: '{args.prompt}'")
        distortion_type = embed(args.prompt)
        
        if distortion_type is None:
            print("Could not map prompt to a known distortion type.")
            print("Try prompts like:")
            print("  - 'remove clouds and brighten this aerial photo'")
            print("  - 'remove the haze from this image'") 
            print("  - 'remove blur from this photo'")
            print("  - 'brighten this dark image'")
            return 1
    else:
        # Auto-detect distortions using degradation encoder
        # Use default checkpoint if none specified
        if args.degradation_encoder_checkpoint is None:
            args.degradation_encoder_checkpoint = "pre-trained/best_model.pt"
            print(f"Using default degradation encoder: {args.degradation_encoder_checkpoint}")
        
        distortion_type = predict_distortions_from_image(
            args.img_path,
            args.degradation_encoder_checkpoint,
            args.class_names_file
        )
        
        if distortion_type is None:
            print("Failed to detect distortions automatically.")
            print("Please provide a prompt using --prompt")
            return 1
        
        print(f"\nAuto-detected distortion type: {distortion_type}")
    
    print()
    
    # Build command for infer.py
    cmd = [
        "python", "infer.py",
        "--prism_checkpoint_path", args.prism_checkpoint_path,
        "--distortion_type", distortion_type,
        "--img_path", args.img_path,
        "--save_root", args.save_root,
        "--num_inference_steps", str(args.num_inference_steps),
        "--seed", str(args.seed),
        "--clip_path", args.clip_path
    ]
    
    print("Running inference with command:")
    print(" ".join(cmd))
    print()
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        print("Success! Check the output in:", args.save_root)
        
    except subprocess.CalledProcessError as e:
        print("Error running inference:")
        print(e.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())