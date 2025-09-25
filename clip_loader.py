"""
CLIP Model Loading Utilities
Handles loading CLIP models from both HuggingFace format and .pt files
"""

import torch
import os
from transformers import CLIPVisionModel, CLIPVisionConfig, CLIPImageProcessor

def load_clip_model(model_path, device='cpu'):
    """
    Universal CLIP model loader - handles both HuggingFace and .pt formats
    
    Args:
        model_path: Path to either HuggingFace model directory or .pt file
        device: Device to load the model on
        
    Returns:
        tuple: (clip_vision_model, clip_processor)
    """
    if model_path.endswith('.pt'):
        return load_clip_from_pt(model_path, device)
    else:
        return load_clip_from_hf(model_path, device)

def load_clip_from_pt(pt_path, device='cpu'):
    """
    Load CLIP model from .pt file
    
    Args:
        pt_path: Path to the .pt file
        device: Device to load the model on
        
    Returns:
        tuple: (clip_vision_model, clip_processor)
    """
    print(f"Loading CLIP model from .pt file: {pt_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(pt_path, map_location=device)
        
        # Create model from config
        config = CLIPVisionConfig.from_dict(checkpoint['config'])
        clip_vision = CLIPVisionModel(config)
        
        # Load state dict
        clip_vision.load_state_dict(checkpoint['model_state_dict'])
        clip_vision.to(device)
        
        # Create processor (use default if no config saved)
        if checkpoint.get('processor_config'):
            # Try to recreate processor from config
            try:
                clip_processor = CLIPImageProcessor(**checkpoint['processor_config'])
            except:
                # Fallback to default processor if recreation fails
                clip_processor = CLIPImageProcessor()
        else:
            # Fallback to default processor
            clip_processor = CLIPImageProcessor()
        
        print("Successfully loaded CLIP model from .pt file")
        return clip_vision, clip_processor
        
    except Exception as e:
        print(f"Error loading model from .pt file: {str(e)}")
        print("💡 Falling back to HuggingFace format...")
        # Try to find a HuggingFace format fallback
        hf_path = pt_path.replace('.pt', '').replace('_model', '')
        if os.path.exists(hf_path):
            return load_clip_from_hf(hf_path, device)
        raise

def load_clip_from_hf(hf_path, device='cpu'):
    """
    Load CLIP model from HuggingFace format
    
    Args:
        hf_path: Path to HuggingFace model directory
        device: Device to load the model on
        
    Returns:
        tuple: (clip_vision_model, clip_processor)
    """
    print(f"Loading CLIP model from HuggingFace format: {hf_path}")
    
    try:
        clip_vision = CLIPVisionModel.from_pretrained(hf_path).to(device)
        clip_processor = CLIPImageProcessor.from_pretrained(hf_path)
        
        print("Successfully loaded CLIP model from HuggingFace format")
        return clip_vision, clip_processor
        
    except Exception as e:
        print(f"Error loading model from HuggingFace format: {str(e)}")
        raise

def get_clip_model_path():
    """
    Get the default CLIP model path (prefer .pt file if available)
    
    Returns:
        str: Path to the CLIP model
    """
    # Check for .pt file first (faster loading)
    pt_path = "pre-trained/clip_vision_model.pt"
    if os.path.exists(pt_path):
        return pt_path
    
    # Fallback to HuggingFace format
    hf_paths = [
        "openai/clip-vit-large-patch14"  # Online fallback
    ]
    
    for path in hf_paths:
        if os.path.exists(path):
            return path
    
    # Final fallback to online model
    return "openai/clip-vit-large-patch14"