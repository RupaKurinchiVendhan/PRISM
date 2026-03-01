"""
Utility functions for loading combined CA-CLIP and classifier weights.

This module provides functions to load weights from either:
1. Combined checkpoint (ca_clip + degradation_classifier)
2. Separate checkpoint files (backward compatible)
"""

import torch
from pathlib import Path
import os


def load_combined_weights(combined_path, device='cpu'):
    """
    Load combined checkpoint containing CA-CLIP and classifier weights.
    
    Args:
        combined_path: Path to combined checkpoint file
        device: Device to load weights to
    
    Returns:
        tuple: (ca_clip_weights, classifier_weights)
    """
    if not os.path.exists(combined_path):
        raise FileNotFoundError(f"Combined checkpoint not found at: {combined_path}")
    
    print(f"Loading combined weights from: {combined_path}")
    checkpoint = torch.load(combined_path, map_location=device)
    
    # Check if this is a combined checkpoint
    if isinstance(checkpoint, dict) and 'ca_clip' in checkpoint and 'degradation_classifier' in checkpoint:
        print("✓ Loading from combined checkpoint format")
        ca_clip_weights = checkpoint['ca_clip']
        classifier_weights = checkpoint['degradation_classifier']
        
        if 'metadata' in checkpoint:
            print(f"  Version: {checkpoint['metadata'].get('combined_version', 'unknown')}")
    else:
        raise ValueError(
            f"Invalid combined checkpoint format. Expected keys 'ca_clip' and 'degradation_classifier', "
            f"but found: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}"
        )
    
    return ca_clip_weights, classifier_weights


def load_ca_clip_weights(path, device='cpu'):
    """
    Load CA-CLIP weights from either combined or separate checkpoint.
    
    Args:
        path: Path to checkpoint (combined or ca_clip.pt)
        device: Device to load weights to
    
    Returns:
        CA-CLIP weights (dict or model state)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    # If combined checkpoint, extract CA-CLIP weights
    if isinstance(checkpoint, dict) and 'ca_clip' in checkpoint:
        print(f"Loading CA-CLIP weights from combined checkpoint: {path}")
        return checkpoint['ca_clip']
    
    # Otherwise assume it's CA-CLIP weights directly
    print(f"Loading CA-CLIP weights from: {path}")
    return checkpoint


def load_classifier_weights(path, device='cpu'):
    """
    Load degradation classifier weights from either combined or separate checkpoint.
    
    Args:
        path: Path to checkpoint (combined or best_model.pt)
        device: Device to load weights to
    
    Returns:
        Classifier weights (dict or model state)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    # If combined checkpoint, extract classifier weights
    if isinstance(checkpoint, dict) and 'degradation_classifier' in checkpoint:
        print(f"Loading classifier weights from combined checkpoint: {path}")
        return checkpoint['degradation_classifier']
    
    # Otherwise assume it's classifier weights directly
    print(f"Loading classifier weights from: {path}")
    return checkpoint


def has_combined_format(path):
    """
    Check if a checkpoint file is in combined format.
    
    Args:
        path: Path to checkpoint file
    
    Returns:
        bool: True if combined format, False otherwise
    """
    if not os.path.exists(path):
        return False
    
    try:
        checkpoint = torch.load(path, map_location='cpu')
        return isinstance(checkpoint, dict) and 'ca_clip' in checkpoint and 'degradation_classifier' in checkpoint
    except:
        return False


def get_checkpoint_info(path):
    """
    Get information about a checkpoint file.
    
    Args:
        path: Path to checkpoint file
    
    Returns:
        dict: Information about the checkpoint
    """
    if not os.path.exists(path):
        return {'exists': False, 'error': 'File not found'}
    
    try:
        checkpoint = torch.load(path, map_location='cpu')
        
        info = {
            'exists': True,
            'path': path,
            'size_mb': os.path.getsize(path) / (1024**2),
        }
        
        if isinstance(checkpoint, dict):
            if 'ca_clip' in checkpoint and 'degradation_classifier' in checkpoint:
                info['format'] = 'combined'
                info['has_ca_clip'] = True
                info['has_classifier'] = True
                if 'metadata' in checkpoint:
                    info['metadata'] = checkpoint['metadata']
            elif 'model_state_dict' in checkpoint or 'state_dict' in checkpoint:
                info['format'] = 'single_model'
                info['keys'] = list(checkpoint.keys())
            else:
                info['format'] = 'unknown_dict'
                info['keys'] = list(checkpoint.keys())
        else:
            info['format'] = 'unknown'
            info['type'] = str(type(checkpoint))
        
        return info
    except Exception as e:
        return {'exists': True, 'error': str(e)}


if __name__ == "__main__":
    # Test the loading functions
    import argparse
    
    parser = argparse.ArgumentParser(description="Test weight loading utilities")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint to inspect")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Checkpoint Information")
    print("=" * 70)
    
    info = get_checkpoint_info(args.checkpoint)
    
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 70)
