#!/usr/bin/env python3
"""
Natural Language Demo for Image Restoration
Simple wrapper that maps natural language prompts to distortion types and calls infer.py
"""

import argparse
import subprocess
import sys
import os
from data_generation.prompts import PROMPT_TO_DISTORTION

def map_prompt_to_distortion(prompt):
    """
    Map natural language prompt to distortion type.
    Uses simple keyword matching with priority for compound tasks.
    """
    prompt_lower = prompt.lower().strip()
    words = prompt_lower.split()
    
    # Priority order for compound tasks (check these first)
    if ("cloud" in words or "clouds" in words) and ("brighten" in words or "bright" in words or "dark" in words or "low light" in prompt_lower):
        return "cloud_low"
    
    if "low contrast" in prompt_lower or "faded" in words:
        return "low_contrast_color"
        
    if "underwater" in words and ("distortion" in words or "geometric" in words or "warp" in words):
        return "unwarp_unrefract"
        
    if ("haze" in words or "dehaze" in words) and ("snow" in words or "desnow" in words):
        return "dehaze_desnow"
    
    if ("blur" in words or "deblur" in words) and ("contrast" in words or "low" in words):
        return "deblur_contrast_low"
        
    if ("noise" in words or "denoise" in words) and ("contrast" in words or "low" in words):
        return "denoise_contrast_low"
        
    if ("superresolve" in words or "super resolution" in prompt_lower) and "noise" in words:
        return "superresolve_denoise"
    
    # Try exact phrase matches
    for keywords, distortion in PROMPT_TO_DISTORTION.items():
        if keywords in prompt_lower:
            return distortion
    
    # Individual task matching (fallback)
    if "haze" in words or "fog" in words or "dehaze" in words:
        return "dehaze_desnow"  # Use compound version if available
    elif "blur" in words or "deblur" in words:
        return "deblur_contrast_low"  # Use compound version
    elif "noise" in words or "grain" in words or "denoise" in words:
        return "denoise"
    elif "cloud" in words or "clouds" in words:
        return "decloud"
    elif "underwater" in words or "unrefract" in words:
        return "unrefract"
    elif "unwarp" in words or "warp" in words:
        return "unwarp"
    elif "defocus" in words or "focus" in words:
        return "defocus"
    elif "decolor" in words or "color" in words:
        return "decolor"
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Natural Language Image Restoration Demo")
    
    # Required arguments
    parser.add_argument("--img_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True, help="Natural language prompt describing what to fix")
    
    # Optional arguments that get passed to infer.py
    parser.add_argument("--save_root", type=str, default="demo_results", help="Directory to save results")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--unified_checkpoint_path", type=str, default="pre-trained/unified_checkpoint.pt", help="Path to unified checkpoint")
    parser.add_argument("--clip_path", type=str, default="pre-trained/clip_vision_model.pt", help="Path to CLIP model")
    
    args = parser.parse_args()
    
    print("🎯 Natural Language Image Restoration Demo")
    print("=" * 50)
    
    # Map prompt to distortion type
    print(f"📝 Input prompt: '{args.prompt}'")
    distortion_type = map_prompt_to_distortion(args.prompt)
    
    if distortion_type is None:
        print("❌ Could not map prompt to a known distortion type.")
        print("Try prompts like:")
        print("  - 'remove clouds and brighten this aerial photo'")
        print("  - 'remove the haze from this image'") 
        print("  - 'remove blur from this photo'")
        print("  - 'brighten this dark image'")
        return 1
    
    print(f"✅ Mapped to distortion type: {distortion_type}")
    print()
    
    # Build command for infer.py
    cmd = [
        "python", "infer.py",
        "--unified_checkpoint_path", args.unified_checkpoint_path,
        "--distortion_type", distortion_type,
        "--img_path", args.img_path,
        "--save_root", args.save_root,
        "--num_inference_steps", str(args.num_inference_steps),
        "--seed", str(args.seed),
        "--clip_path", args.clip_path
    ]
    
    print("🚀 Running inference with command:")
    print(" ".join(cmd))
    print()
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        print("🎉 Success! Check the output in:", args.save_root)
        
    except subprocess.CalledProcessError as e:
        print("❌ Error running inference:")
        print(e.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())