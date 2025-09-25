#!/usr/bin/env python3
"""
Natural Language Demo for Image Restoration
Simple wrapper that maps natural language prompts to distortion types and calls infer.py
"""

import argparse
import subprocess
import sys
import os
from data_generation.prompts import PROMPT_TO_DISTORTION, map_prompt_to_distortion



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
    
    print("Natural Language Image Restoration Demo")
    print("=" * 50)
    
    # Map prompt to distortion type
    print(f"Input prompt: '{args.prompt}'")
    distortion_type = map_prompt_to_distortion(args.prompt)
    
    if distortion_type is None:
        print("Could not map prompt to a known distortion type.")
        print("Try prompts like:")
        print("  - 'remove clouds and brighten this aerial photo'")
        print("  - 'remove the haze from this image'") 
        print("  - 'remove blur from this photo'")
        print("  - 'brighten this dark image'")
        return 1
    
    print(f"Mapped to distortion type: {distortion_type}")
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