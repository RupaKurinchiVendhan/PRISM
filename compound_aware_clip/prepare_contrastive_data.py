"""
Data Preparation Script for CLIP Contrastive Training

This script helps you create the required CSV file and setup data structure
for CLIP contrastive fine-tuning from your existing image restoration datasets.
"""

import os
import pandas as pd
import argparse
from pathlib import Path
import json
from tqdm import tqdm


def create_contrastive_dataset(clean_dir, degraded_dirs, restored_dirs, output_csv, data_root):
    """
    Create contrastive training dataset CSV from organized image directories.
    
    Args:
        clean_dir: Directory containing clean/ground truth images
        degraded_dirs: Dictionary mapping distortion types to their directories
        restored_dirs: Dictionary mapping distortion types to their restored image directories
        output_csv: Output CSV file path
        data_root: Root directory to make paths relative to
    """
    
    data_rows = []
    data_root = Path(data_root)
    
    # Get all clean images
    clean_path = Path(clean_dir)
    clean_images = list(clean_path.glob("*.jpg")) + list(clean_path.glob("*.png"))
    
    print(f"Found {len(clean_images)} clean images")
    
    for clean_img in tqdm(clean_images, desc="Processing images"):
        clean_img_name = clean_img.stem  # filename without extension
        
        # Make clean path relative to data_root
        try:
            clean_rel_path = str(clean_img.relative_to(data_root))
        except ValueError:
            # If not relative, use absolute path
            clean_rel_path = str(clean_img)
        
        # Find corresponding degraded and restored images
        for distortion_type, degraded_dir in degraded_dirs.items():
            degraded_path = Path(degraded_dir)
            
            # Look for matching degraded image
            degraded_candidates = (
                list(degraded_path.glob(f"{clean_img_name}.*")) +
                list(degraded_path.glob(f"*{clean_img_name}*")) +
                list(degraded_path.glob(f"{clean_img_name}_*")) +
                list(degraded_path.glob(f"*_{clean_img_name}.*"))
            )
            
            if not degraded_candidates:
                continue
                
            degraded_img = degraded_candidates[0]  # Take first match
            
            # Make degraded path relative
            try:
                degraded_rel_path = str(degraded_img.relative_to(data_root))
            except ValueError:
                degraded_rel_path = str(degraded_img)
            
            # Find corresponding restored image
            restored_rel_path = degraded_rel_path  # Default to degraded if no restored version
            
            if distortion_type in restored_dirs:
                restored_path = Path(restored_dirs[distortion_type])
                restored_candidates = (
                    list(restored_path.glob(f"{clean_img_name}.*")) +
                    list(restored_path.glob(f"*{clean_img_name}*")) +
                    list(restored_path.glob(f"{clean_img_name}_*")) +
                    list(restored_path.glob(f"*_{clean_img_name}.*"))
                )
                
                if restored_candidates:
                    restored_img = restored_candidates[0]
                    try:
                        restored_rel_path = str(restored_img.relative_to(data_root))
                    except ValueError:
                        restored_rel_path = str(restored_img)
            
            # Add to dataset
            data_rows.append({
                'clean_path': clean_rel_path,
                'degraded_path': degraded_rel_path,
                'restored_path': restored_rel_path,
                'distortion_type': distortion_type
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(data_rows)
    df.to_csv(output_csv, index=False)
    
    print(f"Created dataset with {len(df)} samples")
    print(f"Distortion type distribution:")
    print(df['distortion_type'].value_counts())
    print(f"Saved to: {output_csv}")
    
    return df


def create_sample_distortion_taxonomy():
    """Create a sample distortion taxonomy file."""
    taxonomy = {
        "haze": ["atmospheric_scattering", "reduced_visibility"],
        "dehaze": ["atmospheric_scattering", "reduced_visibility"],
        "rain": ["water_droplets", "streak_artifacts"],
        "derain": ["water_droplets", "streak_artifacts"],
        "snow": ["crystalline_particles", "white_noise"],
        "desnow": ["crystalline_particles", "white_noise"],
        "blur": ["motion_blur", "defocus"],
        "deblur": ["motion_blur", "defocus"],
        "lowlight": ["insufficient_illumination", "noise_amplification"],
        "highlight": ["overexposure", "saturation_clipping"],
        "moire": ["aliasing_patterns", "interference"],
        "demoire": ["aliasing_patterns", "interference"],
        "face": ["facial_artifacts", "identity_preservation"],
        
        # Compositional distortions
        "haze_rain": ["atmospheric_scattering", "reduced_visibility", "water_droplets", "streak_artifacts"],
        "haze_snow": ["atmospheric_scattering", "reduced_visibility", "crystalline_particles", "white_noise"],
        "rain_blur": ["water_droplets", "streak_artifacts", "motion_blur", "defocus"],
        "snow_blur": ["crystalline_particles", "white_noise", "motion_blur", "defocus"],
        "lowlight_noise": ["insufficient_illumination", "noise_amplification", "sensor_noise"],
        "highlight_blur": ["overexposure", "saturation_clipping", "motion_blur", "defocus"]
    }
    return taxonomy


def main():
    parser = argparse.ArgumentParser(description="Prepare data for CLIP contrastive training")
    parser.add_argument("--clean_dir", type=str, required=True,
                       help="Directory containing clean/ground truth images")
    parser.add_argument("--degraded_config", type=str, required=True,
                       help="JSON file mapping distortion types to degraded image directories")
    parser.add_argument("--restored_config", type=str, default=None,
                       help="JSON file mapping distortion types to restored image directories")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory to make paths relative to")
    parser.add_argument("--output_csv", type=str, required=True,
                       help="Output CSV file path")
    parser.add_argument("--taxonomy_output", type=str, default=None,
                       help="Output path for distortion taxonomy JSON")
    
    args = parser.parse_args()
    
    # Load degraded directories config
    with open(args.degraded_config, 'r') as f:
        degraded_dirs = json.load(f)
    
    # Load restored directories config (optional)
    restored_dirs = {}
    if args.restored_config and os.path.exists(args.restored_config):
        with open(args.restored_config, 'r') as f:
            restored_dirs = json.load(f)
    
    print("Degraded directories:", degraded_dirs)
    print("Restored directories:", restored_dirs)
    
    # Create contrastive dataset
    df = create_contrastive_dataset(
        args.clean_dir, degraded_dirs, restored_dirs, 
        args.output_csv, args.data_root
    )
    
    # Create taxonomy file if requested
    if args.taxonomy_output:
        taxonomy = create_sample_distortion_taxonomy()
        with open(args.taxonomy_output, 'w') as f:
            json.dump(taxonomy, f, indent=2)
        print(f"Created sample distortion taxonomy: {args.taxonomy_output}")
        print("Please review and modify the taxonomy to match your specific distortions")


if __name__ == "__main__":
    main()