#!/usr/bin/env python3
"""
WCACLIP Data Preparation Script

This script prepares training data for WCACLIP by:
1. Creating compound-aware degradation taxonomy
2. Organizing images by clean/degraded pairs
3. Generating CSV files for training
"""

import os
import json
import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import itertools


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for WCACLIP training")
    
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing train data")
    parser.add_argument("--output_dir", type=str, default="./wcaclip_data",
                        help="Output directory for prepared data")
    parser.add_argument("--clean_dir_name", type=str, default="clear",
                        help="Name of clean/ground truth directory")
    parser.add_argument("--max_compounds", type=int, default=3,
                        help="Maximum number of degradations in compound")
    parser.add_argument("--min_images_per_degradation", type=int, default=100,
                        help="Minimum number of images per degradation type")
    
    return parser.parse_args()


def discover_degradations(data_root: Path, clean_dir_name: str):
    """Discover all degradation types from directory structure"""
    
    degradations = []
    clean_dir = data_root / clean_dir_name
    
    if not clean_dir.exists():
        raise ValueError(f"Clean directory {clean_dir} not found")
    
    # Find all other directories that are not the clean directory
    for item in data_root.iterdir():
        if item.is_dir() and item.name != clean_dir_name:
            degradations.append(item.name)
    
    print(f"Found {len(degradations)} degradation types: {degradations}")
    return degradations


def parse_compound_degradation(degradation_name: str):
    """Parse compound degradation name into components"""
    # Common separators for compound degradations
    separators = ['_', '-', '+', 'and']
    
    components = [degradation_name]  # Start with the full name
    
    # Try to split by common separators
    for sep in separators:
        if sep in degradation_name:
            parts = degradation_name.split(sep)
            # Clean up parts (remove empty strings, strip whitespace)
            parts = [part.strip() for part in parts if part.strip()]
            if len(parts) > 1:
                components = parts
                break
    
    # Filter out common words that aren't degradations
    stop_words = {'and', 'with', 'plus', 'combined'}
    components = [comp for comp in components if comp.lower() not in stop_words]
    
    return components


def create_degradation_taxonomy(degradations: list, max_compounds: int = 3):
    """Create degradation taxonomy mapping compound names to components"""
    
    taxonomy = {}
    
    # First, identify base degradations (single components)
    base_degradations = set()
    
    for deg in degradations:
        components = parse_compound_degradation(deg)
        if len(components) == 1:
            base_degradations.add(components[0])
        else:
            # For compound degradations, add all components as potential base degradations
            base_degradations.update(components)
    
    print(f"Identified {len(base_degradations)} base degradations: {sorted(base_degradations)}")
    
    # Now map each degradation to its components
    for deg in degradations:
        components = parse_compound_degradation(deg)
        
        # Validate components against known base degradations
        valid_components = []
        for comp in components:
            if comp in base_degradations:
                valid_components.append(comp)
            else:
                # Try fuzzy matching or treat as new base degradation
                base_degradations.add(comp)
                valid_components.append(comp)
        
        taxonomy[deg] = valid_components
    
    return taxonomy, sorted(base_degradations)


def count_images_per_degradation(data_root: Path, degradations: list, clean_dir_name: str):
    """Count number of images available for each degradation type"""
    
    counts = {}
    clean_dir = data_root / clean_dir_name
    
    # Get list of clean images for reference
    clean_images = set()
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
        clean_images.update(img.stem for img in clean_dir.glob(ext))
    
    print(f"Found {len(clean_images)} clean images")
    
    # Count degraded images that have corresponding clean versions
    for deg in degradations:
        deg_dir = data_root / deg
        if not deg_dir.exists():
            counts[deg] = 0
            continue
        
        deg_images = set()
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            deg_images.update(img.stem for img in deg_dir.glob(ext))
        
        # Count only images that have clean counterparts
        valid_images = deg_images.intersection(clean_images)
        counts[deg] = len(valid_images)
    
    return counts


def create_training_csv(
    data_root: Path,
    degradations: list,
    clean_dir_name: str,
    output_path: str,
    min_images_per_degradation: int = 100
):
    """Create CSV file for training with clean/degraded image pairs"""
    
    clean_dir = data_root / clean_dir_name
    training_data = []
    
    # Get clean images
    clean_images = {}
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
        for img_path in clean_dir.glob(ext):
            clean_images[img_path.stem] = img_path
    
    print(f"Processing {len(clean_images)} clean images...")
    
    # Process each degradation type
    valid_degradations = []
    
    for deg in tqdm(degradations, desc="Processing degradations"):
        deg_dir = data_root / deg
        if not deg_dir.exists():
            print(f"Warning: Degradation directory {deg_dir} not found")
            continue
        
        pairs_found = 0
        
        # Find degraded images with clean counterparts
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            for deg_img_path in deg_dir.glob(ext):
                img_stem = deg_img_path.stem
                if img_stem in clean_images:
                    clean_path = clean_images[img_stem]
                    
                    training_data.append({
                        'clean_image': str(clean_path.relative_to(data_root)),
                        'degraded_image': str(deg_img_path.relative_to(data_root)),
                        'degradation_type': deg
                    })
                    pairs_found += 1
        
        print(f"  {deg}: {pairs_found} pairs")
        
        if pairs_found >= min_images_per_degradation:
            valid_degradations.append(deg)
        else:
            print(f"  Warning: {deg} has only {pairs_found} pairs (< {min_images_per_degradation})")
    
    # Create DataFrame and save
    df = pd.DataFrame(training_data)
    df.to_csv(output_path, index=False)
    
    print(f"\nCreated training CSV with {len(training_data)} pairs")
    print(f"Valid degradation types: {len(valid_degradations)}")
    
    return valid_degradations


def create_sample_distortion_taxonomy():
    """Create a sample distortion taxonomy for testing"""
    
    sample_taxonomy = {
        # Single degradations
        "blur": ["blur"],
        "noise": ["noise"],
        "haze": ["haze"],
        "rain": ["rain"],
        "low": ["low"],
        "over": ["over"],
        "compress": ["compress"],
        "defocus": ["defocus"],
        "drops": ["drops"],
        "clouds": ["clouds"],
        "color": ["color"],
        "warp": ["warp"],
        "refract": ["refract"],
        
        # Compound degradations
        "blur_noise": ["blur", "noise"],
        "haze_rain": ["haze", "rain"],
        "low_noise": ["low", "noise"],
        "over_blur": ["over", "blur"],
        "rain_drops": ["rain", "drops"],
        "color_over_rain": ["color", "over", "rain"],
        "rain_noise_compress": ["rain", "noise", "compress"],
        "warp_noise_compress": ["warp", "noise", "compress"],
        "refract_color_over": ["refract", "color", "over"],
        "warp_blur_compress": ["warp", "blur", "compress"],
        "refract_under_compress": ["refract", "under", "compress"],
        "over_clouds_defocus": ["over", "clouds", "defocus"],
        "refract_clouds_drops": ["refract", "clouds", "drops"],
        "warp_drops_noise": ["warp", "drops", "noise"],
        "low_haze_compress": ["low", "haze", "compress"],
    }
    
    return sample_taxonomy


def main():
    args = parse_args()
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Preparing WCACLIP data from: {data_root}")
    print(f"Output directory: {output_dir}")
    
    # Discover degradation types
    try:
        degradations = discover_degradations(data_root, args.clean_dir_name)
    except ValueError as e:
        print(f"Error: {e}")
        print("Creating sample taxonomy instead...")
        taxonomy = create_sample_distortion_taxonomy()
        degradations = list(taxonomy.keys())
    else:
        # Create degradation taxonomy
        taxonomy, base_degradations = create_degradation_taxonomy(
            degradations, args.max_compounds
        )
        
        print(f"\nCreated taxonomy with {len(taxonomy)} degradation types")
        print(f"Base degradations: {base_degradations}")
    
    # Count images per degradation
    if data_root.exists():
        image_counts = count_images_per_degradation(
            data_root, degradations, args.clean_dir_name
        )
        
        print("\nImage counts per degradation:")
        for deg, count in sorted(image_counts.items()):
            print(f"  {deg}: {count}")
        
        # Create training CSV
        csv_path = output_dir / "wcaclip_train.csv"
        valid_degradations = create_training_csv(
            data_root=data_root,
            degradations=degradations,
            clean_dir_name=args.clean_dir_name,
            output_path=str(csv_path),
            min_images_per_degradation=args.min_images_per_degradation
        )
        
        # Filter taxonomy to only include valid degradations
        filtered_taxonomy = {deg: taxonomy[deg] for deg in valid_degradations if deg in taxonomy}
    else:
        print("Data root doesn't exist, using sample taxonomy")
        filtered_taxonomy = taxonomy
        csv_path = output_dir / "wcaclip_train.csv"
    
    # Save taxonomy
    taxonomy_path = output_dir / "distortion_taxonomy.json"
    with open(taxonomy_path, 'w') as f:
        json.dump(filtered_taxonomy, f, indent=2)
    
    print(f"\nSaved distortion taxonomy to: {taxonomy_path}")
    print(f"Saved training CSV to: {csv_path}")
    
    # Create training command example
    train_cmd = f"""
# Example training command:
python train_wcaclip.py \\
    --data_csv {csv_path} \\
    --distortion_taxonomy {taxonomy_path} \\
    --data_root {data_root} \\
    --output_dir ./wcaclip_results \\
    --train_batch_size 16 \\
    --learning_rate 1e-5 \\
    --num_train_epochs 10 \\
    --temperature 0.07 \\
    --quality_loss_weight 1.0
"""
    
    print(train_cmd)
    
    # Save command to file
    with open(output_dir / "train_command.sh", 'w') as f:
        f.write(train_cmd.strip())
    
    print(f"\nData preparation complete!")
    print(f"Files created:")
    print(f"  - {taxonomy_path}")
    print(f"  - {csv_path}")
    print(f"  - {output_dir / 'train_command.sh'}")


if __name__ == "__main__":
    main()