import os
import random
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from itertools import combinations, permutations
import json

# Import transformations from all_transforms.py
from all_transforms import *
        
def load_depth_map(image_path, depth_dir=None):
    """
    Try to load a corresponding depth map for an image.
    Returns a default depth map if none is found.
    """
    if depth_dir is None:
        return np.ones((256, 256), dtype=np.float32) * 0.5
        
    # Try to find a matching depth map
    base_name = os.path.basename(image_path).rsplit(".", 1)[0] + "_depth.jpg"
    depth_path = os.path.join(depth_dir, base_name)
    
    if os.path.exists(depth_path):
        depth = 1.0 - np.array(Image.open(depth_path).convert('L')).astype(np.float32) / 255.0
        return depth
    else:
        return np.ones((256, 256), dtype=np.float32) * 0.5

def setup_transformations():
    """
    Define the transformations to use.
    """
    return {
        # Geometric Distortions
        "warp": ElasticDeformation(),
        "refract": Refraction(),
        "blur": MotionBlur(),

        # Photometric Distortionsc Distortions
        "low": LowLight(),
        "color": ColorJitterTransform(),
        "under": Underexposure(),
        "over": Overexposure(),

        # Occlusions
        "haze": AddHaze(),
        "rain": AddRainAndFog(),
        "snow": AddSnowAndFog(),
        "clouds": AddClouds(),
        "drops": AddRaindrops(),

        # Noise + Resolution Transformations
        "noise": GaussianNoise(),
        "compress": Pixelate(),
        "defocus": DefocusBlur()
    }
    

def get_folder_name(applied_transforms):
    """
    Generate folder name from applied transforms.
    """
    if not applied_transforms:
        return "clear"
    return "_".join(applied_transforms)

def apply_single_transformation(img, depth_map, transform_name, transformations):
    """
    Apply a single transformation to an image.
    """
    transform = transformations[transform_name]
    
    # Check if the transformation requires depth map
    if transform_name in ["haze", "rain", "snow", "blur"]:
        return transform(img, depth_map)
    else:
        return transform(img)

def generate_distortion_sequences(transform_names, max_length=3):
    """
    Generate distortion combinations (without fixed orders - order will be randomized per image).
    """
    sequences = []
    
    # Single transformations
    sequences.extend([[name] for name in transform_names])
    
    # Pairs of transformations (no specific order)
    sequences.extend(list(combinations(transform_names, 2)))
    
    # Triplets of transformations (no specific order)
    sequences.extend(list(combinations(transform_names, 3)))
    
    return sequences

def randomize_sequence_order(seq):
    """
    Randomly shuffle the order of transformations in a sequence.
    """
    seq_copy = list(seq)
    random.shuffle(seq_copy)
    return seq_copy

def generate_partial_restoration_sequences(sequences):
    """
    Generate partial restoration sequences from multi-distortion sequences.
    Returns list of tuples: (full_sequence, partial_sequence, removed_distortions)
    """
    partial_sequences = []
    
    for seq in sequences:
        if len(seq) > 1:  # Only create partial restorations for multi-distortion sequences
            # Generate all possible partial removals
            for i in range(1, len(seq)):  # Remove 1 to len-1 distortions
                for combo in combinations(range(len(seq)), i):
                    # Create partial sequence by removing selected distortions
                    partial_seq = [seq[j] for j in range(len(seq)) if j not in combo]
                    removed = [seq[j] for j in combo]
                    partial_sequences.append((seq, partial_seq, removed))
    
    return partial_sequences

def main(args):
    # Setup output directory structure
    output_base = Path(args.output_dir) / "train"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Create "clear" directory for original images
    clear_dir = output_base / "clear"
    clear_dir.mkdir(exist_ok=True)
    
    # Setup transformations
    transformations = setup_transformations()
    transform_names = list(transformations.keys())
    
    # Generate all distortion sequences (including randomized orders)
    distortion_sequences = generate_distortion_sequences(transform_names, max_length=3)
    
    # Generate partial restoration sequences
    partial_restoration_sequences = generate_partial_restoration_sequences(distortion_sequences)
    
    # Create directories for all sequences
    all_folders = set()
    
    # Regular distortion folders
    for seq in distortion_sequences:
        folder_name = get_folder_name(seq)
        all_folders.add(folder_name)
    
    # Partial restoration folders
    for full_seq, partial_seq, removed in partial_restoration_sequences:
        folder_name = get_folder_name(partial_seq) + "_partial"
        all_folders.add(folder_name)
    
    # Create all directories
    for folder_name in all_folders:
        (output_base / folder_name).mkdir(exist_ok=True)
    
    # Save metadata about sequences
    metadata = {
        "distortion_sequences": [list(seq) for seq in distortion_sequences],
        "partial_restoration_sequences": [
            {
                "full_sequence": list(full_seq),
                "partial_sequence": list(partial_seq),
                "removed_distortions": list(removed)
            }
            for full_seq, partial_seq, removed in partial_restoration_sequences
        ],
        "note": "Distortion orders are randomized per image - folder names represent distortion sets, not specific orders"
    }
    
    with open(output_base / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # List all image files in the input directory
    all_image_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                all_image_files.append(os.path.join(root, file))

    # Sample files
    num_files = min(args.num_images, len(all_image_files))
    sampled_files = random.sample(all_image_files, num_files)
    
    # Process each sampled image
    for i, image_path in enumerate(tqdm(sampled_files, desc="Processing images")):
        try:
            # Load and resize original image
            img = Image.open(image_path).convert('RGB')
            img = img.resize((256, 256), Image.BILINEAR)
            
            # Load depth map
            depth_map = load_depth_map(image_path, args.depth_dir)
            
            # Create standardized filename
            filename = f"{i+1:06d}.png"
            
            # Save original image
            img.save(clear_dir / filename)
            
            # Store intermediate results for iterative application
            intermediate_images = {"clear": img}
            
            # Process regular distortion sequences iteratively
            for seq in tqdm(distortion_sequences, desc=f"Processing distortions for image {i+1}", leave=False):
                try:
                    # Randomize the order of transformations for this image
                    randomized_seq = randomize_sequence_order(seq)
                    
                    folder_name = get_folder_name(seq)  # Folder name based on original sequence (sorted)
                    output_path = output_base / folder_name / filename
                    
                    # Skip if file already exists and not overwriting
                    if output_path.exists() and not args.overwrite:
                        if folder_name not in intermediate_images:
                            # Load existing image for potential use in compound distortions
                            intermediate_images[folder_name] = Image.open(output_path)
                        continue
                    
                    # Find the best starting point for iterative application
                    if len(randomized_seq) == 1:
                        # Single distortion: start from clear image
                        current_img = img.copy()
                        remaining_seq = randomized_seq
                    else:
                        # Multi-distortion: try to build from previous sequence
                        best_base_img = None
                        remaining_seq = randomized_seq
                        
                        # Look for existing intermediate image that matches a prefix of the randomized sequence
                        for prefix_len in range(len(randomized_seq) - 1, 0, -1):
                            prefix_seq = randomized_seq[:prefix_len]
                            # Sort prefix to match folder naming convention
                            prefix_name = get_folder_name(sorted(prefix_seq))
                            
                            if prefix_name in intermediate_images:
                                best_base_img = intermediate_images[prefix_name].copy()
                                remaining_seq = randomized_seq[prefix_len:]
                                break
                        
                        if best_base_img is None:
                            # Start from clear image if no good base found
                            current_img = img.copy()
                            remaining_seq = randomized_seq
                        else:
                            current_img = best_base_img
                    
                    # Apply remaining transformations iteratively in randomized order
                    for transform_name in remaining_seq:
                        current_img = apply_single_transformation(current_img, depth_map, transform_name, transformations)
                    
                    # Save the result
                    current_img.save(output_path)
                    intermediate_images[folder_name] = current_img.copy()
                    
                except Exception as e:
                    print(f"Error applying {seq} to {image_path}: {str(e)}")
                    continue
            
            # Process partial restoration sequences
            for full_seq, partial_seq, removed in tqdm(partial_restoration_sequences, 
                                                     desc=f"Processing partial restorations for image {i+1}", 
                                                     leave=False):
                try:
                    # The "ground truth" for partial restoration is the partially restored image
                    partial_folder_name = get_folder_name(partial_seq) + "_partial"
                    output_path = output_base / partial_folder_name / filename
                    
                    if output_path.exists() and not args.overwrite:
                        continue
                    
                    # For partial restoration, we save the partially restored version
                    if len(partial_seq) == 0:
                        # Complete restoration -> clean image
                        target_img = img.copy()
                    else:
                        # Partial restoration -> apply only remaining distortions in random order
                        randomized_partial_seq = randomize_sequence_order(partial_seq)
                        target_img = img.copy()
                        for transform_name in randomized_partial_seq:
                            target_img = apply_single_transformation(target_img, depth_map, transform_name, transformations)
                    
                    target_img.save(output_path)
                    
                except Exception as e:
                    print(f"Error processing partial restoration {full_seq}->{partial_seq}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    print(f"Processed {num_files} images with {len(distortion_sequences)} distortion sequences.")
    print(f"Generated {len(partial_restoration_sequences)} partial restoration examples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply image distortions and save results")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing images")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory for processed images")
    parser.add_argument("--depth_dir", type=str, default=None, help="Directory containing depth maps (optional)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--num_images", type=int, default=5000, help="Number of images to process")
    args = parser.parse_args()
    
    main(args)