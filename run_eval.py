#!/usr/bin/env python3
"""
Run inference on all degradation types from the benchmark dataset.

This script:
1. Finds all degradation folders in the input directory
2. Uses the unified model checkpoint (pre-trained/prism_model.pt)
3. Runs inference on all images for each degradation type
4. Computes PSNR, SSIM, LPIPS, and FID metrics
"""


import torch
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import lpips
from cleanfid.fid import compute_fid

import os
import sys
import glob
import json
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
from numpy import *
from PIL import Image
from tqdm import tqdm
from modules.utils import *

def get_degradation_folders(input_dir):
    """
    Get list of degradation folders in the input directory.
    
    Returns:
        list of tuples: [(degradation_name, input_path), ...]
    """
    degradation_folders = []
    for item in sorted(os.listdir(input_dir)):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            degradation_folders.append((item, item_path))
    
    return degradation_folders


def get_image_files(folder):
    """Get all image files from a folder."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(files)


def run_inference_for_degradation(degradation, input_folder, prism_checkpoint, output_folder, args):
    """Run inference for a single degradation type."""
    print(f"\n{'='*80}")
    print(f"Processing: {degradation}")
    print(f"  Input: {input_folder}")
    print(f"  Checkpoint: {prism_checkpoint}")
    print(f"  Output: {output_folder}")
    print(f"{'='*80}\n")
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all input images
    input_images = get_image_files(input_folder)
    print(f"Found {len(input_images)} images to process")
    
    if not input_images:
        print(f"No images found in {input_folder}, skipping")
        return False
    
    # Check for special settings based on degradation type
    inp_of_unet_is_random_noise = "low" in degradation or "over" in degradation or "under" in degradation
    
    # Process each image
    success_count = 0
    for img_path in tqdm(input_images, desc=f"Inference ({degradation})"):
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_folder, img_name)
        
        # Skip if already processed
        if os.path.exists(output_path) and not args.overwrite:
            success_count += 1
            continue
        
        # Build inference command using unified checkpoint
        cmd = [
            "python3", "infer.py",
            "--img_path", img_path,
            "--ckpt_path", prism_checkpoint,
            "--distortion_type", degradation,
            "--save_root", output_folder,
            "--pretrained_model_name_or_path", "CompVis/stable-diffusion-v1-4",
            "--clip_path", "openai/clip-vit-large-patch14",
            "--resolution", str(args.resolution),
            "--num_inference_steps", str(args.num_inference_steps),
            "--seed", str(args.seed),
        ]
        
        if inp_of_unet_is_random_noise:
            cmd.append("--inp_of_unet_is_random_noise")
        
        # Run inference
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per image
            )
            
            if result.returncode == 0:
                # Move output to correct location with correct name
                # infer.py might save with a different name, need to check
                temp_outputs = get_image_files(output_folder)
                if temp_outputs:
                    # Get the most recently created file
                    latest = max(temp_outputs, key=os.path.getctime)
                    if latest != output_path:
                        os.rename(latest, output_path)
                    success_count += 1
            else:
                print(f"Error processing {img_name}: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"Timeout processing {img_name}")
        except Exception as e:
            print(f"Exception processing {img_name}: {e}")
    
    print(f"Successfully processed {success_count}/{len(input_images)} images")
    return success_count > 0


def load_image_numpy(img_path):
    """Load image as numpy array for metric computation."""
    img = cv2.imread(img_path)
    if img is None:
        img = array(Image.open(img_path).convert('RGB'))
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def compute_metrics(input_folder, output_folder, gt_folder=None, device='cpu'):
    """
    Compute PSNR, SSIM, LPIPS, and FID metrics.
    
    Args:
        input_folder: Folder with degraded input images
        output_folder: Folder with restored output images
        gt_folder: Optional folder with ground truth clean images
        device: Device for computation
    
    Returns:
        dict: Dictionary of computed metrics
    """
    print(f"\nComputing metrics...")
    
    # Get image files
    output_images = get_image_files(output_folder)
    if not output_images:
        print("No output images found for metric computation")
        return {}
    
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    
    # Initialize LPIPS model
    print("Initializing LPIPS model...")
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    
    # Transform for LPIPS (expects [-1, 1] range)
    to_tensor = transforms.ToTensor()
    
    # Determine reference folder (ground truth if available, else input)
    reference_folder = gt_folder if gt_folder and os.path.exists(gt_folder) else input_folder
    
    for output_path in tqdm(output_images, desc="Computing metrics"):
        img_name = os.path.basename(output_path)
        reference_path = os.path.join(reference_folder, img_name)
        
        if not os.path.exists(reference_path):
            # Try without extension variations
            base_name = os.path.splitext(img_name)[0]
            possible_refs = glob.glob(os.path.join(reference_folder, f"{base_name}.*"))
            if possible_refs:
                reference_path = possible_refs[0]
            else:
                continue
        
        try:
            # Load images as numpy arrays for PSNR/SSIM
            output_img = load_image_numpy(output_path)
            reference_img = load_image_numpy(reference_path)
            
            # Ensure same size
            if output_img.shape != reference_img.shape:
                # Resize output to match reference
                reference_img = cv2.resize(reference_img, 
                                          (output_img.shape[1], output_img.shape[0]),
                                          interpolation=cv2.INTER_LINEAR)
            
            # Compute PSNR (higher is better)
            psnr_val = psnr(reference_img, output_img, data_range=255)
            psnr_scores.append(psnr_val)
            
            # Compute SSIM (higher is better)
            # For multichannel images, we need to specify channel_axis
            ssim_val = ssim(reference_img, output_img, 
                           data_range=255, 
                           channel_axis=2,  # RGB channels on axis 2
                           multichannel=True)
            ssim_scores.append(ssim_val)
            
            # Compute LPIPS (lower is better)
            # Load images as PIL for LPIPS
            output_pil = Image.open(output_path).convert('RGB')
            reference_pil = Image.open(reference_path).convert('RGB')
            
            # Resize if needed
            if output_pil.size != reference_pil.size:
                reference_pil = reference_pil.resize(output_pil.size, Image.BILINEAR)
            
            # Convert to tensor and normalize to [-1, 1]
            output_tensor = to_tensor(output_pil).unsqueeze(0).to(device) * 2 - 1
            reference_tensor = to_tensor(reference_pil).unsqueeze(0).to(device) * 2 - 1
            
            with torch.no_grad():
                lpips_val = lpips_model(output_tensor, reference_tensor).item()
            lpips_scores.append(lpips_val)
            
        except Exception as e:
            print(f"Error computing metrics for {img_name}: {e}")
            continue
    
    # Compute final metrics
    metrics = {}
    if psnr_scores:
        metrics['psnr_mean'] = float(mean(psnr_scores))
        metrics['psnr_std'] = float(std(psnr_scores))
    if ssim_scores:
        metrics['ssim_mean'] = float(mean(ssim_scores))
        metrics['ssim_std'] = float(std(ssim_scores))
    if lpips_scores:
        metrics['lpips_mean'] = float(mean(lpips_scores))
        metrics['lpips_std'] = float(std(lpips_scores))
    
    metrics['num_images'] = len(psnr_scores)
    
    # Compute FID between output and reference folders
    print("Computing FID score...")
    try:
        fid_score = compute_fid(output_folder, reference_folder, device=device)
        metrics['fid'] = float(fid_score)
        print(f"FID: {fid_score:.4f}")
    except Exception as e:
        print(f"Error computing FID: {e}")
        metrics['fid'] = None
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Batch inference and metric computation")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/mdb_benchmark",
        help="Directory containing degradation subfolders"
    )
    parser.add_argument(
        "--prism_checkpoint",
        type=str,
        default="pre-trained/prism_model.pt",
        help="Path to unified PRISM model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_outputs",
        help="Output directory for restored images"
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="Optional: Directory with ground truth clean images"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution for inference"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs"
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip inference, only compute metrics"
    )
    parser.add_argument(
        "--degradations",
        nargs='+',
        default=None,
        help="Specific degradations to process (default: all)"
    )
    
    args = parser.parse_args()
    
    # Verify unified checkpoint exists
    if not os.path.exists(args.prism_checkpoint):
        print(f"Error: PRISM checkpoint not found at {args.prism_checkpoint}")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Using unified checkpoint: {args.prism_checkpoint}")
    
    # Get degradation folders
    degradation_pairs = get_degradation_folders(args.input_dir)
    
    if args.degradations:
        # Filter to specified degradations
        degradation_pairs = [
            (d, inp) for d, inp in degradation_pairs 
            if d in args.degradations
        ]
    
    print(f"\nFound {len(degradation_pairs)} degradation types to process:")
    for degradation, _ in degradation_pairs:
        print(f"  - {degradation}")
    
    if not degradation_pairs:
        print("No degradation types found!")
        return
    
    # Process each degradation type
    all_metrics = {}
    
    for degradation, input_folder in degradation_pairs:
        output_folder = os.path.join(args.output_dir, degradation)
        
        # Run inference
        if not args.skip_inference:
            success = run_inference_for_degradation(
                degradation, 
                input_folder, 
                args.prism_checkpoint, 
                output_folder, 
                args
            )
            if not success:
                continue
        
        # Compute metrics
        metrics = compute_metrics(
            input_folder,
            output_folder,
            args.gt_dir,
            device
        )
        
        all_metrics[degradation] = metrics
        
        # Print metrics
        print(f"\nMetrics for {degradation}:")
        for metric_name, value in metrics.items():
            if value is not None:
                print(f"  {metric_name}: {value:.4f}")
    
    # Save all metrics to JSON
    metrics_file = os.path.join(args.output_dir, "all_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n{'='*80}")
    print(f"All metrics saved to: {metrics_file}")
    print(f"{'='*80}")
    
    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY METRICS")
    print("="*100)
    print(f"{'Degradation':<30} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10} {'FID':>10} {'Count':>10}")
    print("-"*100)
    for degradation in sorted(all_metrics.keys()):
        metrics = all_metrics[degradation]
        psnr = metrics.get('psnr_mean', 0)
        ssim = metrics.get('ssim_mean', 0)
        lpips_val = metrics.get('lpips_mean', 0)
        fid_val = metrics.get('fid', 0) if metrics.get('fid') is not None else 0
        count = metrics.get('num_images', 0)
        print(f"{degradation:<30} {psnr:>10.2f} {ssim:>10.4f} {lpips_val:>10.4f} {fid_val:>10.2f} {count:>10}")
    print("="*100)


if __name__ == '__main__':
    main()
