"""
Pre-degraded Multi-Variant Dataset for CA-CLIP training.
Loads existing degraded images from subfolders for efficient training.
"""
import os
import random
import sys
from typing import List, Dict, Tuple
from collections import defaultdict

from PIL import Image
import cv2
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

# Import util module from current package
from . import util


def clip_transform(np_image, resolution=224):
    pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
    return Compose([
        Resize(resolution, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(resolution),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])(pil_image)


class PreDegradedMultiVariantDataset(data.Dataset):
    """
    Dataset that uses pre-degraded images from subfolders.
    
    Expected structure:
    dataroot/
    ├── clear/              # Clean images (ground truth) - can also be 'clean'
    │   ├── img001.png
    │   ├── img002.png
    │   └── ...
    ├── blur/              # Single degradation
    │   ├── img001.png
    │   └── ...
    ├── hazy/
    ├── blur_hazy/         # Compound degradation (underscore separated)
    ├── blur_hazy_noise/   # Multiple compound
    └── ...
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.size = opt["patch_size"]
        
        # Number of variants to sample per clean image
        self.num_variants = opt.get("num_variants", 256)
        
        # Load clean images (folder name can be 'clean' or 'clear')
        clean_dir_name = opt.get("clean_folder", "clear")  # Default to 'clear'
        clean_dir = os.path.join(opt["dataroot"], clean_dir_name)
        if not os.path.exists(clean_dir):
            raise ValueError(f"Clean directory not found: {clean_dir}")
        
        self.GT_paths = util.get_image_paths(opt["data_type"], clean_dir)
        
        if len(self.GT_paths) == 0:
            raise ValueError(f"No images found in {clean_dir}")
        
        # Build mapping: image_name -> {degradation_type: path}
        self.degradation_map = self._build_degradation_map(opt["dataroot"])
        
        # Get list of available degradation types
        self.degradation_types = sorted(list(set(
            deg_type 
            for deg_dict in self.degradation_map.values() 
            for deg_type in deg_dict.keys()
        )))
        
        print(f"Loaded {len(self.GT_paths)} clean images")
        print(f"Found {len(self.degradation_types)} degradation types")
        print(f"Sample degradation types: {self.degradation_types[:10]}")

    def _build_degradation_map(self, dataroot: str) -> Dict[str, Dict[str, str]]:
        """
        Build mapping from image names to their degraded versions.
        
        Returns:
            Dictionary: {image_name: {degradation_type: full_path}}
        """
        degradation_map = defaultdict(dict)
        
        # Get all subdirectories (except 'clean' or 'clear')
        subdirs = [d for d in os.listdir(dataroot) 
                   if os.path.isdir(os.path.join(dataroot, d)) and d not in ['clean', 'clear']]
        
        for subdir in subdirs:
            deg_type = subdir  # Folder name is degradation type
            deg_dir = os.path.join(dataroot, subdir)
            
            # Get all images in this degradation folder
            try:
                image_paths = util.get_image_paths('img', deg_dir)
                for path in image_paths:
                    img_name = os.path.basename(path)
                    degradation_map[img_name][deg_type] = path
            except Exception as e:
                print(f"Warning: Could not load images from {deg_dir}: {e}")
        
        return degradation_map

    def _sample_variants(self, img_name: str, num_variants: int) -> List[Tuple[str, str]]:
        """
        Sample degraded variants for a given image.
        
        Args:
            img_name: Name of the clean image
            num_variants: Number of variants to sample
            
        Returns:
            List of (degradation_type, path) tuples
        """
        available_degs = self.degradation_map.get(img_name, {})
        
        if len(available_degs) == 0:
            print(f"Warning: No degradations found for {img_name}")
            return []
        
        # Sample with replacement if we need more variants than available
        deg_items = list(available_degs.items())
        
        if len(deg_items) >= num_variants:
            # Sample without replacement
            sampled = random.sample(deg_items, num_variants)
        else:
            # Sample with replacement
            sampled = random.choices(deg_items, k=num_variants)
        
        return sampled

    def __len__(self):
        return len(self.GT_paths)

    def __getitem__(self, index):
        """
        Returns:
            Dictionary containing:
            - 'clean': Clean image tensor [3, H, W]
            - 'variants': Degraded variants tensor [m, 3, H, W]
            - 'variants_clip': CLIP-preprocessed variants [m, 3, 224, 224]
            - 'clean_clip': CLIP-preprocessed clean image [3, 224, 224]
            - 'deg_labels': List of m degradation label strings
        """
        # Load clean image
        GT_path = self.GT_paths[index]
        img_GT = util.read_img(None, GT_path, None)  # [H, W, C] in [0, 1], BGR
        img_name = os.path.basename(GT_path)
        
        if self.opt["phase"] == "train":
            H, W, C = img_GT.shape
            
            # Random crop
            rnd_h = random.randint(0, max(0, H - self.size))
            rnd_w = random.randint(0, max(0, W - self.size))
            img_GT = img_GT[rnd_h : rnd_h + self.size, rnd_w : rnd_w + self.size, :]
            
            # Random flip/rotate
            img_GT = util.augment(img_GT, 
                                  self.opt["use_flip"], 
                                  self.opt["use_rot"],
                                  mode='GT')
        
        # Sample degraded variants
        variant_samples = self._sample_variants(img_name, self.num_variants)
        
        if len(variant_samples) == 0:
            # Fallback: use clean image as variants
            variant_samples = [("clean", GT_path)] * self.num_variants
        
        # Load degraded variants
        variants = []
        variants_clip = []
        deg_labels = []
        
        for deg_type, deg_path in variant_samples:
            # Load degraded image
            img_degraded = util.read_img(None, deg_path, None)  # [H, W, C] in [0, 1], BGR
            
            # Apply same crop as clean image
            if self.opt["phase"] == "train":
                # Use same crop region
                img_degraded = img_degraded[rnd_h : rnd_h + self.size, rnd_w : rnd_w + self.size, :]
                # Apply same augmentation
                img_degraded = util.augment(img_degraded, 
                                           self.opt["use_flip"], 
                                           self.opt["use_rot"],
                                           mode='LQ')
            
            # Store
            variants.append(img_degraded)
            
            # CLIP preprocessing
            if self.opt["color"] == "RGB":
                img_for_clip = cv2.cvtColor(img_degraded, cv2.COLOR_BGR2RGB)
            else:
                img_for_clip = img_degraded
            variants_clip.append(clip_transform(img_for_clip))
            
            # Degradation label: convert folder name to label
            # e.g., "blur_hazy_noise" -> "blur+hazy+noise"
            deg_label = deg_type.replace('_', '+')
            deg_labels.append(deg_label)
        
        # Convert to tensors
        if self.opt["color"] == "RGB":
            img_GT = cv2.cvtColor(img_GT, cv2.COLOR_BGR2RGB)
            variants = [cv2.cvtColor(v, cv2.COLOR_BGR2RGB) for v in variants]
        
        # HWC to CHW, numpy to tensor
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        variants_tensor = torch.stack([
            torch.from_numpy(np.ascontiguousarray(np.transpose(v, (2, 0, 1)))).float()
            for v in variants
        ])  # [m, 3, H, W]
        
        variants_clip_tensor = torch.stack(variants_clip)  # [m, 3, 224, 224]
        
        # CLIP preprocess clean image
        if self.opt["color"] == "RGB":
            clean_for_clip = img_GT.permute(1, 2, 0).numpy()
        else:
            clean_for_clip = cv2.cvtColor(img_GT.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
        clean_clip = clip_transform(clean_for_clip)  # [3, 224, 224]
        
        return {
            'clean': img_GT,                        # [3, H, W]
            'variants': variants_tensor,            # [m, 3, H, W]
            'clean_clip': clean_clip,               # [3, 224, 224]
            'variants_clip': variants_clip_tensor,  # [m, 3, 224, 224]
            'deg_labels': deg_labels,               # List[str] of length m
            'path': GT_path
        }


def collate_fn_predegraded_multivariant(batch: List[Dict]) -> Dict:
    """
    Custom collate function for PreDegradedMultiVariantDataset.
    
    Stacks variants from all images in the batch.
    
    Args:
        batch: List of dictionaries from __getitem__
        
    Returns:
        Collated dictionary with batch dimension
    """
    # Stack clean images: [B, 3, H, W]
    clean = torch.stack([item['clean'] for item in batch])
    clean_clip = torch.stack([item['clean_clip'] for item in batch])
    
    # Stack all variants: [B, m, 3, H, W] -> [B*m, 3, H, W]
    variants = torch.cat([item['variants'] for item in batch], dim=0)
    variants_clip = torch.cat([item['variants_clip'] for item in batch], dim=0)
    
    # Flatten degradation labels: List[List[str]] -> List[str]
    deg_labels = []
    for item in batch:
        deg_labels.extend(item['deg_labels'])
    
    paths = [item['path'] for item in batch]
    
    return {
        'clean': clean,                   # [B, 3, H, W]
        'variants': variants,             # [B*m, 3, H, W]
        'clean_clip': clean_clip,         # [B, 3, 224, 224]
        'variants_clip': variants_clip,   # [B*m, 3, 224, 224]
        'deg_labels': deg_labels,         # List[str] of length B*m
        'paths': paths,
        'num_variants': batch[0]['variants'].shape[0]  # m
    }
