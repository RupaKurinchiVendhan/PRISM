#!/usr/bin/env python3
"""
WCACLIP Training Script

Train a Weighted Compound-Aware CLIP model for image restoration using 
the compound-aware contrastive learning protocol with Jaccard distance weighting.

This script implements the WCACLIP training protocol described in the paper,
including:
1. Compound-aware contrastive loss with Jaccard distance weighting
2. Quality-aware regularization
3. Support for complex degradation taxonomies
"""

import os
import math
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import transformers
from transformers import CLIPVisionModel, CLIPProcessor, CLIPModel
from transformers.optimization import get_scheduler

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="WCACLIP training script")
    
    # Data arguments
    parser.add_argument("--data_csv", type=str, required=True,
                        help="CSV file with columns: clean_image, degraded_image, degradation_type")
    parser.add_argument("--distortion_taxonomy", type=str, required=True,
                        help="JSON file mapping degradation names to component lists")
    parser.add_argument("--data_root", type=str, default="",
                        help="Root directory for image paths")
    
    # Model arguments
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-large-patch14",
                        help="Pretrained CLIP model name")
    parser.add_argument("--freeze_text_encoder", action="store_true", default=False,
                        help="Whether to freeze the text encoder")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./wcaclip_results",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_train_steps", type=int, default=None)
    
    # Loss arguments
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Temperature for contrastive loss")
    parser.add_argument("--quality_loss_weight", type=float, default=1.0,
                        help="Weight for quality-aware regularization loss")
    parser.add_argument("--max_degradations_per_clean", type=int, default=8,
                        help="Maximum number of degraded variants per clean image in a batch")
    
    # Augmentation arguments
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--use_augmentation", action="store_true", default=False)
    
    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    
    # System arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    
    return parser.parse_args()


class WCADataset(Dataset):
    """Dataset for Weighted Compound-Aware CLIP training"""
    
    def __init__(
        self, 
        data_csv: str,
        distortion_taxonomy: Dict[str, List[str]],
        data_root: str = "",
        image_size: int = 224,
        use_augmentation: bool = False
    ):
        self.data_root = Path(data_root) if data_root else Path("")
        self.distortion_taxonomy = distortion_taxonomy
        
        # Load data
        self.data_df = pd.read_csv(data_csv)
        
        # Group by clean images to create compound-aware batches
        self.clean_to_degraded = defaultdict(list)
        for idx, row in self.data_df.iterrows():
            clean_path = str(self.data_root / row['clean_image'])
            degraded_path = str(self.data_root / row['degraded_image'])
            degradation_type = row['degradation_type']
            
            self.clean_to_degraded[clean_path].append({
                'degraded_image': degraded_path,
                'degradation_type': degradation_type,
                'degradation_components': set(self.distortion_taxonomy.get(degradation_type, [degradation_type]))
            })
        
        self.clean_images = list(self.clean_to_degraded.keys())
        
        # Setup transforms
        if use_augmentation:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.clean_images)
    
    def __getitem__(self, idx):
        clean_image_path = self.clean_images[idx]
        degraded_variants = self.clean_to_degraded[clean_image_path]
        
        # Load clean image
        clean_image = Image.open(clean_image_path).convert('RGB')
        clean_tensor = self.transform(image=np.array(clean_image))['image']
        
        # Load degraded variants
        degraded_images = []
        degradation_types = []
        degradation_components = []
        
        for variant in degraded_variants:
            degraded_image = Image.open(variant['degraded_image']).convert('RGB')
            degraded_tensor = self.transform(image=np.array(degraded_image))['image']
            
            degraded_images.append(degraded_tensor)
            degradation_types.append(variant['degradation_type'])
            degradation_components.append(variant['degradation_components'])
        
        return {
            'clean_image': clean_tensor,
            'degraded_images': torch.stack(degraded_images) if len(degraded_images) > 1 else degraded_images[0].unsqueeze(0),
            'degradation_types': degradation_types,
            'degradation_components': degradation_components,
            'clean_image_path': clean_image_path
        }


def jaccard_distance(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard distance between two sets of degradation components"""
    if len(set1) == 0 and len(set2) == 0:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return 1.0 - (intersection / union)


def compute_jaccard_weights(degradation_components: List[Set[str]]) -> torch.Tensor:
    """Compute Jaccard distance weights between all pairs of degradation components"""
    n = len(degradation_components)
    weights = torch.zeros(n, n)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                distance = jaccard_distance(degradation_components[i], degradation_components[j])
                weights[i, j] = math.exp(distance)
    
    return weights


class WCAContrastiveLoss(nn.Module):
    """Weighted Compound-Aware Contrastive Loss"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cosine_sim = nn.CosineSimilarity(dim=-1)
    
    def forward(
        self,
        clean_embeddings: torch.Tensor,  # [batch_size, embed_dim]
        degraded_embeddings: torch.Tensor,  # [batch_size, max_variants, embed_dim]
        degradation_components: List[List[Set[str]]],  # [batch_size][variants][components]
        variant_masks: torch.Tensor  # [batch_size, max_variants] - mask for valid variants
    ) -> torch.Tensor:
        
        batch_size, max_variants, embed_dim = degraded_embeddings.shape
        total_loss = 0.0
        total_anchors = 0
        
        # Create global negatives (all degraded images from other clean images)
        all_degraded = degraded_embeddings.view(-1, embed_dim)  # [batch_size * max_variants, embed_dim]
        global_mask = variant_masks.view(-1)  # [batch_size * max_variants]
        global_negatives = all_degraded[global_mask]  # [num_valid_degraded, embed_dim]
        
        for batch_idx in range(batch_size):
            clean_emb = clean_embeddings[batch_idx]  # [embed_dim]
            
            # Get valid degraded variants for this clean image
            valid_mask = variant_masks[batch_idx]  # [max_variants]
            if valid_mask.sum() == 0:
                continue
                
            valid_degraded = degraded_embeddings[batch_idx][valid_mask]  # [num_valid, embed_dim]
            valid_components = degradation_components[batch_idx]
            
            # Compute Jaccard weights between sibling degradations
            jaccard_weights = compute_jaccard_weights(valid_components).to(degraded_embeddings.device)
            
            # For each degraded variant (anchor)
            for anchor_idx in range(len(valid_degraded)):
                anchor_emb = valid_degraded[anchor_idx]  # [embed_dim]
                
                # Positive: similarity with clean image
                pos_sim = self.cosine_sim(anchor_emb.unsqueeze(0), clean_emb.unsqueeze(0)) / self.temperature
                pos_exp = torch.exp(pos_sim)
                
                # Negative: sibling degradations (weighted by Jaccard distance)
                sibling_negatives = 0.0
                for neg_idx in range(len(valid_degraded)):
                    if neg_idx != anchor_idx:
                        neg_sim = self.cosine_sim(anchor_emb.unsqueeze(0), valid_degraded[neg_idx].unsqueeze(0)) / self.temperature
                        weight = jaccard_weights[anchor_idx, neg_idx]
                        sibling_negatives += weight * torch.exp(neg_sim)
                
                # Negative: global negatives (other images in batch)
                global_sims = self.cosine_sim(anchor_emb.unsqueeze(0), global_negatives) / self.temperature
                global_negatives_sum = torch.exp(global_sims).sum()
                
                # Remove self from global negatives
                current_global_idx = batch_idx * max_variants + anchor_idx
                if current_global_idx < len(global_negatives):
                    self_sim = self.cosine_sim(anchor_emb.unsqueeze(0), anchor_emb.unsqueeze(0)) / self.temperature
                    global_negatives_sum -= torch.exp(self_sim)
                
                # Compute contrastive loss for this anchor
                denominator = sibling_negatives + global_negatives_sum
                anchor_loss = -torch.log(pos_exp / (pos_exp + denominator + 1e-8))
                
                total_loss += anchor_loss
                total_anchors += 1
        
        return total_loss / max(total_anchors, 1)


class QualityRegularizationLoss(nn.Module):
    """Quality-aware regularization loss"""
    
    def __init__(self, embed_dim: int, num_degradation_types: int):
        super().__init__()
        # Learnable degradation probability predictor
        self.degradation_predictor = nn.Linear(embed_dim, num_degradation_types)
    
    def forward(
        self,
        clean_embeddings: torch.Tensor,  # [batch_size, embed_dim]
        degradation_labels: List[List[str]],  # [batch_size][variants]
        degradation_to_idx: Dict[str, int]
    ) -> torch.Tensor:
        
        batch_size = clean_embeddings.shape[0]
        total_loss = 0.0
        
        # Predict degradation probabilities from clean embeddings
        degradation_probs = torch.softmax(self.degradation_predictor(clean_embeddings), dim=-1)
        
        for batch_idx in range(batch_size):
            batch_loss = 0.0
            
            # For each degraded variant of this clean image
            for variant_degradations in degradation_labels[batch_idx]:
                for degradation in variant_degradations:
                    if degradation in degradation_to_idx:
                        deg_idx = degradation_to_idx[degradation]
                        # Penalize high probability of degradation in clean embedding
                        batch_loss += degradation_probs[batch_idx, deg_idx]
            
            total_loss += batch_loss / max(len(degradation_labels[batch_idx]), 1)
        
        return total_loss / batch_size


def collate_fn(batch):
    """Custom collate function to handle variable number of degraded variants"""
    clean_images = torch.stack([item['clean_image'] for item in batch])
    
    # Find maximum number of variants in this batch
    max_variants = max(item['degraded_images'].shape[0] for item in batch)
    
    batch_size = len(batch)
    embed_dim = batch[0]['degraded_images'].shape[-3:]  # (C, H, W)
    
    # Pad degraded images
    degraded_images = torch.zeros(batch_size, max_variants, *embed_dim)
    variant_masks = torch.zeros(batch_size, max_variants, dtype=torch.bool)
    
    degradation_types = []
    degradation_components = []
    clean_image_paths = []
    
    for i, item in enumerate(batch):
        num_variants = item['degraded_images'].shape[0]
        degraded_images[i, :num_variants] = item['degraded_images']
        variant_masks[i, :num_variants] = True
        
        # Pad lists to max_variants
        padded_types = item['degradation_types'] + [''] * (max_variants - num_variants)
        padded_components = item['degradation_components'] + [set()] * (max_variants - num_variants)
        
        degradation_types.append(padded_types)
        degradation_components.append(padded_components)
        clean_image_paths.append(item['clean_image_path'])
    
    return {
        'clean_images': clean_images,
        'degraded_images': degraded_images,
        'variant_masks': variant_masks,
        'degradation_types': degradation_types,
        'degradation_components': degradation_components,
        'clean_image_paths': clean_image_paths
    }


def main():
    args = parse_args()
    
    # Setup accelerator
    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        project_dir=args.output_dir
    )
    
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=os.path.join(args.output_dir, "logs"),
        project_config=accelerator_project_config,
    )
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()
    
    set_seed(args.seed)
    
    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load distortion taxonomy
    with open(args.distortion_taxonomy, 'r') as f:
        distortion_taxonomy = json.load(f)
    
    # Create degradation type to index mapping
    all_degradation_types = set()
    for components in distortion_taxonomy.values():
        all_degradation_types.update(components)
    degradation_to_idx = {deg: idx for idx, deg in enumerate(sorted(all_degradation_types))}
    
    logger.info(f"Found {len(degradation_to_idx)} unique degradation components")
    
    # Create dataset and dataloader
    dataset = WCADataset(
        data_csv=args.data_csv,
        distortion_taxonomy=distortion_taxonomy,
        data_root=args.data_root,
        image_size=args.image_size,
        use_augmentation=args.use_augmentation
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Load CLIP model
    clip_model = CLIPModel.from_pretrained(args.clip_model_name)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model_name)
    
    # Freeze text encoder if specified
    if args.freeze_text_encoder:
        for param in clip_model.text_model.parameters():
            param.requires_grad = False
        logger.info("Frozen text encoder")
    
    # Initialize losses
    contrastive_loss = WCAContrastiveLoss(temperature=args.temperature)
    quality_loss = QualityRegularizationLoss(
        embed_dim=clip_model.vision_model.config.hidden_size,
        num_degradation_types=len(degradation_to_idx)
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        [
            {"params": clip_model.parameters()},
            {"params": quality_loss.parameters()}
        ],
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(dataloader))
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    # Setup scheduler
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    # Prepare everything with accelerator
    clip_model, quality_loss, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        clip_model, quality_loss, optimizer, dataloader, lr_scheduler
    )
    
    # Training info
    logger.info("***** Running WCACLIP training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # Initialize tracker
    if accelerator.is_main_process:
        accelerator.init_trackers("wcaclip", config=vars(args))
    
    # Training loop
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    for epoch in range(args.num_train_epochs):
        clip_model.train()
        quality_loss.train()
        
        for step, batch in enumerate(dataloader):
            clean_images = batch['clean_images']  # [batch_size, 3, H, W]
            degraded_images = batch['degraded_images']  # [batch_size, max_variants, 3, H, W]
            variant_masks = batch['variant_masks']  # [batch_size, max_variants]
            degradation_components = batch['degradation_components']  # [batch_size][variants][components]
            degradation_types = batch['degradation_types']  # [batch_size][variants]
            
            batch_size, max_variants = degraded_images.shape[:2]
            
            # Encode clean images
            clean_embeddings = clip_model.get_image_features(clean_images)
            clean_embeddings = F.normalize(clean_embeddings, p=2, dim=-1)
            
            # Encode degraded images
            degraded_flat = degraded_images.view(-1, *degraded_images.shape[2:])  # [batch_size * max_variants, 3, H, W]
            degraded_embeddings_flat = clip_model.get_image_features(degraded_flat)
            degraded_embeddings_flat = F.normalize(degraded_embeddings_flat, p=2, dim=-1)
            degraded_embeddings = degraded_embeddings_flat.view(batch_size, max_variants, -1)
            
            # Compute contrastive loss
            contrastive_loss_value = contrastive_loss(
                clean_embeddings=clean_embeddings,
                degraded_embeddings=degraded_embeddings,
                degradation_components=degradation_components,
                variant_masks=variant_masks
            )
            
            # Compute quality regularization loss
            quality_loss_value = quality_loss(
                clean_embeddings=clean_embeddings,
                degradation_labels=[[list(comp) for comp in batch_comps] for batch_comps in degradation_components],
                degradation_to_idx=degradation_to_idx
            )
            
            # Total loss
            total_loss = contrastive_loss_value + args.quality_loss_weight * quality_loss_value
            
            # Backward pass
            accelerator.backward(total_loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Logging
            if global_step % args.logging_steps == 0:
                logs = {
                    "contrastive_loss": contrastive_loss_value.item(),
                    "quality_loss": quality_loss_value.item(),
                    "total_loss": total_loss.item(),
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)
            
            # Save checkpoint
            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                
                # Save CLIP model
                unwrapped_clip = accelerator.unwrap_model(clip_model)
                unwrapped_clip.save_pretrained(save_path)
                
                # Save quality loss module
                unwrapped_quality = accelerator.unwrap_model(quality_loss)
                torch.save(unwrapped_quality.state_dict(), os.path.join(save_path, "quality_loss.pt"))
                
                # Save training state
                accelerator.save_state(save_path)
                logger.info(f"Saved checkpoint to {save_path}")
            
            progress_bar.update(1)
            global_step += 1
            
            if global_step >= args.max_train_steps:
                break
        
        if global_step >= args.max_train_steps:
            break
    
    # Final save
    if accelerator.is_main_process:
        final_save_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_save_path, exist_ok=True)
        
        unwrapped_clip = accelerator.unwrap_model(clip_model)
        unwrapped_clip.save_pretrained(final_save_path)
        
        unwrapped_quality = accelerator.unwrap_model(quality_loss)
        torch.save(unwrapped_quality.state_dict(), os.path.join(final_save_path, "quality_loss.pt"))
        
        logger.info(f"Training completed. Final model saved to {final_save_path}")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()