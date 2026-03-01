"""
Compositional-Aware CLIP Loss with Jaccard-weighted contrastive learning.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Optional

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False


def compute_jaccard_weights(degradation_sets_j: List[set], degradation_sets_k: List[set]) -> torch.Tensor:
    """
    Compute Jaccard distance weights between pairs of degradation sets.
    
    Args:
        degradation_sets_j: List of sets of degradation types for anchor samples
        degradation_sets_k: List of sets of degradation types for comparison samples
        
    Returns:
        weight matrix w_jk = exp(1 - jaccard_similarity)
    """
    batch_size_j = len(degradation_sets_j)
    batch_size_k = len(degradation_sets_k)
    
    weights = torch.zeros(batch_size_j, batch_size_k)
    
    for j in range(batch_size_j):
        for k in range(batch_size_k):
            if j == k:
                weights[j, k] = 0.0  # Will be masked out anyway
                continue
                
            set_j = degradation_sets_j[j]
            set_k = degradation_sets_k[k]
            
            intersection = len(set_j & set_k)
            union = len(set_j | set_k)
            
            if union == 0:
                jaccard_sim = 0.0
            else:
                jaccard_sim = intersection / union
            
            # Jaccard distance: 1 - similarity
            jaccard_dist = 1.0 - jaccard_sim
            
            # Weight: exp(jaccard_distance)
            weights[j, k] = torch.exp(torch.tensor(jaccard_dist))
    
    return weights


class CAClipLoss(nn.Module):
    """
    Compositional-Aware CLIP Loss with Jaccard-weighted contrastive learning.
    
    For each degraded variant I_dist^(j), we:
    1. Align it with its clean version I_clean (positive)
    2. Repel from sibling variants (weighted by Jaccard distance)
    3. Repel from variants from other images in batch
    """
    
    def __init__(
            self,
            temperature: float = 0.1,
            local_loss: bool = False,
            gather_with_grad: bool = False,
            rank: int = 0,
            world_size: int = 1,
            use_horovod: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

    def forward(
            self,
            distorted_features: torch.Tensor,      # [B*m, D] features from all degraded variants
            clean_features: torch.Tensor,          # [B, D] features from clean images
            degradation_labels: List[List[str]],   # [B*m] degradation labels for each variant
            num_variants: int = 256,               # m: number of variants per clean image
            output_dict: bool = False
    ):
        """
        Args:
            distorted_features: Features from degraded variants [B*m, D]
            clean_features: Features from clean images [B, D]
            degradation_labels: List of degradation type strings for each variant
            num_variants: Number of variants per clean image (m)
            output_dict: Whether to return dict or scalar
        """
        device = distorted_features.device
        batch_size = clean_features.shape[0]  # B
        total_variants = distorted_features.shape[0]  # B*m
        
        # Normalize features
        distorted_features = F.normalize(distorted_features, dim=-1)
        clean_features = F.normalize(clean_features, dim=-1)
        
        # Parse degradation labels into sets
        degradation_sets = []
        for label in degradation_labels:
            if isinstance(label, str):
                # Handle compound degradations like "hazy+rainy"
                deg_set = set(label.split('+'))
            elif isinstance(label, list):
                deg_set = set(label)
            else:
                deg_set = {label}
            degradation_sets.append(deg_set)
        
        # Compute similarities
        # distorted_features: [B*m, D], clean_features: [B, D]
        # We need to expand clean_features to match each variant
        clean_features_expanded = clean_features.repeat_interleave(num_variants, dim=0)  # [B*m, D]
        
        # Positive similarities: each variant with its clean version
        pos_sim = torch.sum(distorted_features * clean_features_expanded, dim=-1)  # [B*m]
        pos_sim = pos_sim / self.temperature
        
        # Negative similarities: all pairwise comparisons
        neg_sim = torch.matmul(distorted_features, distorted_features.t())  # [B*m, B*m]
        neg_sim = neg_sim / self.temperature
        
        # Compute Jaccard weights for all pairs
        weights = compute_jaccard_weights(degradation_sets, degradation_sets).to(device)  # [B*m, B*m]
        
        # Create mask to exclude self-comparison
        mask = torch.eye(total_variants, dtype=torch.bool, device=device)
        
        # Apply weights to negative similarities
        weighted_neg_sim = neg_sim * weights
        weighted_neg_sim = weighted_neg_sim.masked_fill(mask, float('-inf'))
        
        # Compute loss for each variant
        # L = -log(exp(pos) / (sum of weighted negatives))
        numerator = torch.exp(pos_sim)
        
        # For each variant j, sum over all weighted negatives
        denominator = torch.sum(torch.exp(weighted_neg_sim), dim=-1)  # [B*m]
        
        # Add small epsilon for numerical stability
        loss = -torch.log(numerator / (denominator + 1e-8))
        
        # Average over all variants
        total_loss = loss.mean()
        
        if output_dict:
            return {
                "ca_contrastive_loss": total_loss,
                "pos_sim_mean": pos_sim.mean().detach(),
                "neg_sim_mean": neg_sim[~mask].mean().detach(),
                "weight_mean": weights[~mask].mean().detach()
            }
        
        return total_loss
