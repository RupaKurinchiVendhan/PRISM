#!/usr/bin/env python3
"""
WCACLIP Evaluation Script

Evaluate a trained WCACLIP model by:
1. Computing embedding similarities between clean and degraded images
2. Analyzing compound-aware clustering behavior
3. Visualizing embedding spaces
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Set
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate WCACLIP model")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained WCACLIP model")
    parser.add_argument("--data_csv", type=str, required=True,
                        help="CSV file with test data")
    parser.add_argument("--distortion_taxonomy", type=str, required=True,
                        help="JSON file with distortion taxonomy")
    parser.add_argument("--data_root", type=str, default="",
                        help="Root directory for image paths")
    parser.add_argument("--output_dir", type=str, default="./wcaclip_eval",
                        help="Output directory for evaluation results")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    
    return parser.parse_args()


def jaccard_distance(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard distance between two sets"""
    if len(set1) == 0 and len(set2) == 0:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return 1.0 - (intersection / union)


def load_model_and_processor(model_path: str):
    """Load trained WCACLIP model and processor"""
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    
    model.eval()
    return model, processor


def encode_images(model, processor, image_paths: List[str], data_root: str, batch_size: int = 32):
    """Encode a list of images to embeddings"""
    embeddings = []
    data_root = Path(data_root) if data_root else Path("")
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for path in batch_paths:
            full_path = data_root / path if data_root else Path(path)
            try:
                image = Image.open(full_path).convert('RGB')
                batch_images.append(image)
            except Exception as e:
                print(f"Error loading {full_path}: {e}")
                # Use a dummy black image
                batch_images.append(Image.new('RGB', (224, 224), (0, 0, 0)))
        
        # Process batch
        inputs = processor(images=batch_images, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=-1)
        
        embeddings.extend(image_features.cpu().numpy())
    
    return np.array(embeddings)


def evaluate_compound_awareness(
    clean_embeddings: np.ndarray,
    degraded_embeddings: np.ndarray,
    degradation_components: List[Set[str]],
    degradation_types: List[str]
) -> Dict[str, float]:
    """Evaluate compound-awareness of embeddings"""
    
    results = {}
    
    # 1. Clean-degraded alignment
    clean_degraded_sims = []
    for i in range(len(clean_embeddings)):
        sim = cosine_similarity([clean_embeddings[i]], [degraded_embeddings[i]])[0, 0]
        clean_degraded_sims.append(sim)
    
    results['clean_degraded_similarity'] = np.mean(clean_degraded_sims)
    results['clean_degraded_similarity_std'] = np.std(clean_degraded_sims)
    
    # 2. Compound-aware separation
    compound_separations = []
    jaccard_distances = []
    
    for i in range(len(degraded_embeddings)):
        for j in range(i+1, len(degraded_embeddings)):
            # Compute embedding similarity
            emb_sim = cosine_similarity([degraded_embeddings[i]], [degraded_embeddings[j]])[0, 0]
            
            # Compute Jaccard distance between degradation components
            jaccard_dist = jaccard_distance(degradation_components[i], degradation_components[j])
            
            compound_separations.append(1.0 - emb_sim)  # Convert similarity to distance
            jaccard_distances.append(jaccard_dist)
    
    # Correlation between embedding separation and Jaccard distance
    if len(compound_separations) > 1:
        correlation = np.corrcoef(compound_separations, jaccard_distances)[0, 1]
        results['compound_correlation'] = correlation if not np.isnan(correlation) else 0.0
    else:
        results['compound_correlation'] = 0.0
    
    # 3. Per-degradation type analysis
    degradation_type_sims = {}
    for deg_type in set(degradation_types):
        indices = [i for i, dt in enumerate(degradation_types) if dt == deg_type]
        if len(indices) > 0:
            type_sims = [clean_degraded_sims[i] for i in indices]
            degradation_type_sims[deg_type] = {
                'mean_similarity': np.mean(type_sims),
                'std_similarity': np.std(type_sims),
                'count': len(type_sims)
            }
    
    results['per_degradation_analysis'] = degradation_type_sims
    
    return results


def create_visualizations(
    clean_embeddings: np.ndarray,
    degraded_embeddings: np.ndarray,
    degradation_types: List[str],
    output_dir: Path
):
    """Create visualization plots for embedding analysis"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine embeddings for t-SNE
    all_embeddings = np.vstack([clean_embeddings, degraded_embeddings])
    labels = ['clean'] * len(clean_embeddings) + degradation_types
    
    # Limit to reasonable number for t-SNE
    max_samples = 1000
    if len(all_embeddings) > max_samples:
        indices = np.random.choice(len(all_embeddings), max_samples, replace=False)
        all_embeddings = all_embeddings[indices]
        labels = [labels[i] for i in indices]
    
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Plot t-SNE
    plt.figure(figsize=(12, 8))
    unique_labels = list(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        indices = [j for j, l in enumerate(labels) if l == label]
        if indices:
            x = embeddings_2d[indices, 0]
            y = embeddings_2d[indices, 1]
            plt.scatter(x, y, c=[colors[i]], label=label, alpha=0.6, s=30)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('WCACLIP Embedding Space (t-SNE)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(output_dir / 'embedding_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Similarity heatmap
    if len(degradation_types) <= 20:  # Only for manageable number of types
        unique_types = list(set(degradation_types))
        similarity_matrix = np.zeros((len(unique_types), len(unique_types)))
        
        for i, type1 in enumerate(unique_types):
            indices1 = [j for j, dt in enumerate(degradation_types) if dt == type1]
            for j, type2 in enumerate(unique_types):
                indices2 = [k for k, dt in enumerate(degradation_types) if dt == type2]
                
                if indices1 and indices2:
                    sims = []
                    for idx1 in indices1[:10]:  # Limit for computational efficiency
                        for idx2 in indices2[:10]:
                            sim = cosine_similarity([degraded_embeddings[idx1]], [degraded_embeddings[idx2]])[0, 0]
                            sims.append(sim)
                    similarity_matrix[i, j] = np.mean(sims)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, xticklabels=unique_types, yticklabels=unique_types, 
                   annot=True, fmt='.3f', cmap='viridis')
        plt.title('Inter-Degradation Similarity Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'similarity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model and processor...")
    model, processor = load_model_and_processor(args.model_path)
    
    print("Loading distortion taxonomy...")
    with open(args.distortion_taxonomy, 'r') as f:
        distortion_taxonomy = json.load(f)
    
    print("Loading test data...")
    df = pd.read_csv(args.data_csv)
    
    # Limit samples if specified
    if args.num_samples > 0 and len(df) > args.num_samples:
        df = df.sample(n=args.num_samples, random_state=42).reset_index(drop=True)
    
    print(f"Evaluating on {len(df)} samples...")
    
    # Extract data
    clean_images = df['clean_image'].tolist()
    degraded_images = df['degraded_image'].tolist()
    degradation_types = df['degradation_type'].tolist()
    
    # Convert degradation types to component sets
    degradation_components = []
    for deg_type in degradation_types:
        components = distortion_taxonomy.get(deg_type, [deg_type])
        degradation_components.append(set(components))
    
    print("Encoding clean images...")
    clean_embeddings = encode_images(model, processor, clean_images, args.data_root, args.batch_size)
    
    print("Encoding degraded images...")
    degraded_embeddings = encode_images(model, processor, degraded_images, args.data_root, args.batch_size)
    
    print("Evaluating compound-awareness...")
    results = evaluate_compound_awareness(
        clean_embeddings=clean_embeddings,
        degraded_embeddings=degraded_embeddings,
        degradation_components=degradation_components,
        degradation_types=degradation_types
    )
    
    # Print results
    print("\n" + "="*50)
    print("WCACLIP EVALUATION RESULTS")
    print("="*50)
    print(f"Clean-Degraded Similarity: {results['clean_degraded_similarity']:.4f} ± {results['clean_degraded_similarity_std']:.4f}")
    print(f"Compound Correlation: {results['compound_correlation']:.4f}")
    
    print("\nPer-Degradation Analysis:")
    for deg_type, stats in results['per_degradation_analysis'].items():
        print(f"  {deg_type}: {stats['mean_similarity']:.4f} ± {stats['std_similarity']:.4f} (n={stats['count']})")
    
    # Save detailed results
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                json_results[key] = float(value)
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(
        clean_embeddings=clean_embeddings,
        degraded_embeddings=degraded_embeddings,
        degradation_types=degradation_types,
        output_dir=output_dir
    )
    
    print(f"Visualizations saved to: {output_dir}")
    print("Evaluation complete!")


if __name__ == "__main__":
    main()