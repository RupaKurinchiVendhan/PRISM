"""
This Code provides functions to compute various image quality metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- FID (Fréchet Inception Distance)
- LPIPS (Learned Perceptual Image Patch Similarity)

Also includes statistical analysis tools:
- Paired t-test for comparing methods
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from scipy import stats
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision.transforms as transforms
from torchvision.models import inception_v3
import lpips
from typing import List, Tuple, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class ImageQualityMetrics:
    """
    A comprehensive class for computing image quality metrics.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the metrics calculator.
        
        Args:
            device (str): Device to use for computations ('cuda' or 'cpu')
        """
        self.device = device
        self.lpips_model = None
        self.inception_model = None
        self._setup_models()
    
    def _setup_models(self):
        """Setup deep learning models for FID and LPIPS computation."""
        try:
            # Setup LPIPS model
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
            print(f"LPIPS model loaded on {self.device}")
            
            # Setup Inception model for FID
            self.inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device)
            self.inception_model.eval()
            print(f"Inception model loaded on {self.device}")
            
        except Exception as e:
            print(f"Warning: Could not load deep learning models: {e}")
            print("FID and LPIPS metrics will not be available.")
    
    def load_image(self, image_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Load and preprocess an image.
        
        Args:
            image_path (str): Path to the image
            target_size (tuple, optional): Target size (height, width) for resizing
            
        Returns:
            np.ndarray: Loaded image as numpy array
        """
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path  # Assume it's already a numpy array
        
        if target_size is not None:
            image = cv2.resize(image, (target_size[1], target_size[0]))
        
        return image.astype(np.float32) / 255.0
    
    def compute_psnr(self, image1: Union[str, np.ndarray], image2: Union[str, np.ndarray], 
                     data_range: float = 1.0) -> float:
        """
        Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
        
        Args:
            image1: Reference image (path or numpy array)
            image2: Test image (path or numpy array)
            data_range: Data range of the images (1.0 for [0,1], 255 for [0,255])
            
        Returns:
            float: PSNR value in dB
        """
        img1 = self.load_image(image1) if isinstance(image1, str) else image1
        img2 = self.load_image(image2) if isinstance(image2, str) else image2
        
        # Ensure same dimensions
        if img1.shape != img2.shape:
            min_h = min(img1.shape[0], img2.shape[0])
            min_w = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_h, :min_w]
            img2 = img2[:min_h, :min_w]
        
        return psnr(img1, img2, data_range=data_range)
    
    def compute_ssim(self, image1: Union[str, np.ndarray], image2: Union[str, np.ndarray], 
                     data_range: float = 1.0, multichannel: bool = True) -> float:
        """
        Compute Structural Similarity Index (SSIM) between two images.
        
        Args:
            image1: Reference image (path or numpy array)
            image2: Test image (path or numpy array)
            data_range: Data range of the images
            multichannel: Whether to compute SSIM for multichannel images
            
        Returns:
            float: SSIM value between -1 and 1
        """
        img1 = self.load_image(image1) if isinstance(image1, str) else image1
        img2 = self.load_image(image2) if isinstance(image2, str) else image2
        
        # Ensure same dimensions
        if img1.shape != img2.shape:
            min_h = min(img1.shape[0], img2.shape[0])
            min_w = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_h, :min_w]
            img2 = img2[:min_h, :min_w]
        
        return ssim(img1, img2, data_range=data_range, channel_axis=2 if multichannel else None)
    
    def compute_lpips(self, image1: Union[str, np.ndarray], image2: Union[str, np.ndarray]) -> float:
        """
        Compute Learned Perceptual Image Patch Similarity (LPIPS) between two images.
        
        Args:
            image1: Reference image (path or numpy array)
            image2: Test image (path or numpy array)
            
        Returns:
            float: LPIPS distance (lower is better)
        """
        if self.lpips_model is None:
            raise RuntimeError("LPIPS model not available. Please install lpips package.")
        
        img1 = self.load_image(image1) if isinstance(image1, str) else image1
        img2 = self.load_image(image2) if isinstance(image2, str) else image2
        
        # Ensure same dimensions
        if img1.shape != img2.shape:
            min_h = min(img1.shape[0], img2.shape[0])
            min_w = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_h, :min_w]
            img2 = img2[:min_h, :min_w]
        
        # Convert to torch tensors and normalize to [-1, 1]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        img1_tensor = transform(img1).unsqueeze(0).to(self.device)
        img2_tensor = transform(img2).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            lpips_score = self.lpips_model(img1_tensor, img2_tensor)
        
        return lpips_score.item()
    
    def get_inception_features(self, images: torch.Tensor) -> np.ndarray:
        """
        Extract features from Inception model for FID computation.
        
        Args:
            images: Batch of images as torch tensor
            
        Returns:
            np.ndarray: Feature vectors
        """
        if self.inception_model is None:
            raise RuntimeError("Inception model not available.")
        
        with torch.no_grad():
            features = self.inception_model(images)
            if isinstance(features, tuple):
                features = features[0]  # Get logits
        
        return features.cpu().numpy()
    
    def compute_fid(self, real_images: List[Union[str, np.ndarray]], 
                    generated_images: List[Union[str, np.ndarray]], 
                    batch_size: int = 32) -> float:
        """
        Compute Fréchet Inception Distance (FID) between two sets of images.
        
        Args:
            real_images: List of real images (paths or numpy arrays)
            generated_images: List of generated images (paths or numpy arrays)
            batch_size: Batch size for processing
            
        Returns:
            float: FID score (lower is better)
        """
        if self.inception_model is None:
            raise RuntimeError("Inception model not available for FID computation.")
        
        def preprocess_images(image_list):
            """Preprocess images for Inception model."""
            processed = []
            for img in image_list:
                if isinstance(img, str):
                    img = self.load_image(img, target_size=(299, 299))
                else:
                    img = cv2.resize(img, (299, 299))
                
                # Convert to tensor and normalize for Inception
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
                img_tensor = (img_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
                processed.append(img_tensor)
            
            return torch.stack(processed)
        
        def get_features_batch(image_list):
            """Extract features from a list of images."""
            features_list = []
            
            for i in range(0, len(image_list), batch_size):
                batch = image_list[i:i + batch_size]
                batch_tensor = preprocess_images(batch).to(self.device)
                batch_features = self.get_inception_features(batch_tensor)
                features_list.append(batch_features)
            
            return np.concatenate(features_list, axis=0)
        
        # Get features for both sets
        real_features = get_features_batch(real_images)
        gen_features = get_features_batch(generated_images)
        
        # Compute statistics
        mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
        
        # Compute FID
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid_score = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        
        return fid_score
    
    def compute_all_metrics(self, image1: Union[str, np.ndarray], 
                           image2: Union[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute all available metrics between two images.
        
        Args:
            image1: Reference image
            image2: Test image
            
        Returns:
            dict: Dictionary containing all metric scores
        """
        metrics = {}
        
        try:
            metrics['PSNR'] = self.compute_psnr(image1, image2)
        except Exception as e:
            print(f"Error computing PSNR: {e}")
            metrics['PSNR'] = None
        
        try:
            metrics['SSIM'] = self.compute_ssim(image1, image2)
        except Exception as e:
            print(f"Error computing SSIM: {e}")
            metrics['SSIM'] = None
        
        try:
            metrics['LPIPS'] = self.compute_lpips(image1, image2)
        except Exception as e:
            print(f"Error computing LPIPS: {e}")
            metrics['LPIPS'] = None
        
        return metrics


class StatisticalAnalysis:
    """
    Statistical analysis tools for comparing image quality metrics.
    """
    
    @staticmethod
    def paired_t_test(group1: List[float], group2: List[float], 
                      alternative: str = 'two-sided') -> Dict[str, float]:
        """
        Perform paired t-test between two groups.
        
        Args:
            group1: First group of values
            group2: Second group of values  
            alternative: Type of test ('two-sided', 'less', 'greater')
            
        Returns:
            dict: Test results including t-statistic, p-value, and effect size
        """
        if len(group1) != len(group2):
            raise ValueError("Groups must have the same length for paired t-test")
        
        # Remove None values
        paired_data = [(g1, g2) for g1, g2 in zip(group1, group2) 
                       if g1 is not None and g2 is not None]
        
        if len(paired_data) < 2:
            raise ValueError("Need at least 2 valid pairs for t-test")
        
        g1_clean, g2_clean = zip(*paired_data)
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(g1_clean, g2_clean, alternative=alternative)
        
        # Calculate effect size (Cohen's d for paired samples)
        differences = np.array(g1_clean) - np.array(g2_clean)
        effect_size = np.mean(differences) / np.std(differences, ddof=1)
        
        # Calculate confidence interval for the mean difference
        n = len(differences)
        se = stats.sem(differences)
        ci = stats.t.interval(0.95, n-1, loc=np.mean(differences), scale=se)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'mean_difference': np.mean(differences),
            'std_difference': np.std(differences, ddof=1),
            'confidence_interval_95': ci,
            'n_pairs': n,
            'alternative': alternative
        }
    
    @staticmethod
    def independent_t_test(group1: List[float], group2: List[float], 
                          equal_var: bool = False, alternative: str = 'two-sided') -> Dict[str, float]:
        """
        Perform independent samples t-test between two groups.
        
        Args:
            group1: First group of values
            group2: Second group of values
            equal_var: Whether to assume equal variances
            alternative: Type of test ('two-sided', 'less', 'greater')
            
        Returns:
            dict: Test results including t-statistic, p-value, and effect size
        """
        # Remove None values
        g1_clean = [x for x in group1 if x is not None]
        g2_clean = [x for x in group2 if x is not None]
        
        if len(g1_clean) < 2 or len(g2_clean) < 2:
            raise ValueError("Need at least 2 valid values in each group")
        
        # Perform independent t-test
        t_stat, p_value = stats.ttest_ind(g1_clean, g2_clean, 
                                         equal_var=equal_var, alternative=alternative)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(g1_clean) - 1) * np.var(g1_clean, ddof=1) + 
                             (len(g2_clean) - 1) * np.var(g2_clean, ddof=1)) / 
                            (len(g1_clean) + len(g2_clean) - 2))
        effect_size = (np.mean(g1_clean) - np.mean(g2_clean)) / pooled_std
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'mean_group1': np.mean(g1_clean),
            'mean_group2': np.mean(g2_clean),
            'mean_difference': np.mean(g1_clean) - np.mean(g2_clean),
            'n_group1': len(g1_clean),
            'n_group2': len(g2_clean),
            'alternative': alternative
        }
    
    @staticmethod
    def summary_statistics(values: List[float]) -> Dict[str, float]:
        """
        Compute summary statistics for a list of values.
        
        Args:
            values: List of numeric values
            
        Returns:
            dict: Summary statistics
        """
        clean_values = [x for x in values if x is not None]
        
        if not clean_values:
            return {'error': 'No valid values provided'}
        
        clean_array = np.array(clean_values)
        
        return {
            'count': len(clean_values),
            'mean': np.mean(clean_array),
            'std': np.std(clean_array, ddof=1),
            'min': np.min(clean_array),
            'max': np.max(clean_array),
            'median': np.median(clean_array),
            'q25': np.percentile(clean_array, 25),
            'q75': np.percentile(clean_array, 75),
            'skewness': stats.skew(clean_array),
            'kurtosis': stats.kurtosis(clean_array)
        }


def batch_compute_metrics(reference_dir: str, test_dir: str, 
                         output_file: Optional[str] = None) -> Dict[str, List[float]]:
    """
    Compute metrics for all image pairs in two directories.
    
    Args:
        reference_dir: Directory containing reference images
        test_dir: Directory containing test images
        output_file: Optional file to save results
        
    Returns:
        dict: Dictionary containing lists of metric values
    """
    metrics_calculator = ImageQualityMetrics()
    
    # Get matching image files
    ref_files = sorted([f for f in os.listdir(reference_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    test_files = sorted([f for f in os.listdir(test_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    
    # Find common files
    common_files = list(set(ref_files) & set(test_files))
    common_files.sort()
    
    if not common_files:
        raise ValueError("No common image files found in both directories")
    
    print(f"Processing {len(common_files)} image pairs...")
    
    results = {'PSNR': [], 'SSIM': [], 'LPIPS': [], 'filenames': []}
    
    for filename in common_files:
        ref_path = os.path.join(reference_dir, filename)
        test_path = os.path.join(test_dir, filename)
        
        try:
            metrics = metrics_calculator.compute_all_metrics(ref_path, test_path)
            results['PSNR'].append(metrics.get('PSNR'))
            results['SSIM'].append(metrics.get('SSIM'))
            results['LPIPS'].append(metrics.get('LPIPS'))
            results['filenames'].append(filename)
            
            print(f"Processed {filename}: PSNR={metrics.get('PSNR', 'N/A'):.2f}, "
                  f"SSIM={metrics.get('SSIM', 'N/A'):.3f}, LPIPS={metrics.get('LPIPS', 'N/A'):.3f}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results['PSNR'].append(None)
            results['SSIM'].append(None)
            results['LPIPS'].append(None)
            results['filenames'].append(filename)
    
    # Save results if requested
    if output_file:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    return results
