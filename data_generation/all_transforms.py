"""
Image transformation module for data augmentation in image restoration tasks.
This module contains various transformations for geometric distortions, photometric effects,
weather conditions, and noise patterns.
"""
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import to_tensor, to_pil_image
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter
from skimage import color
from skimage.transform import swirl
from PIL import Image

# Import custom libraries for weather effects
from lib.weather.lime import LIME
from lib.weather.fog_gen import fogAttenuation
from lib.weather.snow_gen import SnowGenUsingNoise
from lib.weather.rain_gen import RainGenUsingNoise
from lib.weather.gen_utils import (
    screen_blend,
    layer_blend,
    alpha_blend,
    illumination2opacity,
    reduce_lightHSV,
    scale_depth
)
from lib.cloud_gen.CloudSimulator import add_cloud_and_shadow
from lib.raindrop.dropgenerator import generateDrops, generate_label
from lib.raindrop.config import cfg


# ------------------ Geometric Distortions ------------------

class ElasticDeformation(torch.nn.Module):
    """
    Applies elastic deformation to an image using Gaussian-filtered displacement fields.

    Args:
        alpha (float): Scaling factor for displacement intensity.
        sigma (float): Standard deviation for Gaussian smoothing.
    """
    def __init__(self, alpha=None, sigma=None):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma

    def forward(self, img):
        # Randomize parameters
        if self.alpha is None:
            self.alpha = random.randint(10, 20)
        if self.sigma is None:
            self.sigma = random.randint(20, 30)
        
        # Convert image to tensor format (C, H, W)
        img = to_tensor(img)
        c, h, w = img.shape

        # Generate random displacement fields
        dx = torch.randn(h, w) * 2 - 1
        dy = torch.randn(h, w) * 2 - 1
        
        # Center the displacement fields to have zero mean
        # This prevents overall translation of the image
        dx = dx - dx.mean()
        dy = dy - dy.mean()

        # Apply Gaussian filtering (smoothing)
        dx = TF.gaussian_blur(dx.unsqueeze(0).unsqueeze(0), 
                             kernel_size=(self.sigma * 2 + 1, self.sigma * 2 + 1)).squeeze()
        dy = TF.gaussian_blur(dy.unsqueeze(0).unsqueeze(0), 
                             kernel_size=(self.sigma * 2 + 1, self.sigma * 2 + 1)).squeeze()

        # Scale displacement
        dx *= self.alpha
        dy *= self.alpha

        # Create coordinate grid
        x_grid, y_grid = torch.meshgrid(torch.arange(w, dtype=torch.float32), 
                                        torch.arange(h, dtype=torch.float32))
        x_grid, y_grid = x_grid.T, y_grid.T  # Transpose to match image dimensions

        # Apply displacement
        x_grid = torch.clamp(x_grid + dx, 0, w - 1)
        y_grid = torch.clamp(y_grid + dy, 0, h - 1)

        # Normalize for grid_sample (range must be [-1, 1])
        x_grid = (x_grid / (w - 1)) * 2 - 1
        y_grid = (y_grid / (h - 1)) * 2 - 1

        # Stack and reshape for grid_sample
        grid = torch.stack((x_grid, y_grid), dim=-1).unsqueeze(0)  # Shape: (1, H, W, 2)

        # Apply warping using grid_sample
        img_deformed = F.grid_sample(img.unsqueeze(0), grid, mode="bilinear", align_corners=False)

        return to_pil_image(img_deformed.squeeze(0).clamp(0, 1))


class Refraction(torch.nn.Module):
    """
    Simulates refraction effect by distorting the image using random displacement maps.
    
    Args:
        strength_range (tuple): Range of possible refraction strengths (min, max).
    """
    def __init__(self, strength = 40):
        super().__init__()
        self.strength = strength

    def forward(self, img):
        # Convert to numpy array if needed
        img = np.array(img)
        h, w = img.shape[:2]
        
        # Generate random strength value
        if self.strength is None:
            self.strength = random.uniform(20, 80)
        
        # Create smooth displacement maps using Gaussian filter
        dx = ndimage.gaussian_filter(np.random.randn(h, w), sigma=10) * self.strength
        dy = ndimage.gaussian_filter(np.random.randn(h, w), sigma=10) * self.strength
        
        # Create coordinate maps
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply displacements and ensure coordinates stay within bounds
        map_x = np.clip(map_x + dx, 0, w - 1).astype(np.float32)
        map_y = np.clip(map_y + dy, 0, h - 1).astype(np.float32)
        
        # Remap the image using the displacement maps
        return Image.fromarray(cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR))


class MotionBlur(torch.nn.Module):
    """
    Applies directional motion blur to simulate camera or object movement.
    Can apply depth-aware blur if depth map is provided.
    
    Args:
        kernel_size (int): Size of the blur kernel.
    """
    def __init__(self, kernel_size=None):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, img, depth_map=None):
        # Convert input to numpy array regardless of input type
        if isinstance(img, torch.Tensor):
            # Convert tensor to numpy array
            if img.ndim == 3 and img.shape[0] in [1, 3]:  # CHW format
                img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            else:  # Assume HWC format
                img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        elif isinstance(img, Image.Image):
            img_np = np.array(img)
        elif isinstance(img, np.ndarray):
            img_np = img.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
            
        # Randomize kernel size and axis
        if self.kernel_size is None:
            self.kernel_size = random.randint(5, 10)
        
        # Ensure kernel_size is odd
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
            
        axis = random.choice(['horizontal', 'vertical', 'diagonal_right', 'diagonal_left'])
        
        # Create kernel based on chosen axis
        kernel = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float32)
        if axis == 'horizontal':
            kernel[int((self.kernel_size - 1) / 2), :] = np.ones(self.kernel_size)
        elif axis == 'vertical':
            kernel[:, int((self.kernel_size - 1) / 2)] = np.ones(self.kernel_size)
        elif axis == 'diagonal_right':
            np.fill_diagonal(kernel, 1)
        else:  # diagonal_left
            np.fill_diagonal(np.fliplr(kernel), 1)
        
        # Normalize kernel
        kernel /= self.kernel_size

        # If no depth map, apply uniform blur
        if depth_map is None:
            result_np = cv2.filter2D(img_np, -1, kernel)
        else:
            # Convert depth map to numpy array if needed
            if isinstance(depth_map, torch.Tensor):
                depth_np = depth_map.squeeze().cpu().numpy()
            elif isinstance(depth_map, Image.Image):
                depth_np = np.array(depth_map)
            else:
                depth_np = depth_map

            # Convert depth map to same size as image if needed
            if depth_np.shape[:2] != img_np.shape[:2]:
                depth_np = cv2.resize(depth_np, (img_np.shape[1], img_np.shape[0]))

            # Normalize depth map to [0,1]
            if depth_np.max() > 0:  # Avoid division by zero
                depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-6)

            # Randomly choose whether to blur foreground or background
            blur_foreground = random.choice([True, False])
            threshold = random.uniform(0.3, 0.7)  # Random threshold to split fore/background

            # Create mask based on depth
            if blur_foreground:
                mask = (depth_np > threshold).astype(np.float32)  # Blur foreground
            else:
                mask = (depth_np <= threshold).astype(np.float32)  # Blur background

            # Handle single channel mask vs 3-channel image
            if mask.ndim == 2 and img_np.ndim == 3:
                mask = np.expand_dims(mask, axis=2)
                if img_np.shape[2] > mask.shape[2]:
                    mask = np.repeat(mask, img_np.shape[2], axis=2)

            # Apply blur to original image
            blurred = cv2.filter2D(img_np, -1, kernel)

            # Blend original and blurred based on mask
            result_np = img_np * (1 - mask) + blurred * mask
        
        # Ensure result is in uint8 format
        result_np = result_np.clip(0, 255).astype(np.uint8)
        
        # Always return a PIL Image
        return Image.fromarray(result_np)


# ------------------ Photometric Distortions ------------------

class LowLight(torch.nn.Module):
    """
    Simulates low-light conditions by reducing image brightness.
    
    Args:
        factor_range (tuple): Range of brightness reduction factors (min, max).
        factor (float, optional): Specific darkness factor to use (overrides random sampling).
    """
    def __init__(self, factor_range=(0.4, 0.9), factor=None):
        super().__init__()
        self.factor = factor
        self.factor_range = factor_range

    def forward(self, img):
        # Use provided factor or randomly select darkness factor
        if self.factor is None:
            self.factor = random.uniform(*self.factor_range)
        
        # Multiply pixel values to reduce brightness
        return Image.fromarray(np.clip(np.array(img) * self.factor, 0, 255).astype(np.uint8))


# class ColorJitterTransform(torch.nn.Module):
#     """
#     Applies random hue shift to simulate color distortion.
    
#     Args:
#         hue (float, optional): Specific hue value to use (overrides random sampling).
#     """
#     def __init__(self, hue=None):
#         super().__init__()
#         self.hue = hue

#     def forward(self, img):
#         # Use provided hue or randomize hue parameter
#         if self.hue is None:
#             self.hue = 0.5
        
#         # Apply color jitter transform
#         transform = T.ColorJitter(hue=self.hue)
#         return transform(img)

class ColorJitterTransform(torch.nn.Module):
    """
    Applies color distortion by shifting RGB channels and adjusting color balance.
    
    Args:
        color_shift (float, optional): Intensity of color channel shifting (0.0 to 1.0).
        color_cast (str, optional): Type of color cast to apply ('warm', 'cool', 'green', 'magenta', 'random').
    """
    def __init__(self, color_shift=None, color_cast=None):
        super().__init__()
        self.color_shift = color_shift
        self.color_cast = color_cast

    def forward(self, img):
        # Convert to numpy array
        if isinstance(img, Image.Image):
            img_array = np.array(img).astype(np.float32) / 255.0
        elif isinstance(img, torch.Tensor):
            img_array = img.permute(1, 2, 0).cpu().numpy()
        else:
            img_array = img.astype(np.float32) / 255.0
        
        # Randomize parameters if not provided
        if self.color_shift is None:
            self.color_shift = random.uniform(0.1, 0.4)
        
        if self.color_cast is None:
            self.color_cast = random.choice(['warm', 'cool', 'green', 'magenta', 'cyan', 'yellow'])
        
        # Apply color channel shifting
        shifted_img = self._apply_channel_shift(img_array)
        
        # Apply color cast
        colored_img = self._apply_color_cast(shifted_img)
        
        # Convert back to PIL Image
        colored_img = np.clip(colored_img * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(colored_img)
    
    def _apply_channel_shift(self, img_array):
        """Apply random shifts to RGB channels independently."""
        shifted = img_array.copy()
        
        # Generate random shifts for each channel
        r_shift = random.uniform(-self.color_shift, self.color_shift)
        g_shift = random.uniform(-self.color_shift, self.color_shift)
        b_shift = random.uniform(-self.color_shift, self.color_shift)
        
        # Apply shifts
        shifted[:, :, 0] = np.clip(shifted[:, :, 0] + r_shift, 0, 1)  # Red
        shifted[:, :, 1] = np.clip(shifted[:, :, 1] + g_shift, 0, 1)  # Green
        shifted[:, :, 2] = np.clip(shifted[:, :, 2] + b_shift, 0, 1)  # Blue
        
        return shifted
    
    def _apply_color_cast(self, img_array):
        """Apply a color cast to simulate different lighting conditions."""
        cast_intensity = random.uniform(0.1, 0.3)
        
        color_matrices = {
            'warm': np.array([1.2, 1.0, 0.8]) * cast_intensity + (1 - cast_intensity),
            'cool': np.array([0.8, 1.0, 1.2]) * cast_intensity + (1 - cast_intensity),
            'green': np.array([0.9, 1.3, 0.9]) * cast_intensity + (1 - cast_intensity),
            'magenta': np.array([1.2, 0.8, 1.1]) * cast_intensity + (1 - cast_intensity),
            'cyan': np.array([0.8, 1.1, 1.2]) * cast_intensity + (1 - cast_intensity),
            'yellow': np.array([1.2, 1.2, 0.7]) * cast_intensity + (1 - cast_intensity)
        }
        
        # Apply the selected color cast
        cast_matrix = color_matrices[self.color_cast]
        colored = img_array * cast_matrix.reshape(1, 1, 3)
        
        return np.clip(colored, 0, 1)


class ContrastTransform(torch.nn.Module):
    """
    Applies random contrast adjustments to the image.
    
    Args:
        contrast (float, optional): Specific contrast value to use (overrides random sampling).
    """
    def __init__(self, contrast=None):
        super().__init__()
        self.contrast = contrast

    def forward(self, img):
        # Use provided contrast or randomize contrast parameter
        if self.contrast is None:
            self.contrast = random.uniform(0.4, 1.0)
        
        # Apply contrast transform
        transform = T.ColorJitter(contrast=self.contrast)
        return transform(img)


class SaturationTransform(torch.nn.Module):
    """
    Applies random saturation adjustments to the image.
    
    Args:
        saturation (float, optional): Specific saturation value to use (overrides random sampling).
    """
    def __init__(self, saturation=None):
        super().__init__()
        self.saturation = saturation

    def forward(self, img):
        # Use provided saturation or randomize saturation parameter
        if self.saturation is None:
            self.saturation = random.uniform(0.4, 1.0)
        
        # Apply saturation transform
        transform = T.ColorJitter(saturation=self.saturation)
        return transform(img)


# ------------------ Occlusion Effects ------------------

class AddHaze(torch.nn.Module):
    """
    Adds haze effect based on depth information.
    
    Args:
        max_opacity (float): Maximum opacity of the haze effect.
    """
    def __init__(self, max_opacity=None):
        super().__init__()
        self.max_opacity = max_opacity

    def forward(self, img, depth_map=None):
        # Randomize opacity
        if self.max_opacity is None:
            self.max_opacity = random.uniform(0.65, 0.9)
        
        # Convert input image to tensor if it's PIL Image
        if isinstance(img, Image.Image):
            img = to_tensor(img)# Convert NumPy array to tensor if needed
        elif isinstance(img, np.ndarray):
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        
        # Handle depth map
        if depth_map is None:
            # Create uniform depth if not provided
            depth_map = torch.ones((1, img.shape[1], img.shape[2])) * 0.5
        else:
            # Convert depth map to tensor if needed
            if isinstance(depth_map, np.ndarray):
                depth_map = torch.from_numpy(depth_map).float()
            elif isinstance(depth_map, Image.Image):
                depth_map = to_tensor(depth_map)
            
            # Ensure depth map is 3D tensor (C,H,W)
            if depth_map.ndim == 2:
                depth_map = depth_map.unsqueeze(0)

        # Resize depth map to match image dimensions
        if depth_map.shape[1:] != img.shape[1:]:
            depth_map = F.interpolate(
                depth_map.unsqueeze(0),
                size=img.shape[1:],
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # Scale depth map by opacity
        depth_scaled = depth_map * self.max_opacity

        # Create haze layer (white)
        haze_layer = torch.ones_like(img)

        # Expand depth map to match image channels if needed
        if depth_scaled.shape[0] == 1:
            depth_scaled = depth_scaled.expand_as(img)

        # Apply haze effect using alpha blending
        hazed = img * (1 - depth_scaled) + haze_layer * depth_scaled
        
        # Convert back to PIL Image
        return to_pil_image(hazed.clamp(0, 1))


class AddRainAndFog(torch.nn.Module):
    """
    Adds realistic rain streaks and fog effects to an image, using depth information when available.
    """
    def __init__(self):
        super().__init__()
        self.raingen = RainEffectGenerator()
        
    def forward(self, img, depth_map=None):
        """
        Args:
            img: PIL.Image or ndarray
            depth_map: ndarray or tensor

        Returns:
            PIL.Image: rainy and foggy output image
        """
        if isinstance(img, Image.Image):
            img = np.array(img)

        if isinstance(depth_map, Image.Image):
            depth_map = np.array(depth_map)

        if depth_map is None:
            # Default flat depth if missing
            depth_map = np.ones(img.shape[:2], dtype=np.float32) * 0.5

        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.squeeze().cpu().numpy()

        # Resize depth to match if necessary
        if img.shape[:2] != depth_map.shape:
            depth_map = np.array(Image.fromarray(depth_map).resize((img.shape[1], img.shape[0])))

        # Apply rain and fog generation
        rainy_img = self.raingen.genEffectFromArrays(img, depth_map)

        return to_pil_image(to_tensor(rainy_img).clamp(0, 1))


# class RainEffectGenerator:
#     """
#     Helper class for generating realistic rain effects with fog and illumination adjustment.
#     """
#     def __init__(self):
#         self._lime = LIME(iterations=25, alpha=1.0)
#         # Adjust these parameters for better visibility
#         self._illumination2darkness = {0: 1, 1: 0.98, 2: 0.95, 3: 0.9}  # Less darkening
#         self._weather2visibility = (3000, 5000)  # Higher visibility range
#         self._illumination2fogcolor = {0: (150, 180), 1: (180, 200), 2: (200, 240), 3: (200, 240)}
#         self._rain_layer_gen = RainGenUsingNoise()

#     def getIlluminationMapCheat(self, img: np.ndarray) -> np.ndarray:
#         """Extract grayscale illumination map from RGB image."""
#         return color.rgb2gray(img)

#     def genRainLayer(self, h=720, w=1280):
#         """Generate rain streak layer with random parameters."""
#         blur_angle = random.choice([-1, 1]) * random.randint(60, 90)
#         layer_large = self._rain_layer_gen.genRainLayer(
#             h=h, w=w, noise_scale=random.uniform(0.35, 0.55),
#             noise_amount=0.2, zoom_layer=random.uniform(1.0, 3.5),
#             blur_kernel_size=random.choice([15, 17, 19, 21, 23]), blur_angle=blur_angle
#         )
#         layer_small = self._rain_layer_gen.genRainLayer(
#             h=h, w=w, noise_scale=random.uniform(0.35, 0.55),
#             noise_amount=0.15, zoom_layer=random.uniform(1.0, 3.5),
#             blur_kernel_size=random.choice([7, 9, 11, 13]), blur_angle=blur_angle
#         )
#         layer = layer_blend(layer_small, layer_large)

#         # Resize if necessary
#         if (h, w) != layer.shape:
#             layer = np.asarray(Image.fromarray(layer).resize((w, h)))
#         return layer

#     def genEffectFromArrays(self, I: np.ndarray, D: np.ndarray):
#         """
#         Generate rain and fog effects on an image using depth information.
        
#         Args:
#             I: Input RGB image as numpy array
#             D: Depth map as numpy array
        
#         Returns:
#             np.ndarray: Image with rain and fog effects
#         """
#         hI, wI, _ = I.shape
#         hD, wD = D.shape

#         # Scale depth map if needed
#         if hI != hD or wI != wD:
#             D = scale_depth(D, hI, wI)

#         # Analyze illumination
#         T = self.getIlluminationMapCheat(I)
#         illumination_array = np.histogram(T, bins=4, range=(0, 1))[0] / (T.size)
#         illumination = illumination_array.argmax()

#         # Apply fog based on illumination
#         if illumination > 0:
#             visibility = random.randint(self._weather2visibility[0], self._weather2visibility[1])
#             fog_color = random.randint(*self._illumination2fogcolor[illumination])
            
#             # Reduce the darkening effect to improve visibility
#             darkness_factor = self._illumination2darkness[illumination]
#             # Apply less darkening for distant objects
#             D_adjusted = D * 0.7  # Scale depth to reduce fog effect on background
            
#             I_dark = reduce_lightHSV(I, sat_red=darkness_factor, val_red=darkness_factor)
#             I_fog = fogAttenuation(I_dark, D_adjusted, visibility=visibility, fog_color=fog_color)
#         else:
#             fog_color = 100  # Lighter fog color
#             visibility = min(D.max() * 1.5, 2000) if D.max() < 1000 else 1500  # Higher visibility
#             I_fog = fogAttenuation(I, D * 0.7, visibility=visibility, fog_color=fog_color)

#         # Add rain effect
#         alpha = illumination2opacity(I, illumination) * random.uniform(0.3, 0.5)
#         rain_layer = self.genRainLayer(h=hI, w=wI)
#         I_rain = alpha_blend(I_fog, rain_layer, alpha)

#         return I_rain.astype(np.uint8)

class RainEffectGenerator:
    """
    Helper class for generating realistic rain effects with minimal fog and lighting changes.
    """
    def __init__(self):
        self._lime = LIME(iterations=25, alpha=1.0)
        # Minimal darkening - keep original brightness
        self._illumination2darkness = {0: 1.0, 1: 1.0, 2: 0.98, 3: 0.95}  # Minimal darkening
        self._weather2visibility = (8000, 15000)  # Much higher visibility range
        self._illumination2fogcolor = {0: (220, 255), 1: (230, 255), 2: (240, 255), 3: (240, 255)}  # Very light fog
        self._rain_layer_gen = RainGenUsingNoise()

    def getIlluminationMapCheat(self, img: np.ndarray) -> np.ndarray:
        """Extract grayscale illumination map from RGB image."""
        return color.rgb2gray(img)

    def genRainLayer(self, h=720, w=1280):
        """Generate rain streak layer with random parameters."""
        blur_angle = random.choice([-1, 1]) * random.randint(60, 90)
        layer_large = self._rain_layer_gen.genRainLayer(
            h=h, w=w, noise_scale=random.uniform(0.35, 0.55),
            noise_amount=0.2, zoom_layer=random.uniform(1.0, 3.5),
            blur_kernel_size=random.choice([15, 17, 19, 21, 23]), blur_angle=blur_angle
        )
        layer_small = self._rain_layer_gen.genRainLayer(
            h=h, w=w, noise_scale=random.uniform(0.35, 0.55),
            noise_amount=0.15, zoom_layer=random.uniform(1.0, 3.5),
            blur_kernel_size=random.choice([7, 9, 11, 13]), blur_angle=blur_angle
        )
        layer = layer_blend(layer_small, layer_large)

        # Resize if necessary
        if (h, w) != layer.shape:
            layer = np.asarray(Image.fromarray(layer).resize((w, h)))
        return layer

    def genEffectFromArrays(self, I: np.ndarray, D: np.ndarray):
        """
        Generate rain effects with minimal fog on an image using depth information.
        
        Args:
            I: Input RGB image as numpy array
            D: Depth map as numpy array
        
        Returns:
            np.ndarray: Image with rain effects and minimal fog
        """
        hI, wI, _ = I.shape
        hD, wD = D.shape

        # Scale depth map if needed
        if hI != hD or wI != wD:
            D = scale_depth(D, hI, wI)

        # Analyze illumination
        T = self.getIlluminationMapCheat(I)
        illumination_array = np.histogram(T, bins=4, range=(0, 1))[0] / (T.size)
        illumination = illumination_array.argmax()

        # Apply minimal fog only to very distant objects
        visibility = random.randint(self._weather2visibility[0], self._weather2visibility[1])
        fog_color = random.randint(*self._illumination2fogcolor[illumination])
        
        # Apply fog only to the most distant parts (top 10% of depth values)
        depth_threshold = np.percentile(D, 90)  # Only affect the most distant 10%
        D_masked = np.where(D > depth_threshold, D * 0.3, 0)  # Very light fog only on distant objects
        
        # Minimal darkening
        darkness_factor = self._illumination2darkness[illumination]
        if darkness_factor < 1.0:
            I_dark = reduce_lightHSV(I, sat_red=darkness_factor, val_red=darkness_factor)
            I_fog = fogAttenuation(I_dark, D_masked, visibility=visibility, fog_color=fog_color)
        else:
            # No darkening, just minimal fog on distant objects
            I_fog = fogAttenuation(I, D_masked, visibility=visibility, fog_color=fog_color)

        # Add rain effect with reduced opacity
        alpha = illumination2opacity(I, illumination) * random.uniform(0.2, 0.4)  # Reduced rain opacity
        rain_layer = self.genRainLayer(h=hI, w=wI)
        I_rain = alpha_blend(I_fog, rain_layer, alpha)

        return I_rain.astype(np.uint8)


class AddSnowAndFog(torch.nn.Module):
    """
    Adds realistic snow and fog effects to an image, using depth information when available.
    """
    def __init__(self):
        super().__init__()
        self.snowgen = SnowEffectGenerator()
        
    def forward(self, img, depth_map=None):
        """
        Args:
            img: PIL.Image or ndarray
            depth_map: ndarray (expected to be 2D or matching img H, W)

        Returns:
            PIL.Image: snowy and foggy image
        """
        if isinstance(img, Image.Image):
            img = np.array(img)

        if isinstance(depth_map, Image.Image):
            depth_map = np.array(depth_map)

        if depth_map is None:
            # Default flat depth if not provided
            depth_map = np.ones(img.shape[:2], dtype=np.float32) * 0.5
        
        # If depth_map is tensor, convert to numpy
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.squeeze().cpu().numpy()

        # Resize depth map to match image if necessary
        if img.shape[:2] != depth_map.shape:
            depth_map = np.array(Image.fromarray(depth_map).resize((img.shape[1], img.shape[0])))

        # Apply snow and fog effect
        snowy_img = self.snowgen.genEffectFromArrays(img, depth_map)

        return to_pil_image(to_tensor(snowy_img).clamp(0, 1))


# class SnowEffectGenerator:
#     """
#     Helper class for generating realistic snow effects with fog and illumination adjustment.
#     """
#     def __init__(self):
#         self._lime = LIME(iterations=25, alpha=1.0)
#         # Adjust these parameters for better visibility
#         self._illumination2darkness = {0: 1, 1: 0.95, 2: 0.9, 3: 0.85}  # Less darkening
#         self._weather2visibility = (2000, 4000)  # Higher visibility range
#         self._illumination2fogcolor = {0: (150, 180), 1: (180, 200), 2: (200, 240), 3: (200, 240)}
#         self._snow_layer_gen = SnowGenUsingNoise()

#     def getIlluminationMapCheat(self, img: np.ndarray) -> np.ndarray:
#         """Extract grayscale illumination map from RGB image."""
#         return color.rgb2gray(img)

#     def genSnowLayer(self, h=720, w=1280):
#         """Generate snow particle layer with random parameters."""
#         num_itr_small = 2
#         num_itr_large = 1
#         blur_angle = random.choice([-1, 1])*random.randint(60, 90)
#         layer_small = self._snow_layer_gen.genSnowMultiLayer(h=h, w=w, blur_angle=blur_angle, 
#                                                            intensity="small", num_itr=num_itr_small)
#         layer_large = self._snow_layer_gen.genSnowMultiLayer(h=h, w=w, blur_angle=blur_angle, 
#                                                            intensity="large", num_itr=num_itr_large)
#         layer = layer_blend(layer_small, layer_large)
        
#         # Resize if necessary
#         hl, wl = layer.shape
#         if h != hl or w != wl:
#             layer = np.asarray(Image.fromarray(layer).resize((w, h)))
#         return layer

#     def genEffectFromArrays(self, I: np.ndarray, D: np.ndarray):
#         """
#         Generate snow and fog effects on an image using depth information.
        
#         Args:
#             I: Input RGB image as numpy array
#             D: Depth map as numpy array
        
#         Returns:
#             np.ndarray: Image with snow and fog effects
#         """
#         hI, wI, _ = I.shape
#         hD, wD = D.shape
        
#         # Scale depth map if needed
#         if hI != hD or wI != wD:
#             D = scale_depth(D, hI, wI)

#         # Analyze illumination
#         T = self.getIlluminationMapCheat(I)
#         illumination_array = np.histogram(T, bins=4, range=(0, 1))[0] / (T.size)
#         illumination = illumination_array.argmax()

#         # Apply fog based on illumination
#         if illumination > 0:
#             visibility = random.randint(self._weather2visibility[0], self._weather2visibility[1])
#             fog_color = random.randint(*self._illumination2fogcolor[illumination])
            
#             # Reduce the darkening effect
#             darkness_factor = self._illumination2darkness[illumination]
#             # Apply less darkening for distant objects
#             D_adjusted = D * 0.6  # Scale depth to reduce fog effect on background
            
#             I_dark = reduce_lightHSV(I, sat_red=darkness_factor, val_red=darkness_factor)
#             I_fog = fogAttenuation(I_dark, D_adjusted, visibility=visibility, fog_color=fog_color)
#         else:
#             fog_color = 100  # Lighter fog color
#             visibility = min(D.max() * 1.5, 2000) if D.max() < 1000 else 1500  # Higher visibility
#             I_fog = fogAttenuation(I, D * 0.6, visibility=visibility, fog_color=fog_color)

#         # Add snow effect
#         snow_layer = self.genSnowLayer(h=hI, w=wI)
#         I_snow = screen_blend(I_fog, snow_layer)
#         return I_snow.astype(np.uint8)
class SnowEffectGenerator:
    """
    Helper class for generating realistic snow effects with minimal fog and lighting changes.
    """
    def __init__(self):
        self._lime = LIME(iterations=25, alpha=1.0)
        # Minimal darkening - keep original brightness
        self._illumination2darkness = {0: 1.0, 1: 1.0, 2: 0.98, 3: 0.95}  # Minimal darkening
        self._weather2visibility = (10000, 20000)  # Much higher visibility range
        self._illumination2fogcolor = {0: (230, 255), 1: (240, 255), 2: (245, 255), 3: (245, 255)}  # Very light fog
        self._snow_layer_gen = SnowGenUsingNoise()

    def getIlluminationMapCheat(self, img: np.ndarray) -> np.ndarray:
        """Extract grayscale illumination map from RGB image."""
        return color.rgb2gray(img)

    def genSnowLayer(self, h=720, w=1280):
        """Generate snow particle layer with random parameters."""
        num_itr_small = 2
        num_itr_large = 1
        blur_angle = random.choice([-1, 1])*random.randint(60, 90)
        layer_small = self._snow_layer_gen.genSnowMultiLayer(h=h, w=w, blur_angle=blur_angle, 
                                                           intensity="small", num_itr=num_itr_small)
        layer_large = self._snow_layer_gen.genSnowMultiLayer(h=h, w=w, blur_angle=blur_angle, 
                                                           intensity="large", num_itr=num_itr_large)
        layer = layer_blend(layer_small, layer_large)
        
        # Resize if necessary
        hl, wl = layer.shape
        if h != hl or w != wl:
            layer = np.asarray(Image.fromarray(layer).resize((w, h)))
        return layer

    def genEffectFromArrays(self, I: np.ndarray, D: np.ndarray):
        """
        Generate snow effects with minimal fog on an image using depth information.
        
        Args:
            I: Input RGB image as numpy array
            D: Depth map as numpy array
        
        Returns:
            np.ndarray: Image with snow effects and minimal fog
        """
        hI, wI, _ = I.shape
        hD, wD = D.shape
        
        # Scale depth map if needed
        if hI != hD or wI != wD:
            D = scale_depth(D, hI, wI)

        # Analyze illumination
        T = self.getIlluminationMapCheat(I)
        illumination_array = np.histogram(T, bins=4, range=(0, 1))[0] / (T.size)
        illumination = illumination_array.argmax()

        # Apply minimal fog only to very distant objects
        visibility = random.randint(self._weather2visibility[0], self._weather2visibility[1])
        fog_color = random.randint(*self._illumination2fogcolor[illumination])
        
        # Apply fog only to the most distant parts (top 5% of depth values)
        depth_threshold = np.percentile(D, 95)  # Only affect the most distant 5%
        D_masked = np.where(D > depth_threshold, D * 0.2, 0)  # Very light fog only on distant objects
        
        # Minimal darkening
        darkness_factor = self._illumination2darkness[illumination]
        if darkness_factor < 1.0:
            I_dark = reduce_lightHSV(I, sat_red=darkness_factor, val_red=darkness_factor)
            I_fog = fogAttenuation(I_dark, D_masked, visibility=visibility, fog_color=fog_color)
        else:
            # No darkening, just minimal fog on distant objects
            I_fog = fogAttenuation(I, D_masked, visibility=visibility, fog_color=fog_color)

        # Add snow effect
        snow_layer = self.genSnowLayer(h=hI, w=wI)
        I_snow = screen_blend(I_fog, snow_layer)
        return I_snow.astype(np.uint8)


class AddClouds(torch.nn.Module):
    """
    Adds realistic cloud and cloud shadow effects to the image.
    """
    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        Args:
            img: PIL Image, numpy array or tensor
        
        Returns:
            PIL.Image: Image with added clouds
        """
        # Randomize parameters
        max_lvl = (random.uniform(0.7, 1.0), random.uniform(0.85, 1.0))
        min_lvl = (random.uniform(0.0, 0.1), random.uniform(0.0, 0.1))
        shadow_max_lvl = [random.uniform(0.2, 0.5), random.uniform(0.4, 0.7)]
        noise_type = 'perlin'
        decay_factor = random.uniform(0.5, 2.5)    
        channel_offset = random.randint(1, 3)
        channel_magnitude_shift = random.uniform(0.01, 0.1)
        blur_scaling = random.uniform(1.0, 3.0)
        cloud_color = random.choice([True, False])

        # Convert to tensor if it's PIL or np.array
        if isinstance(img, np.ndarray):
            img = torch.FloatTensor(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        elif isinstance(img, Image.Image):
            img = T.ToTensor()(img).unsqueeze(0)
        elif isinstance(img, torch.Tensor):
            if img.ndim == 3:
                img = img.unsqueeze(0)
        else:
            raise TypeError("Unsupported image type")

        # Apply cloud effect
        cloudy_tensor = add_cloud_and_shadow(
            img,
            max_lvl=max_lvl,
            min_lvl=min_lvl,
            shadow_max_lvl=shadow_max_lvl,
            noise_type=noise_type,
            decay_factor=decay_factor,
            channel_offset=channel_offset,
            channel_magnitude_shift=channel_magnitude_shift,
            blur_scaling=blur_scaling,
            cloud_color=cloud_color
        )

        # Return as PIL image
        cloudy_img = T.ToPILImage()(cloudy_tensor.squeeze(0).clamp(0, 1))
        return cloudy_img


# ------------------ Noise and Resolution Transformations ------------------

class GaussianNoise(torch.nn.Module):
    """
    Adds Gaussian noise to the image.
    
    Args:
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.
    """
    def __init__(self, mean=None, std=None):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, img):
        # Ensure img is a tensor
        if isinstance(img, Image.Image):
            img = to_tensor(img)
        elif isinstance(img, np.ndarray):
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        # Randomize parameters if None
        if self.std is None:
            self.std = random.uniform(0.05, 0.1)
        if self.mean is None:
            self.mean = 0
        
        # Generate and add Gaussian noise
        noise = torch.randn_like(img) * self.std + self.mean
        
        # Apply noise and clamp values
        noisy_img = torch.clamp(img + noise, 0, 1)
        
        # Return as PIL image
        return to_pil_image(noisy_img)


# class Pixelate(torch.nn.Module):
#     """
#     Simulates compression artifacts by reducing image resolution and then upscaling.artifacts by reducing image resolution and then upscaling.
    
#     Args:
#         scale (float): Downscaling factor.scaling factor.
#     """
#     def __init__(self, scale=None):
#         super().__init__()
#         self.scale = scale

#     def forward(self, img):
#         # Randomize scale each time (between 5 and 10)
#         if self.scale is None:
#             self.scale = random.uniform(1.1, 3.0)

#         # Convert to PIL Image if necessary
#         if isinstance(img, torch.Tensor):
#             img = T.ToPILImage()(img) 
#         elif isinstance(img, np.ndarray):
#             img = Image.fromarray(img)

#         # Get width and height
#         w, h = img.size  # Note: PIL gives (width, height)  

#         # Downscale and upscale to simulate pixelation
#         small = img.resize((int(w // self.scale), int(h // self.scale)), Image.BILINEAR) 
#         resized = small.resize((w, h), Image.NEAREST)
#         return resized

class Pixelate(torch.nn.Module):
    """
    Simulates downsampling for super-resolution by reducing image resolution cleanly.
    This creates low-resolution images that can be used as input for super-resolution models.
    
    Args:
        scale (float): Downscaling factor (e.g., 2.0 means half the resolution).
    """
    def __init__(self, scale=None):
        super().__init__()
        self.scale = scale

    def forward(self, img):
        # Randomize scale factor if not provided
        if self.scale is None:
            self.scale = random.uniform(2.0, 4.0)  # Common super-resolution scales

        # Convert to PIL Image if necessary
        if isinstance(img, torch.Tensor):
            img = T.ToPILImage()(img)
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        # Get original dimensions
        w, h = img.size  # PIL gives (width, height)

        # Calculate new dimensions for downsampling
        new_w = int(w // self.scale)
        new_h = int(h // self.scale)
        
        # Ensure minimum size of 1x1
        new_w = max(1, new_w)
        new_h = max(1, new_h)

        # Downsample using high-quality resampling (bicubic for smooth downsampling)
        # This simulates the effect of a lower resolution sensor or display
        downsampled = img.resize((new_w, new_h), Image.BICUBIC)
        
        # For super-resolution training, we typically want to return the low-res image
        # without upsampling it back. However, if you need it back at original size
        # for visualization purposes, uncomment the next two lines:
        
        # # Upscale back to original size using bicubic interpolation
        # upsampled = downsampled.resize((w, h), Image.BICUBIC)
        # return upsampled
        
        # Return the downsampled image (this is what super-resolution models would use as input)
        return downsampled


class DefocusBlur(torch.nn.Module):
    """
    Applies Gaussian blur to simulate defocus or lens blur.to simulate defocus or lens blur.
    
    Args:    Args:
        blur_strength (int): Controls the size of the blur kernel.: Controls the size of the blur kernel.
    """
    def __init__(self, blur_strength=None):
        super().__init__()
        self.blur_strength = blur_strength

    def forward(self, img):
        # Randomize blur strength
        if self.blur_strength is None:
            self.blur_strength = random.randint(2, 18)

        # Convert to PIL if tensor
        if isinstance(img, torch.Tensor):
            img = T.ToPILImage()(img)

        img = np.array(img)

        # Ensure blur_strength is odd and reasonable
        ksize = max(3, int(self.blur_strength))
        if ksize % 2 == 0: 
            ksize += 1 

        # Apply Gaussian blurr
        blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=0, sigmaY=0)
        return Image.fromarray(blurred)


class Overexposure(torch.nn.Module):
    """
    Simulates overexposure by increasing brightness and reducing contrast,
    with optional blown-out highlights.
    
    Args:
        exposure_factor (float): Factor to increase brightness (>1.0 for overexposure)
        highlight_threshold (float): Threshold above which pixels become blown out (0-1)
        preserve_highlights (bool): Whether to preserve some detail in highlights
    """
    def __init__(self, exposure_factor=None, highlight_threshold=None, preserve_highlights=True):
        super().__init__()
        self.exposure_factor = exposure_factor
        self.highlight_threshold = highlight_threshold
        self.preserve_highlights = preserve_highlights

    def forward(self, img):
        # Convert to numpy array and normalize to [0, 1]
        if isinstance(img, Image.Image):
            img_array = np.array(img).astype(np.float32) / 255.0
        else:
            img_array = np.array(img).astype(np.float32) / 255.0
        
        # Randomize parameters if not provided
        if self.exposure_factor is None:
            self.exposure_factor = random.uniform(1.0, 1.5)  # Overexposure range
        if self.highlight_threshold is None:
            self.highlight_threshold = random.uniform(0.4, 0.9)
        
        # Apply exposure adjustment (gamma correction)
        gamma = 1.0 / self.exposure_factor
        exposed_img = np.power(img_array, gamma)
        
        # Add brightness boost
        brightness_boost = (self.exposure_factor - 1.0) * 0.3
        exposed_img = exposed_img + brightness_boost
        
        # Handle blown-out highlights
        if not self.preserve_highlights:
            # Clip highlights to pure white (blown out)
            highlight_mask = exposed_img > self.highlight_threshold
            exposed_img[highlight_mask] = 1.0
        else:
            # Soft clipping to preserve some detail
            highlight_mask = exposed_img > self.highlight_threshold
            excess = exposed_img - self.highlight_threshold
            # Apply soft compression to highlights
            compressed_excess = excess / (1.0 + excess)
            exposed_img = np.where(highlight_mask, 
                                 self.highlight_threshold + compressed_excess * (1.0 - self.highlight_threshold),
                                 exposed_img)
        
        # Reduce contrast slightly (characteristic of overexposure)
        contrast_reduction = 0.8 + (self.exposure_factor - 1.5) * 0.1
        exposed_img = (exposed_img - 0.5) * contrast_reduction + 0.5
        
        # Clip to valid range and convert back
        exposed_img = np.clip(exposed_img, 0, 1)
        result = (exposed_img * 255).astype(np.uint8)
        
        return Image.fromarray(result)


class Underexposure(torch.nn.Module):
    """
    Simulates underexposure by reducing brightness and increasing noise in shadows,
    with optional crushed blacks.
    
    Args:
        exposure_factor (float): Factor to reduce brightness (<1.0 for underexposure)
        shadow_threshold (float): Threshold below which pixels become crushed (0-1)
        noise_intensity (float): Amount of noise to add in dark areas
        crush_shadows (bool): Whether to crush shadows to pure black
    """
    def __init__(self, exposure_factor=None, shadow_threshold=None, noise_intensity=None, crush_shadows=False):
        super().__init__()
        self.exposure_factor = exposure_factor
        self.shadow_threshold = shadow_threshold
        self.noise_intensity = noise_intensity
        self.crush_shadows = crush_shadows

    def forward(self, img):
        # Convert to numpy array and normalize to [0, 1]
        if isinstance(img, Image.Image):
            img_array = np.array(img).astype(np.float32) / 255.0
        else:
            img_array = np.array(img).astype(np.float32) / 255.0
        
        # Randomize parameters if not provided
        if self.exposure_factor is None:
            self.exposure_factor = random.uniform(0.5, 0.9)  # Underexposure range
        if self.shadow_threshold is None:
            self.shadow_threshold = random.uniform(0.1, 0.3)
        if self.noise_intensity is None:
            self.noise_intensity = random.uniform(0.02, 0.08)
        
        # Apply exposure adjustment (gamma correction)
        gamma = 1.0 / self.exposure_factor
        exposed_img = np.power(img_array, gamma)
        
        # Reduce overall brightness
        brightness_reduction = (1.0 - self.exposure_factor) * 0.4
        exposed_img = exposed_img - brightness_reduction
        
        # Handle crushed shadows
        if self.crush_shadows:
            # Clip shadows to pure black (crushed)
            shadow_mask = exposed_img < self.shadow_threshold
            exposed_img[shadow_mask] = 0.0
        else:
            # Soft clipping to preserve some shadow detail
            shadow_mask = exposed_img < self.shadow_threshold
            # Apply soft compression to shadows
            compressed_shadows = exposed_img[shadow_mask] / (self.shadow_threshold + 1e-6)
            compressed_shadows = compressed_shadows * self.shadow_threshold
            exposed_img[shadow_mask] = compressed_shadows
        
        # Add noise in dark areas (characteristic of underexposure)
        if self.noise_intensity > 0:
            # Create noise mask - more noise in darker areas
            noise_mask = 1.0 - exposed_img  # Inverted luminance
            noise_mask = np.mean(noise_mask, axis=2, keepdims=True) if len(noise_mask.shape) == 3 else noise_mask
            
            # Generate noise
            noise = np.random.normal(0, self.noise_intensity, exposed_img.shape)
            
            # Apply noise more strongly to dark areas
            weighted_noise = noise * noise_mask
            exposed_img = exposed_img + weighted_noise
        
        # Increase contrast slightly in midtones (to compensate for darkness)
        contrast_boost = 1.1 + (1.0 - self.exposure_factor) * 0.2
        exposed_img = (exposed_img - 0.5) * contrast_boost + 0.5
        
        # Clip to valid range and convert back
        exposed_img = np.clip(exposed_img, 0, 1)
        result = (exposed_img * 255).astype(np.uint8)
        
        return Image.fromarray(result)


class AddRaindrops(torch.nn.Module):
    """
    Adds realistic raindrops to the image using the raindrop generator.
    
    Args:
        max_drops (int): Maximum number of raindrops to generate
        min_drops (int): Minimum number of raindrops to generate
        max_radius (int): Maximum radius of raindrops
        min_radius (int): Minimum radius of raindrops
        edge_darkratio (float): Brightness reduction factor for drop edges
    """
    def __init__(self, max_drops=None, min_drops=None, max_radius=None, min_radius=None, edge_darkratio=None):
        super().__init__()
        self.max_drops = max_drops
        self.min_drops = min_drops
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.edge_darkratio = edge_darkratio

    def forward(self, img):
        """
        Args:
            img: PIL Image, numpy array or tensor
        
        Returns:
            PIL.Image: Image with added raindrops
        """
        # Convert to PIL Image if needed
        if isinstance(img, torch.Tensor):
            img = T.ToPILImage()(img)
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        
        # Randomize parameters if not provided
        if self.max_drops is None:
            self.max_drops = 60 #random.randint(10, 50)
        if self.min_drops is None:
            self.min_drops = 20 #random.randint(20, 80)
        if self.max_radius is None:
            self.max_radius = 50 #random.randint(80, 120)
        if self.min_radius is None:
            self.min_radius = 3 #random.randint(5, 15)
        if self.edge_darkratio is None:
            self.edge_darkratio = random.uniform(0.4, 0.8)

        # Create custom config for this instance
        custom_cfg = cfg.copy()
        custom_cfg['maxDrops'] = self.max_drops
        custom_cfg['minDrops'] = self.min_drops
        custom_cfg['maxR'] = self.max_radius
        custom_cfg['minR'] = self.min_radius
        custom_cfg['edge_darkratio'] = self.edge_darkratio
        custom_cfg['return_label'] = False  # We only want the image, not the label
        
        # Save image temporarily to generate raindrops
        import tempfile
        import os
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            img.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Generate raindrops and labels
            h, w = np.array(img).shape[:2]
            listFinalDrops, label_map = generate_label(h, w, custom_cfg)
            
            # Generate the image with raindrops
            result_img = generateDrops(tmp_path, custom_cfg, listFinalDrops)
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        return result_img



# Dictionary mapping transformation names to their corresponding PyTorch implementations# Dictionary mapping transformation names to their corresponding PyTorch implementations
transformation_functions = {
    # Geometric Distortions
    "Warping": ElasticDeformation(),
    "Refraction": Refraction(),
    "Motion Blur": MotionBlur(),

    # Photometric Distortionsc Distortions
    "Low Light": LowLight(),
    "Color Jitter": ColorJitterTransform(),
    "Underexposure": Underexposure(),
    "Overexposure": Overexposure(),
    "Contrast": ContrastTransform(),

    # Occlusions
    "Haze": AddHaze(),
    "Rain": AddRainAndFog(),
    "Snow": AddSnowAndFog(),
    "Clouds": AddClouds(),

    # Noise + Resolution Transformations
    "Gaussian Noise": GaussianNoise(),
    "Compression": Pixelate(),
    "Defocus Blur": DefocusBlur()
}