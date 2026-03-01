"""
CA-CLIP: Compositional-Aware CLIP for Image Restoration

A package for training and using CA-CLIP with Jaccard-weighted contrastive learning.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from . import data
from . import models
from . import utils
from . import open_clip

__all__ = [
    "data",
    "models", 
    "utils",
    "open_clip",
]
