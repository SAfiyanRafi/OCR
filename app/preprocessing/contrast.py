"""
Contrast enhancement module using global histogram stretching and adaptive CLAHE.
"""

from typing import Tuple
import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    For color images, converts to LAB color space and applies CLAHE to Luminance (L) channel.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        lab_enhanced = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(image)


def apply_global_contrast(image: np.ndarray, alpha: float = 1.2, beta: float = 0) -> np.ndarray:
    """
    Apply global linear contrast scaling: result = alpha * image + beta.
    """
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def normalize_contrast_histogram(image: np.ndarray) -> np.ndarray:
    """
    Normalize image contrast using min-max histogram stretching (5th to 95th percentile).
    """
    if image.ndim == 3:
        channels = cv2.split(image)
        norm_channels = []
        for ch in channels:
            p5, p95 = np.percentile(ch, (5, 95))
            if p95 > p5:
                stretched = np.clip((ch - p5) * 255.0 / (p95 - p5), 0, 255).astype(np.uint8)
            else:
                stretched = ch
            norm_channels.append(stretched)
        return cv2.merge(norm_channels)
    else:
        p5, p95 = np.percentile(image, (5, 95))
        if p95 > p5:
            return np.clip((image - p5) * 255.0 / (p95 - p5), 0, 255).astype(np.uint8)
        return image.copy()


# Alias for backward compatibility
enhance_contrast_clahe = apply_clahe
