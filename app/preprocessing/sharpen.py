"""
Sharpening module supporting controlled Unsharp Masking and Laplacian sharpening.
"""

import cv2
import numpy as np


def apply_unsharp_mask(
    image: np.ndarray,
    sigma: float = 1.0,
    amount: float = 0.4,
    threshold: int = 0
) -> np.ndarray:
    """
    Apply Unsharp Masking (USM) for subtle text stroke edge sharpening without halo artifacts.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    if threshold > 0:
        low_contrast_mask = np.abs(image.astype(np.float32) - blurred.astype(np.float32)) < threshold
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        sharpened[low_contrast_mask] = image[low_contrast_mask]
        return sharpened
    else:
        return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


# Alias for backward compatibility
sharpen_unsharp_mask = apply_unsharp_mask


def apply_laplacian_sharpen(image: np.ndarray, strength: float = 0.2) -> np.ndarray:
    """
    Apply mild Laplacian high-pass sharpening.
    """
    kernel = np.array([
        [0, -1, 0],
        [-1, 4 + strength, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    sharpened = cv2.filter2D(image, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)
