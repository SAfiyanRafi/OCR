"""
Illumination and shadow correction module using background estimation.
"""

from typing import Tuple
import cv2
import numpy as np


def correct_illumination_morphology(
    image: np.ndarray,
    kernel_size: int = 51
) -> np.ndarray:
    """
    Equalize uneven lighting and remove shadows using morphological background estimation.
    Divides the original channel by estimated background to normalize slowly varying lighting.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    if image.ndim == 3:
        channels = cv2.split(image)
        corrected_channels = []
        for ch in channels:
            # Estimate background via morphological closing
            bg = cv2.morphologyEx(ch, cv2.MORPH_CLOSE, kernel)
            # Normalize channel: (ch / bg) * 255
            norm = cv2.divide(ch, bg, scale=255.0)
            corrected_channels.append(np.clip(norm, 0, 255).astype(np.uint8))
        return cv2.merge(corrected_channels)
    else:
        bg = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        norm = cv2.divide(image, bg, scale=255.0)
        return np.clip(norm, 0, 255).astype(np.uint8)


def correct_illumination_gaussian(
    image: np.ndarray,
    sigma: float = 30.0
) -> np.ndarray:
    """
    Correct uneven illumination using heavy Gaussian blur background estimation.
    """
    if image.ndim == 3:
        channels = cv2.split(image)
        corrected = []
        for ch in channels:
            bg = cv2.GaussianBlur(ch, (0, 0), sigmaX=sigma, sigmaY=sigma)
            bg = np.maximum(bg, 1)
            norm = cv2.divide(ch, bg, scale=200.0)
            corrected.append(np.clip(norm, 0, 255).astype(np.uint8))
        return cv2.merge(corrected)
    else:
        bg = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
        bg = np.maximum(bg, 1)
        norm = cv2.divide(image, bg, scale=200.0)
        return np.clip(norm, 0, 255).astype(np.uint8)


# Alias for backward compatibility
correct_illumination = correct_illumination_morphology
