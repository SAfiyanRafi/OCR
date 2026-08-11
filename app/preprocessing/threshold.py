"""
Optional binarization thresholding module (Otsu, Adaptive Gaussian, Adaptive Mean).
"""

from typing import Tuple
import cv2
import numpy as np


def apply_otsu_threshold(image: np.ndarray) -> np.ndarray:
    """
    Apply Otsu global automatic thresholding.
    Returns 8-bit single channel binary image (0 or 255).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarized


def apply_adaptive_gaussian_threshold(
    image: np.ndarray,
    block_size: int = 11,
    c: float = 2.0
) -> np.ndarray:
    """
    Apply Adaptive Gaussian Thresholding to handle local illumination gradients.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if block_size % 2 == 0:
        block_size += 1

    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )


def apply_adaptive_mean_threshold(
    image: np.ndarray,
    block_size: int = 11,
    c: float = 2.0
) -> np.ndarray:
    """
    Apply Adaptive Mean Thresholding.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if block_size % 2 == 0:
        block_size += 1

    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c
    )
