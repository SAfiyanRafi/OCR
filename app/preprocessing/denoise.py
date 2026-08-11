"""
Selective denoising module supporting Bilateral, Non-Local Means, Gaussian, and Median filters.
Edge-preserving filters are preferred to protect Urdu dots, diacritics, and fine passport characters.
"""

from typing import Optional
import cv2
import numpy as np


def apply_denoise_bilateral(
    image: np.ndarray,
    d: int = 7,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0
) -> np.ndarray:
    """
    Apply Bilateral Filter: smooths noise while maintaining sharp document text edges.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


# Alias for backward compatibility
denoise_bilateral = apply_denoise_bilateral


def apply_denoise_nlm(
    image: np.ndarray,
    h_luminance: float = 3.0,
    search_window: int = 21,
    block_size: int = 7
) -> np.ndarray:
    """
    Apply Non-Local Means Denoising.
    """
    if image.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(
            image, None, h_luminance, h_luminance, block_size, search_window
        )
    else:
        return cv2.fastNlMeansDenoising(
            image, None, h_luminance, block_size, search_window
        )


def apply_denoise_gaussian(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply mild Gaussian blur.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_denoise_median(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply Median filter to eliminate salt-and-pepper noise.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)
