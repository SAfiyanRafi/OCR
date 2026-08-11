"""
Unit tests for denoising and sharpening operations.
"""

import cv2
import numpy as np
from app.preprocessing.denoise import apply_denoise_bilateral, apply_denoise_gaussian
from app.preprocessing.sharpen import apply_unsharp_mask, apply_laplacian_sharpen


def test_bilateral_denoise(synthetic_cnic_image):
    # Add Gaussian noise
    noise = np.random.normal(0, 15, synthetic_cnic_image.shape).astype(np.int16)
    noisy = np.clip(synthetic_cnic_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    denoised = apply_denoise_bilateral(noisy, d=5, sigma_color=40, sigma_space=40)
    assert denoised.shape == noisy.shape

    # Standard deviation of noisy image should be higher than denoised image
    assert np.std(denoised) < np.std(noisy)


def test_unsharp_mask_sharpening(synthetic_cnic_image):
    sharpened = apply_unsharp_mask(synthetic_cnic_image, sigma=1.0, amount=0.5)
    assert sharpened.shape == synthetic_cnic_image.shape
    assert sharpened.dtype == np.uint8
