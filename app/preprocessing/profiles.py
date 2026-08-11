"""
Field-Specific Preprocessing Profiles.
Applies specialized image transformations tailored for specific field types:
standard, urdu_safe, numeric, date, passport_number, and mrz.
"""

from typing import Dict, Any
import cv2
import numpy as np
from app.preprocessing.contrast import apply_clahe
from app.preprocessing.denoise import apply_denoise_bilateral
from app.preprocessing.sharpen import apply_unsharp_mask
from app.preprocessing.threshold import apply_otsu_threshold


def preprocess_field_profile(image: np.ndarray, profile: str = "standard") -> np.ndarray:
    """
    Apply specialized preprocessing profile to cropped Field ROI image.
    """
    if image is None or image.size == 0:
        return image

    prof = profile.lower().strip()

    if prof == "urdu_safe":
        # Preserve intricate Nastaliq/Urdu stroke details while reducing paper grain noise
        denoised = apply_denoise_bilateral(image, d=5, sigma_color=30, sigma_space=30)
        sharpened = apply_unsharp_mask(denoised, amount=0.3)
        return sharpened

    elif prof == "numeric":
        # High-contrast binarization for CNIC & numbers
        clahe_img = apply_clahe(image, clip_limit=3.0, tile_grid_size=(4, 4))
        gray = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2GRAY) if clahe_img.ndim == 3 else clahe_img
        otsu_img = apply_otsu_threshold(gray)
        return cv2.cvtColor(otsu_img, cv2.COLOR_GRAY2BGR)

    elif prof == "date":
        # Date string contrast enhancement & deskew alignment
        clahe_img = apply_clahe(image, clip_limit=2.5, tile_grid_size=(4, 4))
        sharpened = apply_unsharp_mask(clahe_img, amount=0.5)
        return sharpened

    elif prof == "passport_number":
        # Alphanumeric high-contrast sharpening
        clahe_img = apply_clahe(image, clip_limit=2.5, tile_grid_size=(8, 8))
        sharpened = apply_unsharp_mask(clahe_img, amount=0.6)
        return sharpened

    elif prof == "mrz":
        # OCR-B high-contrast grayscale binarization
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        clahe_gray = apply_clahe(gray, clip_limit=3.5, tile_grid_size=(8, 8))
        denoised = apply_denoise_bilateral(clahe_gray, d=3, sigma_color=25, sigma_space=25)
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    else:
        # Standard profile: mild contrast equalization
        return apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8))
