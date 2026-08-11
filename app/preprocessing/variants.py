"""
Multi-candidate preprocessing variant generator module.
Creates multiple candidate images (Variants A through E) to test with OCR engine.
"""

from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from app.core.models import PreprocessingVariant
from .illumination import correct_illumination_morphology
from .contrast import apply_clahe
from .denoise import apply_denoise_bilateral
from .sharpen import apply_unsharp_mask
from .threshold import apply_otsu_threshold


def generate_candidate_variants(
    base_rectified_image: np.ndarray,
    base_name: str = "variant"
) -> List[PreprocessingVariant]:
    """
    Generate multiple candidate image variants from geometrically rectified image.
    """
    variants: List[PreprocessingVariant] = []

    # Variant A: Baseline rectified image
    variants.append(PreprocessingVariant(
        id="var_01",
        name=f"{base_name}_01_geometry_only",
        image=base_rectified_image.copy(),
        transformations=["perspective_transform", "deskew", "resolution_normalization"]
    ))

    # Variant B: Illumination corrected
    illum_img = correct_illumination_morphology(base_rectified_image, kernel_size=51)
    variants.append(PreprocessingVariant(
        id="var_02",
        name=f"{base_name}_02_illumination_corrected",
        image=illum_img,
        transformations=["perspective_transform", "deskew", "resolution_normalization", "illumination_correction"]
    ))

    # Variant C: CLAHE Contrast Enhanced
    clahe_img = apply_clahe(base_rectified_image, clip_limit=2.0, tile_grid_size=(8, 8))
    variants.append(PreprocessingVariant(
        id="var_03",
        name=f"{base_name}_03_clahe_enhanced",
        image=clahe_img,
        transformations=["perspective_transform", "deskew", "resolution_normalization", "clahe"]
    ))

    # Variant D: Mild Denoise + Mild Sharpening
    denoised = apply_denoise_bilateral(base_rectified_image, d=5, sigma_color=40, sigma_space=40)
    sharpened = apply_unsharp_mask(denoised, amount=0.4)
    variants.append(PreprocessingVariant(
        id="var_04",
        name=f"{base_name}_04_mild_denoise_sharpen",
        image=sharpened,
        transformations=["perspective_transform", "deskew", "resolution_normalization", "bilateral_denoise", "unsharp_mask"]
    ))

    # Variant E: Grayscale / Otsu Binarized
    if base_rectified_image.ndim == 3:
        gray = cv2.cvtColor(base_rectified_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = base_rectified_image.copy()

    otsu_img = apply_otsu_threshold(gray)
    otsu_bgr = cv2.cvtColor(otsu_img, cv2.COLOR_GRAY2BGR)
    variants.append(PreprocessingVariant(
        id="var_05",
        name=f"{base_name}_05_grayscale_binarized",
        image=otsu_bgr,
        transformations=["perspective_transform", "deskew", "resolution_normalization", "grayscale", "otsu_threshold"]
    ))

    return variants


# Alias for backward compatibility
generate_preprocessing_variants = generate_candidate_variants
