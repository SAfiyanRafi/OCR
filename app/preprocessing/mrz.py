"""
Specialized Passport Machine Readable Zone (MRZ) extraction and preprocessing module.
"""

from typing import Tuple, Optional
import cv2
import numpy as np
from .contrast import apply_clahe
from .threshold import apply_otsu_threshold


def crop_mrz_region(
    passport_image: np.ndarray,
    bbox_relative: Tuple[float, float, float, float] = (0.68, 0.0, 1.0, 1.0)
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop the bottom MRZ region of passport page based on relative bounding box [ymin, xmin, ymax, xmax].
    """
    h, w = passport_image.shape[:2]
    ymin = int(round(h * bbox_relative[0]))
    xmin = int(round(w * bbox_relative[1]))
    ymax = int(round(h * bbox_relative[2]))
    xmax = int(round(w * bbox_relative[3]))

    mrz_crop = passport_image[ymin:ymax, xmin:xmax]
    return mrz_crop, (ymin, xmin, ymax, xmax)


def preprocess_mrz(
    mrz_image: np.ndarray,
    target_char_height: int = 35,
    binarize: bool = True
) -> np.ndarray:
    """
    Dedicated MRZ preprocessing:
    1. Grayscale conversion
    2. Geometry scale normalization (upscaling small MRZ text lines)
    3. High-contrast CLAHE enhancement
    4. Otsu binarization (optional)
    """
    if mrz_image.size == 0:
        return mrz_image

    if mrz_image.ndim == 3:
        gray = cv2.cvtColor(mrz_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = mrz_image.copy()

    h, w = gray.shape[:2]
    # MRZ typically has 2 lines of 44 characters (or 3 lines of 30 characters)
    # Estimate current character height (h / 3)
    curr_char_h = max(1, h // 3)
    if curr_char_h < target_char_height:
        scale = float(target_char_height) / float(curr_char_h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Apply CLAHE to boost OCRB font contrast against background security pattern
    enhanced = apply_clahe(gray, clip_limit=3.0, tile_grid_size=(4, 4))

    if binarize:
        binarized = apply_otsu_threshold(enhanced)
        return binarized
    
    return enhanced
