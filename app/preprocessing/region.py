"""
Region-specific document preprocessing module.
Allows applying separate preprocessing parameters to designated document zones (e.g., Urdu zone vs Numeric zone).
"""

from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
from .denoise import apply_denoise_bilateral
from .sharpen import apply_unsharp_mask
from .contrast import apply_clahe


def process_document_regions(
    image: np.ndarray,
    regions_config: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """
    Crop and process defined spatial regions within document.
    
    regions_config structure:
    {
      "urdu_zone": {
         "bbox_relative": [0.35, 0.15, 0.95, 0.45],
         "denoise": "mild",
         "threshold": false
      },
      ...
    }
    """
    h, w = image.shape[:2]
    processed_regions = {}

    for name, cfg in regions_config.items():
        bbox_rel = cfg.get("bbox_relative", [0, 0, 1, 1])
        ymin = int(round(h * bbox_rel[0]))
        xmin = int(round(w * bbox_rel[1]))
        ymax = int(round(h * bbox_rel[2]))
        xmax = int(round(w * bbox_rel[3]))

        crop = image[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            continue

        processed_crop = crop.copy()
        denoise_opt = cfg.get("denoise")
        if denoise_opt == "mild" or denoise_opt == "bilateral":
            processed_crop = apply_denoise_bilateral(processed_crop, d=5, sigma_color=30, sigma_space=30)

        contrast_opt = cfg.get("contrast")
        if contrast_opt:
            processed_crop = apply_clahe(processed_crop, clip_limit=1.5)

        sharpen_opt = cfg.get("sharpen")
        if sharpen_opt:
            processed_crop = apply_unsharp_mask(processed_crop, amount=0.3)

        processed_regions[name] = processed_crop

    return processed_regions
