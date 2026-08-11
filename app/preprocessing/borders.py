"""
Border detection and scanner margin cropping module with configurable safety margin.
"""

from typing import Tuple
import cv2
import numpy as np


def remove_scanner_borders(
    image: np.ndarray,
    margin_percent: float = 1.0,
    dark_threshold: int = 25
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Detect black or solid scanner border margins and crop image to content.
    
    Returns:
        (cropped_image, (ymin, xmin, ymax, xmax))
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h, w = gray.shape[:2]
    
    # Non-black region mask
    mask = gray > dark_threshold
    
    # Find bounding box of non-black content
    coords = cv2.findNonZero(mask.astype(np.uint8))
    if coords is None:
        return image.copy(), (0, 0, h, w)

    x, y, bw, bh = cv2.boundingRect(coords)

    # Add safety margin
    margin_x = int(w * (margin_percent / 100.0))
    margin_y = int(h * (margin_percent / 100.0))

    xmin = max(0, x - margin_x)
    ymin = max(0, y - margin_y)
    xmax = min(w, x + bw + margin_x)
    ymax = min(h, y + bh + margin_y)

    cropped = image[ymin:ymax, xmin:xmax]
    return cropped, (ymin, xmin, ymax, xmax)
