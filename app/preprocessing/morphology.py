"""
Morphological operations module (opening, closing, erosion, dilation).
"""

import cv2
import numpy as np


def apply_morphology(
    image: np.ndarray,
    operation: str = "opening",
    kernel_size: int = 3,
    shape: int = cv2.MORPH_RECT
) -> np.ndarray:
    """
    Apply morphological operation to image.
    
    Operations: "opening", "closing", "erosion", "dilation"
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(shape, (kernel_size, kernel_size))

    op_map = {
        "opening": cv2.MORPH_OPEN,
        "closing": cv2.MORPH_CLOSE,
        "erosion": cv2.MORPH_ERODE,
        "dilation": cv2.MORPH_DILATE
    }

    op_code = op_map.get(operation.lower(), cv2.MORPH_OPEN)
    return cv2.morphologyEx(image, op_code, kernel)
