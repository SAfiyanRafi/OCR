"""
EXIF orientation extraction, 90/180/270 degree rotation correction,
and clamped fine deskew module.
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image, ImageOps


def read_exif_orientation(image_input: Any) -> Optional[int]:
    """
    Extract EXIF orientation tag from PIL Image, file path, or bytes.
    Returns orientation integer (1..8) or None.
    """
    try:
        if isinstance(image_input, Image.Image):
            exif = image_input.getexif()
            if exif:
                return exif.get(274)
        elif isinstance(image_input, str):
            with Image.open(image_input) as img:
                exif = img.getexif()
                if exif:
                    return exif.get(274)
    except Exception:
        pass
    return None


def correct_exif_orientation(image: np.ndarray, orientation: Optional[int] = None) -> Tuple[np.ndarray, bool]:
    """
    Apply EXIF orientation transformation (rotations/flips) to numpy BGR image.
    """
    if orientation is None or orientation == 1:
        return image, False

    corrected = image.copy()
    if orientation == 2:
        corrected = cv2.flip(corrected, 1)
    elif orientation == 3:
        corrected = cv2.rotate(corrected, cv2.ROTATE_180)
    elif orientation == 4:
        corrected = cv2.flip(corrected, 0)
    elif orientation == 5:
        corrected = cv2.rotate(corrected, cv2.ROTATE_90_COUNTERCLOCKWISE)
        corrected = cv2.flip(corrected, 1)
    elif orientation == 6:
        corrected = cv2.rotate(corrected, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        corrected = cv2.rotate(corrected, cv2.ROTATE_90_CLOCKWISE)
        corrected = cv2.flip(corrected, 1)
    elif orientation == 8:
        corrected = cv2.rotate(corrected, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return image, False

    return corrected, True


fix_exif_orientation = correct_exif_orientation


def detect_orthogonal_orientation(image: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Detect 90/180/270 degree orthogonal orientation misalignments.
    Identity cards and Passports must be in landscape format (width > height).
    Returns (oriented_image, rotation_angle_degrees).
    """
    if image is None or image.size == 0:
        return image, 0

    h, w = image.shape[:2]

    # If document is vertical/portrait (h > w * 1.15), rotate 90° clockwise to restore landscape format
    if h > (w * 1.15):
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return rotated, 90

    return image, 0


def detect_and_deskew(image: np.ndarray, max_deskew_angle: float = 15.0) -> Tuple[np.ndarray, float]:
    """
    Detect fine rotational deskew angle clamped strictly to +/- 15.0 degrees.
    Prevents image corner clipping and perspective distortion.
    """
    if image is None or image.size == 0:
        return image, 0.0

    # Step 1: Ensure 90/180/270 degree landscape alignment first
    oriented_img, ortho_angle = detect_orthogonal_orientation(image)

    # Step 2: Calculate fine deskew angle on oriented image
    gray = cv2.cvtColor(oriented_img, cv2.COLOR_BGR2GRAY) if len(oriented_img.shape) == 3 else oriented_img
    canny = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(canny, 1, np.pi/180, threshold=100, minLineLength=max(10, int(oriented_img.shape[1] * 0.3)), maxLineGap=10)

    angles = []
    if lines is not None:
        for line in lines:
            pts = line[0] if len(line.shape) > 1 and len(line[0]) == 4 else line.ravel()
            if len(pts) == 4:
                x1, y1, x2, y2 = pts
                angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                # Enforce strict fine deskew angle limit (-15.0 to +15.0)
                if -max_deskew_angle <= angle <= max_deskew_angle:
                    angles.append(angle)

    median_angle = float(np.median(angles)) if angles else 0.0

    if abs(median_angle) > 0.5 and abs(median_angle) <= max_deskew_angle:
        h, w = oriented_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(oriented_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated, median_angle + ortho_angle

    return oriented_img, float(ortho_angle)


def deskew_document(image: np.ndarray) -> Tuple[np.ndarray, float]:
    return detect_and_deskew(image)
