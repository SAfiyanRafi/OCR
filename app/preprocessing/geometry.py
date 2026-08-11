"""
Document Boundary Detection & Perspective Geometry Transformation.
Calculates 4-point homography and tracks forward/inverse transformation matrices.
Enforces strict 65% minimum area threshold and quad aspect ratio safeguards to prevent warping distortion.
"""

from typing import List, Tuple, Dict, Any, Optional, Union
import cv2
import numpy as np
from app.core.models import DocumentBoundary
from app.preprocessing.orientation import detect_and_deskew, detect_orthogonal_orientation


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 corner points in sequence: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by angle degrees."""
    if abs(angle) < 0.1:
        return image.copy()
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def detect_document_boundary(image: np.ndarray) -> DocumentBoundary:
    """
    Multi-strategy document boundary detector.
    Enforces minimum 65% image area coverage to prevent cropping internal barcode/photo boxes.
    """
    if image is None or image.size == 0:
        return DocumentBoundary(detected=False, confidence=0.0, method="failed")

    h, w = image.shape[:2]
    img_area = float(w * h)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    canny = cv2.Canny(blurred, 50, 150)
    dilated = cv2.dilate(canny, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in sorted_contours[:5]:
            area = cv2.contourArea(cnt)
            # SAFEGUARD: Contour MUST cover at least 65% of the total document image area
            if area < (img_area * 0.65):
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype("float32")
                ordered = order_points(pts)

                # Validate aspect ratio (landscape cards: 1.15 to 2.0)
                rect_w = np.linalg.norm(ordered[1] - ordered[0])
                rect_h = np.linalg.norm(ordered[3] - ordered[0])
                if rect_h > 0:
                    aspect = rect_w / rect_h
                    if 1.15 <= aspect <= 2.0:
                        conf = min(0.95, float(area / img_area) + 0.1)
                        corners_list = [(float(pt[0]), float(pt[1])) for pt in ordered]
                        return DocumentBoundary(
                            corners=corners_list,
                            confidence=conf,
                            detected=True,
                            method="contour"
                        )

    # Fallback to Full Image Bounds to prevent partial cropping
    full_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    corners_list = [(float(pt[0]), float(pt[1])) for pt in full_pts]
    return DocumentBoundary(
        corners=corners_list,
        confidence=0.50,
        detected=False,
        method="fallback_full_image"
    )


def warp_perspective(
    image: np.ndarray,
    boundary: Union[DocumentBoundary, List[Tuple[float, float]], np.ndarray],
    target_aspect_ratio: float = 1.5858
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform 4-point homography perspective warp with distortion safeguards.
    """
    h_img, w_img = image.shape[:2]

    corners = []
    if isinstance(boundary, DocumentBoundary):
        corners = boundary.corners
    elif isinstance(boundary, (list, tuple, np.ndarray)):
        corners = list(boundary)

    if not corners or len(corners) != 4:
        M = np.eye(3, dtype=np.float32)
        return image.copy(), M, M

    pts = np.array(corners, dtype="float32")
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_A), int(width_B))

    height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_A), int(height_B))

    # SAFEGUARD: Aspect ratio distortion check (if quad is degenerate, bypass warp)
    if max_height > 0:
        actual_aspect = float(max_width) / float(max_height)
        if abs(actual_aspect - target_aspect_ratio) / target_aspect_ratio > 0.35:
            M = np.eye(3, dtype=np.float32)
            return image.copy(), M, M

    if target_aspect_ratio > 0:
        max_height = int(max_width / target_aspect_ratio)

    max_width = max(100, max_width)
    max_height = max(100, max_height)

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M_forward = cv2.getPerspectiveTransform(rect, dst)
    M_inverse = cv2.getPerspectiveTransform(dst, rect)

    warped = cv2.warpPerspective(image, M_forward, (max_width, max_height))

    return warped, M_forward, M_inverse


def apply_perspective_transform(
    image: np.ndarray,
    boundary: Union[DocumentBoundary, List[Tuple[float, float]], np.ndarray],
    target_aspect_ratio: float = 1.5858
) -> np.ndarray:
    warped, _, _ = warp_perspective(image, boundary, target_aspect_ratio)
    return warped


def estimate_deskew_angle(image: np.ndarray, max_angle: float = 15.0) -> float:
    _, angle = detect_and_deskew(image, max_deskew_angle=max_angle)
    return angle


def detect_document_boundary_and_warp(image: np.ndarray, target_aspect_ratio: float = 1.5858) -> Tuple[np.ndarray, bool]:
    b = detect_document_boundary(image)
    warped = apply_perspective_transform(image, b.corners, target_aspect_ratio)
    return warped, b.detected


def normalize_resolution(
    image: np.ndarray,
    min_dim: int = 1600,
    max_dim: int = 4000,
    min_width: Optional[int] = None,
    max_width: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    target_min = min_width if min_width is not None else min_dim
    target_max = max_width if max_width is not None else max_dim

    h, w = image.shape[:2]

    if w < target_min:
        scale = float(target_min) / float(w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return resized, scale
    elif w > target_max:
        scale = float(target_max) / float(w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    return image, 1.0


def transform_bbox_inverse(
    bbox_px: List[float],
    inverse_matrix: Optional[np.ndarray] = None
) -> List[float]:
    if inverse_matrix is None or np.array_equal(inverse_matrix, np.eye(3)):
        return bbox_px

    x1, y1, x2, y2 = bbox_px
    pts = np.array([
        [[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]
    ], dtype=np.float32)

    transformed = cv2.perspectiveTransform(pts, inverse_matrix)
    xs = transformed[:, 0, 0]
    ys = transformed[:, 0, 1]

    return [float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))]
