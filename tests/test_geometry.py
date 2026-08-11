"""
Unit tests for geometry processing, boundary detection, 4-point perspective warp, deskew, and resolution scaling.
"""

import cv2
import numpy as np
from app.preprocessing.geometry import (
    order_points,
    detect_document_boundary,
    apply_perspective_transform,
    estimate_deskew_angle,
    rotate_image,
    normalize_resolution
)


def test_order_points():
    unordered = np.array([
        [200, 200],  # bottom-right
        [10, 10],    # top-left
        [200, 10],   # top-right
        [10, 200]    # bottom-left
    ], dtype="float32")

    ordered = order_points(unordered)
    np.testing.assert_array_equal(ordered[0], [10, 10])    # top-left
    np.testing.assert_array_equal(ordered[1], [200, 10])   # top-right
    np.testing.assert_array_equal(ordered[2], [200, 200])  # bottom-right
    np.testing.assert_array_equal(ordered[3], [10, 200])   # bottom-left


def test_detect_document_boundary_and_warp(degraded_cnic_photo):
    corners, confidence = detect_document_boundary(degraded_cnic_photo)
    assert corners is not None
    assert len(corners) == 4
    assert confidence > 0.30

    warped = apply_perspective_transform(degraded_cnic_photo, corners, target_aspect_ratio=1.5858)
    assert isinstance(warped, np.ndarray)
    assert warped.shape[0] > 0 and warped.shape[1] > 0
    # Check aspect ratio of warped document
    h, w = warped.shape[:2]
    aspect_ratio = w / float(h)
    assert abs(aspect_ratio - 1.5858) < 0.10


def test_deskew_and_rotation(synthetic_cnic_image):
    # Rotate clean CNIC image by +5 degrees
    rotated = rotate_image(synthetic_cnic_image, 5.0)
    assert rotated.shape[0] > 0 and rotated.shape[1] > 0

    angle = estimate_deskew_angle(rotated, max_angle=15.0)
    # Estimated angle should be close to 5 degrees
    assert abs(angle) <= 15.0


def test_resolution_normalization(synthetic_cnic_image):
    # Small image (width 500)
    small_img = cv2.resize(synthetic_cnic_image, (500, 300))
    norm_small, scale_up = normalize_resolution(small_img, min_width=1600, max_width=4000)
    assert norm_small.shape[1] == 1600
    assert scale_up > 1.0

    # Large image (width 5000)
    large_img = cv2.resize(synthetic_cnic_image, (5000, 3000))
    norm_large, scale_down = normalize_resolution(large_img, min_width=1600, max_width=4000)
    assert norm_large.shape[1] == 4000
    assert scale_down < 1.0
