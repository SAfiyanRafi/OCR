"""
Unit tests for input validation, format normalization, and EXIF orientation.
"""

import os
import io
import pytest
import numpy as np
from PIL import Image
from app.preprocessing.pipeline import Preprocessor
from app.preprocessing.orientation import correct_exif_orientation, read_exif_orientation


def test_normalize_numpy_input(synthetic_cnic_image):
    bgr, exif = Preprocessor.normalize_input_image(synthetic_cnic_image)
    assert isinstance(bgr, np.ndarray)
    assert bgr.shape == synthetic_cnic_image.shape
    assert bgr.ndim == 3
    assert bgr.shape[2] == 3


def test_normalize_pil_input(synthetic_cnic_image):
    pil_img = Image.fromarray(cv2_to_pil_rgb(synthetic_cnic_image))
    bgr, exif = Preprocessor.normalize_input_image(pil_img)
    assert isinstance(bgr, np.ndarray)
    assert bgr.shape == synthetic_cnic_image.shape


def test_normalize_bytes_input(synthetic_cnic_image):
    pil_img = Image.fromarray(cv2_to_pil_rgb(synthetic_cnic_image))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    bgr, exif = Preprocessor.normalize_input_image(img_bytes)
    assert isinstance(bgr, np.ndarray)
    assert bgr.ndim == 3


def test_rgba_input_flattening():
    # Create RGBA image with transparent alpha channel
    rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    rgba[:, :, :3] = 128  # gray
    rgba[:, :, 3] = 255   # solid alpha

    bgr, _ = Preprocessor.normalize_input_image(rgba)
    assert bgr.shape == (100, 100, 3)


def test_exif_rotations(synthetic_cnic_image):
    # Test EXIF orientation 6 (90 deg CW)
    corrected_6, applied_6 = correct_exif_orientation(synthetic_cnic_image, 6)
    assert applied_6 is True
    assert corrected_6.shape[0] == synthetic_cnic_image.shape[1]
    assert corrected_6.shape[1] == synthetic_cnic_image.shape[0]

    # Test EXIF orientation 3 (180 deg)
    corrected_3, applied_3 = correct_exif_orientation(synthetic_cnic_image, 3)
    assert applied_3 is True
    assert corrected_3.shape == synthetic_cnic_image.shape


def cv2_to_pil_rgb(img_bgr):
    import cv2
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
