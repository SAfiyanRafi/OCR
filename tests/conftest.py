"""
Pytest configuration and synthetic image generators.
"""

import pytest
import cv2
import numpy as np


@pytest.fixture
def synthetic_cnic_image() -> np.ndarray:
    """
    Generate a clean synthetic CNIC front image with text, numbers, and background card structure.
    Aspect ratio ~ 1.585 (856x540).
    """
    h, w = 540, 856
    # Light green / cyan document background
    img = np.ones((h, w, 3), dtype=np.uint8) * 240
    img[:, :, 1] = 248  # Slightly greenish

    # Draw card border frame
    cv2.rectangle(img, (20, 20), (w - 20, h - 20), (80, 120, 80), 3)

    # Draw header text
    cv2.putText(img, "ISLAMIC REPUBLIC OF PAKISTAN", (160, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 80, 20), 2)
    cv2.putText(img, "National Identity Card", (280, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)

    # Draw Name & CNIC Number
    cv2.putText(img, "Name: MUHAMMAD ALI", (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "Father Name: AHMAD KHAN", (200, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "Identity Number: 42101-1234567-1", (200, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 150), 2)

    # Draw Dates
    cv2.putText(img, "Date of Birth: 15.08.1990", (200, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.putText(img, "Date of Issue: 01.01.2020", (200, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.putText(img, "Date of Expiry: 01.01.2030", (500, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # Draw photo placeholder box
    cv2.rectangle(img, (40, 140), (170, 310), (100, 100, 100), -1)

    return img


@pytest.fixture
def synthetic_passport_image() -> np.ndarray:
    """
    Generate a synthetic passport page image with MRZ zone.
    Aspect ratio ~ 1.42 (1000x1420).
    """
    w, h = 1000, 1420
    img = np.ones((h, w, 3), dtype=np.uint8) * 245
    img[:, :, 0] = 235  # Slightly yellowish passport background

    # Outer border
    cv2.rectangle(img, (30, 30), (w - 30, h - 30), (50, 50, 50), 2)

    # Header
    cv2.putText(img, "PAKISTAN PASSPORT", (320, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 0), 3)

    # Fields
    cv2.putText(img, "Type / Code: P / PAK", (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "Passport No: AB1234567", (550, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 0, 0), 2)
    cv2.putText(img, "Surname: KHAN", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "Given Names: MUHAMMAD ALI", (80, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # MRZ Region at bottom (bottom 25%)
    mrz_y = int(h * 0.82)
    cv2.rectangle(img, (40, mrz_y), (w - 40, h - 50), (220, 220, 220), -1)

    # Monospaced MRZ characters
    cv2.putText(img, "P<PAKKHAN<<MUHAMMAD<ALI<<<<<<<<<<<<<<<<<<<<<", (60, mrz_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    cv2.putText(img, "AB12345674PAK9008154M3001018<<<<<<<<<<<<<<02", (60, mrz_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    return img


@pytest.fixture
def degraded_cnic_photo(synthetic_cnic_image) -> np.ndarray:
    """
    Embed synthetic CNIC inside a larger photo canvas with rotation, skew, background margin, and shadow.
    """
    canvas_h, canvas_w = 900, 1200
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 120  # Dark desk background

    # Place synthetic CNIC in center
    h, w = synthetic_cnic_image.shape[:2]
    y_off, x_off = 180, 170
    canvas[y_off:y_off+h, x_off:x_off+w] = synthetic_cnic_image

    # Add gradient shadow across right side
    shadow = np.tile(np.linspace(1.0, 0.4, canvas_w), (canvas_h, 1))
    canvas = (canvas * shadow[:, :, None]).astype(np.uint8)

    return canvas
