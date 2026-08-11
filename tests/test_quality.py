"""
Unit tests for image quality assessment metrics.
"""

import cv2
import numpy as np
from app.preprocessing.quality import (
    analyze_blur,
    analyze_brightness,
    analyze_contrast,
    detect_glare,
    detect_shadow,
    assess_image_quality
)


def test_blur_detection(synthetic_cnic_image):
    gray_sharp = cv2.cvtColor(synthetic_cnic_image, cv2.COLOR_BGR2GRAY)
    var_sharp, is_blurry_sharp, norm_sharp = analyze_blur(gray_sharp)
    assert not is_blurry_sharp
    assert var_sharp > 80.0

    # Apply heavy blur
    gray_blurry = cv2.GaussianBlur(gray_sharp, (25, 25), 0)
    var_blurry, is_blurry_blur, norm_blurry = analyze_blur(gray_blurry)
    assert is_blurry_blur
    assert var_blurry < 80.0


def test_brightness_analysis():
    # Dark image
    dark_img = np.ones((100, 100), dtype=np.uint8) * 20
    b_dark = analyze_brightness(dark_img)
    assert b_dark.underexposed is True

    # Overexposed image
    bright_img = np.ones((100, 100), dtype=np.uint8) * 250
    b_bright = analyze_brightness(bright_img)
    assert b_bright.overexposed is True


def test_contrast_analysis():
    # Low contrast gray box
    low_contrast = np.ones((100, 100), dtype=np.uint8) * 128
    low_contrast[40:60, 40:60] = 135
    c_low = analyze_contrast(low_contrast)
    assert c_low.is_low_contrast is True

    # High contrast box
    high_contrast = np.zeros((100, 100), dtype=np.uint8)
    high_contrast[20:80, 20:80] = 255
    c_high = analyze_contrast(high_contrast)
    assert c_high.is_low_contrast is False


def test_glare_detection(synthetic_cnic_image):
    # Add severe glare patch in middle
    glare_img = synthetic_cnic_image.copy()
    h, w = glare_img.shape[:2]
    glare_img[int(h*0.3):int(h*0.6), int(w*0.3):int(w*0.6)] = 255
    
    ratio, glare_over_text = detect_glare(glare_img)
    assert ratio > 0.02
    assert glare_over_text is True


def test_shadow_detection(synthetic_cnic_image):
    gray = cv2.cvtColor(synthetic_cnic_image, cv2.COLOR_BGR2GRAY)
    shadow_img = gray.copy()
    h, w = shadow_img.shape[:2]
    # Darken bottom right quadrant
    shadow_img[h//2:, w//2:] = (shadow_img[h//2:, w//2:] * 0.2).astype(np.uint8)

    shadow_detected = detect_shadow(shadow_img)
    assert bool(shadow_detected) is True


def test_assess_image_quality_master(synthetic_cnic_image):
    report = assess_image_quality(synthetic_cnic_image)
    assert report.overall_score > 0.60
    assert report.status in ["usable", "low_quality"]
