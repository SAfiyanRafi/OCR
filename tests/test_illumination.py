"""
Unit tests for illumination and shadow correction algorithms.
"""

import cv2
import numpy as np
from app.preprocessing.illumination import correct_illumination_morphology, correct_illumination_gaussian


def test_morphological_illumination_correction(synthetic_cnic_image):
    # Apply synthetic shadow gradient across right side of image
    h, w = synthetic_cnic_image.shape[:2]
    shadow_map = np.tile(np.linspace(1.0, 0.3, w), (h, 1))
    shadowed = (synthetic_cnic_image * shadow_map[:, :, None]).astype(np.uint8)

    corrected = correct_illumination_morphology(shadowed, kernel_size=51)
    assert corrected.shape == shadowed.shape
    
    # Check that dark right-side brightness is restored closer to original
    mean_orig_right = np.mean(synthetic_cnic_image[:, int(w*0.7):])
    mean_shad_right = np.mean(shadowed[:, int(w*0.7):])
    mean_corr_right = np.mean(corrected[:, int(w*0.7):])

    assert mean_corr_right > mean_shad_right


def test_gaussian_illumination_correction(synthetic_cnic_image):
    corrected = correct_illumination_gaussian(synthetic_cnic_image, sigma=30.0)
    assert corrected.shape == synthetic_cnic_image.shape
