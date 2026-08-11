"""
Unit tests for Field-Specific Preprocessing Profiles.
"""

import numpy as np
from app.preprocessing.profiles import preprocess_field_profile


def test_field_preprocessing_profiles():
    dummy_img = np.full((100, 300, 3), 150, dtype=np.uint8)

    standard = preprocess_field_profile(dummy_img, profile="standard")
    assert standard.shape == dummy_img.shape

    urdu = preprocess_field_profile(dummy_img, profile="urdu_safe")
    assert urdu.shape == dummy_img.shape

    numeric = preprocess_field_profile(dummy_img, profile="numeric")
    assert numeric.shape == dummy_img.shape

    passport_no = preprocess_field_profile(dummy_img, profile="passport_number")
    assert passport_no.shape == dummy_img.shape
