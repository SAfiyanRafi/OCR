"""
Unit tests for Passport MRZ region extraction and preprocessing.
"""

from app.preprocessing.mrz import crop_mrz_region, preprocess_mrz


def test_passport_mrz_cropping_and_preprocessing(synthetic_passport_image):
    mrz_crop, bbox = crop_mrz_region(synthetic_passport_image)
    assert mrz_crop.size > 0
    assert mrz_crop.shape[0] < synthetic_passport_image.shape[0]

    preprocessed_mrz = preprocess_mrz(mrz_crop, target_char_height=35, binarize=True)
    assert preprocessed_mrz.size > 0
    # Output should be grayscale / binarized single channel or 2D array
    assert preprocessed_mrz.ndim in [2, 3]
