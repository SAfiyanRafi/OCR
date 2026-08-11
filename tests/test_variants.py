"""
Unit tests for multi-candidate variant generation.
"""

from app.preprocessing.variants import generate_preprocessing_variants


def test_variant_generation(synthetic_cnic_image):
    variants = generate_preprocessing_variants(synthetic_cnic_image)
    assert len(variants) >= 5

    variant_names = [v.name for v in variants]
    assert "variant_01_geometry_only" in variant_names
    assert "variant_02_illumination_corrected" in variant_names
    assert "variant_03_clahe_enhanced" in variant_names
    assert "variant_04_mild_denoise_sharpen" in variant_names
    assert "variant_05_grayscale_binarized" in variant_names

    for v in variants:
        assert v.image.size > 0
        assert len(v.transformations) > 0
