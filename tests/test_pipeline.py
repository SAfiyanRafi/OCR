"""
Integration tests for end-to-end Preprocessor pipeline.
"""

import os
import pytest
from app.preprocessing import Preprocessor


def test_full_pipeline_cnic(synthetic_cnic_image, tmp_path):
    preprocessor = Preprocessor()
    debug_dir = str(tmp_path / "debug")

    result = preprocessor.process(
        input_data=synthetic_cnic_image,
        document_type="cnic_front",
        debug=True,
        debug_dir=debug_dir
    )

    assert result.original_image is not None
    assert result.preprocessed_image is not None
    assert result.best_image is not None
    assert result.quality.overall_score > 0.0
    assert len(result.variants) >= 5

    # Check debug image files
    assert os.path.exists(os.path.join(debug_dir, "01_original.jpg"))
    assert os.path.exists(os.path.join(debug_dir, "10_final.jpg"))


def test_full_pipeline_passport(synthetic_passport_image):
    preprocessor = Preprocessor()
    result = preprocessor.process(
        input_data=synthetic_passport_image,
        document_type="passport",
        debug=False
    )

    assert result.mrz_image is not None
    assert result.mrz_image.size > 0
    assert result.document_type == "passport"
