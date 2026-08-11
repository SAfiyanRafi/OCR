"""
Unit tests for Landmark & Template Alignment Fallback.
"""

import numpy as np
from app.preprocessing.landmark import LandmarkAligner


def test_landmark_template_alignment_fallback():
    dummy_img = np.zeros((800, 1200, 3), dtype=np.uint8)
    cv2_dummy = np.full((800, 1200, 3), 128, dtype=np.uint8)

    canonical_img, M_fw, M_inv, coord_tx, success = LandmarkAligner.align_to_template(
        cv2_dummy, document_type="cnic_front", target_width=2000, target_height=1261
    )

    assert canonical_img is not None
    assert canonical_img.shape[1] == 2000
    assert canonical_img.shape[0] == 1261
    assert M_fw.shape == (3, 3)
    assert M_inv.shape == (3, 3)
    assert coord_tx.canonical_size == (2000, 1261)
