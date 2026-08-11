"""
Unit tests for Canonical Coordinate Transform and Point/BBox mappings.
"""

import numpy as np
from app.core.models import CoordinateTransform
from app.preprocessing.geometry import (
    transform_bbox_canonical_to_original,
    transform_bbox_original_to_canonical,
    transform_point_canonical_to_original,
    transform_point_original_to_canonical
)


def test_coordinate_transform_bidirectional_mapping():
    M_forward = np.array([
        [2.0, 0.0, 10.0],
        [0.0, 2.0, 20.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    M_inverse = np.linalg.inv(M_forward)

    coord_tx = CoordinateTransform(
        original_to_canonical=M_forward,
        canonical_to_original=M_inverse,
        original_size=(1000, 1000),
        canonical_size=(2000, 2000)
    )

    pt_orig = (100.0, 100.0)
    pt_canon = transform_point_original_to_canonical(pt_orig, M_forward)
    assert round(pt_canon[0], 1) == 210.0
    assert round(pt_canon[1], 1) == 220.0

    pt_back = transform_point_canonical_to_original(pt_canon, M_inverse)
    assert round(pt_back[0], 1) == 100.0
    assert round(pt_back[1], 1) == 100.0


def test_transform_bbox_canonical_to_original():
    M_inverse = np.eye(3, dtype=np.float32)
    bbox_canon = [100.0, 200.0, 500.0, 600.0]
    bbox_orig = transform_bbox_canonical_to_original(bbox_canon, M_inverse)
    assert bbox_orig == bbox_canon
