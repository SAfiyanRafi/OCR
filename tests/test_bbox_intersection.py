"""
Unit tests for normalized bounding box region containment and overlap ratio calculation.
"""

from app.ocr.models import OCRToken
from app.extraction.regions import (
    parse_region_bounds,
    normalize_bbox,
    calculate_overlap_ratio,
    extract_tokens_in_region
)


def test_parse_region_bounds():
    r1 = parse_region_bounds({"x1": 0.2, "y1": 0.3, "x2": 0.8, "y2": 0.6})
    assert r1 == (0.2, 0.3, 0.8, 0.6)

    r2 = parse_region_bounds([0.1, 0.2, 0.5, 0.4])
    assert r2 == (0.1, 0.2, 0.5, 0.4)


def test_normalize_bbox():
    # Absolute pixels to normalized 1000x500 image
    bbox_abs = (200.0, 100.0, 400.0, 200.0)
    norm = normalize_bbox(bbox_abs, img_width=1000, img_height=500)
    assert norm == (0.2, 0.2, 0.4, 0.4)


def test_calculate_overlap_ratio():
    token_bbox = (0.2, 0.2, 0.4, 0.4)
    region_bbox = (0.0, 0.0, 0.5, 0.5)

    overlap = calculate_overlap_ratio(token_bbox, region_bbox)
    assert overlap == 1.0  # Completely inside region


def test_extract_tokens_in_region():
    tokens = [
        OCRToken(text="InsideToken", confidence=0.95, bbox=(200, 200, 400, 250), page=1, index=0),
        OCRToken(text="OutsideToken", confidence=0.90, bbox=(800, 800, 950, 850), page=1, index=1)
    ]

    region_cfg = {"x1": 0.15, "y1": 0.15, "x2": 0.50, "y2": 0.50}
    matched = extract_tokens_in_region(tokens, region_cfg, img_width=1000, img_height=1000)

    assert len(matched) == 1
    assert matched[0].text == "InsideToken"
