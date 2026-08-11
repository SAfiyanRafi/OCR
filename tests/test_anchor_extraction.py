"""
Unit tests for directional anchor token searching and extraction.
"""

from app.ocr.models import OCRToken
from app.extraction.anchors import find_anchor_token, extract_tokens_relative_to_anchor


def test_find_anchor_token():
    tokens = [
        OCRToken(text="Name:", confidence=0.95, bbox=(100, 100, 180, 130), page=1, index=0),
        OCRToken(text="MUHAMMAD ALI", confidence=0.92, bbox=(200, 100, 450, 130), page=1, index=1)
    ]

    anchor = find_anchor_token(tokens, "Name")
    assert anchor is not None
    assert anchor.text == "Name:"


def test_extract_tokens_relative_to_anchor_right():
    anchor_tok = OCRToken(text="Identity Number:", confidence=0.95, bbox=(100, 200, 300, 230), page=1, index=0)
    target_tok = OCRToken(text="42101-1234567-1", confidence=0.98, bbox=(320, 200, 600, 230), page=1, index=1)
    other_tok = OCRToken(text="Unrelated", confidence=0.90, bbox=(100, 500, 250, 530), page=1, index=2)

    tokens = [anchor_tok, target_tok, other_tok]
    extracted = extract_tokens_relative_to_anchor(tokens, anchor_tok, direction="right", img_width=1000, img_height=1000)

    assert len(extracted) == 1
    assert extracted[0].text == "42101-1234567-1"


def test_extract_tokens_relative_to_anchor_below():
    anchor_tok = OCRToken(text="Date of Birth", confidence=0.95, bbox=(100, 200, 300, 230), page=1, index=0)
    target_tok = OCRToken(text="15.08.1990", confidence=0.98, bbox=(100, 240, 300, 270), page=1, index=1)

    tokens = [anchor_tok, target_tok]
    extracted = extract_tokens_relative_to_anchor(tokens, anchor_tok, direction="below", img_width=1000, img_height=1000)

    assert len(extracted) == 1
    assert extracted[0].text == "15.08.1990"
