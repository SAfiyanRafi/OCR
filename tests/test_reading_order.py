"""
Unit tests for reading order token sorting.
"""

from app.ocr.models import OCRToken
from app.extraction.reading_order import sort_tokens_reading_order


def test_sort_tokens_reading_order():
    tok1 = OCRToken(text="Word2", confidence=0.9, bbox=(200, 100, 300, 130), page=1, index=0)
    tok2 = OCRToken(text="Word1", confidence=0.9, bbox=(50, 100, 150, 130), page=1, index=1)
    tok3 = OCRToken(text="Line2Word1", confidence=0.9, bbox=(50, 200, 180, 230), page=1, index=2)

    unsorted = [tok1, tok2, tok3]
    sorted_toks = sort_tokens_reading_order(unsorted, line_threshold=15.0)

    assert [t.text for t in sorted_toks] == ["Word1", "Word2", "Line2Word1"]
