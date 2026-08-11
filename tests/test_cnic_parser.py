"""
Unit tests for CNIC Front & Back bilingual field parsers.
"""

from app.ocr.models import RawOCRResult, OCRToken
from app.documents.cnic.parser import CNICParser


def test_cnic_front_bilingual_parser():
    tokens = [
        OCRToken(text="محمد احمد", confidence=0.92, bbox=(200, 210, 500, 240), page=1, index=0),
        OCRToken(text="MUHAMMAD AHMAD", confidence=0.96, bbox=(200, 290, 600, 320), page=1, index=1),
        OCRToken(text="Identity Number:", confidence=0.90, bbox=(200, 520, 350, 550), page=1, index=2),
        OCRToken(text="42101-1234567-1", confidence=0.98, bbox=(360, 520, 650, 550), page=1, index=3),
        OCRToken(text="15.08.1990", confidence=0.94, bbox=(200, 650, 400, 680), page=1, index=4)
    ]

    raw_ocr = RawOCRResult(
        image_width=1000,
        image_height=1000,
        document_type="cnic_front",
        tokens=tokens,
        raw_text=" ".join(t.text for t in tokens)
    )

    parser = CNICParser(doc_side="front")
    result = parser.parse(raw_ocr)

    assert result["document_type"] == "cnic_front"
    assert "fields" in result
    fields = result["fields"]

    # Check cnic_number extraction & validation
    assert fields["cnic_number"]["value"] == "42101-1234567-1"
    assert fields["cnic_number"]["validated"] is True
    assert fields["cnic_number"]["confidence"] > 0.90

    # Check bilingual name representation
    assert result["name"]["en"] == "MUHAMMAD AHMAD"
    assert result["name"]["ur"] == "محمد احمد"
