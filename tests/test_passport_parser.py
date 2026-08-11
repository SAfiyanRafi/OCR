"""
Unit tests for Passport Biodata and MRZ parsers.
"""

from app.ocr.models import RawOCRResult, OCRToken
from app.documents.passport.parser import PassportParser


def test_passport_parser():
    tokens = [
        OCRToken(text="Passport No:", confidence=0.90, bbox=(600, 60, 750, 90), page=1, index=0),
        OCRToken(text="AB1234567", confidence=0.98, bbox=(760, 60, 920, 90), page=1, index=1),
        OCRToken(text="Surname", confidence=0.90, bbox=(300, 120, 450, 145), page=1, index=2),
        OCRToken(text="KHAN", confidence=0.95, bbox=(300, 150, 450, 180), page=1, index=3),
        OCRToken(text="Given Names", confidence=0.90, bbox=(300, 185, 450, 210), page=1, index=4),
        OCRToken(text="MUHAMMAD ALI", confidence=0.94, bbox=(300, 215, 550, 245), page=1, index=5),
        OCRToken(text="P<PAKKHAN<<MUHAMMAD<ALI<<<<<<<<<<<<<<<<<<<<<", confidence=0.99, bbox=(50, 800, 950, 840), page=1, index=6),
        OCRToken(text="AB12345674PAK9008154M3001018<<<<<<<<<<<<<<02", confidence=0.98, bbox=(50, 880, 950, 920), page=1, index=7)
    ]

    raw_ocr = RawOCRResult(
        image_width=1000,
        image_height=1000,
        document_type="passport",
        tokens=tokens,
        raw_text=" ".join(t.text for t in tokens)
    )

    parser = PassportParser()
    result = parser.parse(raw_ocr)

    assert result["document_type"] == "passport"
    fields = result["fields"]

    assert fields["passport_number"]["value"] == "AB1234567"
    assert fields["passport_number"]["validated"] is True
    assert fields["surname"]["value"] == "KHAN"
    assert fields["surname"]["validated"] is True
    assert fields["given_names"]["value"] == "MUHAMMAD ALI"
    assert fields["given_names"]["validated"] is True
