"""
Unit tests for document-aware OCR evaluation and pattern matching.
"""

from app.ocr.base import OCRResultContainer, OCRTextLine
from app.ocr.evaluator import OCREvaluator


def test_cnic_ocr_evaluator():
    ocr_result = OCRResultContainer(
        variant_name="variant_01",
        lines=[
            OCRTextLine(text="PAKISTAN National Identity Card", confidence=0.95, box=[]),
            OCRTextLine(text="Name: MUHAMMAD ALI", confidence=0.92, box=[]),
            OCRTextLine(text="Identity Number: 42101-1234567-1", confidence=0.98, box=[]),
            OCRTextLine(text="Date of Birth: 15.08.1990", confidence=0.94, box=[]),
            OCRTextLine(text="Date of Issue: 01.01.2020", confidence=0.91, box=[])
        ],
        average_confidence=0.94
    )

    scored = OCREvaluator.evaluate(ocr_result, document_type="cnic_front")
    assert "cnic_number" in scored.matched_fields
    assert scored.matched_fields["cnic_number"] == "42101-1234567-1"
    assert scored.field_score > 0.50
    assert scored.total_score > 0.60


def test_passport_mrz_ocr_evaluator():
    ocr_result = OCRResultContainer(
        variant_name="variant_03",
        lines=[
            OCRTextLine(text="PAKISTAN PASSPORT", confidence=0.96, box=[]),
            OCRTextLine(text="Passport No: AB1234567", confidence=0.95, box=[]),
            OCRTextLine(text="P<PAKKHAN<<MUHAMMAD<ALI<<<<<<<<<<<<<<<<<<<<<", confidence=0.99, box=[]),
            OCRTextLine(text="AB12345674PAK9008154M3001018<<<<<<<<<<<<<<02", confidence=0.98, box=[])
        ],
        average_confidence=0.97
    )

    scored = OCREvaluator.evaluate(ocr_result, document_type="passport")
    assert "passport_number" in scored.matched_fields
    assert "mrz_line1" in scored.matched_fields
    assert scored.total_score > 0.70
