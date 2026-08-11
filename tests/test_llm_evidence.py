"""
Unit tests for LLMEvidenceBuilder.
"""

from app.core.models import RawOCRResult, OCRToken, QualityReport, DocumentBoundary
from app.llm.evidence import LLMEvidenceBuilder


def test_build_evidence_lightweight_payload():
    tokens = [
        OCRToken(text="PASSPORT", confidence=0.98, bbox_px=[10, 10, 100, 30], page=1, index=0),
        OCRToken(text="07APR1966", confidence=0.88, bbox_px=[200, 300, 400, 350], page=1, index=1),
        OCRToken(text="23JUL2019", confidence=0.91, bbox_px=[200, 400, 400, 450], page=1, index=2),
        OCRToken(text="22JUL2029", confidence=0.95, bbox_px=[200, 500, 400, 550], page=1, index=3),
    ]

    raw_ocr = RawOCRResult(image_width=2000, image_height=1407, document_type="passport", tokens=tokens, raw_text="PASSPORT 07APR1966 23JUL2019 22JUL2029")

    candidates = {
        "date_of_birth": {
            "value": "07.04.1966",
            "raw_value": "07APR1966",
            "ocr_confidence": 0.88,
            "spatial_confidence": 0.90,
            "confidence": 0.89,
            "validated": True,
            "provenance": {"token_indices": [1]}
        }
    }

    mrz_data = {
        "line1": "P<PAKJAVED<<AKHTER<<<<<<<<<<<<<<<<<<<<<<<<<<",
        "line2": "AG86775647PAK6604071M29072243740616247565<18",
        "parsed": {"passport_number": "AG8677564", "date_of_birth": "660407", "date_of_expiry": "290722"}
    }

    evidence = LLMEvidenceBuilder.build_evidence(
        document_type="passport",
        raw_ocr=raw_ocr,
        candidate_fields=candidates,
        mrz_data=mrz_data
    )

    assert evidence.document_type == "passport"
    assert len(evidence.ocr_tokens) == 4
    assert evidence.ocr_tokens[1].index == 1
    assert evidence.ocr_tokens[1].text == "07APR1966"
    assert evidence.mrz.parsed["passport_number"] == "AG8677564"
    assert evidence.candidate_fields["date_of_birth"].token_indices == [1]
