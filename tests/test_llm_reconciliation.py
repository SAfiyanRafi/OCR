"""
Unit tests for LLMReconciler hierarchy, conflict resolution, date normalization, and review routing.
"""

from app.core.models import RawOCRResult, OCRToken
from app.llm.evidence import LLMEvidenceBuilder
from app.llm.reconciliation import LLMReconciler, normalize_to_iso_date, format_cnic


def test_date_iso_normalization():
    assert normalize_to_iso_date("07APR1966") == "1966-04-07"
    assert normalize_to_iso_date("23JUL2019") == "2019-07-23"
    assert normalize_to_iso_date("22JUL2029") == "2029-07-22"
    assert normalize_to_iso_date("07.04.1966") == "1966-04-07"
    assert normalize_to_iso_date("660407") == "1966-04-07"
    assert normalize_to_iso_date("290722") == "2029-07-22"


def test_format_cnic_number():
    assert format_cnic("3740616247565") == "37406-1624756-5"
    assert format_cnic("37406-1624756-5") == "37406-1624756-5"


def test_reconcile_passport_evidence():
    tokens = [
        OCRToken(text="PASSPORT", confidence=0.98, bbox_px=[10, 10, 100, 30], page=1, index=0),
        OCRToken(text="07APR1966", confidence=0.88, bbox_px=[200, 300, 400, 350], page=1, index=1),
        OCRToken(text="23JUL2019", confidence=0.91, bbox_px=[200, 400, 400, 450], page=1, index=2),
        OCRToken(text="22JUL2029", confidence=0.95, bbox_px=[200, 500, 400, 550], page=1, index=3),
    ]

    raw_ocr = RawOCRResult(image_width=2000, image_height=1407, document_type="passport", tokens=tokens, raw_text="PASSPORT 07APR1966 23JUL2019 22JUL2029")

    candidates = {
        "date_of_birth": {"value": "07.04.1966", "raw_value": "07APR1966", "ocr_confidence": 0.88, "spatial_confidence": 0.90, "confidence": 0.89, "validated": True, "provenance": {"token_indices": [1]}},
        "date_of_issue": {"value": "07.04.1966", "raw_value": "07APR1966", "ocr_confidence": 0.70, "spatial_confidence": 0.75, "confidence": 0.72, "validated": False, "provenance": {"token_indices": [1]}},
        "date_of_expiry": {"value": "22.07.2029", "raw_value": "22JUL2029", "ocr_confidence": 0.95, "spatial_confidence": 0.95, "confidence": 0.95, "validated": True, "provenance": {"token_indices": [3]}}
    }

    mrz_data = {
        "line1": "P<PAKJAVED<<AKHTER<<<<<<<<<<<<<<<<<<<<<<<<<<",
        "line2": "AG86775647PAK6604071M29072243740616247565<18",
        "parsed": {"passport_number": "AG8677564", "surname": "JAVED", "given_names": "AKHTER", "date_of_birth": "660407", "date_of_expiry": "290722"}
    }

    evidence = LLMEvidenceBuilder.build_evidence(
        document_type="passport",
        raw_ocr=raw_ocr,
        candidate_fields=candidates,
        mrz_data=mrz_data
    )

    result = LLMReconciler.reconcile_document(evidence)

    assert result.fields["date_of_birth"].value == "1966-04-07"
    assert result.fields["date_of_issue"].value == "2019-07-23"
    assert result.fields["date_of_expiry"].value == "2029-07-22"
    assert result.fields["passport_number"].value == "AG8677564"
    assert result.fields["surname"].value == "JAVED"
    assert result.fields["given_names"].value == "AKHTER"
