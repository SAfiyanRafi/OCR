"""
MANDATORY REGRESSION TEST (Section 32).
Tests the exact passport scenario where existing parser produces duplicate dates:
  date_of_birth = 07.04.1966
  date_of_issue = 07.04.1966
  date_of_expiry = 22.07.2029
while visual OCR contains:
  07APR1966
  23JUL2019
  22JUL2029
and MRZ contains:
  DOB = 660407
  EXPIRY = 290722

The reconciliation layer MUST resolve:
  date_of_birth: 1966-04-07
  date_of_issue: 2019-07-23
  date_of_expiry: 2029-07-22
"""

from app.core.models import RawOCRResult, OCRToken
from app.llm.evidence import LLMEvidenceBuilder
from app.llm.nemotron import NemotronAdapter
from app.documents.passport.pipeline import PassportPipeline


def test_mandatory_passport_date_reconciliation_regression():
    # 1. Prepare visual OCR tokens simulating the imperfect scan
    tokens = [
        OCRToken(text="PASSPORT", confidence=0.99, bbox_px=[100, 50, 300, 90], page=1, index=0),
        OCRToken(text="JAVED", confidence=0.96, bbox_px=[300, 120, 500, 160], page=1, index=1),
        OCRToken(text="AKHTER", confidence=0.97, bbox_px=[300, 180, 500, 220], page=1, index=2),
        OCRToken(text="07APR1966", confidence=0.88, bbox_px=[300, 270, 550, 310], page=1, index=3),
        OCRToken(text="23JUL2019", confidence=0.92, bbox_px=[300, 450, 550, 490], page=1, index=4),
        OCRToken(text="22JUL2029", confidence=0.95, bbox_px=[300, 520, 550, 560], page=1, index=5),
        OCRToken(text="P<PAKJAVED<<AKHTER<<<<<<<<<<<<<<<<<<<<<<<<<<", confidence=0.99, bbox_px=[50, 800, 950, 840], page=1, index=6),
        OCRToken(text="AG86775647PAK6604071M29072243740616247565<18", confidence=0.99, bbox_px=[50, 850, 950, 890], page=1, index=7),
    ]

    raw_ocr = RawOCRResult(
        image_width=2000,
        image_height=1407,
        document_type="passport",
        tokens=tokens,
        raw_text="PASSPORT JAVED AKHTER 07APR1966 23JUL2019 22JUL2029 P<PAKJAVED<<AKHTER<<<<<<<<<<<<<<<<<<<<<<<<<< AG86775647PAK6604071M29072243740616247565<18"
    )

    # 2. Candidate fields matching the old parser output (with duplicate DOB assigned to issue date)
    flawed_candidate_fields = {
        "surname": {"value": "JAVED", "raw_value": "JAVED", "ocr_confidence": 0.96, "spatial_confidence": 0.90, "confidence": 0.93, "validated": True, "provenance": {"token_indices": [1]}},
        "given_names": {"value": "AKHTER", "raw_value": "AKHTER", "ocr_confidence": 0.97, "spatial_confidence": 0.90, "confidence": 0.94, "validated": True, "provenance": {"token_indices": [2]}},
        "date_of_birth": {"value": "07.04.1966", "raw_value": "07APR1966", "ocr_confidence": 0.88, "spatial_confidence": 0.85, "confidence": 0.86, "validated": True, "provenance": {"token_indices": [3]}},
        "date_of_issue": {"value": "07.04.1966", "raw_value": "07APR1966", "ocr_confidence": 0.70, "spatial_confidence": 0.60, "confidence": 0.65, "validated": False, "provenance": {"token_indices": [3]}},
        "date_of_expiry": {"value": "22.07.2029", "raw_value": "22JUL2029", "ocr_confidence": 0.95, "spatial_confidence": 0.90, "confidence": 0.92, "validated": True, "provenance": {"token_indices": [5]}},
        "passport_number": {"value": "AG8677564", "raw_value": "AG8677564", "ocr_confidence": 0.98, "spatial_confidence": 0.95, "confidence": 0.96, "validated": True, "provenance": {"token_indices": [7]}}
    }

    mrz_data = {
        "line1": "P<PAKJAVED<<AKHTER<<<<<<<<<<<<<<<<<<<<<<<<<<",
        "line2": "AG86775647PAK6604071M29072243740616247565<18",
        "parsed": {
            "passport_number": "AG8677564",
            "surname": "JAVED",
            "given_names": "AKHTER",
            "date_of_birth": "660407",
            "date_of_expiry": "290722"
        }
    }

    # 3. Build LLM Document Evidence Payload
    evidence = LLMEvidenceBuilder.build_evidence(
        document_type="passport",
        raw_ocr=raw_ocr,
        candidate_fields=flawed_candidate_fields,
        mrz_data=mrz_data
    )

    # 4. Run Nemotron Reconciliation Adapter
    adapter = NemotronAdapter()
    result = adapter.reconcile(evidence)

    # 5. MANDATORY ASSERTIONS
    assert result.fields["date_of_birth"].value == "1966-04-07", f"Expected 1966-04-07, got {result.fields['date_of_birth'].value}"
    assert result.fields["date_of_issue"].value == "2019-07-23", f"Expected 2019-07-23, got {result.fields['date_of_issue'].value}"
    assert result.fields["date_of_expiry"].value == "2029-07-22", f"Expected 2029-07-22, got {result.fields['date_of_expiry'].value}"
    assert result.fields["passport_number"].value == "AG8677564"
    assert result.fields["surname"].value == "JAVED"
    assert result.fields["given_names"].value == "AKHTER"
    assert result.status == "success"
