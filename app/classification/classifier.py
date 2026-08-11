"""
Document Classifier Component.
Classifies input document into cnic_front, cnic_back, passport, or unknown.
Separates document classification from field extraction.
"""

from typing import Optional
import re
from app.core.models import RawOCRResult, ClassificationResult


class DocumentClassifier:
    """
    Automated document type classifier using keyword signatures and structural layout.
    """

    @staticmethod
    def classify(raw_ocr: RawOCRResult, explicit_type: Optional[str] = None) -> ClassificationResult:
        """
        Classify document and return ClassificationResult.
        """
        # 1. Explicit user override
        if explicit_type and explicit_type.strip() in ["cnic_front", "cnic_back", "passport"]:
            return ClassificationResult(
                document_type=explicit_type.strip(),
                confidence=1.0,
                method="explicit_override"
            )

        full_text = raw_ocr.raw_text.upper()

        # 2. Passport Detection Signatures
        passport_keywords = ["PASSPORT", "ISLAMIC REPUBLIC OF PAKISTAN", "P<PAK", "SURNAME", "GIVEN NAMES", "NATIONALITY"]
        passport_matches = sum(1 for kw in passport_keywords if kw in full_text)
        if passport_matches >= 2 or "P<PAK" in full_text:
            conf = min(0.98, 0.50 + passport_matches * 0.15)
            return ClassificationResult(
                document_type="passport",
                confidence=conf,
                method="keyword_mrz_signature"
            )

        # 3. CNIC Front Detection Signatures
        cnic_front_keywords = ["NATIONAL IDENTITY CARD", "ISLAMIC REPUBLIC OF PAKISTAN", "NAME", "FATHER NAME", "IDENTITY NUMBER"]
        has_cnic_num = bool(re.search(r"\b[1-7]\d{4}[-\s]?\d{7}[-\s]?\d{1}\b", full_text))
        has_urdu = bool(re.search(r"[\u0600-\u06FF]", full_text))
        
        cnic_front_matches = sum(1 for kw in cnic_front_keywords if kw in full_text)
        if cnic_front_matches >= 2 or (has_cnic_num and has_urdu):
            conf = min(0.98, 0.40 + cnic_front_matches * 0.15 + (0.20 if has_cnic_num else 0.0))
            return ClassificationResult(
                document_type="cnic_front",
                confidence=conf,
                method="keyword_bilingual_signature"
            )

        # 4. CNIC Back Detection Signatures
        cnic_back_keywords = ["FAMILY NUMBER", "ADDRESS", "REGISTRAR GENERAL"]
        cnic_back_matches = sum(1 for kw in cnic_back_keywords if kw in full_text)
        if cnic_back_matches >= 1 or ("FAMILY" in full_text and has_urdu):
            return ClassificationResult(
                document_type="cnic_back",
                confidence=0.85,
                method="keyword_signature"
            )

        # 5. Unknown / Low Confidence
        return ClassificationResult(
            document_type="unknown",
            confidence=0.30,
            method="fallback_unknown"
        )
