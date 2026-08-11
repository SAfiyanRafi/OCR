"""
Weighted OCR Candidate Evaluator.
Scores preprocessing variants without permanently coupling to full extraction.
Score = mean_conf * 0.25 + text_cov * 0.20 + field_cov * 0.25 + critical_score * 0.20 + format_score * 0.10.
"""

from typing import List, Dict, Any, Tuple
import re
import numpy as np
from app.core.models import OCRToken, RawOCRResult, OCRCandidateScore, PreprocessingVariant


class OCRCandidateEvaluator:
    """
    Evaluates candidate OCR results using multi-metric weighted scoring.
    """

    @staticmethod
    def evaluate(raw_ocr: Any, document_type: str = "generic") -> OCRCandidateScore:
        """
        Evaluate candidate OCR tokens / lines and return OCRCandidateScore.
        """
        tokens = []
        full_text_list = []
        confidences = []

        if hasattr(raw_ocr, "tokens") and raw_ocr.tokens:
            tokens = raw_ocr.tokens
            full_text_list = [t.text for t in tokens]
            confidences = [t.confidence for t in tokens]
        elif hasattr(raw_ocr, "lines") and raw_ocr.lines:
            full_text_list = [line.text for line in raw_ocr.lines]
            confidences = [line.confidence for line in raw_ocr.lines]

        full_text = " ".join(full_text_list)
        mean_conf = float(np.mean(confidences)) if confidences else getattr(raw_ocr, "average_confidence", 0.0)

        # Matched field detection via regex
        matched_fields: Dict[str, str] = {}
        
        # CNIC regex
        cnic_match = re.search(r"\b([1-7]\d{4}[-\s]?\d{7}[-\s]?\d{1})\b", full_text)
        if cnic_match:
            matched_fields["cnic_number"] = cnic_match.group(1).replace(" ", "-")

        # Passport number regex
        passport_match = re.search(r"\b([A-Z]{1,3}\d{6,8})\b", full_text)
        if passport_match and "PASSPORT" in full_text.upper():
            matched_fields["passport_number"] = passport_match.group(1)

        # MRZ Line 1 regex
        mrz1_match = re.search(r"(P<[A-Z0-9<]{30,})", full_text)
        if mrz1_match:
            matched_fields["mrz_line1"] = mrz1_match.group(1)

        # Date regex
        date_match = re.search(r"\b(\d{2}[\.\/-]\d{2}[\.\/-]\d{4})\b", full_text)
        if date_match:
            matched_fields["date_of_birth"] = date_match.group(1)

        # Metrics
        total_chars = sum(len(txt) for txt in full_text_list)
        text_coverage = min(1.0, total_chars / 150.0)

        distinct_y_bands = set(int(t.bbox_norm[1] * 20.0) for t in tokens if hasattr(t, "bbox_norm") and t.bbox_norm)
        expected_field_coverage = min(1.0, len(distinct_y_bands) / 6.0) if tokens else min(1.0, len(full_text_list) / 5.0)

        critical_score = min(1.0, len(matched_fields) / 3.0) if matched_fields else 0.40
        format_score = min(1.0, len(full_text_list) / 4.0)

        total_score = float(np.clip(
            mean_conf * 0.25 +
            text_coverage * 0.20 +
            expected_field_coverage * 0.25 +
            critical_score * 0.20 +
            format_score * 0.10,
            0.0, 1.0
        ))

        return OCRCandidateScore(
            mean_confidence=round(mean_conf, 4),
            text_coverage=round(text_coverage, 4),
            expected_field_coverage=round(expected_field_coverage, 4),
            critical_field_score=round(critical_score, 4),
            format_score=round(format_score, 4),
            spatial_score=0.80,
            total_score=round(total_score, 4),
            field_score=round(critical_score, 4),
            matched_fields=matched_fields
        )


OCREvaluator = OCRCandidateEvaluator
