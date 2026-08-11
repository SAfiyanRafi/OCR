"""
Calibrated Confidence Calculation Engine.
Calculates evidence-based 2-decimal rounded confidence scores combining:
OCR token confidence, spatial fit, format validation, MRZ agreement, and candidate competition.
"""

from typing import Dict, Any, Optional, List


def calculate_calibrated_field_confidence(
    ocr_confidence: float,
    spatial_confidence: float,
    format_valid: bool = False,
    mrz_agreed: bool = False,
    is_critical: bool = False,
    candidate_competing: bool = False
) -> float:
    """
    Compute a calibrated 2-decimal rounded field confidence score.
    Returns float in range [0.00, 1.00].
    """
    ocr_c = max(0.0, min(1.0, float(ocr_confidence)))
    spat_c = max(0.0, min(1.0, float(spatial_confidence)))

    val_bonus = 0.35 if format_valid else 0.0
    mrz_bonus = 0.25 if mrz_agreed else 0.0
    competition_penalty = 0.15 if candidate_competing else 0.0

    score = (0.25 * ocr_c) + (0.15 * spat_c) + val_bonus + mrz_bonus - competition_penalty

    if is_critical and not format_valid and not mrz_agreed:
        score = min(score, 0.40)

    final_score = max(0.0, min(1.0, score))
    return round(final_score, 2)
