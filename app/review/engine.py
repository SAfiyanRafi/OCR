"""
Human Review Engine.
Evaluates field confidence, validation status, and warnings to assign review state:
AUTO_ACCEPT, NEEDS_REVIEW, or AUTO_REJECT.
"""

from typing import Dict, Any, List, Tuple


def evaluate_review_state(
    fields: Dict[str, Any],
    warnings: List[str],
    classification_conf: float = 1.0,
    accept_threshold: float = 0.90,
    review_threshold: float = 0.60
) -> Tuple[str, List[str]]:
    """
    Evaluate extracted fields and return (review_state, review_reasons).
    """
    reasons: List[str] = []

    if classification_conf < 0.70:
        reasons.append(f"Low document classification confidence ({classification_conf * 100:.0f}%)")
        return "NEEDS_REVIEW", reasons

    unvalidated_count = 0
    low_conf_count = 0
    total_fields = 0
    conf_sum = 0.0

    for key, field_obj in fields.items():
        if isinstance(field_obj, dict) and "confidence" in field_obj:
            total_fields += 1
            conf = field_obj.get("confidence", 0.0)
            val = field_obj.get("validated", False)
            conf_sum += conf

            if not val:
                unvalidated_count += 1
                reasons.append(f"Field '{key}' failed format validation")
            if conf < review_threshold:
                low_conf_count += 1
                reasons.append(f"Field '{key}' has low OCR confidence ({conf * 100:.0f}%)")

    if warnings:
        reasons.extend(warnings)

    mean_conf = (conf_sum / max(1, total_fields)) if total_fields else 0.0

    if total_fields > 0 and unvalidated_count == 0 and mean_conf >= accept_threshold and not warnings:
        return "AUTO_ACCEPT", ["All critical fields validated with high confidence"]
    elif mean_conf < review_threshold or unvalidated_count >= 3:
        return "AUTO_REJECT", reasons
    else:
        return "NEEDS_REVIEW", reasons
