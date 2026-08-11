"""
Field Candidate Scoring Engine.
Calculates multi-dimensional layered field confidence scores combining:
ocr_confidence + spatial_confidence + anchor_confidence + validation_confidence + script_confidence + checksum_confidence.
"""

from typing import Dict, Any, List, Optional, Tuple
import re
import numpy as np
from app.core.models import FieldResult, FieldProvenance


def calculate_field_candidate_score(
    raw_val: str,
    norm_val: str,
    ocr_confidence: float,
    strategy: str = "region",
    is_valid: bool = False,
    script: str = "latin",
    checksum_valid: bool = False,
    is_anchor_matched: bool = False
) -> Dict[str, float]:
    """
    Compute multi-dimensional layered field confidence scores.
    Returns dict containing:
    - ocr_confidence
    - spatial_confidence
    - anchor_confidence
    - validation_confidence
    - field_confidence (total weighted score)
    """
    raw_clean = raw_val.strip()
    norm_clean = norm_val.strip()

    # 1. OCR Confidence
    ocr_conf = max(0.0, min(1.0, float(ocr_confidence)))

    # 2. Spatial Confidence
    spatial_conf = 0.90 if strategy in ("region", "hybrid") and raw_clean else 0.60

    # 3. Anchor Confidence
    anchor_conf = 0.95 if is_anchor_matched else (0.70 if strategy == "hybrid" else 0.50)

    # 4. Validation Confidence
    val_conf = 1.0 if is_valid and norm_clean else (0.30 if norm_clean else 0.0)

    # 5. Checksum & Script Bonuses
    checksum_bonus = 0.15 if checksum_valid else 0.0
    script_bonus = 0.10 if script in ("latin", "urdu", "numeric") else 0.0

    # Weighted Field Confidence Score Calculation
    field_conf = (
        (0.35 * ocr_conf) +
        (0.20 * spatial_conf) +
        (0.15 * anchor_conf) +
        (0.30 * val_conf) +
        checksum_bonus +
        script_bonus
    )

    field_conf = max(0.0, min(1.0, round(field_conf, 4)))

    return {
        "ocr_confidence": round(ocr_conf, 4),
        "spatial_confidence": round(spatial_conf, 4),
        "anchor_confidence": round(anchor_conf, 4),
        "validation_confidence": round(val_conf, 4),
        "field_confidence": field_conf
    }


def select_best_field_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Select the single highest-scoring field candidate from candidate list.
    """
    if not candidates:
        return {}

    best_cand = None
    best_score = -1.0

    for cand in candidates:
        raw = str(cand.get("raw", cand.get("raw_value", ""))).strip()
        norm = str(cand.get("normalized", cand.get("value", ""))).strip()
        is_valid = bool(cand.get("validated", False))
        ocr_conf = float(cand.get("ocr_confidence", cand.get("confidence", 0.0)))
        strategy = cand.get("strategy", "region")

        scores = calculate_field_candidate_score(
            raw_val=raw,
            norm_val=norm,
            ocr_confidence=ocr_conf,
            strategy=strategy,
            is_valid=is_valid,
            checksum_valid=cand.get("checksum_valid", False),
            is_anchor_matched=cand.get("is_anchor_matched", False)
        )

        total_score = scores["field_confidence"]
        if total_score > best_score:
            best_score = total_score
            best_cand = {**cand, **scores}

    return best_cand or candidates[0]
