"""
Directional Anchor Extractor Component.
Supports fuzzy anchor matching, OCR typo tolerance, max distance, and vertical tolerance.
"""

from typing import List, Optional, Dict, Any, Tuple
import re
from app.core.models import OCRToken, AnchorConfig


def compute_string_similarity(s1: str, s2: str) -> float:
    """
    Simple character overlap similarity score (0.0 to 1.0) for fuzzy anchor matching.
    """
    s1_clean = re.sub(r"[^\w]", "", s1.lower())
    s2_clean = re.sub(r"[^\w]", "", s2.lower())

    if not s1_clean or not s2_clean:
        return 0.0
    if s1_clean == s2_clean or s1_clean in s2_clean or s2_clean in s1_clean:
        return 1.0

    common = set(s1_clean).intersection(set(s2_clean))
    return float(len(common)) / float(max(len(set(s1_clean)), len(set(s2_clean))))


def find_anchor_token(
    tokens: List[OCRToken],
    keyword: str,
    confidence_threshold: float = 0.65
) -> Optional[OCRToken]:
    """
    Find best matching anchor token using exact or fuzzy similarity matching.
    """
    if not keyword or not tokens:
        return None

    kw_clean = keyword.lower().strip()
    best_tok = None
    best_sim = 0.0

    for t in tokens:
        t_clean = t.text.lower().strip()

        # Exact substring match
        if kw_clean in t_clean or t_clean in kw_clean:
            return t

        # Fuzzy typo-tolerant match
        sim = compute_string_similarity(kw_clean, t_clean)
        if sim > best_sim and sim >= confidence_threshold:
            best_sim = sim
            best_tok = t

    return best_tok


def extract_tokens_relative_to_anchor(
    tokens: List[OCRToken],
    anchor_token: OCRToken,
    direction: str = "right",
    max_distance_px: float = 500.0,
    vertical_tolerance_px: float = 50.0,
    img_width: int = 1000,
    img_height: int = 1000
) -> List[OCRToken]:
    """
    Extract target OCR tokens relative to anchor token with spatial distance & tolerance limits.
    """
    if not anchor_token or not tokens:
        return []

    ax1, ay1, ax2, ay2 = anchor_token.bbox_px
    acx = (ax1 + ax2) / 2.0
    acy = (ay1 + ay2) / 2.0

    matched: List[OCRToken] = []

    for t in tokens:
        if t.index == anchor_token.index:
            continue

        tx1, ty1, tx2, ty2 = t.bbox_px
        tcx = (tx1 + tx2) / 2.0
        tcy = (ty1 + ty2) / 2.0

        dx = tcx - acx
        dy = tcy - acy
        dist = (dx**2 + dy**2) ** 0.5

        if dist > max_distance_px:
            continue

        if direction == "right":
            # Target must be to the right and vertically aligned
            if tx1 >= ax1 - 10.0 and abs(tcy - acy) <= vertical_tolerance_px:
                matched.append(t)
        elif direction == "below":
            # Target must be below bottom of anchor (ay2) and within vertical_tolerance_px distance
            if ty1 >= ay2 - 5.0 and (ty1 - ay2) <= vertical_tolerance_px and abs(tcx - acx) <= max_distance_px * 0.5:
                matched.append(t)
        elif direction == "left":
            if tx2 <= ax2 + 10.0 and abs(tcy - acy) <= vertical_tolerance_px:
                matched.append(t)
        elif direction == "above":
            if ty2 <= ay1 + 5.0 and (ay1 - ty2) <= vertical_tolerance_px and abs(tcx - acx) <= max_distance_px * 0.5:
                matched.append(t)

    return matched
