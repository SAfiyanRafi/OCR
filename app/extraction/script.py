"""
Script Detector Component.
Detects character script of OCR tokens (Urdu, Latin, Numeric, Mixed).
"""

import re
from typing import List
from app.core.models import OCRToken


def classify_text_script(text: str) -> str:
    """
    Classify character script of text string.
    """
    if not text:
        return "unknown"

    has_urdu = bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\u8A00-\u8D03]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    has_digits = bool(re.search(r"\d", text))

    if has_urdu and not has_latin:
        return "urdu"
    elif has_latin and not has_urdu:
        return "latin"
    elif has_digits and not has_urdu and not has_latin:
        return "numeric"
    elif has_urdu and has_latin:
        return "mixed"
    return "latin" if has_latin else "numeric" if has_digits else "unknown"
