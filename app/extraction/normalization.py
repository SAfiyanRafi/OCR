"""
Field-specific normalization utilities.
Normalizes CNIC numbers, dates, passport numbers, and text strings based on configuration rules.
"""

import re
from typing import Optional

MONTH_MAP = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
    "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
}

LABEL_NOISE_PATTERNS = [
    r"^date\s*of\s*birth", r"^date\s*of\s*issue", r"^date\s*of\s*expiry",
    r"^ssteafbirt", r"^surname", r"^given\s*names", r"^given\s*name", r"^passport\s*no",
    r"^nationality", r"^type", r"^country\s*code", r"^identity\s*no", r"^father\s*name", r"^name"
]


def strip_label_noise(raw_text: str) -> str:
    """
    Remove common anchor label words that leak into extracted regions.
    """
    clean = raw_text.strip()
    for pat in LABEL_NOISE_PATTERNS:
        clean = re.sub(pat, "", clean, flags=re.IGNORECASE).strip()
    return clean


def normalize_cnic_number(raw_text: str) -> str:
    """
    Normalize CNIC number to standard XXXXX-XXXXXXX-X format.
    """
    digits_match = re.search(r"\b\d{5}[-\s]?\d{7}[-\s]?\d{1}\b|\b\d{13}\b", raw_text)
    if digits_match:
        digits = re.sub(r"\D", "", digits_match.group(0))
        if len(digits) == 13:
            return f"{digits[:5]}-{digits[5:12]}-{digits[12]}"

    digits = re.sub(r"\D", "", raw_text)
    if len(digits) == 13:
        return f"{digits[:5]}-{digits[5:12]}-{digits[12]}"
    return raw_text.strip()


def normalize_date(raw_text: str) -> str:
    """
    Normalize date strings to DD.MM.YYYY format, supporting alphanumeric month names.
    
    Examples:
        "15/08/1990" -> "15.08.1990"
        "ssteafBirt 07APR1966" -> "07.04.1966"
        "23 JUL 2019" -> "23.07.2019"
    """
    clean = strip_label_noise(raw_text)

    # 1. Match DD APR YYYY or DDAPR1966 (e.g., 07APR1966, 23 JUL 2019)
    match_alpha = re.search(r"\b(\d{1,2})[\s\./-]*([A-Za-z]{3})[\s\./-]*(\d{4})\b", clean)
    if match_alpha:
        d, m_str, y = match_alpha.groups()
        m_upper = m_str.upper()
        if m_upper in MONTH_MAP:
            return f"{int(d):02d}.{MONTH_MAP[m_upper]}.{y}"

    # 2. Format DD.MM.YYYY or DD/MM/YYYY
    match_dmy = re.search(r"\b(\d{1,2})[\./-](\d{1,2})[\./-](\d{4})\b", clean)
    if match_dmy:
        d, m, y = match_dmy.groups()
        return f"{int(d):02d}.{int(m):02d}.{y}"

    # 3. Format YYYY.MM.DD
    match_ymd = re.search(r"\b(\d{4})[\./-](\d{1,2})[\./-](\d{1,2})\b", clean)
    if match_ymd:
        y, m, d = match_ymd.groups()
        return f"{int(d):02d}.{int(m):02d}.{y}"

    return clean


def normalize_father_name(raw_text: str) -> str:
    """
    Normalize father name.
    If comma exists (e.g. 'SAFDER, GHULAM'), invert words to 'GHULAM SAFDER'.
    If no comma, keep clean text as is.
    """
    clean = strip_label_noise(raw_text)
    if "," in clean:
        parts = [p.strip() for p in clean.split(",") if p.strip()]
        if len(parts) >= 2:
            return f"{parts[1]} {parts[0]}".upper()
    return clean.upper().strip()


def normalize_passport_number(raw_text: str) -> str:
    """
    Normalize Passport number (e.g. AB1234567).
    """
    clean = raw_text.upper().strip()
    match = re.search(r"\b[A-Z]{1,3}\d{6,8}\b", clean)
    if match:
        return match.group(0)
    return re.sub(r"[^\w]", "", clean)


def normalize_digits_only(raw_text: str) -> str:
    """
    Extract digits only.
    """
    return re.sub(r"\D", "", raw_text)


def normalize_uppercase_mrz(raw_text: str) -> str:
    """
    Normalize MRZ text characters.
    """
    clean = re.sub(r"[^A-Z0-9<]", "", raw_text.upper())
    return clean


def apply_field_normalization(raw_text: str, strategy: str) -> str:
    """
    Apply specified field normalization strategy.
    """
    if not raw_text:
        return ""

    strategy = (strategy or "none").lower().strip()

    if strategy == "cnic":
        return normalize_cnic_number(raw_text)
    elif strategy == "date":
        return normalize_date(raw_text)
    elif strategy == "passport_number":
        return normalize_passport_number(raw_text)
    elif strategy in ["father_name_invert", "father_name"]:
        return normalize_father_name(raw_text)
    elif strategy == "digits_only":
        return normalize_digits_only(raw_text)
    elif strategy == "uppercase":
        clean = strip_label_noise(raw_text)
        return clean.upper().strip()
    elif strategy == "uppercase_mrz":
        return normalize_uppercase_mrz(raw_text)
    elif strategy == "uppercase_alphanumeric":
        clean = strip_label_noise(raw_text)
        return re.sub(r"[^\w\s]", "", clean).upper().strip()
    else:
        return raw_text.strip()
