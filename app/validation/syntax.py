"""
Syntax Validators.
Validates string format and regex patterns.
"""

import re


def validate_cnic_syntax(value: str) -> bool:
    """Validate CNIC format: XXXXX-XXXXXXX-X."""
    return bool(re.match(r"^[1-7]\d{4}-\d{7}-\d{1}$", value.strip()))


def validate_passport_number_syntax(value: str) -> bool:
    """Validate Pakistani passport number syntax (e.g. AB1234567)."""
    return bool(re.match(r"^[A-Z]{1,3}\d{6,8}$", value.strip()))


def validate_date_syntax(value: str) -> bool:
    """Validate DD.MM.YYYY date syntax."""
    return bool(re.match(r"^\d{2}\.\d{2}\.\d{4}$", value.strip()))


def validate_mrz_syntax(value: str) -> bool:
    """Validate MRZ line syntax (30-44 uppercase alphanumeric and <)."""
    clean = value.strip()
    return len(clean) >= 30 and bool(re.match(r"^[A-Z0-9<]+$", clean))
