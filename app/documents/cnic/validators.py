"""
Validators for Pakistani CNIC documents.
"""

import re


def validate_cnic_number(value: str) -> bool:
    """
    Validate 13-digit Pakistani CNIC number format: XXXXX-XXXXXXX-X.
    Valid province codes: 1..7 (1=KPK, 2=FATA, 3=Punjab, 4=Sindh, 5=Balochistan, 6=Islamabad, 7=GB).
    """
    pattern = r"^[1-7]\d{4}-\d{7}-\d{1}$"
    return bool(re.match(pattern, value.strip()))


def validate_date(value: str) -> bool:
    """
    Validate DD.MM.YYYY date pattern.
    """
    pattern = r"^\d{2}\.\d{2}\.\d{4}$"
    if not re.match(pattern, value.strip()):
        return False
    parts = value.strip().split(".")
    d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
    return (1 <= d <= 31) and (1 <= m <= 12) and (1900 <= y <= 2100)


def validate_text_non_empty(value: str) -> bool:
    """
    Validate non-empty string.
    """
    return len(value.strip()) > 0


def run_cnic_validator(validator_name: str, value: str) -> bool:
    """
    Execute CNIC validator by name.
    """
    name = (validator_name or "none").lower().strip()
    if name == "cnic_number" or name == "cnic":
        return validate_cnic_number(value)
    elif name == "date":
        return validate_date(value)
    elif name == "text_non_empty":
        return validate_text_non_empty(value)
    elif name == "none":
        return True
    return True
