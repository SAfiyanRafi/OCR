"""
Validators for Pakistani Passport documents.
"""

import re
from app.documents.cnic.validators import validate_date, validate_text_non_empty


def validate_passport_number(value: str) -> bool:
    """
    Validate Pakistani passport number (e.g. 2 letters + 7 digits or 8-9 uppercase alphanumeric).
    """
    pattern = r"^[A-Z]{2}\d{7}$|^[A-Z0-9]{8,9}$"
    return bool(re.match(pattern, value.strip()))


def validate_mrz_line(value: str) -> bool:
    """
    Validate Passport MRZ line length (30-44 characters) and format.
    """
    clean = value.strip()
    return len(clean) >= 30 and bool(re.match(r"^[A-Z0-9<]+$", clean))


def run_passport_validator(validator_name: str, value: str) -> bool:
    """
    Execute Passport validator by name.
    """
    name = (validator_name or "none").lower().strip()
    if name == "passport_number":
        return validate_passport_number(value)
    elif name == "mrz_line":
        return validate_mrz_line(value)
    elif name == "date":
        return validate_date(value)
    elif name == "text_non_empty":
        return validate_text_non_empty(value)
    elif name == "none":
        return True
    return True
