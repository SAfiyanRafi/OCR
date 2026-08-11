"""
Pakistani Passport document profile specification.
"""

from typing import List, Dict
from .base import DocumentProfile


class PassportProfile(DocumentProfile):
    name = "passport"
    aspect_ratio = 1.4205  # ID-3 standard (125 mm x 88 mm)
    target_width = 2500
    target_height = 1760
    is_mrz_document = True
    
    expected_fields = [
        "passport_number",
        "mrz_line1",
        "mrz_line2",
        "date_of_birth",
        "date_of_expiry"
    ]
    
    field_patterns = {
        "passport_number": r"\b[A-Z]{2}\d{7}\b|\b[A-Z0-9]{8,9}\b",
        "mrz_line1": r"P<PAK[A-Z<]+",
        "mrz_line2": r"[A-Z0-9<]{30,44}",
        "date_of_birth": r"\b\d{2}[\./-]\d{2}[\./-]\d{4}\b|\b\d{6}\b",
        "date_of_expiry": r"\b\d{2}[\./-]\d{2}[\./-]\d{4}\b|\b\d{6}\b"
    }
