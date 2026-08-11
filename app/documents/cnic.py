"""
Pakistani CNIC document profiles (Front & Back).
"""

from typing import List, Dict
from .base import DocumentProfile


class CNICFrontProfile(DocumentProfile):
    name = "cnic_front"
    aspect_ratio = 1.5858  # ID-1 standard (85.60 mm x 53.98 mm)
    target_width = 2048
    target_height = 1291
    is_mrz_document = False
    
    expected_fields = [
        "cnic_number",
        "date_of_birth",
        "date_of_issue",
        "date_of_expiry"
    ]
    
    field_patterns = {
        "cnic_number": r"\b\d{5}[-\s]?\d{7}[-\s]?\d{1}\b",
        "date_of_birth": r"\b\d{2}[\./-]\d{2}[\./-]\d{4}\b",
        "date_of_issue": r"\b\d{2}[\./-]\d{2}[\./-]\d{4}\b",
        "date_of_expiry": r"\b\d{2}[\./-]\d{2}[\./-]\d{4}\b"
    }


class CNICBackProfile(DocumentProfile):
    name = "cnic_back"
    aspect_ratio = 1.5858
    target_width = 2048
    target_height = 1291
    is_mrz_document = False
    
    expected_fields = [
        "family_number",
        "date_of_issue"
    ]
    
    field_patterns = {
        "family_number": r"\b\d{6,10}\b",
        "date_of_issue": r"\b\d{2}[\./-]\d{2}[\./-]\d{4}\b"
    }
