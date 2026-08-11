"""
Privacy and Security Module.
Provides logging wrappers and redaction utilities for sensitive identity fields.
"""

import re
import logging

logger = logging.getLogger("pak_identity_ocr.security")

# Regex patterns for CNIC numbers and Passport numbers
CNIC_PATTERN = re.compile(r"\b([1-7]\d{4})[-.\s]?(\d{7})[-.\s]?(\d{1})\b")
PASSPORT_PATTERN = re.compile(r"\b([A-Z]{2})\d{7}\b")


def redact_cnic(text: str) -> str:
    """
    Redact CNIC number: 42101-1234567-1 -> *****-*******-1
    """
    if not text:
        return ""
    return CNIC_PATTERN.sub(r"*****-*******-\3", text)


def redact_passport(text: str) -> str:
    """
    Redact Passport number: AB1234567 -> AB*******
    """
    if not text:
        return ""
    return PASSPORT_PATTERN.sub(r"\1*******", text)


def redact_sensitive_text(text: str) -> str:
    """
    Redact sensitive identity numbers from log output.
    """
    if not text:
        return ""
    text = redact_cnic(text)
    text = redact_passport(text)
    return text


class PrivacyLogFilter(logging.Filter):
    """
    Logging filter that automatically redacts CNIC and Passport numbers.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_sensitive_text(record.msg)
        return True
