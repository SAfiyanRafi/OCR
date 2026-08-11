"""
Privacy & Log Redaction Utility Module.
Protects Personally Identifiable Information (PII) such as CNIC numbers,
passport numbers, dates of birth, and names from appearing in production logs.
"""

from typing import Any, Dict, List, Union
import re
import os

SECURE_DEBUG_MODE = os.getenv("SECURE_DEBUG_MODE", "false").lower() in ("true", "1", "yes")

CNIC_REGEX = re.compile(r"\b\d{5}[-\s]?\d{7}[-\s]?\d{1}\b")
PASSPORT_REGEX = re.compile(r"\b[A-Z]{1,3}\d{6,8}\b", re.IGNORECASE)
DATE_REGEX = re.compile(r"\b\d{2}[\.\/\-]\d{2}[\.\/\-]\d{4}\b")


def redact_text(text: str) -> str:
    """
    Redact PII patterns from text strings unless SECURE_DEBUG_MODE is enabled.
    """
    if SECURE_DEBUG_MODE or not text:
        return text

    redacted = CNIC_REGEX.sub("[REDACTED_CNIC]", text)
    redacted = PASSPORT_REGEX.sub("[REDACTED_PASSPORT]", redacted)
    redacted = DATE_REGEX.sub("[REDACTED_DATE]", redacted)
    return redacted


def sanitize_dict_for_logging(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively sanitize dict fields for production logs.
    """
    if SECURE_DEBUG_MODE:
        return data

    sensitive_keys = {"cnic_number", "passport_number", "father_name", "surname", "given_names", "name", "raw_text", "raw_value", "date_of_birth"}
    sanitized: Dict[str, Any] = {}

    for k, v in data.items():
        if k in sensitive_keys:
            if isinstance(v, str):
                sanitized[k] = redact_text(v)
            elif isinstance(v, dict):
                sanitized[k] = {sub_k: redact_text(str(sub_v)) for sub_k, sub_v in v.items()}
            else:
                sanitized[k] = "[REDACTED_PII]"
        elif isinstance(v, dict):
            sanitized[k] = sanitize_dict_for_logging(v)
        elif isinstance(v, list):
            sanitized[k] = [sanitize_dict_for_logging(item) if isinstance(item, dict) else redact_text(str(item)) for item in v]
        else:
            sanitized[k] = v

    return sanitized
