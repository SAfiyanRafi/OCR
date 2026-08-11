"""
Unit tests for Privacy & Log Redaction Utility.
"""

from app.core.privacy import redact_text, sanitize_dict_for_logging


def test_privacy_log_redaction():
    text_sample = "Customer CNIC is 37406-1624756-5 and DOB is 07.04.1966 with Passport AG8677564."
    redacted = redact_text(text_sample)

    assert "37406-1624756-5" not in redacted
    assert "AG8677564" not in redacted
    assert "[REDACTED_CNIC]" in redacted
    assert "[REDACTED_PASSPORT]" in redacted
    assert "[REDACTED_DATE]" in redacted


def test_sanitize_dict_for_logging():
    data = {
        "status": "success",
        "cnic_number": "37406-1624756-5",
        "name": "MUHAMMAD AKHTER"
    }

    sanitized = sanitize_dict_for_logging(data)
    assert "37406-1624756-5" not in str(sanitized["cnic_number"])
