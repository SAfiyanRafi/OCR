"""
Unit tests for CNIC, Date, Passport normalizers and validators.
"""

from app.extraction.normalization import (
    normalize_cnic_number,
    normalize_date,
    normalize_passport_number,
    apply_field_normalization
)
from app.documents.cnic.validators import validate_cnic_number, validate_date
from app.documents.passport.validators import validate_passport_number, validate_mrz_line


def test_cnic_normalization_and_validation():
    raw1 = "12345 1234567 1"
    norm1 = normalize_cnic_number(raw1)
    assert norm1 == "12345-1234567-1"
    assert validate_cnic_number("42101-1234567-1") is True
    assert validate_cnic_number("invalid-cnic") is False


def test_date_normalization_and_validation():
    raw_date = "15/08/1990"
    norm_date = normalize_date(raw_date)
    assert norm_date == "15.08.1990"
    assert validate_date(norm_date) is True
    assert validate_date("99.99.9999") is False


def test_passport_normalization_and_validation():
    raw_p = "ab1234567"
    norm_p = normalize_passport_number(raw_p)
    assert norm_p == "AB1234567"
    assert validate_passport_number(norm_p) is True

    mrz = "P<PAKKHAN<<MUHAMMAD<ALI<<<<<<<<<<<<<<<<<<<<<"
    assert validate_mrz_line(mrz) is True
