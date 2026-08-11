"""
Unit tests for Father Name comma inversion normalization rule.
"""

from app.extraction.normalization import normalize_father_name


def test_father_name_comma_inversion():
    # Test case 1: Comma present (SAFDER, GHULAM -> GHULAM SAFDER)
    raw1 = "SAFDER, GHULAM"
    assert normalize_father_name(raw1) == "GHULAM SAFDER"

    # Test case 2: Comma with spaces (SAFDER , GHULAM -> GHULAM SAFDER)
    raw2 = "SAFDER , GHULAM"
    assert normalize_father_name(raw2) == "GHULAM SAFDER"

    # Test case 3: Label prefix present (Father Name: SAFDER, GHULAM -> GHULAM SAFDER)
    raw3 = "Father Name SAFDER, GHULAM"
    assert normalize_father_name(raw3) == "GHULAM SAFDER"

    # Test case 4: No comma present (SAFDER GHULAM -> SAFDER GHULAM)
    raw4 = "SAFDER GHULAM"
    assert normalize_father_name(raw4) == "SAFDER GHULAM"
