"""
Cross-Field Validation Component.
Validates logical relationships between fields (DOB < Issue < Expiry, visible passport == MRZ passport).
Generates explicit warnings without changing raw text.
"""

from typing import Dict, Any, List
from app.validation.semantic import parse_date_parts


def validate_cross_fields(fields: Dict[str, Any], document_type: str = "generic") -> List[str]:
    """
    Run cross-field validation rules and return list of warnings.
    """
    warnings: List[str] = []

    # 1. Date Chronology Rules (DOB < Issue < Expiry)
    dob_val = fields.get("date_of_birth", {}).get("value", "")
    issue_val = fields.get("date_of_issue", {}).get("value", "")
    expiry_val = fields.get("date_of_expiry", {}).get("value", "")

    dob_parts = parse_date_parts(str(dob_val))
    issue_parts = parse_date_parts(str(issue_val))
    expiry_parts = parse_date_parts(str(expiry_val))

    if dob_parts and issue_parts:
        if dob_parts[2] >= issue_parts[2]:
            warnings.append(f"Cross-field warning: Date of birth year ({dob_parts[2]}) must be before date of issue year ({issue_parts[2]})")

    if issue_parts and expiry_parts:
        if issue_parts[2] >= expiry_parts[2]:
            warnings.append(f"Cross-field warning: Date of issue year ({issue_parts[2]}) must be before date of expiry year ({expiry_parts[2]})")

    # 2. Passport Visible vs MRZ Number Consistency Rule
    if "passport" in document_type:
        p_num = fields.get("passport_number", {}).get("value", "")
        mrz1 = fields.get("mrz_line1", {}).get("value", "")
        mrz2 = fields.get("mrz_line2", {}).get("value", "")

        if p_num and mrz2:
            p_num_clean = str(p_num).strip().upper()
            mrz2_clean = str(mrz2).strip().upper()
            if len(p_num_clean) >= 7 and len(mrz2_clean) >= 9:
                if p_num_clean[:7] not in mrz2_clean[:9]:
                    warnings.append(f"Cross-field warning: Visible passport number '{p_num_clean}' does not match MRZ line 2 passport segment '{mrz2_clean[:9]}'")

    return warnings
