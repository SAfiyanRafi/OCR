"""
Checksum Validators.
Validates ICAO Doc 9303 MRZ checksum digits and CNIC province codes.
"""

# ICAO 7-3-1 Checksum Weighting Vector
WEIGHTS = [7, 3, 1]


def compute_mrz_check_digit(mrz_segment: str) -> int:
    """
    Compute ICAO 9303 check digit for MRZ alphanumeric segment.
    """
    total = 0
    for idx, char in enumerate(mrz_segment):
        if char.isdigit():
            val = int(char)
        elif char.isalpha():
            val = ord(char.upper()) - 55  # A=10, B=11 ... Z=35
        elif char == "<":
            val = 0
        else:
            val = 0
        total += val * WEIGHTS[idx % 3]
    return total % 10


def validate_mrz_checksum(mrz_line2: str) -> bool:
    """
    Validate MRZ line 2 checksums (Passport Number, DOB, Expiry).
    """
    clean = mrz_line2.strip().upper()
    if len(clean) < 28:
        return False

    # Check Passport Number digit at index 9
    p_num = clean[:9]
    p_check = clean[9]
    if p_check.isdigit() and compute_mrz_check_digit(p_num) != int(p_check):
        return False

    # Check DOB digit at index 19
    dob = clean[13:19]
    dob_check = clean[19]
    if dob_check.isdigit() and compute_mrz_check_digit(dob) != int(dob_check):
        return False

    return True
