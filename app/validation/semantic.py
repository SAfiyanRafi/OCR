"""
Semantic Validators.
Validates logical bounds, calendar dates, and ranges.
"""

from typing import Tuple, Optional


def parse_date_parts(date_str: str) -> Optional[Tuple[int, int, int]]:
    """Parse DD.MM.YYYY into (day, month, year)."""
    try:
        parts = date_str.strip().split(".")
        if len(parts) == 3:
            return int(parts[0]), int(parts[1]), int(parts[2])
    except Exception:
        pass
    return None


def validate_date_semantic(value: str) -> bool:
    """Validate calendar bounds (1 <= day <= 31, 1 <= month <= 12, 1900 <= year <= 2100)."""
    parts = parse_date_parts(value)
    if not parts:
        return False
    d, m, y = parts
    return (1 <= d <= 31) and (1 <= m <= 12) and (1900 <= y <= 2100)
