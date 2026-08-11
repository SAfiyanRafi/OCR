"""
Base document profile specification interface.
"""

from typing import Dict, Any, List, Optional
import re


class DocumentProfile:
    """Base class for document profiles."""
    name: str = "generic"
    aspect_ratio: float = 1.5
    target_width: int = 2000
    target_height: int = 1333
    expected_fields: List[str] = []
    field_patterns: Dict[str, str] = {}
    is_mrz_document: bool = False

    @classmethod
    def evaluate_text_content(cls, texts: List[str]) -> Dict[str, Any]:
        """
        Evaluate extracted OCR text items against expected document fields and patterns.
        
        Returns:
            {
               "critical_fields_found": int,
               "total_fields_expected": int,
               "field_matches": Dict[str, str],
               "field_score": float
            }
        """
        matched_fields = {}
        full_text = " ".join(texts)

        for field, pattern in cls.field_patterns.items():
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                matched_fields[field] = match.group(0)

        found_count = len(matched_fields)
        total_expected = max(1, len(cls.expected_fields))
        score = float(found_count) / float(total_expected)

        return {
            "critical_fields_found": found_count,
            "total_fields_expected": total_expected,
            "field_matches": matched_fields,
            "field_score": score
        }
