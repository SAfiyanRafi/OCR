"""
Reading order sorting module.
Sorts OCRTokens top-to-bottom, left-to-right into natural reading order.
"""

from typing import List
from app.ocr.models import OCRToken


def sort_tokens_reading_order(
    tokens: List[OCRToken],
    line_threshold: float = 15.0
) -> List[OCRToken]:
    """
    Sort tokens by vertical Y position into text lines, then left-to-right within each line.
    """
    if not tokens:
        return []

    # Sort primarily by Y coordinate
    sorted_y = sorted(tokens, key=lambda t: t.bbox[1])

    lines: List[List[OCRToken]] = []
    current_line: List[OCRToken] = []
    current_y = None

    for tok in sorted_y:
        y1 = tok.bbox[1]
        if current_y is None:
            current_y = y1
            current_line.append(tok)
        elif abs(y1 - current_y) <= line_threshold:
            current_line.append(tok)
        else:
            # Sort previous line left-to-right by X coordinate
            current_line.sort(key=lambda t: t.bbox[0])
            lines.append(current_line)
            current_line = [tok]
            current_y = y1

    if current_line:
        current_line.sort(key=lambda t: t.bbox[0])
        lines.append(current_line)

    flattened: List[OCRToken] = []
    for line in lines:
        flattened.extend(line)

    return flattened
