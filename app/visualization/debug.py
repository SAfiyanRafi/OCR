"""
Multi-layer Debug Visualization Renderer.
Renders raw OCR token bounding boxes (green), configured regions (blue),
extracted fields (yellow), and validation/review badges.
"""

from typing import Dict, Any, List
import cv2
import numpy as np


def draw_debug_visualization(image: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    """
    Render multi-layer debug annotations over preprocessed image safely.
    """
    canvas = image.copy()
    h, w = canvas.shape[:2]

    raw_ocr = result.get("raw_ocr", {})
    if isinstance(raw_ocr, dict):
        tokens = raw_ocr.get("tokens", [])
    else:
        tokens = getattr(raw_ocr, "tokens", [])

    fields = result.get("fields", {})
    review_state = result.get("review_state", "AUTO_ACCEPT")

    # Layer 1: Raw OCR Token Bounding Boxes (Thin Green Rectangles)
    for t in tokens:
        if isinstance(t, dict):
            bbox_px = t.get("bbox_px") or t.get("bbox")
        else:
            bbox_px = getattr(t, "bbox_px", getattr(t, "bbox", None))

        if bbox_px and len(bbox_px) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox_px]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 128), 1)

    # Layer 2: Extracted Field Bounding Boxes (Yellow/Green Rectangles)
    for key, field_obj in fields.items():
        if isinstance(field_obj, dict):
            bbox = field_obj.get("bbox")
            val = field_obj.get("value", "")
            validated = field_obj.get("validated", False)

            if bbox and len(bbox) == 4 and (bbox[2] - bbox[0]) > 0:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                color = (0, 230, 115) if validated else (0, 165, 255)

                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

                label_str = f"{key}: {val}" if isinstance(val, str) else f"{key}"
                label_str = label_str[:35]
                badge = "[V]" if validated else "[?]"

                cv2.putText(canvas, f"{badge} {label_str}", (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Layer 3: Header Review Status Banner
    banner_color = (0, 180, 80) if review_state == "AUTO_ACCEPT" else (0, 140, 240) if review_state == "NEEDS_REVIEW" else (0, 0, 220)
    cv2.rectangle(canvas, (0, 0), (w, 32), banner_color, -1)
    doc_type = result.get("document_type", "document").upper()
    banner_text = f"{doc_type} | REVIEW STATE: {review_state}"
    cv2.putText(canvas, banner_text, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas
