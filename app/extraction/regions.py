"""
Region Token Containment Extractor.
Supports center-point, bounding box overlap, and Intersection-over-Union (IoU) containment modes.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from app.core.models import OCRToken, RegionConfig


def normalize_bbox(bbox: Tuple[float, float, float, float], img_width: float = 1000.0, img_height: float = 1000.0) -> Tuple[float, float, float, float]:
    """Normalize pixel bounding box [x1, y1, x2, y2] to 0.0 - 1.0."""
    return (
        round(bbox[0] / max(1.0, img_width), 4),
        round(bbox[1] / max(1.0, img_height), 4),
        round(bbox[2] / max(1.0, img_width), 4),
        round(bbox[3] / max(1.0, img_height), 4)
    )


def parse_region_bounds(region: Any, w: int = 1000, h: int = 1000) -> Tuple[float, float, float, float]:
    """Parse region bounds into tuple (x1, y1, x2, y2)."""
    if isinstance(region, RegionConfig):
        return (region.x1, region.y1, region.x2, region.y2)
    elif isinstance(region, dict):
        return (float(region.get("x1", 0.0)), float(region.get("y1", 0.0)), float(region.get("x2", 1.0)), float(region.get("y2", 1.0)))
    elif isinstance(region, (list, tuple)) and len(region) == 4:
        return (float(region[0]), float(region[1]), float(region[2]), float(region[3]))
    return (0.0, 0.0, 1.0, 1.0)


def calculate_overlap_ratio(bboxA: Tuple[float, float, float, float], bboxB: Tuple[float, float, float, float]) -> float:
    """
    Calculate area overlap ratio of bboxA relative to bboxA's total area.
    """
    xA = max(bboxA[0], bboxB[0])
    yA = max(bboxA[1], bboxB[1])
    xB = min(bboxA[2], bboxB[2])
    yB = min(bboxA[3], bboxB[3])

    inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxA_area = (bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1])

    if boxA_area <= 0.0:
        return 0.0
    return inter_area / boxA_area


def calculate_iou(bboxA: Tuple[float, float, float, float], bboxB: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection-over-Union (IoU) ratio.
    """
    xA = max(bboxA[0], bboxB[0])
    yA = max(bboxA[1], bboxB[1])
    xB = min(bboxA[2], bboxB[2])
    yB = min(bboxA[3], bboxB[3])

    inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxA_area = (bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1])
    boxB_area = (bboxB[2] - bboxB[0]) * (bboxB[3] - bboxB[1])
    union_area = boxA_area + boxB_area - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def is_point_inside_rect(px: float, py: float, rect: List[float]) -> bool:
    """Check if point (px, py) is inside normalized rectangle [x1, y1, x2, y2]."""
    return rect[0] <= px <= rect[2] and rect[1] <= py <= rect[3]


def extract_tokens_in_region(
    tokens: List[OCRToken],
    region: Union[Dict[str, Any], RegionConfig],
    img_width: int = 1000,
    img_height: int = 1000
) -> List[OCRToken]:
    """
    Extract OCR tokens inside a configured region.
    """
    if isinstance(region, RegionConfig):
        rx1, ry1, rx2, ry2 = region.x1, region.y1, region.x2, region.y2
        mode = region.containment
        min_overlap = region.minimum_overlap
    else:
        rx1 = float(region.get("x1", 0.0))
        ry1 = float(region.get("y1", 0.0))
        rx2 = float(region.get("x2", 1.0))
        ry2 = float(region.get("y2", 1.0))
        mode = region.get("containment", "center")
        min_overlap = float(region.get("minimum_overlap", 0.50))

    region_rect = [rx1, ry1, rx2, ry2]
    matched: List[OCRToken] = []

    for t in tokens:
        bx1, by1, bx2, by2 = t.bbox_norm if hasattr(t, "bbox_norm") and t.bbox_norm else [t.bbox[0]/img_width, t.bbox[1]/img_height, t.bbox[2]/img_width, t.bbox[3]/img_height]
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0

        if mode == "center":
            if is_point_inside_rect(cx, cy, region_rect):
                matched.append(t)
        elif mode == "overlap":
            ratio = calculate_overlap_ratio((bx1, by1, bx2, by2), (rx1, ry1, rx2, ry2))
            if ratio >= min_overlap:
                matched.append(t)
        elif mode == "iou":
            iou = calculate_iou((bx1, by1, bx2, by2), (rx1, ry1, rx2, ry2))
            if iou >= min_overlap:
                matched.append(t)
        else:
            if is_point_inside_rect(cx, cy, region_rect):
                matched.append(t)

    return matched
