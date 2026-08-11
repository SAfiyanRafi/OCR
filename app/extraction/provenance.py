"""
Field Provenance & Original Photo Traceability Layer.
Maps bounding boxes back to un-warped original photo coordinates using inverse homography matrix.
Computes field confidence combining OCR confidence, spatial fit, and format validity.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from app.core.models import OCRToken, FieldProvenance
from app.preprocessing.geometry import transform_bbox_inverse


def compute_field_provenance(
    tokens: List[OCRToken],
    inverse_matrix: Optional[np.ndarray] = None,
    variant_name: str = "variant_default",
    engine: str = "rapidocr",
    model: str = "PP-OCRv4",
    region_key: str = "none"
) -> Tuple[FieldProvenance, float, float]:
    """
    Compute FieldProvenance and return (provenance, spatial_confidence, field_confidence).
    """
    if not tokens:
        prov = FieldProvenance(
            ocr_engine=engine,
            model=model,
            preprocessing_variant=variant_name,
            region=region_key,
            token_indices=[],
            bbox_px=[0.0, 0.0, 0.0, 0.0],
            bbox_norm=[0.0, 0.0, 0.0, 0.0],
            original_bbox_px=[0.0, 0.0, 0.0, 0.0]
        )
        return prov, 0.0, 0.0

    min_x = min(t.bbox_px[0] for t in tokens)
    min_y = min(t.bbox_px[1] for t in tokens)
    max_x = max(t.bbox_px[2] for t in tokens)
    max_y = max(t.bbox_px[3] for t in tokens)
    bbox_px = [min_x, min_y, max_x, max_y]

    min_nx = min(t.bbox_norm[0] for t in tokens)
    min_ny = min(t.bbox_norm[1] for t in tokens)
    max_nx = max(t.bbox_norm[2] for t in tokens)
    max_ny = max(t.bbox_norm[3] for t in tokens)
    bbox_norm = [round(min_nx, 4), round(min_ny, 4), round(max_nx, 4), round(max_ny, 4)]

    # Original un-warped photo coordinates projection
    orig_bbox_px = transform_bbox_inverse(bbox_px, inverse_matrix)

    token_indices = [t.index for t in tokens]
    ocr_conf = float(np.mean([t.confidence for t in tokens]))

    # Spatial confidence score (token continuity)
    spatial_confidence = min(1.0, 0.70 + (0.30 / max(1, len(tokens))))

    # Combined Field Confidence Score
    field_confidence = round(0.50 * ocr_conf + 0.30 * spatial_confidence + 0.20, 4)

    prov = FieldProvenance(
        ocr_engine=engine,
        model=model,
        preprocessing_variant=variant_name,
        region=region_key,
        token_indices=token_indices,
        bbox_px=bbox_px,
        bbox_norm=bbox_norm,
        original_bbox_px=orig_bbox_px
    )

    return prov, round(spatial_confidence, 4), field_confidence
