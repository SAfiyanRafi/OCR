"""
Pydantic Schemas and Dataclasses for Core Data Structures.
Enforces dual pixel/normalized coordinate tracking, Pydantic YAML configuration validation,
quality reports, decision plans, candidate scoring, CoordinateTransform, FieldROI,
and layered field provenance.
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Literal
from dataclasses import dataclass, field
import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class Point(BaseModel):
    x: float
    y: float


@dataclass
class DocumentBoundary:
    corners: List[Tuple[float, float]] = field(default_factory=list)
    confidence: float = 0.0
    detected: bool = False
    method: str = "fallback"

    def __iter__(self):
        return iter([self.corners, self.confidence])


@dataclass
class CoordinateTransform:
    """
    Transformation tracking container preserving bidirectional coordinate mappings
    between original input space and canonical document space.
    """
    original_to_canonical: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    canonical_to_original: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    original_size: Tuple[int, int] = (1000, 1000)  # (width, height)
    canonical_size: Tuple[int, int] = (2000, 1261) # (width, height)


@dataclass
class FieldROI:
    """
    Field-specific Region of Interest (ROI) cropped from canonical image.
    """
    field_name: str
    image: np.ndarray
    bbox_canonical: List[int] = field(default_factory=list)  # [x1, y1, x2, y2] px
    bbox_original: List[int] = field(default_factory=list)   # [x1, y1, x2, y2] px
    bbox_norm: List[float] = field(default_factory=list)    # [x1, y1, x2, y2] norm
    source: str = "region"


@dataclass
class QualityReport:
    blur_score: float = 0.0
    contrast_score: float = 0.0
    brightness_score: float = 0.0
    noise_score: float = 0.0
    glare_score: float = 0.0
    shadow_score: float = 0.0
    perspective_score: float = 0.0
    resolution_score: float = 0.0
    overall_score: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "usable" if self.overall_score >= 0.50 else "low_quality"


@dataclass
class PreprocessingPlan:
    perspective_correction: bool = True
    deskew: bool = True
    illumination_correction: bool = True
    denoise: Optional[str] = None
    contrast: Optional[str] = None
    sharpening: Optional[str] = None
    threshold: Optional[str] = None
    generate_variants: bool = True
    reasons: List[str] = field(default_factory=list)


@dataclass
class ImageStage:
    name: str
    image: np.ndarray
    parent_stage: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class OCRToken(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str
    confidence: float
    bbox_px: List[float] = Field(default_factory=list)
    bbox_norm: List[float] = Field(default_factory=list)
    image_width: int = 1000
    image_height: int = 1000
    page: int = 1
    index: int = 0
    script: str = "unknown"

    def __init__(self, **data):
        if "bbox" in data and not data.get("bbox_px"):
            data["bbox_px"] = list(data["bbox"])
        if "bbox_px" in data and not data.get("bbox_norm"):
            w = data.get("image_width", 1000)
            h = data.get("image_height", 1000)
            px = data["bbox_px"]
            if len(px) == 4:
                data["bbox_norm"] = [round(px[0]/max(1, w), 4), round(px[1]/max(1, h), 4), round(px[2]/max(1, w), 4), round(px[3]/max(1, h), 4)]
        super().__init__(**data)

    @property
    def bbox(self) -> List[float]:
        return self.bbox_px

    @property
    def center(self) -> Tuple[float, float]:
        px = self.bbox_px
        if len(px) == 4:
            return ((px[0] + px[2]) / 2.0, (px[1] + px[3]) / 2.0)
        return (0.0, 0.0)

    @property
    def width(self) -> float:
        px = self.bbox_px
        return abs(px[2] - px[0]) if len(px) == 4 else 0.0

    @property
    def height(self) -> float:
        px = self.bbox_px
        return abs(px[3] - px[1]) if len(px) == 4 else 0.0


@dataclass
class RawOCRResult:
    image_width: int
    image_height: int
    document_type: str
    tokens: List[OCRToken] = field(default_factory=list)
    raw_text: str = ""
    source: str = "engine"
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "document_type": self.document_type,
            "tokens": [t.model_dump() for t in self.tokens],
            "raw_text": self.raw_text,
            "source": self.source,
            "processing_time_ms": self.processing_time_ms
        }


@dataclass
class PreprocessingVariant:
    id: str
    name: str
    image: np.ndarray
    transformations: List[str] = field(default_factory=list)
    source_stage: str = "original"


@dataclass
class OCRCandidate:
    variant_id: str
    tokens: List[OCRToken]
    engine: str
    model: str
    runtime: str
    processing_time_ms: float = 0.0


@dataclass
class OCRCandidateScore:
    mean_confidence: float = 0.0
    text_coverage: float = 0.0
    expected_field_coverage: float = 0.0
    critical_field_score: float = 0.0
    format_score: float = 0.0
    spatial_score: float = 0.0
    total_score: float = 0.0
    field_score: float = 0.0
    matched_fields: Dict[str, str] = field(default_factory=dict)


class FieldProvenance(BaseModel):
    ocr_engine: str = "rapidocr"
    model: str = "PP-OCRv4"
    preprocessing_profile: str = "standard"
    variant: str = "variant_default"
    region: str = "none"
    anchor: Optional[str] = None
    token_indices: List[int] = Field(default_factory=list)
    bbox_canonical: List[float] = Field(default_factory=list)
    bbox_original: List[float] = Field(default_factory=list)
    bbox_norm: List[float] = Field(default_factory=list)
    bbox_px: List[float] = Field(default_factory=list)


class FieldResult(BaseModel):
    raw: Union[str, Dict[str, str]] = ""
    normalized: Union[str, Dict[str, str]] = ""
    value: Union[str, Dict[str, str]] = ""
    raw_value: Union[str, Dict[str, str]] = ""
    ocr_confidence: float = 0.0
    spatial_confidence: float = 0.0
    anchor_confidence: float = 0.0
    validation_confidence: float = 0.0
    field_confidence: float = 0.0
    confidence: float = 0.0
    validated: bool = False
    validation_errors: List[str] = Field(default_factory=list)
    source: Optional[FieldProvenance] = None
    script: str = "unknown"

    def __init__(self, **data):
        if "raw" in data and not data.get("raw_value"):
            data["raw_value"] = data["raw"]
        elif "raw_value" in data and not data.get("raw"):
            data["raw"] = data["raw_value"]

        if "normalized" in data and not data.get("value"):
            data["value"] = data["normalized"]
        elif "value" in data and not data.get("normalized"):
            data["normalized"] = data["value"]

        if "field_confidence" in data and not data.get("confidence"):
            data["confidence"] = data["field_confidence"]
        elif "confidence" in data and not data.get("field_confidence"):
            data["field_confidence"] = data["confidence"]

        super().__init__(**data)


class ClassificationResult(BaseModel):
    document_type: str
    confidence: float = 0.0
    method: str = "rule_based"


class RegionConfig(BaseModel):
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 1.0
    y2: float = 1.0
    margin: float = 0.0
    containment: Literal["center", "overlap", "iou"] = "center"
    minimum_overlap: float = 0.50


class AnchorConfig(BaseModel):
    keyword: str
    text: Optional[str] = None
    direction: Literal["right", "below", "left", "above"] = "right"
    max_distance: float = 500.0
    vertical_tolerance: float = 50.0
    horizontal_tolerance: float = 50.0
    fallback_to_region: bool = True
    fuzzy_match: bool = True

    def __init__(self, **data):
        if "text" in data and not data.get("keyword"):
            data["keyword"] = data["text"]
        elif "keyword" in data and not data.get("text"):
            data["text"] = data["keyword"]
        super().__init__(**data)


class FieldConfig(BaseModel):
    label: str
    language: Literal["en", "ur", "both"] = "en"
    script: Literal["latin", "urdu", "numeric", "mixed"] = "latin"
    strategy: Literal["region", "anchor", "hybrid"] = "region"
    preprocessing_profile: str = "standard"
    region: Optional[RegionConfig] = None
    anchor: Optional[AnchorConfig] = None
    normalization: str = "none"
    validator: str = "none"
    critical: bool = False


class CanonicalConfig(BaseModel):
    width: int = 2000
    height: int = 1261
    aspect_ratio: float = 1.5858


class GeometryConfig(BaseModel):
    expected_aspect_ratio: float = 1.5858
    aspect_ratio_tolerance: float = 0.08
    perspective_enabled: bool = True
    confidence_threshold: float = 0.60


class DocumentConfig(BaseModel):
    config_version: str = "v3.0"
    document_type: str
    language: str = "en"
    canonical: Optional[CanonicalConfig] = Field(default_factory=CanonicalConfig)
    geometry: Optional[GeometryConfig] = None
    fields: Dict[str, FieldConfig] = Field(default_factory=dict)
