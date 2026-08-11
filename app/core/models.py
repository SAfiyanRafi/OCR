"""
Pydantic Schemas and Dataclasses for Core Data Structures.
Enforces dual pixel/normalized coordinate tracking, Pydantic YAML configuration validation,
quality reports, decision plans, candidate scoring, and field provenance.
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
    preprocessing_variant: str = "variant_default"
    region: str = "none"
    token_indices: List[int] = Field(default_factory=list)
    bbox_px: List[float] = Field(default_factory=list)
    bbox_norm: List[float] = Field(default_factory=list)
    original_bbox_px: List[float] = Field(default_factory=list)


class FieldResult(BaseModel):
    value: Union[str, Dict[str, str]]
    raw_value: Union[str, Dict[str, str]]
    confidence: float = 0.0
    spatial_confidence: float = 0.0
    field_confidence: float = 0.0
    validated: bool = False
    validation_errors: List[str] = Field(default_factory=list)
    source: Optional[FieldProvenance] = None


class ClassificationResult(BaseModel):
    document_type: str
    confidence: float = 0.0
    method: str = "rule_based"


class RegionConfig(BaseModel):
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 1.0
    y2: float = 1.0
    containment: Literal["center", "overlap", "iou"] = "center"
    minimum_overlap: float = 0.50


class AnchorConfig(BaseModel):
    keyword: str
    direction: Literal["right", "below", "left", "above"] = "right"
    max_distance: float = 500.0
    vertical_tolerance: float = 50.0
    fallback_to_region: bool = True
    fuzzy_match: bool = True


class FieldConfig(BaseModel):
    label: str
    language: Literal["en", "ur", "both"] = "en"
    strategy: Literal["region", "anchor", "hybrid"] = "region"
    region: Optional[RegionConfig] = None
    anchor: Optional[AnchorConfig] = None
    normalization: str = "none"
    validator: str = "none"
    critical: bool = False


class GeometryConfig(BaseModel):
    expected_aspect_ratio: float = 1.5858
    aspect_ratio_tolerance: float = 0.08
    perspective_enabled: bool = True
    confidence_threshold: float = 0.80


class DocumentConfig(BaseModel):
    config_version: str = "v3.0"
    document_type: str
    language: str = "en"
    geometry: Optional[GeometryConfig] = None
    fields: Dict[str, FieldConfig] = Field(default_factory=dict)
