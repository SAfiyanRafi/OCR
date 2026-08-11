"""
Data models and containers for Document Image Preprocessing Pipeline.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field


class BrightnessMetrics(BaseModel):
    """Metrics assessing image illumination and exposure."""
    mean: float
    median: float
    percentile_10: float
    percentile_90: float
    dark_region_percent: float
    overexposed_region_percent: float
    underexposed: bool
    overexposed: bool
    uneven_illumination: bool


class ContrastMetrics(BaseModel):
    """Metrics assessing image contrast."""
    rms_contrast: float
    michelson_contrast: float
    std_dev: float
    is_low_contrast: bool


class QualityReport(BaseModel):
    """Comprehensive image quality assessment report."""
    blur_score: float  # Laplacian variance
    is_blurry: bool
    brightness: BrightnessMetrics
    contrast: ContrastMetrics
    noise_level: float  # Estimated noise variance
    glare_ratio: float  # Saturated high-brightness low-variance pixel ratio
    glare_over_text: bool
    shadow_detected: bool
    perspective_distorted: bool
    overall_score: float  # 0.0 to 1.0 normalized quality score
    status: str  # "usable", "low_quality", "unusable"
    warnings: List[str] = Field(default_factory=list)


class TransformationMeta(BaseModel):
    """Audit log metadata tracking all preprocessing operations performed."""
    exif_orientation_corrected: bool = False
    exif_tag_found: Optional[int] = None
    
    document_detection: Dict[str, Any] = Field(default_factory=lambda: {
        "applied": False,
        "confidence": 0.0,
        "corners": None
    })
    
    perspective_correction: Dict[str, Any] = Field(default_factory=lambda: {
        "applied": False,
        "target_aspect_ratio": None
    })
    
    rotation: Dict[str, Any] = Field(default_factory=lambda: {
        "applied": False,
        "angle": 0.0
    })
    
    resize: Dict[str, Any] = Field(default_factory=lambda: {
        "applied": False,
        "original_size": [0, 0],
        "new_size": [0, 0],
        "scale_factor": 1.0
    })
    
    illumination_correction: Dict[str, Any] = Field(default_factory=lambda: {
        "applied": False,
        "method": None
    })
    
    clahe: Dict[str, Any] = Field(default_factory=lambda: {
        "applied": False,
        "clip_limit": 0.0,
        "tile_grid_size": [8, 8]
    })
    
    denoise: Dict[str, Any] = Field(default_factory=lambda: {
        "applied": False,
        "method": None
    })
    
    sharpen: Dict[str, Any] = Field(default_factory=lambda: {
        "applied": False,
        "method": None,
        "strength": None
    })
    
    threshold: Dict[str, Any] = Field(default_factory=lambda: {
        "applied": False,
        "method": None
    })
    
    morphology: Dict[str, Any] = Field(default_factory=lambda: {
        "applied": False,
        "operation": None
    })
    
    border_removal: Dict[str, Any] = Field(default_factory=lambda: {
        "applied": False,
        "crop_box": None
    })


class ImageVariantContainer:
    """Wrapper holding a preprocessing variant candidate image and metadata."""
    def __init__(
        self,
        name: str,
        image: np.ndarray,
        description: str,
        transformations: List[str],
        score: float = 0.0,
        ocr_result: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.image = image
        self.description = description
        self.transformations = transformations
        self.score = score
        self.ocr_result = ocr_result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "transformations": self.transformations,
            "score": self.score,
            "ocr_result": self.ocr_result
        }


class PreprocessingResult:
    """Master container returned by Preprocessor.process()."""
    def __init__(
        self,
        original_image: np.ndarray,
        preprocessed_image: np.ndarray,
        quality: QualityReport,
        transformations: TransformationMeta,
        best_image: Optional[np.ndarray] = None,
        variants: Optional[List[ImageVariantContainer]] = None,
        warnings: Optional[List[str]] = None,
        mrz_image: Optional[np.ndarray] = None,
        document_type: str = "generic"
    ):
        self.original_image = original_image
        self.preprocessed_image = preprocessed_image
        self.best_image = best_image if best_image is not None else preprocessed_image
        self.quality = quality
        self.transformations = transformations
        self.variants = variants if variants is not None else []
        self.warnings = warnings if warnings is not None else []
        self.mrz_image = mrz_image
        self.document_type = document_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_type": self.document_type,
            "quality": self.quality.model_dump(),
            "transformations": self.transformations.model_dump(),
            "warnings": self.warnings,
            "variants_count": len(self.variants),
            "mrz_extracted": self.mrz_image is not None
        }
