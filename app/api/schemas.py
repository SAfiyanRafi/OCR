"""
Production FastAPI API Request & Response Schemas.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    type: str
    confidence: float = 1.0
    method: str = "classifier"


class OCRInfo(BaseModel):
    engine: str = "rapidocr"
    model: str = "PP-OCRv4"
    selected_variant: str = "default"


class FieldValueSchema(BaseModel):
    value: Union[str, Dict[str, str]]
    raw_value: Union[str, Dict[str, str]]
    ocr_confidence: float = 0.0
    spatial_confidence: float = 0.0
    confidence: float = 0.0
    validated: bool = False
    validation_errors: List[str] = Field(default_factory=list)
    bbox: List[float] = Field(default_factory=list)
    bbox_norm: List[float] = Field(default_factory=list)
    provenance: Optional[Dict[str, Any]] = None


class ProductionOCRResponse(BaseModel):
    status: str = "success"
    document_type: str
    review_state: str = "AUTO_ACCEPT"  # AUTO_ACCEPT, NEEDS_REVIEW, AUTO_REJECT
    review_reasons: List[str] = Field(default_factory=list)
    name: Optional[Dict[str, str]] = None
    father_name: Optional[Dict[str, str]] = None
    fields: Dict[str, Any]
    quality_report: Optional[Dict[str, Any]] = None
    preprocessing_plan: Optional[Dict[str, Any]] = None
    preprocessing_metadata: Optional[Dict[str, Any]] = None
    raw_ocr: Optional[Dict[str, Any]] = None
    audit_trail: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)


class ClassifyResponse(BaseModel):
    document_type: str
    confidence: float
    method: str


class PreprocessResponse(BaseModel):
    status: str = "success"
    quality_report: Dict[str, Any]
    preprocessing_plan: Dict[str, Any]
    stages: List[str]
    selected_variant: str
