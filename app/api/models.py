"""
FastAPI Request & Response Schemas.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class FieldResult(BaseModel):
    value: str
    raw_value: Optional[str] = ""
    confidence: float = 0.0
    validated: bool = False
    bbox: List[float] = Field(default_factory=list)


class OCRResponse(BaseModel):
    document_type: str
    fields: Dict[str, Any]
    name: Optional[Dict[str, str]] = None
    father_name: Optional[Dict[str, str]] = None
    quality_report: Optional[Dict[str, Any]] = None
    preprocessing_metadata: Optional[Dict[str, Any]] = None
    raw_ocr: Optional[Dict[str, Any]] = None


class CreateConfigRequest(BaseModel):
    config_name: str  # e.g. driving_license.yaml
    document_type: str  # e.g. driving_license
    language: Optional[str] = "en"


class SaveFieldRequest(BaseModel):
    config_name: str  # cnic_front.yaml, passport.yaml, etc.
    field_key: str
    label: str
    language: Optional[str] = "en"
    strategy: Optional[str] = "region"
    x1: float
    y1: float
    x2: float
    y2: float
    anchor_keyword: Optional[str] = None
    anchor_direction: Optional[str] = "right"
    normalization: Optional[str] = "none"
    validator: Optional[str] = "none"


class RegionConfigRequest(BaseModel):
    config_name: str
    field_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    label: Optional[str] = None
    strategy: Optional[str] = "region"
    normalization: Optional[str] = "none"
    validator: Optional[str] = "none"
