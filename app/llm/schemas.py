"""
Pydantic Schemas for Nemotron LLM Evidence, Reconciliation, and Public API Contracts.
Strictly typing token evidence, candidate fields, MRZ verification, field decisions,
provenance tracking, and public API representations.
"""

from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, ConfigDict


class LLMTokenEvidence(BaseModel):
    index: int
    text: str
    confidence: float
    bbox_px: List[float] = Field(default_factory=list)
    bbox_norm: List[float] = Field(default_factory=list)
    script: str = "latin"


class LLMCandidateValidation(BaseModel):
    passed: bool = False
    errors: List[str] = Field(default_factory=list)


class LLMCandidateField(BaseModel):
    candidate_value: Optional[str] = None
    raw_value: Optional[str] = None
    ocr_confidence: float = 0.0
    spatial_confidence: float = 0.0
    combined_confidence: float = 0.0
    validation: LLMCandidateValidation = Field(default_factory=LLMCandidateValidation)
    token_indices: List[int] = Field(default_factory=list)
    bbox: List[float] = Field(default_factory=list)


class LLMMRZChecks(BaseModel):
    document_number_check: bool = False
    dob_check: bool = False
    expiry_check: bool = False
    composite_check: bool = False


class LLMMRZEvidence(BaseModel):
    line1: Optional[str] = None
    line2: Optional[str] = None
    parsed: Dict[str, Any] = Field(default_factory=dict)
    checks: LLMMRZChecks = Field(default_factory=LLMMRZChecks)


class LLMDocumentGeometry(BaseModel):
    canonical_width: int = 2000
    canonical_height: int = 1261
    perspective_corrected: bool = True
    deskewed: bool = True


class LLMQualityEvidence(BaseModel):
    overall_score: float = 0.0
    blur_score: float = 0.0
    contrast_score: float = 0.0
    glare_score: float = 0.0
    shadow_score: float = 0.0


class LLMDocumentEvidence(BaseModel):
    document_type: str
    document_side: Optional[str] = None
    image: Dict[str, int] = Field(default_factory=lambda: {"width": 2000, "height": 1261})
    document_geometry: LLMDocumentGeometry = Field(default_factory=LLMDocumentGeometry)
    quality: LLMQualityEvidence = Field(default_factory=LLMQualityEvidence)
    candidate_fields: Dict[str, LLMCandidateField] = Field(default_factory=dict)
    ocr_tokens: List[LLMTokenEvidence] = Field(default_factory=list)
    mrz: LLMMRZEvidence = Field(default_factory=LLMMRZEvidence)
    validation_results: Dict[str, Any] = Field(default_factory=dict)
    document_schema: Dict[str, Any] = Field(default_factory=dict)


class LLMFieldValidation(BaseModel):
    format_valid: bool = False
    checksum_valid: bool = False
    errors: List[str] = Field(default_factory=list)


class LLMFieldResult(BaseModel):
    value: Optional[Union[str, Dict[str, str]]] = None
    raw_value: Optional[Union[str, Dict[str, str]]] = None
    normalized_value: Optional[Union[str, Dict[str, str]]] = None
    decision: Literal["ACCEPT", "REVIEW", "UNKNOWN"] = "REVIEW"
    confidence: float = 0.0
    source: str = "visual_ocr"
    source_token_indices: List[int] = Field(default_factory=list)
    validation: LLMFieldValidation = Field(default_factory=LLMFieldValidation)
    language: str = "en"
    script: str = "latin"


class LLMEvidenceSummary(BaseModel):
    mrz_consistency: bool = False
    spatial_consistency: bool = False
    cross_field_consistency: bool = False


class LLMModelMetadata(BaseModel):
    model: str = "Nemotron-30B"
    pipeline_version: str = "3.0.0"
    reconciliation_mode: str = "evidence_driven"


class LLMDocumentResult(BaseModel):
    document_type: str
    status: Literal["success", "review", "failed"] = "success"
    review_required: bool = False
    fields: Dict[str, LLMFieldResult] = Field(default_factory=dict)
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    review_reasons: List[str] = Field(default_factory=list)
    evidence_summary: LLMEvidenceSummary = Field(default_factory=LLMEvidenceSummary)
    model_metadata: LLMModelMetadata = Field(default_factory=LLMModelMetadata)


class FinalDocumentField(BaseModel):
    value: Optional[Union[str, Dict[str, str]]] = None
    raw_value: Optional[Union[str, Dict[str, str]]] = None
    normalized_value: Optional[Union[str, Dict[str, str]]] = None
    decision: str = "ACCEPT"
    confidence: float = 0.0
    source: str = "visual_ocr"
    validated: bool = True
    script: str = "latin"


class FinalDocument(BaseModel):
    document_type: str
    status: str = "success"
    review_state: str = "AUTO_ACCEPT"
    review_required: bool = False
    review_reasons: List[str] = Field(default_factory=list)
    fields: Dict[str, FinalDocumentField] = Field(default_factory=dict)
    name: Optional[Union[str, Dict[str, str]]] = None
    father_name: Optional[Union[str, Dict[str, str]]] = None
    audit_trail: Dict[str, Any] = Field(default_factory=dict)
