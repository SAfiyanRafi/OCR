"""
Nemotron 30B Document Reasoning & Reconciliation Package.
Provides evidence building, prompt generation, Nemotron LLM adapter,
evidential reconciliation, calibrated confidence estimation, and structured JSON output.
"""

from app.llm.schemas import (
    LLMTokenEvidence,
    LLMCandidateField,
    LLMMRZEvidence,
    LLMDocumentEvidence,
    LLMFieldResult,
    LLMDocumentResult,
    FinalDocument
)
from app.llm.evidence import LLMEvidenceBuilder
from app.llm.reconciliation import LLMReconciler
from app.llm.nemotron import NemotronAdapter

__all__ = [
    "LLMTokenEvidence",
    "LLMCandidateField",
    "LLMMRZEvidence",
    "LLMDocumentEvidence",
    "LLMFieldResult",
    "LLMDocumentResult",
    "FinalDocument",
    "LLMEvidenceBuilder",
    "LLMReconciler",
    "NemotronAdapter"
]
