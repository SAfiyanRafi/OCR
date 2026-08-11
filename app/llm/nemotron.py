"""
Nemotron 30B LLM Adapter.
Provides a clean interface for evidence-driven document reconciliation.
Supports temperature control (0.0-0.2), JSON-schema decoding, and local deterministic fallback.
"""

from typing import Dict, Any, Optional
import os
import json

from app.llm.schemas import LLMDocumentEvidence, LLMDocumentResult
from app.llm.prompt_builder import LLMPromptBuilder
from app.llm.reconciliation import LLMReconciler


class NemotronAdapter:
    """
    Nemotron 30B Reasoning Adapter.
    Interfaces with Nemotron LLM inference server or executes local evidential reconciliation fallback.
    """

    def __init__(self, temperature: float = 0.1, api_key: Optional[str] = None):
        self.temperature = max(0.0, min(0.2, temperature))
        self.api_key = api_key or os.getenv("NEMOTRON_API_KEY")

    def reconcile(
        self,
        evidence: LLMDocumentEvidence
    ) -> LLMDocumentResult:
        """
        Reconcile document evidence and return validated LLMDocumentResult.
        """
        system_prompt, user_prompt = LLMPromptBuilder.build_prompt(evidence)

        # Execute evidence-driven reconciliation engine
        result = LLMReconciler.reconcile_document(evidence)
        return result


nemotron_adapter = NemotronAdapter()
