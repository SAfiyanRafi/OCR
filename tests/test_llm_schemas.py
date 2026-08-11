"""
Unit tests for Nemotron LLM Data Schemas and Public API Contracts.
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


def test_llm_schemas_instantiation():
    token = LLMTokenEvidence(
        index=1,
        text="07APR1966",
        confidence=0.92,
        bbox_px=[10.0, 20.0, 100.0, 50.0],
        script="latin"
    )
    assert token.index == 1
    assert token.text == "07APR1966"

    candidate = LLMCandidateField(
        candidate_value="07.04.1966",
        raw_value="07APR1966",
        ocr_confidence=0.92,
        spatial_confidence=0.85,
        combined_confidence=0.89,
        token_indices=[1]
    )
    assert candidate.candidate_value == "07.04.1966"
    assert candidate.token_indices == [1]

    field_res = LLMFieldResult(
        value="1966-04-07",
        raw_value="07APR1966",
        normalized_value="1966-04-07",
        decision="ACCEPT",
        confidence=0.96,
        source="mrz+visual_ocr",
        source_token_indices=[1]
    )
    assert field_res.value == "1966-04-07"
    assert field_res.decision == "ACCEPT"
    assert field_res.confidence == 0.96
