"""
LLM Prompt Engineering & Document Schema Builder.
Centralizes document schema definitions (passport, cnic_front, cnic_back)
and constructs strict evidence-driven system prompts for Nemotron 30B reasoning.
"""

from typing import Dict, Any, List
import json
from app.llm.schemas import LLMDocumentEvidence


PASSPORT_SCHEMA = {
    "document_type": "passport",
    "critical_fields": ["passport_number", "surname", "given_names", "date_of_birth", "date_of_expiry"],
    "fields": {
        "passport_number": {
            "type": "passport_number",
            "authoritative_sources": ["mrz", "visual_ocr"],
            "format": "^[A-Z0-9]{7,9}$"
        },
        "surname": {
            "type": "person_name",
            "authoritative_sources": ["mrz", "visual_ocr"],
            "format": "uppercase_text"
        },
        "given_names": {
            "type": "person_name",
            "authoritative_sources": ["mrz", "visual_ocr"],
            "format": "uppercase_text"
        },
        "father_name": {
            "type": "person_name",
            "authoritative_sources": ["visual_ocr"],
            "format": "uppercase_text"
        },
        "date_of_birth": {
            "type": "date",
            "authoritative_sources": ["mrz", "visual_ocr"],
            "format": "YYYY-MM-DD"
        },
        "date_of_issue": {
            "type": "date",
            "authoritative_sources": ["visual_ocr"],
            "format": "YYYY-MM-DD"
        },
        "date_of_expiry": {
            "type": "date",
            "authoritative_sources": ["mrz", "visual_ocr"],
            "format": "YYYY-MM-DD"
        },
        "cnic_number": {
            "type": "cnic_number",
            "authoritative_sources": ["mrz", "visual_ocr"],
            "format": "^\\d{5}-\\d{7}-\\d{1}$"
        }
    }
}


CNIC_FRONT_SCHEMA = {
    "document_type": "cnic_front",
    "critical_fields": ["cnic_number", "name", "date_of_birth"],
    "fields": {
        "name_en": {
            "type": "person_name",
            "language": "en",
            "script": "latin",
            "authoritative_sources": ["visual_ocr"]
        },
        "name_ur": {
            "type": "person_name",
            "language": "ur",
            "script": "urdu",
            "authoritative_sources": ["visual_ocr"]
        },
        "father_name_en": {
            "type": "person_name",
            "language": "en",
            "script": "latin",
            "authoritative_sources": ["visual_ocr"]
        },
        "father_name_ur": {
            "type": "person_name",
            "language": "ur",
            "script": "urdu",
            "authoritative_sources": ["visual_ocr"]
        },
        "cnic_number": {
            "type": "cnic_number",
            "authoritative_sources": ["visual_ocr"],
            "format": "^\\d{5}-\\d{7}-\\d{1}$"
        },
        "date_of_birth": {
            "type": "date",
            "authoritative_sources": ["visual_ocr"],
            "format": "YYYY-MM-DD"
        },
        "date_of_issue": {
            "type": "date",
            "authoritative_sources": ["visual_ocr"],
            "format": "YYYY-MM-DD"
        },
        "date_of_expiry": {
            "type": "date",
            "authoritative_sources": ["visual_ocr"],
            "format": "YYYY-MM-DD"
        }
    }
}


CNIC_BACK_SCHEMA = {
    "document_type": "cnic_back",
    "critical_fields": ["cnic_number"],
    "fields": {
        "cnic_number": {
            "type": "cnic_number",
            "authoritative_sources": ["visual_ocr"],
            "format": "^\\d{5}-\\d{7}-\\d{1}$"
        }
    }
}


def get_document_schema(doc_type: str) -> Dict[str, Any]:
    """Retrieve canonical document schema by type."""
    dt = doc_type.lower().strip()
    if "passport" in dt:
        return PASSPORT_SCHEMA
    elif "back" in dt:
        return CNIC_BACK_SCHEMA
    else:
        return CNIC_FRONT_SCHEMA


class LLMPromptBuilder:
    """
    Constructs strict prompt payloads for Nemotron 30B document reconciliation.
    """

    SYSTEM_INSTRUCTIONS = (
        "You are an expert identity-document evidence reconciliation engine for Pakistani CNICs and Passports.\n"
        "Your task is to analyze the OCR evidence, bounding boxes, spatial candidate extractions, and MRZ checksums\n"
        "to produce the most defensible, evidence-backed interpretation of the document.\n\n"
        "STRICT GUIDELINES:\n"
        "1. NEVER HALLUCINATE: Do not invent text, numbers, or names not supported by OCR/MRZ tokens.\n"
        "2. CANDIDATES ARE NOT GROUND TRUTH: Existing candidate assignments may be incorrect due to spatial misalignment.\n"
        "3. SPATIAL REASONING: Use token bounding boxes, line positions, and spatial proximity to determine correct fields.\n"
        "4. EVIDENTIARY HIERARCHY:\n"
        "   - Validated MRZ (checksum pass) > High-confidence visual OCR > Spatial label proximity > Deterministic candidate.\n"
        "5. DATE NORMALIZATION: All dates MUST be normalized to ISO-8601 YYYY-MM-DD (e.g. 07APR1966 -> 1966-04-07).\n"
        "   Preserve original raw text in raw_value.\n"
        "6. URDU PRESERVATION: Retain exact Urdu Unicode characters. NEVER machine-translate or fabricate English transliterations.\n"
        "7. CNIC FORMAT: Format CNIC numbers as XXXXX-XXXXXXX-X.\n"
        "8. DECISION ROUTING: Assign ACCEPT if evidence is strong; assign REVIEW if conflicting, ambiguous, or low confidence.\n"
        "9. JSON ONLY: Output ONLY valid JSON adhering strictly to the required schema. No Markdown fences, no explanation text.\n"
    )

    @staticmethod
    def build_prompt(evidence: LLMDocumentEvidence) -> Tuple[str, str]:
        """
        Construct (system_prompt, user_prompt) tuple for Nemotron adapter.
        """
        doc_schema = get_document_schema(evidence.document_type)

        payload = {
            "document_type": evidence.document_type,
            "document_geometry": evidence.document_geometry.model_dump(),
            "quality": evidence.quality.model_dump(),
            "candidate_fields": {k: v.model_dump() for k, v in evidence.candidate_fields.items()},
            "ocr_tokens": [t.model_dump() for t in evidence.ocr_tokens],
            "mrz": evidence.mrz.model_dump(),
            "document_schema": doc_schema
        }

        user_prompt = (
            f"Analyze the following evidence payload for a {evidence.document_type} and perform evidence-driven field reconciliation:\n\n"
            f"```json\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n```\n\n"
            "Produce the final JSON output matching LLMDocumentResult structure."
        )

        return LLMPromptBuilder.SYSTEM_INSTRUCTIONS, user_prompt
