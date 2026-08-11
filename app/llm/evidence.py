"""
LLM Document Evidence Builder.
Transforms raw OCR tokens, spatial candidate extractions, MRZ decoder output,
and image quality reports into a lightweight LLMDocumentEvidence payload for Nemotron reasoning.
"""

from typing import Dict, Any, List, Optional
from app.core.models import RawOCRResult, QualityReport, DocumentBoundary
from app.llm.schemas import (
    LLMDocumentEvidence,
    LLMTokenEvidence,
    LLMCandidateField,
    LLMCandidateValidation,
    LLMMRZEvidence,
    LLMMRZChecks,
    LLMDocumentGeometry,
    LLMQualityEvidence
)


class LLMEvidenceBuilder:
    """
    Evidence builder constructing structured evidence for LLM reasoning.
    Strips internal image arrays while retaining spatial tokens, candidates, and MRZ checksums.
    """

    @staticmethod
    def build_evidence(
        document_type: str,
        raw_ocr: RawOCRResult,
        candidate_fields: Dict[str, Any],
        mrz_data: Optional[Dict[str, Any]] = None,
        quality_report: Optional[QualityReport] = None,
        boundary: Optional[DocumentBoundary] = None,
        document_schema: Optional[Dict[str, Any]] = None
    ) -> LLMDocumentEvidence:
        """
        Build LLMDocumentEvidence container.
        """
        # 1. OCR Tokens Representation
        ocr_tokens_evidence: List[LLMTokenEvidence] = []
        for idx, t in enumerate(raw_ocr.tokens):
            tok_idx = getattr(t, "index", idx)
            tok_text = str(getattr(t, "text", "")).strip()
            tok_conf = float(getattr(t, "confidence", 0.0))
            bbox_px = [float(x) for x in getattr(t, "bbox_px", getattr(t, "bbox", []))]
            bbox_norm = [float(x) for x in getattr(t, "bbox_norm", [])]
            script = getattr(t, "script", "latin")

            ocr_tokens_evidence.append(LLMTokenEvidence(
                index=tok_idx,
                text=tok_text,
                confidence=round(tok_conf, 4),
                bbox_px=bbox_px,
                bbox_norm=bbox_norm,
                script=script
            ))

        # 2. Spatial Candidate Fields Representation
        candidate_fields_evidence: Dict[str, LLMCandidateField] = {}
        for f_name, f_data in candidate_fields.items():
            if isinstance(f_data, dict):
                cand_val = f_data.get("value")
                cand_val_str = cand_val.get("en", str(cand_val)) if isinstance(cand_val, dict) else str(cand_val) if cand_val is not None else None
                raw_val = f_data.get("raw_value")
                raw_val_str = str(raw_val) if raw_val is not None else None

                ocr_c = float(f_data.get("ocr_confidence", 0.0))
                spat_c = float(f_data.get("spatial_confidence", 0.0))
                comb_c = float(f_data.get("confidence", 0.0))
                is_valid = bool(f_data.get("validated", False))
                val_errs = list(f_data.get("validation_errors", []))

                prov = f_data.get("provenance") or f_data.get("source") or {}
                prov_dict = prov if isinstance(prov, dict) else (prov.model_dump() if hasattr(prov, "model_dump") else {})
                token_indices = prov_dict.get("token_indices", [])
                bbox = f_data.get("bbox") or f_data.get("bbox_canonical") or []

                candidate_fields_evidence[f_name] = LLMCandidateField(
                    candidate_value=cand_val_str,
                    raw_value=raw_val_str,
                    ocr_confidence=round(ocr_c, 4),
                    spatial_confidence=round(spat_c, 4),
                    combined_confidence=round(comb_c, 4),
                    validation=LLMCandidateValidation(passed=is_valid, errors=val_errs),
                    token_indices=token_indices,
                    bbox=[float(x) for x in bbox]
                )

        # 3. MRZ Evidence Representation
        mrz_dict = mrz_data or {}
        line1 = mrz_dict.get("line1")
        line2 = mrz_dict.get("line2")
        parsed_mrz = mrz_dict.get("parsed", {})
        checks_dict = mrz_dict.get("checks", {})

        mrz_evidence = LLMMRZEvidence(
            line1=line1,
            line2=line2,
            parsed=parsed_mrz,
            checks=LLMMRZChecks(
                document_number_check=bool(checks_dict.get("document_number_check", True if parsed_mrz.get("passport_number") else False)),
                dob_check=bool(checks_dict.get("dob_check", True if parsed_mrz.get("date_of_birth") else False)),
                expiry_check=bool(checks_dict.get("expiry_check", True if parsed_mrz.get("date_of_expiry") else False)),
                composite_check=bool(checks_dict.get("composite_check", True if parsed_mrz else False))
            )
        )

        # 4. Geometry & Quality
        w = raw_ocr.image_width
        h = raw_ocr.image_height
        persp = boundary.detected if boundary else True

        geom_evidence = LLMDocumentGeometry(
            canonical_width=w,
            canonical_height=h,
            perspective_corrected=persp,
            deskewed=True
        )

        qual_evidence = LLMQualityEvidence(
            overall_score=round(quality_report.overall_score, 4) if quality_report else 0.85,
            blur_score=round(quality_report.blur_score, 2) if quality_report else 250.0,
            contrast_score=round(quality_report.contrast_score, 4) if quality_report else 0.60,
            glare_score=round(quality_report.glare_score, 4) if quality_report else 0.0,
            shadow_score=round(quality_report.shadow_score, 4) if quality_report else 0.10
        )

        return LLMDocumentEvidence(
            document_type=document_type,
            image={"width": w, "height": h},
            document_geometry=geom_evidence,
            quality=qual_evidence,
            candidate_fields=candidate_fields_evidence,
            ocr_tokens=ocr_tokens_evidence,
            mrz=mrz_evidence,
            document_schema=document_schema or {}
        )
