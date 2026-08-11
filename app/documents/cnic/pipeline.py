"""
Master Canonicalized Field-Specific Pipeline for Pakistani CNIC (Front & Back).
Integrates Canonicalization, Landmark Fallback, Global OCR, Field ROI Resolution,
Field-Specific Preprocessing Profiles (Urdu-Safe, Numeric, Date), Candidate Scoring,
Nemotron 30B LLM Reconciliation, and Review Decisions.
"""

from typing import Dict, Any, Optional, List
import time
import numpy as np

from app.core.versioning import PIPELINE_VERSION, CONFIG_VERSION_DEFAULT, PARSER_VERSION_DEFAULT
from app.core.models import FieldROI, FieldResult, FieldProvenance
from app.preprocessing.pipeline import AdaptivePreprocessor
from app.preprocessing.profiles import preprocess_field_profile
from app.ocr.paddle import PaddleOCRAdapter
from app.ocr.evaluator import OCRCandidateEvaluator
from app.classification.classifier import DocumentClassifier
from app.documents.cnic.parser import CNICParser
from app.extraction.regions import resolve_field_roi
from app.extraction.scoring import calculate_field_candidate_score, select_best_field_candidate
from app.validation.cross_field import validate_cross_fields
from app.review.engine import evaluate_review_state
from app.core.privacy import sanitize_dict_for_logging

# LLM Nemotron 30B Reconciliation Imports
from app.llm.evidence import LLMEvidenceBuilder
from app.llm.nemotron import NemotronAdapter
from app.llm.schemas import FinalDocument, FinalDocumentField


class CNICPipeline:
    """
    Master pipeline for CNIC front and back document processing with Nemotron 30B Reconciliation.
    """

    def __init__(self, performance_mode: str = "balanced"):
        self.performance_mode = performance_mode
        self.preprocessor = AdaptivePreprocessor(performance_mode=performance_mode)
        self.ocr_engine = PaddleOCRAdapter()
        self.parser_front = CNICParser(doc_side="front")
        self.parser_back = CNICParser(doc_side="back")
        self.nemotron = NemotronAdapter()

    def process(
        self,
        image_input: Any,
        doc_side: str = "front",
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Execute CNIC processing end-to-end using Canonical Field-Specific ROIs & Nemotron 30B Reconciliation.
        """
        start_t = time.time()
        expected_type = "cnic_back" if doc_side == "back" else "cnic_front"

        # 1. Adaptive Preprocessing & Canonicalization
        prep_res = self.preprocessor.process(image_input, document_type=expected_type)
        canonical_img = prep_res["canonical_image"]
        coord_tx = prep_res["coordinate_transform"]
        M_inverse = prep_res["processed_to_original_matrix"]

        # 2. Global OCR (for verification, anchors, layout analysis, debugging)
        global_ocr = self.ocr_engine.extract_tokens(canonical_img, document_type=expected_type)
        classification = DocumentClassifier.classify(global_ocr, explicit_type=expected_type)

        # 3. Primary Field Parsing & Candidate Generation
        parser = self.parser_back if doc_side == "back" else self.parser_front
        parsed_res = parser.parse(
            global_ocr,
            inverse_matrix=M_inverse,
            variant_name="canonical"
        )

        fields_cfg = parser.config.get("fields", {})
        fused_fields: Dict[str, Any] = parsed_res["fields"].copy()

        # 4. Field ROI Resolution & Field-Specific OCR
        for field_name, f_cfg in fields_cfg.items():
            reg_cfg = f_cfg.get("region")
            profile_name = f_cfg.get("preprocessing_profile", "standard")
            strategy = f_cfg.get("strategy", "region")

            if reg_cfg:
                field_roi = resolve_field_roi(
                    canonical_image=canonical_img,
                    field_name=field_name,
                    region=reg_cfg,
                    M_inverse=M_inverse,
                    source=strategy
                )

                processed_roi_img = preprocess_field_profile(field_roi.image, profile=profile_name)
                field_ocr = self.ocr_engine.extract_tokens(processed_roi_img, document_type=expected_type)
                roi_text = field_ocr.raw_text.strip()

                if roi_text:
                    cand_parsed = parser.parse(field_ocr, inverse_matrix=M_inverse, variant_name=f"roi_{profile_name}")
                    roi_field_obj = cand_parsed["fields"].get(field_name)

                    if roi_field_obj and roi_field_obj.get("value"):
                        best_cand = select_best_field_candidate([
                            {**fused_fields.get(field_name, {}), "strategy": strategy},
                            {**roi_field_obj, "strategy": strategy, "is_anchor_matched": False}
                        ])

                        provenance = FieldProvenance(
                            ocr_engine="paddleocr_ppocrv4",
                            model="PP-OCRv4",
                            preprocessing_profile=profile_name,
                            variant=f"canonical_roi_{profile_name}",
                            region=field_name,
                            bbox_canonical=[float(x) for x in field_roi.bbox_canonical],
                            bbox_original=[float(x) for x in field_roi.bbox_original],
                            bbox_norm=field_roi.bbox_norm,
                            bbox_px=[float(x) for x in field_roi.bbox_canonical]
                        )

                        best_cand["source"] = provenance.model_dump()
                        best_cand["bbox"] = field_roi.bbox_canonical
                        best_cand["bbox_norm"] = field_roi.bbox_norm
                        best_cand["bbox_canonical"] = field_roi.bbox_canonical
                        best_cand["bbox_original"] = field_roi.bbox_original

                        fused_fields[field_name] = best_cand

        parsed_res["fields"] = fused_fields

        # 5. Nemotron 30B LLM Evidence Building & Reconciliation
        llm_evidence = LLMEvidenceBuilder.build_evidence(
            document_type=expected_type,
            raw_ocr=global_ocr,
            candidate_fields=fused_fields,
            quality_report=prep_res["quality_report"],
            boundary=prep_res["boundary"]
        )

        llm_result = self.nemotron.reconcile(llm_evidence)

        # 6. Cross-field validation & Human Review Routing
        warnings = prep_res["quality_report"].warnings.copy()
        warnings.extend(llm_result.review_reasons)

        review_state, review_reasons = evaluate_review_state(
            fields=parsed_res["fields"],
            warnings=warnings,
            classification_conf=classification.confidence
        )

        if llm_result.review_required:
            review_state = "NEEDS_REVIEW" if review_state == "AUTO_ACCEPT" else review_state

        elapsed_ms = (time.time() - start_t) * 1000.0

        audit_trail = {
            "pipeline_version": PIPELINE_VERSION,
            "config_version": CONFIG_VERSION_DEFAULT,
            "parser_version": PARSER_VERSION_DEFAULT,
            "document_type": expected_type,
            "classification": classification.model_dump(),
            "canonical_dimensions": {"width": canonical_img.shape[1], "height": canonical_img.shape[0]},
            "ocr_engine": "paddleocr_ppocrv4",
            "nemotron_model": "Nemotron-30B",
            "processing_time_ms": round(elapsed_ms, 2)
        }

        # Build public FinalDocument representation
        final_fields_api: Dict[str, Any] = {}
        for fk, f_res in llm_result.fields.items():
            final_fields_api[fk] = {
                "value": f_res.value,
                "raw_value": f_res.raw_value,
                "normalized_value": f_res.normalized_value,
                "decision": f_res.decision,
                "confidence": f_res.confidence,
                "source": f_res.source,
                "validated": f_res.validation.format_valid,
                "script": f_res.script,
                "bbox": fused_fields.get(fk, {}).get("bbox", [])
            }

        name_en = llm_result.fields.get("name_en", {}).value or ""
        name_ur = llm_result.fields.get("name_ur", {}).value or ""
        father_en = llm_result.fields.get("father_name_en", {}).value or ""
        father_ur = llm_result.fields.get("father_name_ur", {}).value or ""

        result = {
            "status": "success",
            "document_type": expected_type,
            "review_state": review_state,
            "review_required": llm_result.review_required,
            "review_reasons": warnings,
            "name": {"en": str(name_en), "ur": str(name_ur)},
            "father_name": {"en": str(father_en), "ur": str(father_ur)},
            "fields": final_fields_api,
            "llm_reconciliation": llm_result.model_dump(),
            "quality_report": prep_res["quality_report"].__dict__,
            "preprocessing_plan": prep_res["preprocessing_plan"].__dict__,
            "raw_ocr": global_ocr.__dict__,
            "audit_trail": audit_trail,
            "warnings": warnings
        }

        return result
