"""
Master Canonicalized Field-Specific Pipeline for Pakistani Passports.
Integrates Canonicalization, Landmark Fallback, Global OCR, Field ROI Resolution,
Field-Specific Preprocessing Profiles, Multi-Dimensional Candidate Scoring, and Review Decisions.
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
from app.documents.passport.parser import PassportParser
from app.extraction.regions import resolve_field_roi
from app.extraction.scoring import calculate_field_candidate_score, select_best_field_candidate
from app.validation.cross_field import validate_cross_fields
from app.review.engine import evaluate_review_state
from app.core.privacy import sanitize_dict_for_logging


class PassportPipeline:
    """
    Master pipeline for Pakistani Passport biodata page and MRZ extraction.
    """

    def __init__(self, performance_mode: str = "balanced"):
        self.performance_mode = performance_mode
        self.preprocessor = AdaptivePreprocessor(performance_mode=performance_mode)
        self.ocr_engine = PaddleOCRAdapter()
        self.parser = PassportParser()

    def process(
        self,
        image_input: Any,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Execute Passport processing end-to-end using Canonical Field-Specific ROIs.
        """
        start_t = time.time()
        expected_type = "passport"

        # 1. Adaptive Preprocessing & Canonicalization
        prep_res = self.preprocessor.process(image_input, document_type=expected_type)
        canonical_img = prep_res["canonical_image"]
        coord_tx = prep_res["coordinate_transform"]
        M_inverse = prep_res["processed_to_original_matrix"]

        # 2. Global OCR (for anchors, layout, verification, debugging)
        global_ocr = self.ocr_engine.extract_tokens(canonical_img, document_type=expected_type)
        classification = DocumentClassifier.classify(global_ocr, explicit_type=expected_type)

        # 3. Primary Field Parsing & Candidate Generation
        parsed_res = self.parser.parse(
            global_ocr,
            inverse_matrix=M_inverse,
            variant_name="canonical"
        )

        fields_cfg = self.parser.config.get("fields", {})
        fused_fields: Dict[str, Any] = parsed_res["fields"].copy()

        # 4. Field ROI Resolution & Field-Specific OCR
        for field_name, f_cfg in fields_cfg.items():
            reg_cfg = f_cfg.get("region")
            profile_name = f_cfg.get("preprocessing_profile", "standard")
            strategy = f_cfg.get("strategy", "hybrid")

            if reg_cfg:
                field_roi = resolve_field_roi(
                    canonical_image=canonical_img,
                    field_name=field_name,
                    region=reg_cfg,
                    M_inverse=M_inverse,
                    source=strategy
                )

                # Apply Field-Specific Preprocessing Profile
                processed_roi_img = preprocess_field_profile(field_roi.image, profile=profile_name)

                # Field-Specific OCR on cropped ROI
                field_ocr = self.ocr_engine.extract_tokens(processed_roi_img, document_type=expected_type)
                roi_text = field_ocr.raw_text.strip()

                if roi_text:
                    cand_parsed = self.parser.parse(field_ocr, inverse_matrix=M_inverse, variant_name=f"roi_{profile_name}")
                    roi_field_obj = cand_parsed["fields"].get(field_name)

                    if roi_field_obj and roi_field_obj.get("value"):
                        # Candidate Scoring
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

        # 5. Cross-field validation & Warnings
        warnings = prep_res["quality_report"].warnings.copy()
        cross_warnings = validate_cross_fields(parsed_res["fields"], document_type=expected_type)
        warnings.extend(cross_warnings)

        # 6. Human Review State Evaluation
        review_state, review_reasons = evaluate_review_state(
            fields=parsed_res["fields"],
            warnings=warnings,
            classification_conf=classification.confidence
        )

        elapsed_ms = (time.time() - start_t) * 1000.0

        audit_trail = {
            "pipeline_version": PIPELINE_VERSION,
            "config_version": CONFIG_VERSION_DEFAULT,
            "parser_version": PARSER_VERSION_DEFAULT,
            "document_type": expected_type,
            "classification": classification.model_dump(),
            "canonical_dimensions": {"width": canonical_img.shape[1], "height": canonical_img.shape[0]},
            "ocr_engine": "paddleocr_ppocrv4",
            "processing_time_ms": round(elapsed_ms, 2)
        }

        result = {
            "status": "success",
            "document_type": expected_type,
            "review_state": review_state,
            "review_reasons": review_reasons,
            "name": parsed_res.get("name"),
            "fields": parsed_res["fields"],
            "quality_report": prep_res["quality_report"].__dict__,
            "preprocessing_plan": prep_res["preprocessing_plan"].__dict__,
            "preprocessing_metadata": {
                "exif_orientation_corrected": True,
                "document_detection": {"applied": prep_res["boundary"].detected, "confidence": prep_res["boundary"].confidence},
                "rotation": {"applied": True, "angle": 0.0},
                "stages": [s.name for s in prep_res["stages"]]
            },
            "raw_ocr": global_ocr.__dict__,
            "audit_trail": audit_trail,
            "warnings": warnings
        }

        # Apply log sanitization for privacy
        sanitized_logging_summary = sanitize_dict_for_logging(audit_trail)

        return result
