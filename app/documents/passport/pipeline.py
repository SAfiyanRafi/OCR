"""
Master Pipeline for Pakistani Passports (Biodata Page & MRZ).
Employs Multi-Variant Cross-Candidate Field Fusion, Document Classifier, Traceability, and Human Review.
"""

from typing import Dict, Any, Optional, List
import time
from app.core.versioning import PIPELINE_VERSION, CONFIG_VERSION_DEFAULT, PARSER_VERSION_DEFAULT
from app.preprocessing.pipeline import AdaptivePreprocessor
from app.ocr.paddle import PaddleOCRAdapter
from app.ocr.evaluator import OCRCandidateEvaluator
from app.classification.classifier import DocumentClassifier
from app.documents.passport.parser import PassportParser
from app.validation.cross_field import validate_cross_fields
from app.review.engine import evaluate_review_state


class PassportPipeline:
    """
    Master pipeline for Passport biodata page and MRZ extraction with Multi-Variant Fusion.
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
        Execute Passport processing end-to-end with Multi-Variant Candidate Fusion.
        """
        start_t = time.time()
        expected_type = "passport"

        # 1. Adaptive Preprocessing
        prep_res = self.preprocessor.process(image_input, document_type=expected_type)

        # 2. Multi-Variant Field Fusion: Parse OCR across all image variants
        all_variant_results: List[Dict[str, Any]] = []
        best_ocr = None
        best_variant = prep_res["variants"][0]
        best_score = -1.0

        for variant in prep_res["variants"]:
            raw_ocr = self.ocr_engine.extract_tokens(variant.image, document_type=expected_type)
            score_obj = OCRCandidateEvaluator.evaluate(raw_ocr, document_type=expected_type)

            parsed_res = self.parser.parse(
                raw_ocr,
                inverse_matrix=prep_res["processed_to_original_matrix"],
                variant_name=variant.name
            )

            all_variant_results.append({
                "variant": variant,
                "raw_ocr": raw_ocr,
                "score": score_obj.total_score,
                "parsed_res": parsed_res
            })

            if score_obj.total_score > best_score:
                best_score = score_obj.total_score
                best_ocr = raw_ocr
                best_variant = variant

        if not all_variant_results:
            raw_ocr = self.ocr_engine.extract_tokens(prep_res["best_image"], document_type=expected_type)
            parsed_res = self.parser.parse(raw_ocr, inverse_matrix=prep_res["processed_to_original_matrix"], variant_name="default")
            all_variant_results.append({"variant": best_variant, "raw_ocr": raw_ocr, "score": 0.5, "parsed_res": parsed_res})
            best_ocr = raw_ocr

        # 3. Fuse Best Fields Across Variants
        fused_fields = all_variant_results[0]["parsed_res"]["fields"].copy()

        target_field_keys = ["passport_number", "surname", "given_names", "father_name", "date_of_birth", "date_of_issue", "date_of_expiry", "cnic_number"]

        for key in target_field_keys:
            best_field_cand = None
            best_field_score = -1.0

            for v_res in all_variant_results:
                f_obj = v_res["parsed_res"]["fields"].get(key)
                if not f_obj:
                    continue

                val = str(f_obj.get("value", "")).strip()
                is_valid = f_obj.get("validated", False)
                conf = float(f_obj.get("confidence", 0.0))

                # Scoring formula per field candidate: validated bonus (+2.0) + confidence
                f_score = (2.0 if (is_valid and val) else 0.0) + (1.0 if val else 0.0) + conf

                if f_score > best_field_score:
                    best_field_score = f_score
                    best_field_cand = f_obj

            if best_field_cand and best_field_cand.get("value"):
                fused_fields[key] = best_field_cand

        parsed_res = {
            "document_type": expected_type,
            "name": {
                "en": f"{fused_fields.get('given_names', {}).get('value', '')} {fused_fields.get('surname', {}).get('value', '')}".strip()
            },
            "fields": fused_fields
        }

        # 4. Document Classification
        classification = DocumentClassifier.classify(best_ocr, explicit_type=expected_type)

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

        # Audit metadata
        audit_trail = {
            "pipeline_version": PIPELINE_VERSION,
            "config_version": CONFIG_VERSION_DEFAULT,
            "parser_version": PARSER_VERSION_DEFAULT,
            "document_type": expected_type,
            "classification": classification.model_dump(),
            "selected_variant": best_variant.name,
            "fused_variants_count": len(all_variant_results),
            "ocr_engine": "paddleocr_ppocrv4",
            "candidate_score": best_score,
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
            "raw_ocr": best_ocr.__dict__,
            "audit_trail": audit_trail,
            "warnings": warnings
        }

        return result
