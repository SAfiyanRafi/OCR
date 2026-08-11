"""
Master Pipeline for Pakistani CNIC (Front & Back).
Integrates Adaptive Preprocessing, OCRCandidateEvaluator, DocumentClassifier,
Bilingual Field Association, Traceability, and Human Review State Evaluation.
"""

from typing import Dict, Any, Optional
import time
from app.core.versioning import PIPELINE_VERSION, CONFIG_VERSION_DEFAULT, PARSER_VERSION_DEFAULT
from app.preprocessing.pipeline import AdaptivePreprocessor
from app.ocr.paddle import PaddleOCRAdapter
from app.ocr.evaluator import OCRCandidateEvaluator
from app.classification.classifier import DocumentClassifier
from app.documents.cnic.parser import CNICParser
from app.validation.cross_field import validate_cross_fields
from app.review.engine import evaluate_review_state


class CNICPipeline:
    """
    Master pipeline for CNIC front and back document processing.
    """

    def __init__(self, performance_mode: str = "balanced"):
        self.performance_mode = performance_mode
        self.preprocessor = AdaptivePreprocessor(performance_mode=performance_mode)
        self.ocr_engine = PaddleOCRAdapter()
        self.parser_front = CNICParser(doc_side="front")
        self.parser_back = CNICParser(doc_side="back")

    def process(
        self,
        image_input: Any,
        doc_side: str = "front",
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Execute CNIC processing end-to-end and return structured JSON.
        """
        start_t = time.time()
        expected_type = "cnic_back" if doc_side == "back" else "cnic_front"

        # 1. Adaptive Preprocessing
        prep_res = self.preprocessor.process(image_input, document_type=expected_type)

        # 2. Candidate Evaluation across preprocessing variants
        best_ocr = None
        best_variant = None
        best_score = -1.0

        for variant in prep_res["variants"]:
            raw_ocr = self.ocr_engine.extract_tokens(variant.image, document_type=expected_type)
            score_obj = OCRCandidateEvaluator.evaluate(raw_ocr, document_type=expected_type)
            if score_obj.total_score > best_score:
                best_score = score_obj.total_score
                best_ocr = raw_ocr
                best_variant = variant

        if best_ocr is None:
            best_ocr = self.ocr_engine.extract_tokens(prep_res["best_image"], document_type=expected_type)
            best_variant = prep_res["variants"][0]

        # 3. Document Classification
        classification = DocumentClassifier.classify(best_ocr, explicit_type=expected_type)

        # 4. Field Parsing
        parser = self.parser_back if doc_side == "back" else self.parser_front
        parsed_res = parser.parse(best_ocr, inverse_matrix=prep_res["processed_to_original_matrix"], variant_name=best_variant.name)

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
            "ocr_engine": self.ocr_engine.rapid_adapter.model_version if self.ocr_engine.rapid_adapter.engine else "paddleocr",
            "candidate_score": best_score,
            "processing_time_ms": round(elapsed_ms, 2)
        }

        result = {
            "status": "success",
            "document_type": expected_type,
            "review_state": review_state,
            "review_reasons": review_reasons,
            "name": parsed_res.get("name"),
            "father_name": parsed_res.get("father_name"),
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
