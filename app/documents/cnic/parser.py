"""
Bilingual CNIC Field Parser (Front & Back).
Applies region containment, directional anchors, reading order sorting,
field normalization, syntax validation, field provenance, and bilingual association.
"""

from typing import Dict, Any, List, Optional
import os
import yaml
import numpy as np

from app.core.models import OCRToken, RawOCRResult
from app.extraction.regions import extract_tokens_in_region
from app.extraction.anchors import find_anchor_token, extract_tokens_relative_to_anchor
from app.extraction.reading_order import sort_tokens_reading_order
from app.extraction.normalization import apply_field_normalization, strip_label_noise
from app.extraction.bilingual import associate_bilingual_fields
from app.extraction.provenance import compute_field_provenance
from app.documents.cnic.validators import run_cnic_validator


class CNICParser:
    """
    Parser for Pakistani CNIC documents.
    """

    def __init__(self, doc_side: str = "front", config_path: Optional[str] = None):
        self.doc_side = doc_side
        if config_path is None:
            cfg_name = "cnic_back.yaml" if doc_side == "back" else "cnic_front.yaml"
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "configs", cfg_name
            )

        self.config = self._load_config(config_path)

    def _load_config(self, path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
        return {}

    def parse(
        self,
        raw_ocr: RawOCRResult,
        inverse_matrix: Optional[np.ndarray] = None,
        variant_name: str = "default"
    ) -> Dict[str, Any]:
        """
        Parse raw OCR tokens into structured CNIC JSON.
        """
        fields_cfg = self.config.get("fields", {})
        extracted_fields: Dict[str, Any] = {}

        w = raw_ocr.image_width
        h = raw_ocr.image_height
        tokens = raw_ocr.tokens

        for field_name, cfg in fields_cfg.items():
            matched_tokens: List[OCRToken] = []

            strategy = cfg.get("strategy", "region")
            region_cfg = cfg.get("region")
            anchor_cfg = cfg.get("anchor")

            # 1. Region extraction
            if region_cfg:
                matched_tokens = extract_tokens_in_region(tokens, region_cfg, img_width=w, img_height=h)

            # 2. Anchor fallback
            if (not matched_tokens or strategy == "anchor") and anchor_cfg:
                keyword = anchor_cfg.get("keyword")
                direction = anchor_cfg.get("direction", "right")
                anchor_tok = find_anchor_token(tokens, keyword) if keyword else None
                if anchor_tok:
                    rel_tokens = extract_tokens_relative_to_anchor(tokens, anchor_tok, direction, img_width=w, img_height=h)
                    if rel_tokens:
                        matched_tokens = rel_tokens

            # Sort tokens in natural reading order
            sorted_tokens = sort_tokens_reading_order(matched_tokens)
            raw_text_val = " ".join([t.text for t in sorted_tokens])

            # Apply normalization
            norm_rule = cfg.get("normalization", "none")
            normalized_val = apply_field_normalization(raw_text_val, norm_rule)

            # Run validator
            val_rule = cfg.get("validator", "none")
            validated = run_cnic_validator(val_rule, normalized_val)

            validation_errors = [] if validated else ["invalid_format"]

            # Compute Field Provenance & Inverse Photo Projection
            provenance, spatial_conf, field_conf = compute_field_provenance(
                tokens=sorted_tokens,
                inverse_matrix=inverse_matrix,
                variant_name=variant_name,
                engine=raw_ocr.source,
                region_key=field_name
            )

            ocr_conf = float(np.mean([t.confidence for t in sorted_tokens])) if sorted_tokens else 0.0

            extracted_fields[field_name] = {
                "value": normalized_val,
                "raw_value": raw_text_val,
                "ocr_confidence": round(ocr_conf, 4),
                "spatial_confidence": spatial_conf,
                "confidence": field_conf,
                "validated": validated,
                "validation_errors": validation_errors,
                "bbox": provenance.bbox_px,
                "bbox_norm": provenance.bbox_norm,
                "provenance": provenance.model_dump(),
                "source": raw_ocr.source
            }

        # Associate Urdu and English name fields into bilingual objects
        results_with_bilingual = associate_bilingual_fields(extracted_fields)

        return {
            "document_type": f"cnic_{self.doc_side}",
            "name": results_with_bilingual.get("name"),
            "father_name": results_with_bilingual.get("father_name"),
            "fields": extracted_fields
        }
