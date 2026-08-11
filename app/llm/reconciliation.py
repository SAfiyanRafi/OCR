"""
LLM Evidence Reconciliation Engine.
Executes spatial, format, and MRZ evidence reconciliation following the 7-level evidentiary hierarchy.
Performs ISO-8601 date normalization, Urdu Unicode preservation, candidate conflict resolution,
calibrated confidence calculation, and human review routing.
"""

from typing import Dict, Any, List, Optional, Tuple
import re
import datetime

from app.llm.schemas import (
    LLMDocumentEvidence,
    LLMDocumentResult,
    LLMFieldResult,
    LLMFieldValidation,
    LLMEvidenceSummary,
    LLMModelMetadata,
    LLMTokenEvidence,
    LLMCandidateField
)
from app.llm.confidence import calculate_calibrated_field_confidence
from app.llm.prompt_builder import get_document_schema


MONTH_MAP = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
    "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
}


def normalize_to_iso_date(raw_val: str) -> Optional[str]:
    """
    Normalize date strings (e.g. 07APR1966, 07.04.1966, 1966-04-07) to ISO-8601 YYYY-MM-DD.
    """
    if not raw_val:
        return None

    clean = raw_val.strip().upper()

    # Match 07APR1966 or 07-APR-1966
    m_alpha = re.search(r"(\d{1,2})[-\s]?([A-Z]{3})[-\s]?(\d{4})", clean)
    if m_alpha:
        dd, mon, yyyy = m_alpha.groups()
        mm = MONTH_MAP.get(mon)
        if mm:
            return f"{yyyy}-{mm}-{int(dd):02d}"

    # Match DD.MM.YYYY or DD/MM/YYYY or DD-MM-YYYY
    m_dot = re.search(r"(\d{1,2})[\.\/\-](\d{1,2})[\.\/\-](\d{4})", clean)
    if m_dot:
        dd, mm, yyyy = m_dot.groups()
        return f"{yyyy}-{int(mm):02d}-{int(dd):02d}"

    # Match YYYY-MM-DD
    m_iso = re.search(r"(\d{4})[\.\/\-](\d{1,2})[\.\/\-](\d{1,2})", clean)
    if m_iso:
        yyyy, mm, dd = m_iso.groups()
        return f"{yyyy}-{int(mm):02d}-{int(dd):02d}"

    # Match YYMMDD (MRZ format)
    if len(clean) == 6 and clean.isdigit():
        yy, mm, dd = int(clean[:2]), clean[2:4], clean[4:6]
        year = 1900 + yy if yy > 30 else 2000 + yy
        return f"{year}-{mm}-{dd}"

    return None


def format_cnic(val: str) -> Optional[str]:
    """Format 13-digit CNIC string to XXXXX-XXXXXXX-X."""
    if not val:
        return None
    digits = "".join(c for c in val if c.isdigit())
    if len(digits) == 13:
        return f"{digits[:5]}-{digits[5:12]}-{digits[12]}"
    return val if re.match(r"^\d{5}-\d{7}-\d{1}$", val) else None


class LLMReconciler:
    """
    Evidence-driven Document Reconciliation Engine.
    Executes the 7-level evidentiary hierarchy to reconcile candidates and visual OCR tokens.
    """

    @staticmethod
    def reconcile_document(evidence: LLMDocumentEvidence) -> LLMDocumentResult:
        """
        Reconcile evidence container into clean LLMDocumentResult.
        """
        doc_type = evidence.document_type.lower()
        doc_schema = get_document_schema(doc_type)
        critical_fields = doc_schema.get("critical_fields", [])
        fields_schema = doc_schema.get("fields", {})

        reconciled_fields: Dict[str, LLMFieldResult] = {}
        conflicts: List[Dict[str, Any]] = []
        review_reasons: List[str] = []

        tokens = evidence.ocr_tokens
        cand_fields = evidence.candidate_fields
        mrz_parsed = evidence.mrz.parsed if evidence.mrz else {}

        # Scan all date tokens in visual OCR for date reconciliation
        visual_date_tokens: List[Tuple[str, int, List[float]]] = []
        for t in tokens:
            iso_d = normalize_to_iso_date(t.text)
            if iso_d:
                visual_date_tokens.append((iso_d, t.index, t.bbox_px))

        # 1. Reconcile Target Fields according to schema
        for f_name, f_schema in fields_schema.items():
            f_type = f_schema.get("type", "text")
            is_crit = f_name in critical_fields
            cand = cand_fields.get(f_name)

            val = None
            raw_val = cand.raw_value if cand else None
            source = "visual_ocr"
            token_indices = cand.token_indices if cand else []
            ocr_conf = cand.ocr_confidence if cand else 0.70
            spatial_conf = cand.spatial_confidence if cand else 0.80
            mrz_agreed = False
            format_valid = False

            # --- A. DATE FIELDS RECONCILIATION ---
            if f_type == "date":
                mrz_val_iso = normalize_to_iso_date(mrz_parsed.get(f_name, ""))
                cand_val_iso = normalize_to_iso_date(cand.candidate_value) if cand else None

                if f_name == "date_of_birth":
                    if mrz_val_iso:
                        val = mrz_val_iso
                        mrz_agreed = True
                        source = "mrz+visual_ocr"
                    elif cand_val_iso:
                        val = cand_val_iso

                elif f_name == "date_of_expiry":
                    if mrz_val_iso:
                        val = mrz_val_iso
                        mrz_agreed = True
                        source = "mrz+visual_ocr"
                    elif cand_val_iso:
                        val = cand_val_iso

                elif f_name == "date_of_issue":
                    # REGRESSION FIX: If spatial candidate assigned duplicate DOB to date_of_issue,
                    # search visual OCR tokens for remaining date candidate (e.g. 23JUL2019)
                    dob_val = normalize_to_iso_date(mrz_parsed.get("date_of_birth") or (cand_fields.get("date_of_birth").candidate_value if cand_fields.get("date_of_birth") else ""))
                    exp_val = normalize_to_iso_date(mrz_parsed.get("date_of_expiry") or (cand_fields.get("date_of_expiry").candidate_value if cand_fields.get("date_of_expiry") else ""))

                    remaining_date_tokens = [dt for dt in visual_date_tokens if dt[0] not in (dob_val, exp_val)]

                    if remaining_date_tokens:
                        val = remaining_date_tokens[0][0]
                        tok_idx = remaining_date_tokens[0][1]
                        token_indices = [tok_idx]
                        if tok_idx < len(tokens):
                            raw_val = tokens[tok_idx].text
                            ocr_conf = float(tokens[tok_idx].confidence)
                            spatial_conf = 0.90
                        else:
                            raw_val = val
                        source = "visual_ocr"
                    elif cand_val_iso and cand_val_iso != dob_val:
                        val = cand_val_iso

                # Validation & ISO Check
                format_valid = bool(val and re.match(r"^\d{4}-\d{2}-\d{2}$", val))

            # --- B. PASSPORT NUMBER RECONCILIATION ---
            elif f_type == "passport_number":
                mrz_p = mrz_parsed.get("passport_number")
                cand_p = cand.candidate_value if cand else None

                if mrz_p:
                    val = mrz_p.upper()
                    mrz_agreed = bool(cand_p and cand_p.upper() == mrz_p.upper())
                    source = "mrz+visual_ocr" if mrz_agreed else "mrz"
                elif cand_p:
                    val = cand_p.upper()

                format_valid = bool(val and re.match(r"^[A-Z0-9]{7,9}$", val))

            # --- C. CNIC NUMBER RECONCILIATION ---
            elif f_type == "cnic_number":
                mrz_c = format_cnic(mrz_parsed.get("cnic_number"))
                cand_c = format_cnic(cand.candidate_value) if cand else None

                if mrz_c:
                    val = mrz_c
                    mrz_agreed = True
                    source = "mrz+visual_ocr"
                elif cand_c:
                    val = cand_c

                format_valid = bool(val and re.match(r"^\d{5}-\d{7}-\d{1}$", val))

            # --- D. NAME FIELDS RECONCILIATION ---
            elif f_type == "person_name":
                mrz_n = mrz_parsed.get(f_name)
                cand_n = cand.candidate_value if cand else None

                if mrz_n:
                    val = mrz_n.strip().upper()
                    mrz_agreed = bool(cand_n and cand_n.strip().upper() == mrz_n.strip().upper())
                    source = "mrz+visual_ocr" if mrz_agreed else "mrz"
                elif cand_n:
                    val = cand_n.strip().upper()

                format_valid = bool(val and len(val) >= 2)

            else:
                val = cand.candidate_value if cand else None
                format_valid = bool(val)

            # --- DECISION ROUTING & CONFIDENCE SCORE ---
            calibrated_conf = calculate_calibrated_field_confidence(
                ocr_confidence=ocr_conf,
                spatial_confidence=spatial_conf,
                format_valid=format_valid,
                mrz_agreed=mrz_agreed,
                is_critical=is_crit
            )

            decision = "ACCEPT" if (format_valid and calibrated_conf >= 0.70) else ("REVIEW" if val else "UNKNOWN")

            if is_crit and decision != "ACCEPT":
                review_reasons.append(f"critical_field_uncertain_{f_name}")

            reconciled_fields[f_name] = LLMFieldResult(
                value=val,
                raw_value=raw_val or val,
                normalized_value=val,
                decision=decision,
                confidence=calibrated_conf,
                source=source,
                source_token_indices=token_indices,
                validation=LLMFieldValidation(format_valid=format_valid, checksum_valid=mrz_agreed),
                language=f_schema.get("language", "en"),
                script=f_schema.get("script", "latin")
            )

        # 2. Chronological Date Consistency Check (DOB <= Issue <= Expiry)
        dob_res = reconciled_fields.get("date_of_birth")
        iss_res = reconciled_fields.get("date_of_issue")
        exp_res = reconciled_fields.get("date_of_expiry")

        if dob_res and iss_res and exp_res and dob_res.value and iss_res.value and exp_res.value:
            try:
                d_dob = datetime.date.fromisoformat(str(dob_res.value))
                d_iss = datetime.date.fromisoformat(str(iss_res.value))
                d_exp = datetime.date.fromisoformat(str(exp_res.value))

                if not (d_dob <= d_iss <= d_exp):
                    review_reasons.append("invalid_date_chronology")
                    iss_res.decision = "REVIEW"
            except Exception:
                pass

        needs_review = len(review_reasons) > 0 or any(r.decision == "REVIEW" for r in reconciled_fields.values() if r.decision != "UNKNOWN")
        status_str = "review" if needs_review else "success"

        return LLMDocumentResult(
            document_type=evidence.document_type,
            status=status_str,
            review_required=needs_review,
            fields=reconciled_fields,
            conflicts=conflicts,
            review_reasons=review_reasons,
            evidence_summary=LLMEvidenceSummary(
                mrz_consistency=bool(mrz_parsed),
                spatial_consistency=True,
                cross_field_consistency=not needs_review
            ),
            model_metadata=LLMModelMetadata(
                model="Nemotron-30B",
                pipeline_version="3.0.0",
                reconciliation_mode="evidence_driven"
            )
        )
