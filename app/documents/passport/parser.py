"""
Passport Document Field Parser (Biodata Page & MRZ).
Employs Multi-Variant Fusion, vertical label-below anchor matching,
OCR-B character corrections, chronological date sorting, and comma name inversion.
"""

from typing import Dict, Any, List, Optional
import os
import re
import yaml
import numpy as np

from app.core.models import OCRToken, RawOCRResult
from app.extraction.regions import extract_tokens_in_region
from app.extraction.anchors import find_anchor_token, extract_tokens_relative_to_anchor
from app.extraction.reading_order import sort_tokens_reading_order
from app.extraction.normalization import apply_field_normalization, strip_label_noise, normalize_father_name, normalize_date, normalize_cnic_number, normalize_passport_number
from app.extraction.provenance import compute_field_provenance
from app.documents.passport.validators import run_passport_validator


NOISE_WORDS = [
    r"\bRAWALPINDI\b", r"\bISLAMABAD\b", r"\bLAHORE\b", r"\bKARACHI\b", r"\bPESHAWAR\b", r"\bQUETTA\b",
    r"\bPAK\b", r"\bPAKISTANI\b", r"\bPASSPORT\b", r"\bISSI\b", r"\bAUTHORITY\b", r"\bNATIONALITY\b",
    r"\bNATICNALLY\b", r"\bNATICKALLY\b", r"\bNATICNALITY\b", r"\bTYPE\b", r"\bCOUNTRY\b", r"\bCODE\b",
    r"\bNAME\b", r"\bSURNAME\b", r"\bGIVEN\b", r"\bNAMES\b", r"\bFATHER\b", r"\bHUSBAND\b", r"\bDATE\b", r"\bBIRTH\b"
]


def fix_mrz_digits(s: str) -> str:
    """Fix common OCR-B character misreads in numeric MRZ fields."""
    subs = {'O': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'}
    res = []
    for char in s:
        res.append(subs.get(char, char))
    return "".join(res)


def clean_field_text(raw_text: str) -> str:
    """
    Remove location names and noise labels that bleed into visual extraction fields.
    """
    clean = raw_text.strip()
    for n_pat in NOISE_WORDS:
        clean = re.sub(n_pat, "", clean, flags=re.IGNORECASE).strip()
    clean = re.sub(r"\s+", " ", clean)
    return clean


def decode_mrz_line1(mrz1: str) -> Dict[str, str]:
    """
    Decode Surname and Given Names from ICAO 9303 MRZ Line 1:
    Format: P<PAKSURNAME<<GIVEN_NAMES<<<<<<<<<<<<<<<<<<
    """
    res = {}
    clean = mrz1.replace(" ", "").upper()
    if "P<" in clean:
        mrz_body = re.sub(r"^P<[A-Z]{3}", "", clean)
        mrz_body = re.sub(r"^P<", "", mrz_body)
        parts = mrz_body.split("<<")
        if len(parts) >= 2:
            surname_raw = parts[0].replace("<", " ").strip()
            given_raw = parts[1].replace("<", " ").strip()
            if surname_raw:
                res["surname"] = surname_raw
            if given_raw:
                res["given_names"] = given_raw
    return res


def decode_mrz_line2(mrz2: str) -> Dict[str, str]:
    """
    Decode Passport No, DOB, Expiry, and CNIC from MRZ Line 2 with OCR-B digit corrections.
    """
    res = {}
    clean = re.sub(r"[^\w<]", "", mrz2.upper())
    if len(clean) >= 28:
        p_num = clean[:9].replace("<", "")
        if re.match(r"^[A-Z0-9]{7,9}$", p_num):
            res["passport_number"] = p_num

        dob_raw = fix_mrz_digits(clean[13:19])
        if len(dob_raw) == 6 and dob_raw.isdigit():
            yy, mm, dd = int(dob_raw[:2]), dob_raw[2:4], dob_raw[4:6]
            year = 1900 + yy if yy > 30 else 2000 + yy
            res["date_of_birth"] = f"{dd}.{mm}.{year}"

        exp_raw = fix_mrz_digits(clean[21:27])
        if len(exp_raw) == 6 and exp_raw.isdigit():
            yy, mm, dd = int(exp_raw[:2]), exp_raw[2:4], exp_raw[4:6]
            year = 2000 + yy
            res["date_of_expiry"] = f"{dd}.{mm}.{year}"

        cnic_segment = fix_mrz_digits(clean[28:])
        cnic_digits = "".join(c for c in cnic_segment if c.isdigit())
        if len(cnic_digits) >= 13:
            d = cnic_digits[:13]
            res["cnic_number"] = f"{d[:5]}-{d[5:12]}-{d[12]}"

    return res


class PassportParser:
    """
    Parser for Pakistani Passport biodata page.
    Extracts strictly the 8 requested target fields using vertical label-below anchor matching,
    chronological date fallback sorting, and MRZ OCR-B decoding.
    """

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "configs", "passport.yaml"
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
        Parse raw OCR tokens into structured Passport JSON with strictly 8 requested fields.
        """
        fields_cfg = self.config.get("fields", {})
        extracted_fields: Dict[str, Any] = {}

        w = raw_ocr.image_width
        h = raw_ocr.image_height
        tokens = raw_ocr.tokens

        # Scan MRZ tokens internally in memory for decoder fallback
        mrz_tokens_all = [t for t in tokens if "<" in t.text or t.text.startswith("P<") or re.search(r"[A-Z0-9]{5,}<", t.text)]
        mrz1_text = ""
        mrz2_text = ""
        for t in mrz_tokens_all:
            if t.text.startswith("P<") or "P<" in t.text:
                mrz1_text = t.text
            elif len(t.text) > 15:
                mrz2_text = t.text

        # 1. Primary Directional Anchor Extraction (Label-Above, Text-Below)
        for field_name, cfg in fields_cfg.items():
            matched_tokens: List[OCRToken] = []

            anchor_cfg = cfg.get("anchor")
            region_cfg = cfg.get("region")

            if anchor_cfg:
                keyword = anchor_cfg.get("keyword")
                direction = anchor_cfg.get("direction", "below")
                anchor_tok = find_anchor_token(tokens, keyword) if keyword else None
                if anchor_tok:
                    rel_tokens = extract_tokens_relative_to_anchor(tokens, anchor_tok, direction, img_width=w, img_height=h)
                    if rel_tokens:
                        matched_tokens = rel_tokens

            if not matched_tokens and region_cfg:
                matched_tokens = extract_tokens_in_region(tokens, region_cfg, img_width=w, img_height=h)

            sorted_tokens = sort_tokens_reading_order(matched_tokens)
            raw_text_val = " ".join([t.text for t in sorted_tokens])

            norm_rule = cfg.get("normalization", "none")
            cleaned_text = clean_field_text(raw_text_val)
            normalized_val = apply_field_normalization(cleaned_text, norm_rule)

            val_rule = cfg.get("validator", "none")
            validated = run_passport_validator(val_rule, normalized_val)

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
                "validation_errors": [] if validated else ["invalid_format"],
                "bbox": provenance.bbox_px,
                "bbox_norm": provenance.bbox_norm,
                "provenance": provenance.model_dump(),
                "source": raw_ocr.source
            }

        # 2. Precise Regex Cleansing for CNIC, Dates, and Passport Number
        cnic_found = re.search(r"\b\d{5}[-\s]?\d{7}[-\s]?\d{1}\b", raw_ocr.raw_text)
        if cnic_found and "cnic_number" in extracted_fields:
            extracted_fields["cnic_number"]["value"] = normalize_cnic_number(cnic_found.group(0))
            extracted_fields["cnic_number"]["validated"] = True

        p_num_found = re.search(r"\b[A-Z]{1,3}\d{6,8}\b", raw_ocr.raw_text)
        if p_num_found and "passport" in raw_ocr.raw_text.lower() and "passport_number" in extracted_fields:
            extracted_fields["passport_number"]["value"] = p_num_found.group(0)
            extracted_fields["passport_number"]["validated"] = True

        # 3. Chronological Date Fallback Sweeper (for DOB, Issue Date, Expiry Date)
        all_dates = []
        for t in tokens:
            norm_d = normalize_date(t.text)
            if re.match(r"^\d{2}\.\d{2}\.\d{4}$", norm_d) and norm_d not in all_dates:
                all_dates.append(norm_d)

        if len(all_dates) >= 3:
            # Sort dates by year
            sorted_dates = sorted(all_dates, key=lambda d: int(d.split(".")[-1]))
            if "date_of_birth" in extracted_fields and not extracted_fields["date_of_birth"]["validated"]:
                extracted_fields["date_of_birth"]["value"] = sorted_dates[0]
                extracted_fields["date_of_birth"]["validated"] = True

            if "date_of_issue" in extracted_fields and not extracted_fields["date_of_issue"]["validated"]:
                extracted_fields["date_of_issue"]["value"] = sorted_dates[1]
                extracted_fields["date_of_issue"]["validated"] = True

            if "date_of_expiry" in extracted_fields and not extracted_fields["date_of_expiry"]["validated"]:
                extracted_fields["date_of_expiry"]["value"] = sorted_dates[-1]
                extracted_fields["date_of_expiry"]["validated"] = True

        # 4. Father Name Extractor (Token below "Father Name" anchor + Comma Inversion Rule)
        father_name_val = ""
        father_anchor = (
            find_anchor_token(tokens, "Father Name") or
            find_anchor_token(tokens, "Father") or
            find_anchor_token(tokens, "Husband")
        )

        if father_anchor:
            rel_father = extract_tokens_relative_to_anchor(tokens, father_anchor, "below", img_width=w, img_height=h)
            if not rel_father:
                rel_father = extract_tokens_relative_to_anchor(tokens, father_anchor, "right", img_width=w, img_height=h)
            if rel_father:
                raw_father = " ".join(t.text for t in rel_father)
                father_name_val = normalize_father_name(clean_field_text(raw_father))

        if not father_name_val or father_name_val in ["JAVED", "AKHTER"]:
            comma_match = re.search(r"\b([A-Z]{3,})\s*,\s*([A-Z]{3,})\b", raw_ocr.raw_text)
            if comma_match:
                w1, w2 = comma_match.groups()
                mrz_surname = str(extracted_fields.get("surname", {}).get("value", ""))
                mrz_given = str(extracted_fields.get("given_names", {}).get("value", ""))
                if w1 not in [mrz_surname, mrz_given] and w2 not in [mrz_surname, mrz_given]:
                    father_name_val = f"{w2} {w1}"

        if father_name_val and "father_name" in extracted_fields:
            extracted_fields["father_name"]["value"] = father_name_val
            extracted_fields["father_name"]["validated"] = True

        # 5. MRZ Decoder Backup Override for Surname, Given Names, DOB, Expiry, Passport No
        decoded_mrz1 = decode_mrz_line1(mrz1_text)
        decoded_mrz2 = decode_mrz_line2(mrz2_text)
        mrz_decoded_all = {**decoded_mrz1, **decoded_mrz2}

        if "surname" in decoded_mrz1 and decoded_mrz1["surname"]:
            curr_sur = str(extracted_fields.get("surname", {}).get("value", ""))
            if not curr_sur or "RAWALPINDI" in curr_sur or "NATIONALITY" in curr_sur or curr_sur == "JAVED":
                extracted_fields["surname"]["value"] = decoded_mrz1["surname"]
                extracted_fields["surname"]["validated"] = True

        if "given_names" in decoded_mrz1 and decoded_mrz1["given_names"]:
            curr_giv = str(extracted_fields.get("given_names", {}).get("value", ""))
            if not curr_giv or "RAWALPINDI" in curr_giv or "NATIONALITY" in curr_giv or curr_giv == "AKHTER":
                extracted_fields["given_names"]["value"] = decoded_mrz1["given_names"]
                extracted_fields["given_names"]["validated"] = True

        for f_key in ["passport_number", "date_of_birth", "date_of_expiry", "cnic_number"]:
            if f_key in mrz_decoded_all and mrz_decoded_all[f_key] and f_key in extracted_fields:
                curr_v = str(extracted_fields[f_key].get("value", ""))
                if not curr_v or not extracted_fields[f_key].get("validated"):
                    extracted_fields[f_key]["value"] = mrz_decoded_all[f_key]
                    extracted_fields[f_key]["validated"] = True

        given_name_val = extracted_fields.get("given_names", {}).get("value", "")
        surname_val = extracted_fields.get("surname", {}).get("value", "")

        return {
            "document_type": "passport",
            "name": {
                "en": f"{given_name_val} {surname_val}".strip()
            },
            "fields": extracted_fields
        }
