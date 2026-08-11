"""
RapidOCR ONNX Runtime Adapter Engine.
Executes PaddleOCR PP-OCRv4 models natively via ONNX Runtime at high resolution (1600px).
"""

from typing import List, Optional, Dict, Any
import time
import logging
import re
import numpy as np
from app.core.models import OCRToken, RawOCRResult

logger = logging.getLogger(__name__)

RAPID_AVAILABLE = False
try:
    from rapidocr_onnxruntime import RapidOCR
    RAPID_AVAILABLE = True
except ImportError:
    RAPID_AVAILABLE = False


def detect_script(text: str) -> str:
    """
    Detect script of OCR token string (urdu, latin, numeric, mixed).
    """
    if not text:
        return "unknown"

    has_urdu = bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\u8A00-\u8D03]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    has_digits = bool(re.search(r"\d", text))

    if has_urdu and not has_latin:
        return "urdu"
    elif has_latin and not has_urdu:
        return "latin"
    elif has_digits and not has_urdu and not has_latin:
        return "numeric"
    elif has_urdu and has_latin:
        return "mixed"
    return "latin" if has_latin else "numeric" if has_digits else "unknown"


class RapidOCRAdapter:
    """
    RapidOCR ONNX Engine Adapter executing PaddleOCR PP-OCRv4 models with high-precision parameters.
    """

    def __init__(self, model_version: str = "ppocrv4"):
        self.model_version = model_version
        self.engine = None
        if RAPID_AVAILABLE:
            try:
                # Optimized High-Precision Detection Parameters for PaddleOCR PP-OCRv4
                self.engine = RapidOCR(
                    det_limit_side_len=1600,
                    det_limit_type="max",
                    det_db_box_thresh=0.3,
                    det_db_unclip_ratio=1.8
                )
            except Exception as e:
                try:
                    self.engine = RapidOCR()
                except Exception as ex:
                    logger.warning(f"Failed to initialize RapidOCR: {ex}")

    def extract_tokens(self, image: np.ndarray, document_type: str = "generic") -> RawOCRResult:
        """
        Run PaddleOCR PP-OCRv4 engine and return RawOCRResult.
        """
        start_t = time.time()
        h, w = image.shape[:2]
        tokens: List[OCRToken] = []

        if self.engine is not None:
            try:
                results, _ = self.engine(image)
                if results:
                    for idx, res in enumerate(results):
                        box = res[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        text = str(res[1]).strip()
                        conf = float(res[2])

                        xs = [pt[0] for pt in box]
                        ys = [pt[1] for pt in box]
                        x1_px, y1_px, x2_px, y2_px = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))

                        # Dual Coordinate System: Pixel + Normalized
                        bbox_px = [x1_px, y1_px, x2_px, y2_px]
                        bbox_norm = [
                            round(x1_px / max(1, w), 4),
                            round(y1_px / max(1, h), 4),
                            round(x2_px / max(1, w), 4),
                            round(y2_px / max(1, h), 4)
                        ]

                        script = detect_script(text)

                        tokens.append(OCRToken(
                            text=text,
                            confidence=conf,
                            bbox_px=bbox_px,
                            bbox_norm=bbox_norm,
                            image_width=w,
                            image_height=h,
                            page=1,
                            index=idx,
                            script=script
                        ))
            except Exception as e:
                logger.error(f"Error during RapidOCR execution: {e}")

        elapsed_ms = (time.time() - start_t) * 1000.0
        raw_text = " ".join([t.text for t in tokens])

        return RawOCRResult(
            image_width=w,
            image_height=h,
            document_type=document_type,
            tokens=tokens,
            raw_text=raw_text,
            source="paddleocr_ppocrv4",
            processing_time_ms=elapsed_ms
        )
