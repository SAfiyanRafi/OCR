"""
PaddleOCR Engine Adapter.
Supports native PaddleOCR SDK and delegates to RapidOCR ONNX if installed.
Enforces dual pixel (bbox_px) and normalized (bbox_norm) coordinates.
"""

from typing import List, Optional, Dict, Any, Tuple
import time
import logging
import numpy as np

from app.core.models import OCRToken, RawOCRResult
from app.ocr.rapidocr import RapidOCRAdapter, detect_script

logger = logging.getLogger(__name__)

PADDLE_AVAILABLE = False
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False


class PaddleOCRAdapter:
    """
    Adapter for PaddleOCR SDK and RapidOCR ONNX engine.
    Exposes identical dual-coordinate token schema regardless of engine.
    """

    def __init__(self, lang: str = "en", use_angle_cls: bool = True):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.rapid_adapter = RapidOCRAdapter()
        self.paddle_engine = None

        if PADDLE_AVAILABLE:
            try:
                self.paddle_engine = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, show_log=False)
            except Exception as e:
                logger.warning(f"Failed to initialize PaddleOCR: {e}")

    def extract_tokens(self, image: np.ndarray, document_type: str = "generic") -> RawOCRResult:
        """
        Run OCR engine and return RawOCRResult.
        """
        start_t = time.time()
        h, w = image.shape[:2]

        # Use RapidOCR if available (preferred on Python 3.14)
        if self.rapid_adapter.engine is not None:
            return self.rapid_adapter.extract_tokens(image, document_type=document_type)

        tokens: List[OCRToken] = []

        if self.paddle_engine is not None:
            try:
                raw_results = self.paddle_engine.ocr(image, cls=self.use_angle_cls)
                if raw_results and raw_results[0]:
                    for idx, res in enumerate(raw_results[0]):
                        box = res[0]
                        text, conf = res[1]
                        text = str(text).strip()

                        xs = [pt[0] for pt in box]
                        ys = [pt[1] for pt in box]
                        x1_px, y1_px, x2_px, y2_px = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))

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
                            confidence=float(conf),
                            bbox_px=bbox_px,
                            bbox_norm=bbox_norm,
                            image_width=w,
                            image_height=h,
                            page=1,
                            index=idx,
                            script=script
                        ))
            except Exception as e:
                logger.error(f"Error during PaddleOCR execution: {e}")

        elapsed_ms = (time.time() - start_t) * 1000.0
        raw_text = " ".join([t.text for t in tokens])

        return RawOCRResult(
            image_width=w,
            image_height=h,
            document_type=document_type,
            tokens=tokens,
            raw_text=raw_text,
            source="paddleocr" if self.paddle_engine else "mock_fallback",
            processing_time_ms=elapsed_ms
        )
