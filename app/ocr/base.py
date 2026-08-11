"""
OCR Engine Abstraction Protocol & Candidate Models.
Ensures extraction layer never depends directly on any specific OCR engine.
"""

from typing import Protocol, List, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np
from app.core.models import OCRToken, RawOCRResult, OCRCandidate


@dataclass
class OCRTextLine:
    text: str
    confidence: float
    box: List[List[float]] = field(default_factory=list)


@dataclass
class OCRResultContainer:
    variant_name: str
    lines: List[OCRTextLine] = field(default_factory=list)
    average_confidence: float = 0.0
    total_score: float = 0.0

    def model_dump(self) -> Dict[str, Any]:
        return {
            "variant_name": self.variant_name,
            "average_confidence": self.average_confidence,
            "total_score": self.total_score,
            "line_count": len(self.lines)
        }


class OCREngineProtocol(Protocol):
    """
    Common protocol interface for OCR engine adapters (PaddleOCR, RapidOCR ONNX).
    """

    def extract_tokens(self, image: np.ndarray, document_type: str = "generic") -> RawOCRResult:
        """
        Run OCR text detection & recognition and return RawOCRResult.
        """
        ...
