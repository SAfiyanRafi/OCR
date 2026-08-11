"""
OCR Engine and Candidate Evaluator Package.
"""

from .paddle import PaddleOCRAdapter
from .rapidocr import RapidOCRAdapter, detect_script
from .evaluator import OCRCandidateEvaluator
from .base import OCREngineProtocol
