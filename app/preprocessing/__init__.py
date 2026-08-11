"""
Adaptive Document Image Preprocessing Module.
"""

from .pipeline import AdaptivePreprocessor, Preprocessor
from .quality import QualityAnalyzer, analyze_blur, analyze_brightness, analyze_contrast, detect_glare, detect_shadows, assess_image_quality
from .decision import PreprocessingPlanner
from .geometry import detect_document_boundary, warp_perspective, transform_bbox_inverse
from .orientation import fix_exif_orientation, detect_and_deskew, correct_exif_orientation
