"""
Quality Analyzer Component.
Responsible ONLY for measuring image quality metrics and returning a QualityReport.
Must NOT decide which filters or transformations to apply.
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import cv2
import numpy as np
from app.core.models import QualityReport


@dataclass
class BrightnessAnalysis:
    brightness: float
    underexposed: bool
    overexposed: bool


@dataclass
class ContrastAnalysis:
    contrast: float
    is_low_contrast: bool


def analyze_blur(image: np.ndarray) -> Tuple[float, bool, float]:
    """Returns (laplacian_var, is_blurry, normalized_score)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurry = var < 80.0
    norm_score = min(1.0, var / 200.0)
    return var, is_blurry, norm_score


def analyze_brightness(image: np.ndarray) -> BrightnessAnalysis:
    """Returns BrightnessAnalysis object."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    val = float(np.mean(gray))
    score = val / 255.0
    return BrightnessAnalysis(
        brightness=score,
        underexposed=score < 0.25,
        overexposed=score > 0.85
    )


def analyze_contrast(image: np.ndarray) -> ContrastAnalysis:
    """Returns ContrastAnalysis object."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    std = float(np.std(gray))
    score = min(1.0, std / 64.0)
    return ContrastAnalysis(
        contrast=score,
        is_low_contrast=score < 0.25
    )


def detect_glare(image: np.ndarray) -> Tuple[float, bool]:
    """Returns (glare_ratio, is_glare_severe)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    glare_pixels = np.sum(gray > 245)
    ratio = float(glare_pixels) / float(gray.size)
    return ratio, ratio > 0.08


def detect_shadows(image: np.ndarray) -> float:
    """Detect shadow variance across block grid."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape[:2]
    grid_h, grid_w = max(1, h // 4), max(1, w // 4)
    block_means = []
    for r in range(0, h, grid_h):
        for c in range(0, w, grid_w):
            block_means.append(np.mean(gray[r:r+grid_h, c:c+grid_w]))
    return min(1.0, float(np.std(block_means)) / 50.0)


detect_shadow = detect_shadows


def assess_image_quality(image: np.ndarray) -> QualityReport:
    """Alias for QualityAnalyzer.analyze."""
    return QualityAnalyzer.analyze(image)


class QualityAnalyzer:
    """
    Measures physical image metrics without filter decision coupling.
    """

    @staticmethod
    def analyze(image: np.ndarray, boundary_confidence: float = 1.0) -> QualityReport:
        """
        Analyze image metrics and return QualityReport.
        """
        if image is None or image.size == 0:
            return QualityReport(warnings=["Empty image received"])

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        warnings: List[str] = []

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < 80.0:
            warnings.append(f"Image appears blurry (Laplacian var: {lap_var:.1f})")

        brightness_val = float(np.mean(gray))
        brightness_score = brightness_val / 255.0
        if brightness_score < 0.25:
            warnings.append("Image is under-exposed / dark")
        elif brightness_score > 0.85:
            warnings.append("Image is over-exposed / bright")

        contrast_std = float(np.std(gray))
        contrast_score = min(1.0, contrast_std / 64.0)
        if contrast_score < 0.25:
            warnings.append("Image has low contrast")

        blur_diff = cv2.absdiff(gray, cv2.GaussianBlur(gray, (5, 5), 0))
        noise_score = min(1.0, float(np.mean(blur_diff)) / 25.0)

        glare_pixels = np.sum(gray > 245)
        glare_ratio = float(glare_pixels) / float(w * h)
        if glare_ratio > 0.08:
            warnings.append(f"High glare detected ({glare_ratio * 100:.1f}%)")

        grid_h, grid_w = max(1, h // 4), max(1, w // 4)
        block_means = []
        for r in range(0, h, grid_h):
            for c in range(0, w, grid_w):
                block_means.append(np.mean(gray[r:r+grid_h, c:c+grid_w]))
        shadow_score = min(1.0, float(np.std(block_means)) / 50.0)

        min_dim = min(h, w)
        if min_dim < 600:
            resolution_score = min_dim / 600.0
            warnings.append("Low resolution scan")
        else:
            resolution_score = 1.0

        perspective_score = boundary_confidence

        overall_score = float(np.clip(
            0.30 * min(1.0, lap_var / 200.0) +
            0.20 * contrast_score +
            0.20 * (1.0 - abs(brightness_score - 0.5) * 2.0) +
            0.15 * resolution_score +
            0.15 * perspective_score,
            0.0, 1.0
        ))

        return QualityReport(
            blur_score=lap_var,
            contrast_score=contrast_score,
            brightness_score=brightness_score,
            noise_score=noise_score,
            glare_score=glare_ratio,
            shadow_score=shadow_score,
            perspective_score=perspective_score,
            resolution_score=resolution_score,
            overall_score=overall_score,
            warnings=warnings
        )
