"""
Preprocessing Decision Engine.
Consumes QualityReport and document configuration to produce a PreprocessingPlan.
Enforces safety limits to apply only required, non-destructive transformations.
"""

from typing import List, Optional, Dict, Any
from app.core.models import QualityReport, PreprocessingPlan, DocumentConfig


class PreprocessingPlanner:
    """
    Determines optimal, adaptive preprocessing plan based on quality metrics and safety limits.
    """

    @staticmethod
    def plan(
        quality: QualityReport,
        config: Optional[DocumentConfig] = None,
        performance_mode: str = "balanced"
    ) -> PreprocessingPlan:
        """
        Produce a tailored PreprocessingPlan with explicit safeguards.
        """
        reasons: List[str] = []

        # Default toggles
        perspective = True
        deskew = True
        illumination = False
        denoise: Optional[str] = None
        contrast: Optional[str] = None
        sharpening: Optional[str] = None
        threshold: Optional[str] = None
        generate_variants = True

        # Safety Limit 1: Do not perspective correct if boundary confidence is low
        if quality.perspective_score < 0.60:
            perspective = False
            reasons.append("Bypassed perspective correction: low boundary confidence (< 0.60)")
        else:
            reasons.append("Enabled perspective warp: high boundary confidence")

        # Safety Limit 2: Do not perform illumination correction if lighting is uniform
        if quality.shadow_score >= 0.15 or quality.brightness_score < 0.35 or quality.brightness_score > 0.75:
            illumination = True
            reasons.append(f"Enabled illumination correction: shadow score {quality.shadow_score:.2f}")
        else:
            illumination = False
            reasons.append("Bypassed illumination correction: lighting is uniform")

        # Safety Limit 3: Do not denoise if noise score is low
        if quality.noise_score > 0.35:
            denoise = "bilateral"
            reasons.append(f"Enabled bilateral denoise: noise score {quality.noise_score:.2f}")
        else:
            denoise = None
            reasons.append("Bypassed denoise: noise is low")

        # Safety Limit 4: Do not enhance contrast if contrast is already high
        if quality.contrast_score < 0.40 or quality.brightness_score < 0.35:
            contrast = "clahe"
            reasons.append(f"Enabled CLAHE contrast: low contrast score {quality.contrast_score:.2f}")
        else:
            contrast = None
            reasons.append("Bypassed contrast enhancement: contrast is sufficient")

        # Safety Limit 5: Do not sharpen if image is already very sharp
        if quality.blur_score < 140.0:
            sharpening = "usm"
            reasons.append(f"Enabled USM sharpening: blur score {quality.blur_score:.1f}")
        else:
            sharpening = None
            reasons.append(f"Bypassed sharpening: image is sharp (blur score {quality.blur_score:.1f})")

        # Safety Limit 6: Thresholding only for dedicated MRZ or high contrast
        threshold = None

        # Mode overrides
        if performance_mode == "fast":
            generate_variants = False
            reasons.append("Performance mode 'fast': disabled variant generation")

        return PreprocessingPlan(
            perspective_correction=perspective,
            deskew=deskew,
            illumination_correction=illumination,
            denoise=denoise,
            contrast=contrast,
            sharpening=sharpening,
            threshold=threshold,
            generate_variants=generate_variants,
            reasons=reasons
        )
