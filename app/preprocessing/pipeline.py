"""
Adaptive Document Image Preprocessing & Canonicalization Pipeline.
Executes quality analysis, boundary detection, landmark template fallback alignment,
and produces the Canonical Image along with CoordinateTransform tracking.
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import cv2
import numpy as np

from app.core.models import (
    QualityReport, PreprocessingPlan, ImageStage, DocumentBoundary, DocumentConfig, CoordinateTransform
)
from app.preprocessing.quality import QualityAnalyzer
from app.preprocessing.decision import PreprocessingPlanner
from app.preprocessing.geometry import detect_document_boundary, warp_perspective, normalize_resolution
from app.preprocessing.landmark import LandmarkAligner
from app.preprocessing.orientation import fix_exif_orientation, detect_and_deskew, read_exif_orientation
from app.preprocessing.illumination import correct_illumination
from app.preprocessing.contrast import enhance_contrast_clahe
from app.preprocessing.denoise import denoise_bilateral
from app.preprocessing.sharpen import sharpen_unsharp_mask
from app.preprocessing.variants import generate_candidate_variants, PreprocessingVariant


class PreprocessingResultContainer(dict):
    """
    Result container allowing both dictionary key indexing and property attribute access.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    @property
    def preprocessed_image(self) -> np.ndarray:
        return self.get("best_image")

    @property
    def quality(self) -> QualityReport:
        return self.get("quality_report")

    @property
    def mrz_image(self) -> np.ndarray:
        img = self.get("best_image")
        if img is not None and img.size > 0:
            h = img.shape[0]
            return img[int(h * 0.70):, :]  # Crop bottom 30% MRZ area
        return img


class AdaptivePreprocessor:
    """
    Master adaptive preprocessor converting all incoming document photos into a Canonical Document Image.
    """

    def __init__(self, performance_mode: str = "balanced"):
        self.performance_mode = performance_mode

    @staticmethod
    def normalize_input_image(image_input: Any) -> Tuple[np.ndarray, bool]:
        """
        Normalize input (numpy, PIL Image, bytes) to BGR numpy array and apply EXIF orientation.
        """
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Unable to read image path: {image_input}")
            exif_tag = read_exif_orientation(image_input)
            bgr, applied = fix_exif_orientation(img, exif_tag)
            return bgr, applied

        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Unable to decode image bytes")
            exif_tag = read_exif_orientation(image_input)
            bgr, applied = fix_exif_orientation(img, exif_tag)
            return bgr, applied

        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3 and image_input.shape[2] == 4:
                bgr = cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
            elif image_input.ndim == 2:
                bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            else:
                bgr = image_input.copy()
            return bgr, False

        elif hasattr(image_input, "convert"):  # PIL Image
            exif_tag = read_exif_orientation(image_input)
            img_rgb = np.array(image_input.convert("RGB"))
            bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            bgr_fixed, applied = fix_exif_orientation(bgr, exif_tag)
            return bgr_fixed, applied

        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    @staticmethod
    def normalize_image_format(image: np.ndarray, target_width: int = 2000) -> np.ndarray:
        """
        Standardized Image Normalization Pipeline:
        1. Normalize resolution to target_width (2000px)
        2. Remove uneven background shadows via morphological division
        3. Standardize contrast in LAB color space via CLAHE
        """
        if image is None or image.size == 0:
            return image

        res_normalized, _ = normalize_resolution(image, min_width=target_width, max_width=target_width)
        shadow_free = correct_illumination(res_normalized)
        contrast_equalized = enhance_contrast_clahe(shadow_free)
        return contrast_equalized

    def process(
        self,
        image_input: Any = None,
        input_data: Any = None,
        document_type: str = "generic",
        debug: bool = False,
        debug_dir: Optional[str] = None,
        config: Optional[DocumentConfig] = None
    ) -> PreprocessingResultContainer:
        """
        Execute adaptive pipeline producing the Canonical Image and CoordinateTransform container.
        """
        target_input = image_input if image_input is not None else input_data
        if target_input is None:
            raise ValueError("No image input provided to preprocessor.")

        original_img, _ = self.normalize_input_image(target_input)
        h_orig, w_orig = original_img.shape[:2]
        stages: List[ImageStage] = []

        stages.append(ImageStage(
            name="original",
            image=original_img.copy(),
            parent_stage=None,
            metadata={"width": w_orig, "height": h_orig}
        ))

        exif_fixed, exif_applied = fix_exif_orientation(original_img)
        curr_img = exif_fixed
        stages.append(ImageStage(
            name="orientation_corrected",
            image=curr_img.copy(),
            parent_stage="original",
            metadata={"applied": exif_applied}
        ))

        # Quality Analysis
        boundary = detect_document_boundary(curr_img)
        quality = QualityAnalyzer.analyze(curr_img, boundary_confidence=boundary.confidence)
        plan = PreprocessingPlanner.plan(quality, config=config, performance_mode=self.performance_mode)

        target_aspect = 1.5858 if "cnic" in document_type else 1.4205
        target_w = 2000
        target_h = int(target_w / target_aspect)

        canonical_img = None
        M_forward = np.eye(3, dtype=np.float32)
        M_inverse = np.eye(3, dtype=np.float32)
        coord_tx = None

        # 1. Perspective Transformation when boundary confidence >= 0.60
        if plan.perspective_correction and boundary.detected and boundary.confidence >= 0.60:
            warped, M_fw, M_inv, coord_tx = warp_perspective(
                curr_img, boundary, target_aspect_ratio=target_aspect, target_width=target_w
            )
            canonical_img = warped
            M_forward = M_fw
            M_inverse = M_inv
            stages.append(ImageStage(
                name="perspective_corrected",
                image=canonical_img.copy(),
                parent_stage="orientation_corrected",
                metadata={"boundary_method": boundary.method, "confidence": boundary.confidence}
            ))
        else:
            # 2. Template / Landmark Fallback Alignment when boundary confidence < 0.60
            canonical_img, M_fw, M_inv, coord_tx, landmark_success = LandmarkAligner.align_to_template(
                curr_img, document_type=document_type, target_width=target_w, target_height=target_h
            )
            M_forward = M_fw
            M_inverse = M_inv
            stages.append(ImageStage(
                name="landmark_aligned",
                image=canonical_img.copy(),
                parent_stage="orientation_corrected",
                metadata={"landmark_matched": landmark_success}
            ))

        if plan.deskew:
            deskewed_img, angle = detect_and_deskew(canonical_img)
            canonical_img = deskewed_img
            stages.append(ImageStage(
                name="deskewed",
                image=canonical_img.copy(),
                parent_stage=stages[-1].name,
                metadata={"rotation_angle": angle}
            ))

        # Standardized Format Normalization
        formatted_canonical = self.normalize_image_format(canonical_img, target_width=target_w)
        stages.append(ImageStage(
            name="canonical_normalized",
            image=formatted_canonical.copy(),
            parent_stage=stages[-1].name,
            metadata={"width": target_w, "height": target_h}
        ))

        curr_img = formatted_canonical

        variants: List[PreprocessingVariant] = []
        if plan.generate_variants:
            variants = generate_candidate_variants(curr_img, base_name="canonical")
            variants.append(PreprocessingVariant(
                id="var_00_original",
                name="canonical_00_original",
                image=original_img.copy(),
                transformations=["original"]
            ))
        else:
            variants = [PreprocessingVariant(id="var_0", name="default", image=curr_img)]

        if debug and debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, "01_original.jpg"), original_img)
            cv2.imwrite(os.path.join(debug_dir, "10_canonical.jpg"), curr_img)
            cv2.imwrite(os.path.join(debug_dir, "10_final.jpg"), curr_img)

        return PreprocessingResultContainer({
            "best_image": curr_img,
            "canonical_image": curr_img,
            "original_image": original_img,
            "coordinate_transform": coord_tx,
            "document_type": document_type,
            "stages": stages,
            "quality_report": quality,
            "preprocessing_plan": plan,
            "boundary": boundary,
            "variants": variants,
            "original_to_processed_matrix": M_forward,
            "processed_to_original_matrix": M_inverse
        })

    def _load_image(self, image_input: Any) -> np.ndarray:
        bgr, _ = self.normalize_input_image(image_input)
        return bgr


Preprocessor = AdaptivePreprocessor
