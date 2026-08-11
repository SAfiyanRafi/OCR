"""
Landmark & Template Alignment Fallback Engine.
Executes computer-vision feature matching (ORB/ECC) to align documents to canonical templates
when boundary contour detection confidence is low (< 0.60).
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from app.core.models import CoordinateTransform


class LandmarkAligner:
    """
    Template Landmark Aligner for CNIC Front, CNIC Back, and Passport.
    """

    @staticmethod
    def align_to_template(
        image: np.ndarray,
        document_type: str = "cnic_front",
        target_width: int = 2000,
        target_height: int = 1261
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, CoordinateTransform, bool]:
        """
        Align image to canonical document template dimensions using ORB feature matching.
        Returns (canonical_image, M_forward, M_inverse, coordinate_transform, success).
        """
        h_orig, w_orig = image.shape[:2]

        if image is None or image.size == 0:
            M = np.eye(3, dtype=np.float32)
            coord_tx = CoordinateTransform(M, M, (w_orig, h_orig), (target_width, target_height))
            return image, M, M, coord_tx, False

        # Create target canonical reference canvas (synthetic landmark template if no file)
        canonical_ref = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        # Draw high-contrast landmark corners and border guides in canonical space
        cv2.rectangle(canonical_ref, (50, 50), (target_width - 50, target_height - 50), (255, 255, 255), 10)
        cv2.circle(canonical_ref, (100, 100), 30, (255, 255, 255), -1)
        cv2.circle(canonical_ref, (target_width - 100, 100), 30, (255, 255, 255), -1)
        cv2.circle(canonical_ref, (100, target_height - 100), 30, (255, 255, 255), -1)
        cv2.circle(canonical_ref, (target_width - 100, target_height - 100), 30, (255, 255, 255), -1)

        gray_src = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray_dst = cv2.cvtColor(canonical_ref, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(gray_src, None)
        kp2, des2 = orb.detectAndCompute(gray_dst, None)

        if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) >= 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)

                M_forward, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M_forward is not None and M_forward.shape == (3, 3):
                    try:
                        M_inverse = np.linalg.inv(M_forward)
                        canonical_img = cv2.warpPerspective(image, M_forward, (target_width, target_height))
                        coord_tx = CoordinateTransform(
                            original_to_canonical=M_forward,
                            canonical_to_original=M_inverse,
                            original_size=(w_orig, h_orig),
                            canonical_size=(target_width, target_height)
                        )
                        return canonical_img, M_forward, M_inverse, coord_tx, True
                    except Exception:
                        pass

        # Fallback to direct resize homography transformation matrix
        M_forward = np.array([
            [float(target_width) / max(1, w_orig), 0, 0],
            [0, float(target_height) / max(1, h_orig), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        M_inverse = np.linalg.inv(M_forward)
        canonical_img = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

        coord_tx = CoordinateTransform(
            original_to_canonical=M_forward,
            canonical_to_original=M_inverse,
            original_size=(w_orig, h_orig),
            canonical_size=(target_width, target_height)
        )

        return canonical_img, M_forward, M_inverse, coord_tx, False


align_to_landmark_template = LandmarkAligner.align_to_template
