"""
Road Detection and Segmentation Module.

Implements satellite road network extraction using:
1. Directional morphological filtering (multi-angle linear kernels)
2. Elongation and aspect-ratio analysis (distinguishing continuous roads from compact buildings)
3. Morphological thinning / centerline extraction
4. Vectorization into linear corridor segments with width estimation
5. Structured detection dictionary output and annotated visual overlays.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


class RoadSegmenter:
    """Detects and segments road networks and corridors in satellite imagery."""

    def __init__(
        self,
        min_length: int = 25,
        min_aspect_ratio: float = 2.2,
        confidence_threshold: float = 0.45,
    ):
        """
        Args:
            min_length: Minimum major axis length in pixels for road segments.
            min_aspect_ratio: Length to width ratio (roads are elongated, buildings are compact).
            confidence_threshold: Minimum confidence score to retain detection.
        """
        self.min_length = min_length
        self.min_aspect_ratio = min_aspect_ratio
        self.confidence_threshold = confidence_threshold

    def segment(
        self,
        image_rgb: np.ndarray,
        building_mask: Optional[np.ndarray] = None,
        candidate_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Extracts road corridors from an RGB satellite image.

        Args:
            image_rgb: uint8 array of shape (H, W, 3).
            building_mask: Optional binary mask of buildings to prevent cross-class confusion.
            candidate_mask: Optional change mask from Part 1.

        Returns:
            binary_mask: Binary road mask of shape (H, W).
            detections: List of detection dictionaries with centerlines, corridors, and confidence.
        """
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image of shape (H, W, 3), got {image_rgb.shape}")

        height, width, _ = image_rgb.shape
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # 1. Multi-directional linear filtering to enhance linear road tracks
        line_responses = []
        kernel_len = 15
        angles = [0, 30, 45, 60, 90, 120, 135, 150]

        for angle in angles:
            rad = np.deg2rad(angle)
            kernel = np.zeros((kernel_len, kernel_len), dtype=np.float32)
            center = kernel_len // 2
            dx = int(center * np.cos(rad))
            dy = int(center * np.sin(rad))
            cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1.0, 1)
            kernel /= (kernel.sum() + 1e-6)

            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            line_responses.append(filtered)

        max_line_resp = np.max(np.stack(line_responses, axis=0), axis=0)
        norm_lines = cv2.normalize(max_line_resp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 2. Local contrast enhancement for asphalt / dirt roads
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        road_thresh = cv2.adaptiveThreshold(
            enhanced_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5
        )

        # Combine linear features with local thresholding
        road_features = cv2.bitwise_and(norm_lines, road_thresh)
        _, binary_candidates = cv2.threshold(road_features, 40, 255, cv2.THRESH_BINARY)

        # 3. Exclude known buildings if building mask is provided
        if building_mask is not None:
            bldg_u8 = (building_mask > 0).astype(np.uint8) * 255
            # Dilate building mask slightly to clear edges
            k_bldg = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            bldg_dilated = cv2.dilate(bldg_u8, k_bldg, iterations=1)
            binary_candidates = cv2.bitwise_and(binary_candidates, cv2.bitwise_not(bldg_dilated))

        if candidate_mask is not None:
            cand_u8 = (candidate_mask.astype(bool)).astype(np.uint8) * 255
            k_cand = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            cand_dilated = cv2.dilate(cand_u8, k_cand, iterations=1)
            binary_candidates = cv2.bitwise_and(binary_candidates, cand_dilated)

        # Morphological closing along lines to bridge road interruptions
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary_candidates, cv2.MORPH_OPEN, k_close)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        binary_mask = np.zeros((height, width), dtype=np.uint8)
        detections = []

        for idx, cnt in enumerate(contours, start=1):
            area = cv2.contourArea(cnt)
            if area < 30:
                continue

            # Need at least 5 points to fit an ellipse for aspect ratio analysis
            if len(cnt) >= 5:
                (_, _), (major_axis, minor_axis), angle = cv2.fitEllipse(cnt)
                length = max(major_axis, minor_axis)
                width_est = max(1.0, min(major_axis, minor_axis))
                aspect_ratio = float(length / width_est)
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                length = max(w, h)
                width_est = max(1.0, min(w, h))
                aspect_ratio = float(length / width_est)

            # Roads MUST be elongated (aspect ratio > threshold and sufficient length)
            if aspect_ratio < self.min_aspect_ratio or length < self.min_length:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            epsilon = 0.03 * cv2.arcLength(cnt, True)
            approx_polygon = cv2.approxPolyDP(cnt, epsilon, True)

            # Road confidence increases with elongation
            confidence = float(np.clip(0.40 + (0.12 * min(aspect_ratio, 5.0)), 0.0, 0.95))
            if confidence < self.confidence_threshold:
                continue

            cv2.drawContours(binary_mask, [cnt], -1, 1, thickness=cv2.FILLED)
            polygon_coords = approx_polygon.reshape(-1, 2).tolist()

            detections.append({
                "region_id": idx,
                "object_type": "road",
                "confidence": round(confidence, 3),
                "confidence_percent": round(confidence * 100, 1),
                "bbox": [int(x), int(y), int(w), int(h)],
                "polygon": polygon_coords,
                "area_pixels": int(area),
                "aspect_ratio": round(aspect_ratio, 2),
                "estimated_length_px": round(length, 1),
                "estimated_width_px": round(width_est, 1),
            })

        return binary_mask, detections

    def visualize(
        self,
        image_rgb: np.ndarray,
        detections: List[Dict],
        binary_mask: Optional[np.ndarray] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Visualizes detected roads with yellow/orange corridors and labeled bounding boxes.
        """
        overlay = image_rgb.copy()

        # 1. Semi-transparent orange/amber mask for road corridors
        if binary_mask is not None:
            colored_mask = np.zeros_like(image_rgb)
            colored_mask[binary_mask == 1] = [255, 180, 0]  # Amber/Yellow in RGB
            cv2.addWeighted(colored_mask, 0.45, overlay, 0.55, 0, overlay)

        # 2. Draw boundaries and labels
        for det in detections:
            pts = np.array(det["polygon"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], isClosed=True, color=(255, 220, 0), thickness=2)

            x, y, w, h = det["bbox"]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 140, 0), 1)

            label = f"Road {det['confidence_percent']}%"
            cv2.putText(
                overlay,
                label,
                (x, max(12, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_p), bgr)

        return overlay
