"""
Building Detection and Segmentation Module.

Implements satellite building footprint extraction using a hybrid computer vision
and remote sensing pipeline:
1. Spectral contrast and rooftop reflectance thresholding
2. Morphological structural filtering (disconnecting rooftop structures from roads and soil)
3. Geometric filtering (aspect ratio < 3.0, solidity > 0.50, area thresholds)
4. Polygon approximation using Ramer-Douglas-Peucker (approxPolyDP)
5. Structured detection dictionary output and annotated visual overlays.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


class BuildingSegmenter:
    """Detects and segments building footprints in satellite imagery."""

    def __init__(
        self,
        min_area: int = 20,
        max_area: int = 8000,
        min_solidity: float = 0.52,
        max_aspect_ratio: float = 3.2,
        confidence_threshold: float = 0.45,
        epsilon_factor: float = 0.02,
    ):
        """
        Args:
            min_area: Minimum building footprint area in pixels.
            max_area: Maximum building footprint area in pixels.
            min_solidity: Contour area / Convex hull area (filters out jagged non-building noise).
            max_aspect_ratio: Length to width ratio (buildings are compact, aspect ratio < 3.2).
            confidence_threshold: Minimum confidence score to retain detection.
            epsilon_factor: Polygon approximation precision (approxPolyDP).
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_solidity = min_solidity
        self.max_aspect_ratio = max_aspect_ratio
        self.confidence_threshold = confidence_threshold
        self.epsilon_factor = epsilon_factor

    def segment(
        self,
        image_rgb: np.ndarray,
        ndvi: Optional[np.ndarray] = None,
        candidate_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Runs building segmentation on an RGB satellite image.

        Args:
            image_rgb: uint8 array of shape (H, W, 3).
            ndvi: Optional Normalized Difference Vegetation Index array (-1 to 1).
            candidate_mask: Optional binary mask (e.g. from Part 1 change detection) to focus analysis.

        Returns:
            binary_mask: Binary mask of shape (H, W) where 1 indicates building.
            detections: List of detection dictionaries with bbox, polygon, confidence, and metrics.
        """
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image of shape (H, W, 3), got {image_rgb.shape}")

        height, width, _ = image_rgb.shape
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # 1. Contrast enhancement via CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 2. Extract built-up / rooftop features via percentile thresholding
        valid_pixels = enhanced[enhanced > 10]
        if len(valid_pixels) > 0:
            threshold_val = np.percentile(valid_pixels, 72)
        else:
            threshold_val = 110.0

        _, binary_buildings = cv2.threshold(enhanced, threshold_val, 255, cv2.THRESH_BINARY)

        # 3. Morphological opening to disconnect buildings from roads, fences, and background
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        opened = cv2.morphologyEx(binary_buildings, cv2.MORPH_OPEN, kernel_open)

        # Morphological closing to fill small holes inside rooftop structures
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

        # 4. Suppress vegetation if NDVI is available
        if ndvi is not None:
            vegetation_mask = (ndvi > 0.20).astype(np.uint8) * 255
            closed = cv2.bitwise_and(closed, cv2.bitwise_not(vegetation_mask))

        # Focus on candidate change areas if provided from Part 1
        if candidate_mask is not None:
            cand_u8 = (candidate_mask.astype(bool)).astype(np.uint8) * 255
            kernel_cand = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            cand_dilated = cv2.dilate(cand_u8, kernel_cand, iterations=1)
            closed = cv2.bitwise_and(closed, cand_dilated)

        # 5. Contour detection and polygon extraction
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        binary_mask = np.zeros((height, width), dtype=np.uint8)
        detections = []

        for idx, cnt in enumerate(contours, start=1):
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = max(w, h) / max(1.0, min(w, h))

            # Buildings are compact and polygonal; roads are elongated
            if aspect_ratio > self.max_aspect_ratio:
                continue

            # Geometric shape validation: Solidity & Rectangularity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area / hull_area) if hull_area > 0 else 0.0

            if solidity < self.min_solidity:
                continue

            rect_area = w * h
            rectangularity = float(area / rect_area) if rect_area > 0 else 0.0

            # Approximate polygonal boundary (Ramer-Douglas-Peucker)
            epsilon = self.epsilon_factor * cv2.arcLength(cnt, True)
            approx_polygon = cv2.approxPolyDP(cnt, epsilon, True)

            # Confidence score based on solidity, rectangularity, and contrast
            confidence = float(np.clip(0.50 + (0.28 * solidity) + (0.20 * rectangularity), 0.0, 0.98))
            if confidence < self.confidence_threshold:
                continue

            # Draw accepted building onto binary mask
            cv2.drawContours(binary_mask, [cnt], -1, 1, thickness=cv2.FILLED)

            polygon_coords = approx_polygon.reshape(-1, 2).tolist()

            detections.append({
                "region_id": idx,
                "object_type": "building",
                "confidence": round(confidence, 3),
                "confidence_percent": round(confidence * 100, 1),
                "bbox": [int(x), int(y), int(w), int(h)],
                "polygon": polygon_coords,
                "area_pixels": int(area),
                "solidity": round(solidity, 2),
                "rectangularity": round(rectangularity, 2),
                "aspect_ratio": round(aspect_ratio, 2),
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
        Generates annotated visualization with building masks, polygon outlines, and bounding boxes.
        """
        overlay = image_rgb.copy()

        # 1. Semi-transparent cyan mask for segmented roofs
        if binary_mask is not None:
            colored_mask = np.zeros_like(image_rgb)
            colored_mask[binary_mask == 1] = [0, 200, 255]  # Cyan in RGB
            cv2.addWeighted(colored_mask, 0.45, overlay, 0.55, 0, overlay)

        # 2. Draw polygon boundaries and labels
        for det in detections:
            pts = np.array(det["polygon"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255), thickness=1)

            x, y, w, h = det["bbox"]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 165, 255), 1)

            label = f"Bldg {det['confidence_percent']}%"
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
