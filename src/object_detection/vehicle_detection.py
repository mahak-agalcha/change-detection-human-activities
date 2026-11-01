"""
Vehicle Detection and Spatial Resolution Feasibility Module.

Scientific Constraint:
Vehicle detection from overhead imagery is governed by the sensor's Ground Sample Distance (GSD):
- Medium-resolution sensors (SpaceNet 7 PlanetScope @ 3.0m, Sentinel-2 @ 10.0m) cannot resolve
  passenger vehicles (~4.5m x 1.8m), which span only 1-2 pixels. Attempting detection at this
  scale produces massive false positive rates.
- Very High Resolution (VHR) sensors (<0.5m GSD, e.g. WorldView-3, xView, aerial orthophotos)
  render vehicles across 50-300 pixels, enabling reliable detection.

This module enforces resolution checks and executes vehicle detection only when physically feasible.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


class VehicleDetector:
    """Detects vehicles in high-resolution overhead imagery with strict GSD validation."""

    VHR_MAX_GSD_METERS = 0.60  # Maximum GSD supporting optical vehicle detection

    def __init__(
        self,
        default_gsd: float = 3.0,  # Default SpaceNet 7 GSD
        min_vehicle_area: int = 15,
        max_vehicle_area: int = 400,
        confidence_threshold: float = 0.50,
    ):
        """
        Args:
            default_gsd: Estimated Ground Sample Distance (meters per pixel).
            min_vehicle_area: Minimum pixel area for vehicle bounding box.
            max_vehicle_area: Maximum pixel area for vehicle bounding box.
            confidence_threshold: Minimum detection confidence.
        """
        self.default_gsd = default_gsd
        self.min_vehicle_area = min_vehicle_area
        self.max_vehicle_area = max_vehicle_area
        self.confidence_threshold = confidence_threshold

    def check_feasibility(self, gsd_meters: Optional[float] = None) -> Dict:
        """
        Evaluates whether satellite resolution physically supports reliable vehicle detection.
        """
        gsd = gsd_meters if gsd_meters is not None else self.default_gsd
        car_length_meters = 4.5
        car_width_meters = 1.8

        pixel_length = car_length_meters / gsd
        pixel_width = car_width_meters / gsd
        pixel_area = pixel_length * pixel_width

        is_feasible = gsd <= self.VHR_MAX_GSD_METERS

        if is_feasible:
            reason = (
                f"Resolution is FEASIBLE ({gsd:.2f} m/px). "
                f"A typical vehicle spans ~{pixel_length:.1f}x{pixel_width:.1f} px ({pixel_area:.0f} px^2), "
                f"which provides sufficient spatial detail for reliable detection."
            )
        else:
            reason = (
                f"Resolution is INSUFFICIENT ({gsd:.2f} m/px). "
                f"A standard vehicle spans only ~{pixel_length:.1f}x{pixel_width:.1f} px ({pixel_area:.1f} px^2), "
                f"falling below the spatial sampling threshold. "
                f"Reliable vehicle detection requires GSD <= {self.VHR_MAX_GSD_METERS}m (e.g. WorldView-3 / xView)."
            )

        return {
            "is_feasible": is_feasible,
            "gsd_meters": gsd,
            "expected_vehicle_pixels": round(pixel_area, 1),
            "reason": reason,
        }

    def detect(
        self,
        image_rgb: np.ndarray,
        gsd_meters: Optional[float] = None,
        road_mask: Optional[np.ndarray] = None,
        force_detection: bool = False,
    ) -> Tuple[List[Dict], Dict]:
        """
        Runs vehicle detection if resolution allows or if forced on high-res crop.

        Args:
            image_rgb: uint8 array of shape (H, W, 3).
            gsd_meters: Ground sample distance in meters.
            road_mask: Optional road corridor mask (vehicles are constrained to roads/parking).
            force_detection: Overrides resolution check (useful for high-res validation crops).

        Returns:
            detections: List of detection dictionaries.
            diagnostic: Report explaining feasibility and parameters.
        """
        feasibility = self.check_feasibility(gsd_meters)

        if not feasibility["is_feasible"] and not force_detection:
            # Scientifically correct response: suppress false detections on coarse imagery
            return [], feasibility

        height, width, _ = image_rgb.shape
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Overhead vehicle detection via local contrast & morphological top-hat filter
        # Top-hat isolates bright/dark objects on contrasting road background
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        vehicle_contrast = cv2.bitwise_or(tophat, blackhat)

        # Constrain to roads if road mask is available
        if road_mask is not None:
            r_mask = (road_mask > 0).astype(np.uint8) * 255
            k_expand = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            r_dilated = cv2.dilate(r_mask, k_expand, iterations=1)
            vehicle_contrast = cv2.bitwise_and(vehicle_contrast, r_dilated)

        # Threshold contrasting candidate vehicle blobs
        _, thresh = cv2.threshold(vehicle_contrast, 35, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for idx, cnt in enumerate(contours, start=1):
            area = cv2.contourArea(cnt)
            if area < self.min_vehicle_area or area > self.max_vehicle_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = max(w, h) / max(1, min(w, h))

            # Vehicles viewed from above have aspect ratio typically between 1.5 and 3.5
            if aspect_ratio > 4.0:
                continue

            # Confidence based on intensity contrast and size consistency
            roi = vehicle_contrast[y:y + h, x:x + w]
            contrast_score = float(roi.mean()) / 255.0
            confidence = float(np.clip(0.50 + (0.45 * contrast_score), 0.0, 0.95))

            if confidence < self.confidence_threshold:
                continue

            detections.append({
                "region_id": idx,
                "object_type": "vehicle",
                "confidence": round(confidence, 3),
                "confidence_percent": round(confidence * 100, 1),
                "bbox": [int(x), int(y), int(w), int(h)],
                "center": [int(x + w // 2), int(y + h // 2)],
                "area_pixels": int(area),
                "aspect_ratio": round(aspect_ratio, 2),
            })

        feasibility["detected_count"] = len(detections)
        return detections, feasibility

    def visualize(
        self,
        image_rgb: np.ndarray,
        detections: List[Dict],
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Draws red bounding boxes around detected vehicles.
        """
        overlay = image_rgb.copy()

        for det in detections:
            x, y, w, h = det["bbox"]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 1)  # Red box
            label = f"Veh {det['confidence_percent']}%"
            cv2.putText(
                overlay,
                label,
                (x, max(10, y - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
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
