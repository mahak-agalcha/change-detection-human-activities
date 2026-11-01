"""
Unified Object Detection and Segmentation Pipeline.

Integrates:
1. Multi-format image loading (PNG, GeoTIFF, NPZ)
2. Building footprint segmentation and polygon extraction
3. Road corridor and network extraction
4. Vehicle detection with GSD feasibility analysis
5. Geospatial transformation (CRS, Affine, GeoJSON export)
6. Multi-class composite visual overlay generation
7. Output contract compliance for Part 3 and Part 4 integration.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.object_detection.image_loader import SatelliteImageLoader
from src.object_detection.building_segmentation import BuildingSegmenter
from src.object_detection.road_segmentation import RoadSegmenter
from src.object_detection.vehicle_detection import VehicleDetector
from src.geospatial.geo_converter import GeospatialConverter


class ObjectDetectionPipeline:
    """End-to-end Part 2 pipeline for satellite object detection and segmentation."""

    def __init__(
        self,
        default_crs: str = "EPSG:4326",
        default_gsd: float = 3.0,
        building_min_area: int = 15,
        road_min_length: int = 25,
        confidence_threshold: float = 0.50,
    ):
        self.default_crs = default_crs
        self.default_gsd = default_gsd
        self.loader = SatelliteImageLoader(default_crs=default_crs)
        self.building_segmenter = BuildingSegmenter(
            min_area=building_min_area,
            confidence_threshold=confidence_threshold,
        )
        self.road_segmenter = RoadSegmenter(
            min_length=road_min_length,
            confidence_threshold=max(0.40, confidence_threshold - 0.05),
        )
        self.vehicle_detector = VehicleDetector(
            default_gsd=default_gsd,
            confidence_threshold=confidence_threshold,
        )
        self.geo_converter = GeospatialConverter(default_crs=default_crs)

    def run_image(
        self,
        image_path: Union[str, Path, np.ndarray],
        output_dir: Optional[Union[str, Path]] = None,
        candidate_mask: Optional[np.ndarray] = None,
        gsd_meters: Optional[float] = None,
        force_vehicle_detection: bool = False,
    ) -> Dict:
        """
        Runs the complete object detection and segmentation pipeline on a single image.

        Args:
            image_path: Path to satellite image (or preloaded array).
            output_dir: Folder where visual overlays and GeoJSON outputs will be saved.
            candidate_mask: Optional binary mask of candidate changed regions (from Part 1).
            gsd_meters: Optional Ground Sample Distance in meters/pixel.
            force_vehicle_detection: If True, executes vehicle detector regardless of GSD.

        Returns:
            Dictionary containing detected objects, masks, GeoJSON, and file paths.
        """
        # 1. Load and standardize image
        rgb_image, metadata = self.loader.load_image(image_path)
        active_gsd = gsd_meters if gsd_meters is not None else self.default_gsd
        ndvi = metadata.get("spectral_indices", {}).get("ndvi")

        # 2. Building segmentation
        building_mask, building_detections = self.building_segmenter.segment(
            image_rgb=rgb_image,
            ndvi=ndvi,
            candidate_mask=candidate_mask,
        )

        # 3. Road segmentation (excluding detected buildings)
        road_mask, road_detections = self.road_segmenter.segment(
            image_rgb=rgb_image,
            building_mask=building_mask,
            candidate_mask=candidate_mask,
        )

        # 4. Vehicle detection with spatial resolution assessment
        vehicle_detections, vehicle_diag = self.vehicle_detector.detect(
            image_rgb=rgb_image,
            gsd_meters=active_gsd,
            road_mask=road_mask,
            force_detection=force_vehicle_detection,
        )

        # Combine all detections with unified region indexing
        all_detections = []
        region_counter = 1

        for det in building_detections:
            item = dict(det)
            item["region_id"] = region_counter
            region_counter += 1
            all_detections.append(item)

        for det in road_detections:
            item = dict(det)
            item["region_id"] = region_counter
            region_counter += 1
            all_detections.append(item)

        for det in vehicle_detections:
            item = dict(det)
            item["region_id"] = region_counter
            region_counter += 1
            all_detections.append(item)

        # 5. Geospatial vectorization
        transform = metadata["transform"]
        geojson_data = self.geo_converter.to_geojson_features(
            detections=all_detections,
            transform=transform,
            crs=metadata["crs"],
            source_image=Path(image_path).name if isinstance(image_path, (str, Path)) else "array",
            gsd_meters=active_gsd,
        )

        # Prepare contract output (for backend/app/services/inference.py compatibility)
        contract_features = []
        for feature in geojson_data["features"]:
            props = feature["properties"]
            contract_features.append({
                "region_id": props["region_id"],
                "object_type": props["object_type"],
                "cause_label": props["cause_label"],
                "confidence": props["confidence"],
                "geometry": feature["geometry"],
                "area_sq_m": props["estimated_area_sq_m"],
            })

        results = {
            "metadata": metadata,
            "counts": {
                "buildings": len(building_detections),
                "roads": len(road_detections),
                "vehicles": len(vehicle_detections),
                "total": len(all_detections),
            },
            "vehicle_diagnostic": vehicle_diag,
            "detections": all_detections,
            "geojson": geojson_data,
            "contract_features": contract_features,
            "masks": {
                "building": building_mask,
                "road": road_mask,
            },
            "saved_files": {},
        }

        # 6. Save visual overlays and data files if output directory specified
        if output_dir is not None:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            stem = Path(image_path).stem if isinstance(image_path, (str, Path)) else "analysis"

            # Composite multi-class visual overlay
            composite_overlay = self._create_composite_overlay(
                rgb_image=rgb_image,
                building_mask=building_mask,
                road_mask=road_mask,
                detections=all_detections,
            )

            overlay_file = out_path / f"{stem}_composite_detection.png"
            bgr = cv2.cvtColor(composite_overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(overlay_file), bgr)
            results["saved_files"]["composite_overlay"] = str(overlay_file)

            # Separate Building visual overlay
            bldg_overlay_file = out_path / f"{stem}_building_segmentation.png"
            self.building_segmenter.visualize(
                rgb_image, building_detections, building_mask, bldg_overlay_file
            )
            results["saved_files"]["building_overlay"] = str(bldg_overlay_file)

            # Separate Road visual overlay
            road_overlay_file = out_path / f"{stem}_road_segmentation.png"
            self.road_segmenter.visualize(
                rgb_image, road_detections, road_mask, road_overlay_file
            )
            results["saved_files"]["road_overlay"] = str(road_overlay_file)

            # GeoJSON export
            geojson_file = out_path / f"{stem}_objects.geojson"
            self.geo_converter.save_geojson(geojson_data, geojson_file)
            results["saved_files"]["geojson"] = str(geojson_file)

            # CSV detections export
            csv_file = out_path / f"{stem}_detections.csv"
            self._save_summary_csv(all_detections, csv_file)
            results["saved_files"]["csv"] = str(csv_file)

            # Contract JSON export (for Part 3 / Part 4)
            contract_file = out_path / f"{stem}_part2_contract.json"
            with contract_file.open("w", encoding="utf-8") as f:
                json.dump(contract_features, f, indent=2)
            results["saved_files"]["contract_json"] = str(contract_file)

        return results

    def _create_composite_overlay(
        self,
        rgb_image: np.ndarray,
        building_mask: np.ndarray,
        road_mask: np.ndarray,
        detections: List[Dict],
    ) -> np.ndarray:
        """Draws multi-class overlays with custom colors per object class."""
        overlay = rgb_image.copy()

        # Buildings: Cyan [0, 200, 255]
        mask_buildings = np.zeros_like(rgb_image)
        mask_buildings[building_mask == 1] = [0, 200, 255]
        cv2.addWeighted(mask_buildings, 0.40, overlay, 0.60, 0, overlay)

        # Roads: Amber/Yellow [255, 175, 0]
        mask_roads = np.zeros_like(rgb_image)
        mask_roads[road_mask == 1] = [255, 175, 0]
        cv2.addWeighted(mask_roads, 0.45, overlay, 0.55, 0, overlay)

        # Outlines and bounding labels
        for det in detections:
            obj_type = det["object_type"]
            conf = det.get("confidence_percent", 0.0)
            x, y, w, h = det["bbox"]

            if obj_type == "building":
                color = (0, 255, 255)  # Cyan/yellow border
                label = f"Bldg {conf}%"
            elif obj_type == "road":
                color = (255, 200, 0)  # Amber border
                label = f"Road {conf}%"
            elif obj_type == "vehicle":
                color = (255, 50, 50)  # Red border
                label = f"Veh {conf}%"
            else:
                color = (200, 200, 200)
                label = f"{obj_type} {conf}%"

            if det.get("polygon") and len(det["polygon"]) >= 3:
                pts = np.array(det["polygon"], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=1)

            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
            cv2.putText(
                overlay,
                label,
                (x, max(12, y - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Add top title banner
        banner = np.zeros((30, overlay.shape[1], 3), dtype=np.uint8)
        banner[:] = [30, 30, 30]
        text = "Part 2: Multi-Class Detection [Cyan: Building | Yellow: Road | Red: Vehicle]"
        cv2.putText(banner, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        composite = np.vstack([banner, overlay])

        return composite

    @staticmethod
    def _save_summary_csv(detections: List[Dict], output_path: Path) -> None:
        """Saves tabular detection summary to CSV."""
        fieldnames = [
            "region_id",
            "object_type",
            "confidence",
            "confidence_percent",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "area_pixels",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for det in detections:
                x, y, w, h = det["bbox"]
                writer.writerow({
                    "region_id": det["region_id"],
                    "object_type": det["object_type"],
                    "confidence": det["confidence"],
                    "confidence_percent": det["confidence_percent"],
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_w": w,
                    "bbox_h": h,
                    "area_pixels": det.get("area_pixels", w * h),
                })
