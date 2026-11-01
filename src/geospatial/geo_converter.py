"""
Geospatial Vectorization and GeoJSON Export Module.

Provides:
1. Pixel-to-Geographic Affine transformation (lon, lat)
2. Shapely polygon construction and topological validation (is_valid, buffer(0))
3. GeoJSON FeatureCollection generation conforming to docs/api/output-contract.md
   and backend/app/services/inference.py contract.
4. Physical ground area estimation (square meters).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from shapely.geometry import Polygon, mapping
from shapely.validation import make_valid


class GeospatialConverter:
    """Converts pixel detection coordinates to valid geospatial GeoJSON geometries."""

    def __init__(self, default_crs: str = "EPSG:4326"):
        self.default_crs = default_crs

    def pixel_to_geo_coords(
        self,
        pixel_coords: List[Tuple[float, float]],
        transform: Tuple[float, float, float, float, float, float],
    ) -> List[List[float]]:
        """
        Transforms pixel coordinates (col, row) to geographic coordinates (x_geo, y_geo).

        Transform tuple format: (a, b, c, d, e, f)
            x_geo = c + a * col + b * row
            y_geo = f + d * col + e * row
        """
        a, b, c, d, e, f = transform
        geo_coords = []
        for pt in pixel_coords:
            col, row = float(pt[0]), float(pt[1])
            x_geo = c + (a * col) + (b * row)
            y_geo = f + (d * col) + (e * row)
            # GeoJSON expects [longitude, latitude]
            geo_coords.append([round(x_geo, 6), round(y_geo, 6)])

        # Ensure polygon ring is closed (first point equals last point)
        if geo_coords and geo_coords[0] != geo_coords[-1]:
            geo_coords.append(geo_coords[0])

        return geo_coords

    def bbox_to_geo_polygon(
        self,
        bbox: List[int],
        transform: Tuple[float, float, float, float, float, float],
    ) -> List[List[float]]:
        """Converts a bounding box [x, y, w, h] to a closed geographic polygon ring."""
        x, y, w, h = bbox
        corners = [
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h),
            (x, y)
        ]
        return self.pixel_to_geo_coords(corners, transform)

    def to_geojson_features(
        self,
        detections: List[Dict],
        transform: Tuple[float, float, float, float, float, float],
        crs: Optional[str] = None,
        source_image: Optional[str] = None,
        gsd_meters: float = 3.0,
    ) -> Dict:
        """
        Converts a list of detection dictionaries into a valid GeoJSON FeatureCollection.

        Conforms strictly to:
        1. docs/api/output-contract.md:
           - object type ("building", "road", "vehicle")
           - object confidence
           - geometry
           - region ID
        2. backend/app/services/inference.py contract.
        """
        active_crs = crs or self.default_crs
        features = []

        for det in detections:
            # Prefer detailed polygon outline if available, otherwise use bounding box
            if det.get("polygon") and len(det["polygon"]) >= 3:
                geo_ring = self.pixel_to_geo_coords(det["polygon"], transform)
            elif det.get("bbox"):
                geo_ring = self.bbox_to_geo_polygon(det["bbox"], transform)
            else:
                continue

            if len(geo_ring) < 4:
                continue

            # Validate geometry with Shapely
            raw_poly = Polygon(geo_ring)
            if not raw_poly.is_valid:
                poly = make_valid(raw_poly)
            else:
                poly = raw_poly

            geom_dict = mapping(poly)

            # Estimate physical ground area in square meters
            pixel_area = det.get("area_pixels", (det.get("bbox", [0, 0, 0, 0])[2] * det.get("bbox", [0, 0, 0, 0])[3]))
            area_sq_m = float(pixel_area * (gsd_meters ** 2))

            feature = {
                "type": "Feature",
                "geometry": geom_dict,
                "properties": {
                    "region_id": det.get("region_id", len(features) + 1),
                    "object_type": det.get("object_type", "unknown"),
                    "confidence": float(det.get("confidence", 0.0)),
                    "confidence_percent": det.get("confidence_percent", round(det.get("confidence", 0.0) * 100, 1)),
                    "cause_label": det.get("cause_label", "human" if det.get("object_type") in ("building", "road") else "uncertain"),
                    "area_pixels": pixel_area,
                    "estimated_area_sq_m": round(area_sq_m, 2),
                    "source_image": source_image or "unknown",
                }
            }
            features.append(feature)

        feature_collection = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": active_crs}
            },
            "features": features
        }

        return feature_collection

    def save_geojson(
        self,
        geojson_data: Dict,
        output_path: Union[str, Path]
    ) -> Path:
        """Saves GeoJSON FeatureCollection to disk with pretty formatting."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=2)
        return path
