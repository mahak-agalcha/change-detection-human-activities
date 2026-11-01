"""
Automated Test Suite for Part 2: Object Detection and Segmentation.

Tests:
1. Image loading and validation
2. Remote sensing band handling (NDVI, RGB normalization)
3. Building segmentation and polygon contour extraction
4. Road network corridor segmentation and elongation filtering
5. Vehicle detection and spatial resolution (GSD) feasibility analysis
6. Geospatial Affine transformation and GeoJSON schema compliance
7. End-to-end pipeline execution and file creation
8. Backend inference contract compatibility
"""

import json
from pathlib import Path
import sys
import numpy as np
import pytest

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.object_detection.image_loader import SatelliteImageLoader
from src.object_detection.building_segmentation import BuildingSegmenter
from src.object_detection.road_segmentation import RoadSegmenter
from src.object_detection.vehicle_detection import VehicleDetector
from src.geospatial.geo_converter import GeospatialConverter
from src.object_detection.pipeline import ObjectDetectionPipeline
from run_part2_pipeline import run_inference


# -------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------

@pytest.fixture
def synthetic_satellite_scene():
    """Generates a synthetic 256x256 RGB satellite patch with known structures."""
    img = np.full((256, 256, 3), 60, dtype=np.uint8)  # Dark ground/soil

    # Draw 2 rectangular bright buildings
    img[30:70, 30:80] = [210, 215, 220]    # Building 1: 40x50 px
    img[120:170, 140:190] = [200, 205, 210]  # Building 2: 50x50 px

    # Draw an elongated linear road across the scene
    img[90:105, :] = [140, 135, 130]      # Horizontal road corridor: 15 px high, 256 px wide

    return img


@pytest.fixture
def synthetic_4band_scene():
    """Generates a synthetic 4-band image (R, G, B, NIR) mimicking PlanetScope."""
    img = np.zeros((128, 128, 4), dtype=np.float32)
    img[:, :, 0] = 50.0   # Red
    img[:, :, 1] = 60.0   # Green
    img[:, :, 2] = 40.0   # Blue
    img[:, :, 3] = 180.0  # High NIR (vegetation)
    return img


# -------------------------------------------------------------
# 1. Image Loading and Preprocessing Tests
# -------------------------------------------------------------

def test_image_loader_rgb_array(synthetic_satellite_scene):
    loader = SatelliteImageLoader()
    rgb, meta = loader.load_image(synthetic_satellite_scene)

    assert rgb.shape == (256, 256, 3)
    assert rgb.dtype == np.uint8
    assert meta["height"] == 256
    assert meta["width"] == 256
    assert meta["crs"] == "EPSG:4326"
    assert len(meta["transform"]) == 6


def test_image_loader_4band_ndvi(synthetic_4band_scene):
    loader = SatelliteImageLoader()
    rgb, meta = loader.load_image(synthetic_4band_scene)

    assert rgb.shape == (128, 128, 3)
    assert meta["has_nir"] is True
    assert "ndvi" in meta["spectral_indices"]
    ndvi = meta["spectral_indices"]["ndvi"]
    assert ndvi.shape == (128, 128)
    assert ndvi.min() >= -1.0 and ndvi.max() <= 1.0


def test_image_loader_missing_file_raises():
    loader = SatelliteImageLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_image("non_existent_satellite_file.tif")


def test_image_loader_invalid_extension_raises(tmp_path):
    bad_file = tmp_path / "test.txt"
    bad_file.write_text("not an image")
    loader = SatelliteImageLoader()
    with pytest.raises(ValueError, match="Unsupported image extension"):
        loader.load_image(bad_file)


# -------------------------------------------------------------
# 2. Building Segmentation Tests
# -------------------------------------------------------------

def test_building_segmentation_extraction(synthetic_satellite_scene):
    segmenter = BuildingSegmenter(min_area=50, max_area=5000)
    mask, detections = segmenter.segment(synthetic_satellite_scene)

    assert mask.shape == (256, 256)
    assert mask.dtype == np.uint8
    assert len(detections) >= 1

    for det in detections:
        assert det["object_type"] == "building"
        assert 0.0 <= det["confidence"] <= 1.0
        assert len(det["bbox"]) == 4  # [x, y, w, h]
        assert len(det["polygon"]) >= 3  # Valid polygon ring
        assert det["area_pixels"] >= 50


def test_building_segmentation_invalid_shape():
    segmenter = BuildingSegmenter()
    invalid_img = np.zeros((100, 100), dtype=np.uint8)  # 2D instead of 3D RGB
    with pytest.raises(ValueError, match="Expected RGB image"):
        segmenter.segment(invalid_img)


# -------------------------------------------------------------
# 3. Road Segmentation Tests
# -------------------------------------------------------------

def test_road_segmentation_elongation(synthetic_satellite_scene):
    segmenter = RoadSegmenter(min_length=20, min_aspect_ratio=2.0)
    mask, detections = segmenter.segment(synthetic_satellite_scene)

    assert mask.shape == (256, 256)
    assert isinstance(detections, list)
    for det in detections:
        assert det["object_type"] == "road"
        assert det["aspect_ratio"] >= 2.0
        assert 0.0 <= det["confidence"] <= 1.0


# -------------------------------------------------------------
# 4. Vehicle Detection and Resolution Feasibility Tests
# -------------------------------------------------------------

def test_vehicle_coarse_resolution_suppression(synthetic_satellite_scene):
    detector = VehicleDetector()
    # At 3.0m GSD (SpaceNet 7 PlanetScope), vehicles span < 1 pixel -> must be suppressed
    dets, diag = detector.detect(synthetic_satellite_scene, gsd_meters=3.0)

    assert diag["is_feasible"] is False
    assert len(dets) == 0
    assert "INSUFFICIENT" in diag["reason"]


def test_vehicle_vhr_resolution_feasibility():
    detector = VehicleDetector()
    # At 0.3m GSD (WorldView-3 / xView), vehicles span ~15x6 pixels -> feasible!
    diag = detector.check_feasibility(gsd_meters=0.3)

    assert diag["is_feasible"] is True
    assert "FEASIBLE" in diag["reason"]
    assert diag["expected_vehicle_pixels"] > 30


def test_vehicle_forced_detection(synthetic_satellite_scene):
    detector = VehicleDetector()
    # Force detection option executes detector on high-res crop
    dets, diag = detector.detect(synthetic_satellite_scene, force_detection=True)
    assert isinstance(dets, list)


# -------------------------------------------------------------
# 5. Geospatial Conversion & GeoJSON Tests
# -------------------------------------------------------------

def test_geospatial_affine_transform():
    converter = GeospatialConverter()
    # Transform: x_geo = 75.8 + 0.0001*col, y_geo = 22.7 - 0.0001*row
    transform = (0.0001, 0.0, 75.8, 0.0, -0.0001, 22.7)
    pixel_coords = [(0, 0), (10, 0), (10, 10), (0, 10)]

    geo_coords = converter.pixel_to_geo_coords(pixel_coords, transform)

    assert len(geo_coords) == 5  # Ring closed
    assert geo_coords[0] == [75.8, 22.7]
    assert geo_coords[0] == geo_coords[-1]  # Closed polygon


def test_geospatial_geojson_schema(tmp_path):
    converter = GeospatialConverter()
    transform = (0.0001, 0.0, 75.8, 0.0, -0.0001, 22.7)
    detections = [
        {
            "region_id": 1,
            "object_type": "building",
            "confidence": 0.88,
            "confidence_percent": 88.0,
            "bbox": [10, 10, 20, 20],
            "polygon": [[10, 10], [30, 10], [30, 30], [10, 30]],
            "area_pixels": 400,
        }
    ]

    geojson_data = converter.to_geojson_features(detections, transform)
    assert geojson_data["type"] == "FeatureCollection"
    assert len(geojson_data["features"]) == 1

    feature = geojson_data["features"][0]
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Polygon"
    assert feature["properties"]["object_type"] == "building"
    assert feature["properties"]["confidence"] == 0.88

    # Test saving
    out_file = tmp_path / "test.geojson"
    converter.save_geojson(geojson_data, out_file)
    assert out_file.exists()

    with out_file.open(encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["type"] == "FeatureCollection"


# -------------------------------------------------------------
# 6. End-to-End Pipeline & Contract Tests
# -------------------------------------------------------------

def test_pipeline_end_to_end(synthetic_satellite_scene, tmp_path):
    pipeline = ObjectDetectionPipeline()
    results = pipeline.run_image(
        image_path=synthetic_satellite_scene,
        output_dir=tmp_path / "part2_test_outputs",
    )

    assert "counts" in results
    assert "geojson" in results
    assert "contract_features" in results
    assert "saved_files" in results

    # Verify generated files exist
    assert Path(results["saved_files"]["composite_overlay"]).exists()
    assert Path(results["saved_files"]["geojson"]).exists()
    assert Path(results["saved_files"]["csv"]).exists()
    assert Path(results["saved_files"]["contract_json"]).exists()


def test_run_inference_service_compatibility(synthetic_satellite_scene, tmp_path):
    """Verifies that run_inference strictly satisfies backend/app/services/inference.py contract."""
    from PIL import Image
    test_img_path = tmp_path / "test_satellite.png"
    Image.fromarray(synthetic_satellite_scene).save(test_img_path)

    # Call run_inference
    contract_output = run_inference(str(test_img_path), str(test_img_path))

    assert isinstance(contract_output, list)
    if len(contract_output) > 0:
        item = contract_output[0]
        # Required keys by backend/app/services/inference.py
        assert "object_type" in item
        assert item["object_type"] in ("building", "road", "vehicle", "unknown")
        assert "cause_label" in item
        assert "confidence" in item
        assert 0.0 <= item["confidence"] <= 1.0
        assert "geometry" in item
        assert "type" in item["geometry"]
        assert "coordinates" in item["geometry"]
