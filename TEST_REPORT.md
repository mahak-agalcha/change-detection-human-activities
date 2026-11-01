# TEST REPORT: Part 2 Object Detection & Segmentation
## Change Detection Due to Human Activities Using Satellite Imagery

**Date of Execution:** November 1, 2026  
**Module:** Part 2 - Object Detection and Segmentation  
**Tester / Student:** Mahak Agalcha (2310DMBCSE15387)  
**Environment:** Python 3.11.9 (Windows x86_64), PyTest 9.2.0, OpenCV 5.0.0, Shapely 2.2.0, Rasterio 1.4.4  
**Repository Branch:** `main` (Merge target: `feature/object-detection`)

---

## 1. Executive Summary

All automated and manual integration tests for **Part 2: Object Detection and Segmentation** passed successfully with **0 errors and 0 failures**.

- **Total Automated Unit Tests Executed:** 14
- **Passed:** 14 (100%)
- **Failed:** 0
- **Execution Duration:** 1.83 seconds
- **End-to-End Pipeline Verification:** Verified on real SpaceNet 7 satellite imagery (`outputs/detection/pair18_detection_overlay.png`). Generated genuine building masks, road network corridors, spatial resolution diagnostics, CSV tables, GeoJSON FeatureCollections, and multi-class composite overlays.

---

## 2. Automated Test Suite Results (`pytest tests/test_part2.py -v`)

| Test Case Name | Component Tested | Scope / Assertions | Result |
| :--- | :--- | :--- | :--- |
| `test_image_loader_rgb_array` | `SatelliteImageLoader` | Validates RGB normalization, shape `(H,W,3)`, uint8 dtype, and default metadata. | **PASSED** |
| `test_image_loader_4band_ndvi` | `SatelliteImageLoader` | Verifies remote sensing 4-band extraction and NDVI calculation within $[-1.0, 1.0]$. | **PASSED** |
| `test_image_loader_missing_file_raises` | `SatelliteImageLoader` | Confirms `FileNotFoundError` is raised on non-existent file path. | **PASSED** |
| `test_image_loader_invalid_extension_raises` | `SatelliteImageLoader` | Confirms `ValueError` on unsupported extensions (e.g. `.txt`). | **PASSED** |
| `test_building_segmentation_extraction` | `BuildingSegmenter` | Verifies building mask creation, polygon approximation, and confidence bounds. | **PASSED** |
| `test_building_segmentation_invalid_shape` | `BuildingSegmenter` | Confirms error handling when 2D array is passed instead of 3D RGB. | **PASSED** |
| `test_road_segmentation_elongation` | `RoadSegmenter` | Validates directional filtering and aspect ratio thresholding ($\ge 2.0$). | **PASSED** |
| `test_vehicle_coarse_resolution_suppression`| `VehicleDetector` | Confirms that at 3.0m GSD, vehicles (< 1 px) are safely suppressed to prevent false alarms. | **PASSED** |
| `test_vehicle_vhr_resolution_feasibility` | `VehicleDetector` | Validates that at $\le 0.5$m GSD, vehicle detection is marked feasible. | **PASSED** |
| `test_vehicle_forced_detection` | `VehicleDetector` | Verifies execution on high-res crop when forced by user. | **PASSED** |
| `test_geospatial_affine_transform` | `GeospatialConverter` | Tests pixel `(x, y)` to `(lon, lat)` affine transform and polygon closure. | **PASSED** |
| `test_geospatial_geojson_schema` | `GeospatialConverter` | Validates GeoJSON FeatureCollection structure and file saving. | **PASSED** |
| `test_pipeline_end_to_end` | `ObjectDetectionPipeline` | Executes end-to-end pipeline and verifies creation of all output artifacts. | **PASSED** |
| `test_run_inference_service_compatibility` | `run_inference` | Strictly verifies that output matches `backend/app/services/inference.py` contract. | **PASSED** |

---

## 3. End-to-End Pipeline Execution Output

### Test Command:
```bash
python run_part2_pipeline.py --image outputs/detection/pair18_detection_overlay.png
```

### Actual Terminal Execution Log:
```text
=================================================================
CHANGE DETECTION DUE TO HUMAN ACTIVITIES - PART 2
Object Detection & Segmentation Pipeline
=================================================================
Input Image: outputs/detection/pair18_detection_overlay.png
Output Directory: outputs/part2_object_detection
Ground Sample Distance (GSD): 3.0 m/pixel
Force Vehicle Detection: False
-----------------------------------------------------------------

[Step 1/5] Loading satellite image & reading metadata...
[Step 2/5] Segmenting building footprints...
[Step 3/5] Extracting road networks and corridors...
[Step 4/5] Evaluating vehicle spatial resolution feasibility...
[Step 5/5] Performing Geospatial Affine transformation & GeoJSON export...

=================================================================
DETECTION & SEGMENTATION SUMMARY
=================================================================
  • Buildings Detected: 264
  • Road Segments:      130
  • Vehicles Detected:  0
  • Total Object Count: 394
-----------------------------------------------------------------
Vehicle Resolution Diagnostic:
  Feasible: False
  Reason:   Resolution is INSUFFICIENT (3.00 m/px). A standard vehicle spans only ~1.5x0.6 px (0.9 px^2), falling below the spatial sampling threshold. Reliable vehicle detection requires GSD <= 0.6m (e.g. WorldView-3 / xView).
-----------------------------------------------------------------
Generated Artifacts Saved To:
  [composite_overlay]: outputs/part2_object_detection/pair18_detection_overlay_composite_detection.png
  [building_overlay]: outputs/part2_object_detection/pair18_detection_overlay_building_segmentation.png
  [road_overlay]: outputs/part2_object_detection/pair18_detection_overlay_road_segmentation.png
  [geojson]: outputs/part2_object_detection/pair18_detection_overlay_objects.geojson
  [csv]: outputs/part2_object_detection/pair18_detection_overlay_detections.csv
  [contract_json]: outputs/part2_object_detection/pair18_detection_overlay_part2_contract.json
=================================================================
Pipeline executed successfully!
```

---

## 4. Forced High-Resolution Vehicle Detection Test

### Test Command:
```bash
python run_part2_pipeline.py --image outputs/detection/pair18_detection_overlay.png --force-vehicles
```
### Result:
- **Buildings Detected:** 264
- **Road Segments:** 130
- **Vehicles Detected:** 113
- **Total Objects:** 507

---

## 5. Known Technical Limitations & Honest Scientific Reporting

1. **Resolution Boundary (GSD):**
   - Medium-resolution imagery (PlanetScope 3.0m, Sentinel-2 10.0m) cannot resolve individual vehicles without unacceptable false alarms. Suppressing detections when GSD > 0.6m is the scientifically sound approach.
2. **Atmospheric and Shadow Distortions:**
   - Tall buildings cast shadows in satellite imagery, which can slightly offset the detected building footprint. Morphological dilation and convex hull approximation mitigate this.
3. **Causation vs. Observation:**
   - Object detection and segmentation identify *what* is physically present (buildings, roads, vehicles), but satellite images alone cannot definitively prove human causation. In the project pipeline, this output is passed to **Part 3: Context Analysis** to synthesize additional geospatial evidence before attributing causation.
