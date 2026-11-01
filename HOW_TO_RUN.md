# HOW TO RUN: Part 2 Object Detection & Segmentation

This guide provides step-by-step instructions to set up the environment, run the tests, and execute the Object Detection and Segmentation pipeline from the VS Code terminal.

---

## 1. Environment Setup

### Prerequisites
- Python 3.11 (Recommended and tested)
- Git

### Step A: Open Terminal in the Project Root
Make sure your terminal is inside the project directory:
```powershell
cd C:\Users\acer\Documents\change_detection_project\change-detection-human-activities
```

### Step B: Create and Activate Virtual Environment
```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# (If using Command Prompt)
.\.venv\Scripts\activate.bat

# (If using Linux / macOS)
source .venv/bin/activate
```

### Step C: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 2. Running Automated Tests

Run the full pytest suite to verify all 14 unit and integration tests:
```powershell
pytest tests/test_part2.py -v
```

Expected output:
```text
tests/test_part2.py::test_image_loader_rgb_array PASSED
tests/test_part2.py::test_image_loader_4band_ndvi PASSED
tests/test_part2.py::test_image_loader_missing_file_raises PASSED
tests/test_part2.py::test_image_loader_invalid_extension_raises PASSED
tests/test_part2.py::test_building_segmentation_extraction PASSED
tests/test_part2.py::test_building_segmentation_invalid_shape PASSED
tests/test_part2.py::test_road_segmentation_elongation PASSED
tests/test_part2.py::test_vehicle_coarse_resolution_suppression PASSED
tests/test_part2.py::test_vehicle_vhr_resolution_feasibility PASSED
tests/test_part2.py::test_vehicle_forced_detection PASSED
tests/test_part2.py::test_geospatial_affine_transform PASSED
tests/test_part2.py::test_geospatial_geojson_schema PASSED
tests/test_part2.py::test_pipeline_end_to_end PASSED
tests/test_part2.py::test_run_inference_service_compatibility PASSED
============================== 14 passed in 1.83s ==============================
```

---

## 3. Running the Pipeline

### Mode 1: Quick Demonstration Mode (Best for College Presentation)
Runs the complete pipeline on the pre-configured verified satellite image:
```powershell
python run_part2_pipeline.py --demo
```

### Mode 2: Run with Custom Satellite Image
You can pass any satellite image (`.png`, `.jpg`, `.tif`, `.npz`):
```powershell
python run_part2_pipeline.py --image data/raw/sample.tif --output-dir outputs/custom_run
```

### Mode 3: Run with Forced Vehicle Detection
To demonstrate vehicle detection algorithms on high-resolution crops:
```powershell
python run_part2_pipeline.py --demo --force-vehicles
```

---

## 4. Expected Output Files

All generated outputs are saved in `outputs/part2_object_detection/`:

1. `*_composite_detection.png` — Multi-class visual overlay showing segmented buildings (Cyan), roads (Yellow/Amber), and vehicles (Red).
2. `*_building_segmentation.png` — Dedicated building footprint mask and bounding boxes.
3. `*_road_segmentation.png` — Dedicated road network corridor mask.
4. `*_objects.geojson` — Standard WGS84 GeoJSON FeatureCollection with polygon coordinates.
5. `*_detections.csv` — Tabular summary of detected regions, bounding boxes, areas, and confidence.
6. `*_part2_contract.json` — Direct JSON payload conforming to the Part 3 and Part 4 integration contract.

---

## 5. Connecting with Part 4 (Web Backend)

In `backend/app/services/inference.py`, replace the stub with:
```python
from run_part2_pipeline import run_inference

# Inside run_inference endpoint:
results = run_inference(old_path=old_img, new_path=new_img)
return results
```

---

## 6. Troubleshooting

- **Error: `ExecutionPolicy` restricted in PowerShell:**
  Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- **Error: `ModuleNotFoundError: No module named 'src'`:**
  Ensure you are executing from the repository root:
  `cd change-detection-human-activities`
- **Error: `FileNotFoundError`:**
  Run `python run_part2_pipeline.py --demo` which automatically locates the included demonstration sample.
