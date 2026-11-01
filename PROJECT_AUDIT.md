# PROJECT AUDIT: Change Detection Due to Human Activities
## Part 2: Object Detection and Segmentation Module Audit

**Date:** November 1, 2026  
**Module:** Part 2 - Object Detection and Segmentation  
**Student:** Mahak Agalcha (Final Year B.Tech CSE)  
**Project Guide:** Dr. Sandeep Kumar Jain  
**Repository:** `https://github.com/mahak-agalcha/change-detection-human-activities.git`  
**Audited Location:** `c:\Users\acer\Documents\change_detection_project\change-detection-human-activities`

---

## 1. Executive Summary

This audit evaluates the codebase for **Part 2: Object Detection and Segmentation** within the multi-module project *"Change Detection Due to Human Activities Using Satellite Imagery"*.

The overall project has 4 interconnected modules:
1. **Part 1:** Data Pipeline & Change Detection (Preprocessing, Siamese CNN for temporal change mask).
2. **Part 2:** Object Detection and Segmentation (**Assigned Module** - Buildings, Roads, Vehicles, Geospatial Vectorization).
3. **Part 3:** Change Interpretation, Evidence, and Geospatial Database (Context analysis: Human vs Natural vs Uncertain).
4. **Part 4:** Backend, Frontend, and Deployment (FastAPI, React + Leaflet, Docker).

Prior work has implemented Part 1 (Siamese CNN change detection) and a basic connected-components bounding-box script in `src/object_detection/bounding_boxes.py`. Part 2's core responsibilities—building segmentation, road segmentation, vehicle resolution assessment, and real geospatial (CRS/GeoJSON) processing—remain incomplete.

---

## 2. Directory Structure and Workspace Inspection

### 2.1 Workspace Discrepancy Identified
- **Opened Workspace:** `c:\Users\acer\Documents\change_detection_project`
- **Actual Git Repository:** `c:\Users\acer\Documents\change_detection_project\change-detection-human-activities` (Contains `.git`, configuration, source code, and outputs).
- **Stray artifacts in parent folder:**
  - `data/` (empty `raw` and `processed` folders)
  - `models/` (empty `docs/` folder, 0-byte file `object_detection`)
  - `src/` (0-byte files `object_detection` and `segmentation`)
  *(Recommendation: Keep all development inside `change-detection-human-activities` to maintain Git integrity. Do not delete parent files without approval.)*

### 2.2 Git Repository State
- **Active Branch:** `main` (commit `aa8ffbf`)
- **Remote Branches:**
  - `origin/main`
  - `origin/feature/data-pipeline`
  - `origin/feature/object-detection` (currently identical to `aa8ffbf` on `main`)
  - `origin/feature/context-database`
  - `origin/feature/frontend-backend`
- **Working Tree:** Clean (no untracked or uncommitted changes prior to audit).

### 2.3 Existing Files Overview
```text
change-detection-human-activities/
├── .dockerignore
├── .gitignore
├── ARCHITECTURE.md                  # High-level Git branch diagram
├── Dockerfile                       # Python 3.11-slim base
├── docker-compose.yml
├── PROJECT_CONTRACT.md              # Team details and 4-part pipeline contract
├── README.md                        # Project overview
├── requirements.txt                 # Contains only: numpy, pandas
├── inspect_labels.py                # Inspects SpaceNet 7 building GeoJSON labels
├── data/
│   ├── raw/README.md                # Explains SpaceNet 7 raw storage rules
│   └── processed/README.md          # Explains processed data rules
├── docs/api/output-contract.md      # Inter-module data exchange contract
├── models/README.md                 # Explains model checkpoint exclusions
├── notebooks/exploration/           # Exploration notebooks
├── outputs/
│   ├── baseline/                    # Visualizations & CSVs for change baseline
│   └── detection/                   # pair18_detections.csv, overlay PNGs
└── src/
    ├── change_detection/
    │   └── baseline.py              # SiameseChangeCNN PyTorch model
    ├── object_detection/
    │   └── bounding_boxes.py        # Connected components on change mask (Pixel-level)
    └── preprocessing/               # 18 scripts for SpaceNet 7 data preparation
```

---

## 3. What Already Works

1. **Part 1 Change Detection Pipeline:**
   - Siamese CNN architecture in `src/change_detection/baseline.py` takes two 4-band satellite image patches and outputs a change probability map.
   - Comprehensive data preparation and evaluation scripts in `src/preprocessing/`.
2. **Baseline Detection Stub:**
   - `src/object_detection/bounding_boxes.py` extracts bounding boxes around connected regions in binary change masks using `cv2.connectedComponentsWithStats`.
   - Exports pixel-level CSV outputs (`outputs/detection/pair18_detections.csv`).
3. **Architecture & Contracts:**
   - `docs/api/output-contract.md` clearly states what Part 2 must deliver to Part 3 and Part 4:
     - `object type` ("building", "road", "vehicle")
     - `object confidence` (float 0.0–1.0)
     - `geometry` (Geospatial coordinates / GeoJSON polygon or MultiPolygon)
     - `region ID` (matching Part 1 candidate regions)
   - `backend/app/services/inference.py` on `origin/feature/frontend-backend` directly expects:
     ```python
     {
         "object_type": "building" | "road" | "vehicle",
         "cause_label": "human" | "natural" | "unknown",
         "confidence": float 0..1,
         "geometry": GeoJSON geometry dict (lon, lat)
     }
     ```

---

## 4. What Is Incomplete or Missing (Part 2 Responsibilities)

1. **Building Segmentation:**
   - Currently, `bounding_boxes.py` only wraps change detection regions in generic boxes labeled `predicted_change_region`. There is no actual building segmentation or footprint extraction from satellite imagery.
   - We must provide a robust building segmentation pipeline that accepts satellite imagery, computes building probability/masks, vectorizes them into building footprint polygons, and generates visual overlays.
2. **Road Segmentation:**
   - Completely missing in the current repository.
   - We must implement a road segmentation and extraction module capable of identifying linear road infrastructure, producing binary road masks, centerline/corridors, and vector geometries.
3. **Vehicle Detection Assessment:**
   - Missing in the codebase.
   - *Technical constraint to document honestly:* Typical medium-resolution satellite imagery (e.g., SpaceNet 7 PlanetScope, Sentinel-2) has a Ground Sample Distance (GSD) of 3.0m–10m per pixel. A standard passenger car is ~4.5m x 1.8m, occupying only 1–2 pixels, rendering optical vehicle detection technically unreliable. Vehicle detection is only scientifically feasible on Very High Resolution (VHR) imagery (GSD < 0.5m, e.g., WorldView-3 or xView).
   - We will implement a VHR vehicle detector baseline, provide sample high-res test imagery, and scientifically document the resolution cutoff.
4. **Geospatial Processing Engine (GIS):**
   - Currently, detections only use pixel coordinates `(x, y, w, h)`.
   - Missing: Geospatial georeferencing, affine transformations (`Affine`), CRS preservation (e.g., EPSG:4326 WGS84, EPSG:3857 Web Mercator), Shapely polygon conversion, and standard GeoJSON export.
5. **Project Integration CLI:**
   - Missing a single unified CLI for Part 2 that takes input images, runs segmentation and detection, and exports outputs conforming to `output-contract.md`.
6. **Dependencies:**
   - `requirements.txt` has only `numpy` and `pandas`. Missing `torch`, `torchvision`, `opencv-python`, `shapely`, `rasterio`, `geopandas`, `matplotlib`, `pillow`.
7. **Testing & Verification:**
   - No automated unit tests or integration test suite exist for Part 2.
8. **Documentation:**
   - Missing viva prep, student explanation, test reports, and dataset guides.

---

## 5. Architectural Issues & Corrective Actions

| Issue | Details | Corrective Action |
| :--- | :--- | :--- |
| **Workspace Ambiguity** | Parent folder contains empty duplicate files (`src/segmentation`, `models/object_detection`). | Focus all code and commits inside `change-detection-human-activities`. |
| **Missing GIS Engine** | Outputs are pixel coordinates only; no GeoJSON or real-world coordinates. | Build a dedicated `src/geospatial/` module with Affine transform and GeoJSON serialization. |
| **Sparse requirements.txt** | Key libraries are imported in code but not listed in requirements. | Update `requirements.txt` with tested, compatible dependencies. |
| **Scientific Claim Integrity** | Need to separate observed spatial change from inferred human causation. | Ensure Part 2 strictly outputs object classifications and geometries; flag causation as uncertain without ground truth. |

---

## 6. Proposed Implementation Plan (4–5 Hour Schedule)

### Hour 1: Audit, Environment & Data Setup (Current)
- Complete project audit (`PROJECT_AUDIT.md`).
- Set up Python 3.11 environment with required CV and GIS dependencies.
- Prepare public sample imagery (SpaceNet 7 GeoTIFF / high-res sample) and create `DATASET_GUIDE.md`.

### Hour 2: Building Detection & Segmentation Module
- Implement `src/object_detection/building_segmentation.py`.
- Support mask generation, morphological refinement, contour-to-polygon conversion, and annotated visual output.

### Hour 3: Road Segmentation, Vehicle Detection & Geospatial Engine
- Implement `src/object_detection/road_segmentation.py` (spectral/structural linear extraction).
- Implement `src/object_detection/vehicle_detection.py` (high-res object detector + resolution feasibility check).
- Implement `src/geospatial/geo_converter.py` (Rasterio/Affine coordinate transforms, CRS handling, GeoJSON export).

### Hour 4: Integration Pipeline & Automated Testing
- Create `src/object_detection/pipeline.py` and CLI `run_part2_pipeline.py`.
- Write automated tests in `tests/test_part2.py`.
- Run pipeline and generate actual output artifacts in `outputs/part2_object_detection/`.
- Compile `TEST_REPORT.md`.

### Hour 5: Documentation, Viva Prep & Git Integration
- Write `MY_MODULE_EXPLANATION.md`, `HOW_TO_RUN.md`, `VIVA_QUESTIONS.md`.
- Review git status and staging.
- Commit changes cleanly and verify push to GitHub remote.
- Provide student presentation walkthrough and 2-minute pitch script.
