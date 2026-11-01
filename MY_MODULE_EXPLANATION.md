# MY MODULE EXPLANATION
## Part 2: Object Detection and Segmentation
**Student:** Mahak Agalcha | **Guide:** Dr. Sandeep Kumar Jain  
**Project:** Change Detection Due to Human Activities Using Satellite Imagery

---

## 1. What My Module Does

My module is **Part 2: Object Detection and Segmentation**.

In this major project, my role is to identify and delineate specific physical infrastructure in satellite imagery:
1. **Building Detection & Footprint Segmentation:** Locates individual buildings, extracts their polygonal rooftop boundaries, and calculates physical ground areas.
2. **Road Network & Corridor Extraction:** Identifies linear transportation corridors and road surfaces, separating them from compact structures.
3. **Vehicle Detection & Resolution Feasibility Analysis:** Evaluates whether satellite Ground Sample Distance (GSD) physically allows vehicle detection and detects vehicles on high-resolution imagery.
4. **Geospatial Processing (GIS):** Converts image pixel coordinates $(x, y)$ into real-world geographic coordinates $(\text{longitude}, \text{latitude})$ and exports standard **GeoJSON FeatureCollections** and annotated visual overlays.

---

## 2. Why My Module Is Needed in the Overall Project

The complete project has 4 modules:
- **Part 1:** Data Pipeline & Change Detection (Teammate)
- **Part 2:** Object Detection & Segmentation (**MY MODULE**)
- **Part 3:** Change Interpretation, Evidence & Geospatial Database (Teammate)
- **Part 4:** Backend, Frontend & Deployment (Teammate)

### The Core Problem:
Part 1 outputs a **binary change mask** (i.e., "these pixels changed between 2018 and 2019").
**However, raw pixel change alone does not tell you WHAT changed!**
- Did a forest get cut down?
- Was a new residential colony built?
- Was a new highway paved?
- Or did grass simply dry up in summer?

**My module answers "WHAT is there?":**
By segmenting buildings and roads from satellite imagery, my module transforms meaningless pixel blobs into classified, vectorized objects (`object_type: "building"`, `object_type: "road"`). This allows Part 3 to correlate the changed pixels with actual human structures and determine whether human activity caused the change.

---

## 3. How My Module Works (Step-by-Step Flow)

```text
Satellite Image (GeoTIFF / PNG / NPZ)
                 │
                 ▼
[Step 1] SatelliteImageLoader
  - Standardizes bands (RGB, extracts NIR if present)
  - Computes remote sensing indices (NDVI / NDWI)
  - Reads Georeferencing metadata (CRS, Affine transform)
                 │
                 ▼
[Step 2] BuildingSegmenter
  - Rooftop spectral & contrast thresholding (CLAHE)
  - Morphological opening to disconnect adjacent roofs
  - Geometric filtering (Solidity > 0.52, Aspect Ratio < 3.2)
  - Polygon approximation (Ramer-Douglas-Peucker algorithm)
                 │
                 ▼
[Step 3] RoadSegmenter
  - Multi-angle directional linear filtering
  - Elongation analysis (Aspect Ratio >= 2.0, Length >= 25 px)
  - Masks out buildings to prevent cross-class confusion
                 │
                 ▼
[Step 4] VehicleDetector
  - Ground Sample Distance (GSD) validation
  - If GSD > 0.6m: Suppresses detections to prevent false alarms
  - If VHR (<=0.6m): Detects vehicle contrast blobs on road corridors
                 │
                 ▼
[Step 5] GeospatialConverter
  - Affine transformation: (col, row) -> (longitude, latitude)
  - Shapely polygon topology validation (is_valid, buffer(0))
  - Calculates ground area in square meters (m²)
  - Exports standard GeoJSON FeatureCollection
                 │
                 ▼
Output Artifacts (GeoJSON, CSV, Composite Overlays, Backend Contract JSON)
```

---

## 4. Fundamental Computer Vision & GIS Concepts (Viva Essentials)

### A. Object Detection vs. Segmentation
- **Object Detection:** Identifies *where* an object is by placing a rectangular bounding box `(x, y, width, height)` around it. It does not know the exact shape.
- **Segmentation:** Classifies every individual pixel belonging to the object, revealing its exact organic or polygonal boundary.

### B. Semantic Segmentation vs. Instance Segmentation
- **Semantic Segmentation:** Labels pixels by class (all building pixels are colored cyan, but adjacent buildings are treated as one continuous class).
- **Instance Segmentation:** Distinguishes individual object instances (Building #1 is separate from Building #2, even if they touch).

### C. Bounding Box vs. Mask
- **Bounding Box:** A 4-parameter rectangle enclosing the object limits. Fast, but includes empty background corners.
- **Mask:** A 2D binary matrix where `1` represents object pixels and `0` represents background.

### D. Raster Data vs. Vector Data
- **Raster:** A regular grid of pixels with numeric values (e.g., satellite imagery, `.tif`, `.png`).
- **Vector:** Mathematical geometries defined by coordinate points: Points, Lines (Road centerlines), and Polygons (Building footprints, `.geojson`, `.shp`).

### E. GeoTIFF vs. GeoJSON
- **GeoTIFF:** A raster image format containing embedded geospatial metadata: Coordinate Reference System (CRS) and Affine georeferencing transform.
- **GeoJSON:** A lightweight, human-readable JSON format for encoding geographic vector data (points, lines, polygons) using WGS84 `[longitude, latitude]` pairs.

### F. Training vs. Inference
- **Training:** The offline optimization process where model parameters/weights are updated using annotated ground-truth datasets.
- **Inference:** Running the trained or configured pipeline on new, unseen satellite images to produce predictions.

---

## 5. Technology Choices and Justification

| Technology | Purpose | Why Chosen |
| :--- | :--- | :--- |
| **Python 3.11** | Core Language | Stable, standard environment matching the project's Docker container. |
| **OpenCV (`cv2`)** | Image Processing & Filtering | High-performance C++ backend for morphological filtering, contours, and CLAHE. |
| **Shapely** | Vector Geometry | Industry standard for geospatial polygon creation, validation, and area computation. |
| **Rasterio / Affine**| Geospatial Raster I/O | Standard library for reading GeoTIFF metadata and computing affine coordinate transforms. |
| **PyTest** | Automated Testing | Industry standard unit testing framework ensuring 100% test pass rate. |

---

## 6. Crucial Scientific & Ethical Requirement

> **Causation vs. Correlation:**  
> Our system detects physical objects and multi-temporal changes from satellite images. However, **satellite images alone cannot prove the intent or root cause of change**.  
> For example:
> - A cleared patch of land might be illegal deforestation, or it might be authorized government road construction, or wildfire aftermath.
> 
> Therefore, my module strictly outputs observed physical objects (`building`, `road`, `vehicle`) and their geometric footprints. The task of synthesizing contextual evidence (land ownership, zoning laws, news reports) is delegated to **Part 3: Context Analysis**. We never falsely claim that every observed change is definitively human-caused without corroborated evidence.

---

## 7. Known Limitations

1. **Resolution Constraint (GSD):** Standard satellite imagery (PlanetScope @ 3.0m) cannot reliably detect passenger vehicles (4.5m x 1.8m), as they occupy less than 1 pixel. Vehicle detection is only physically feasible on Very High Resolution (VHR $\le 0.5$m) sensors.
2. **Solar Shadow Parallax:** High sun angles create building shadows that can merge with adjacent dark roads. Morphological opening and aspect ratio filtering help separate these.
