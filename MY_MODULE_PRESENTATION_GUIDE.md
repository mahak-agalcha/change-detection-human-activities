# PRESENTATION & VIVA GUIDE: PART 2
## Object Detection and Segmentation in Satellite Imagery
**Student:** Mahak Agalcha (2310DMBCSE15387)  
**Project Guide:** Dr. Sandeep Kumar Jain  
**Module:** Part 2 - Object Detection and Segmentation

---

## 1. 2-Minute Quick Presentation Pitch

Use this exact script when introducing your contribution to your college faculty:

> *"Good morning respected Guide and Faculty members. In our major project, 'Change Detection Due to Human Activities Using Satellite Imagery', my assigned module is **Part 2: Object Detection and Segmentation**.*
>
> *While Part 1 of our project detects temporal pixel changes, raw pixel change alone does not explain what actually changed on the ground. A change could be a new building, a newly paved road, or merely seasonal vegetation drying up. My module gives semantic and spatial understanding to these changes by detecting, segmenting, and vectorizing specific human infrastructure—primarily buildings and road networks.*
>
> *I designed and implemented a modular Python architecture using OpenCV, Shapely, and Rasterio. My module takes multi-band satellite imagery, computes spectral indices like NDVI to filter out vegetation, applies local contrast enhancement and morphological structural kernels, and uses the Ramer-Douglas-Peucker algorithm to vectorize building footprints into clean geometric polygons.*
>
> *To maintain scientific integrity, I integrated a Ground Sample Distance (GSD) analyzer. For medium-resolution imagery like SpaceNet 7 PlanetScope at 3.0 meters per pixel, passenger vehicles occupy less than one pixel, so vehicle detection is suppressed to prevent false alarms. However, for Very High Resolution imagery below 0.6 meters, vehicle detection is fully supported.*
>
> *Finally, through an Affine transformation engine, my module converts image pixels into real-world geographic coordinates and exports standardized GeoJSON FeatureCollections that feed directly into our team's web map dashboard. My implementation has been thoroughly verified with 14 automated unit tests, all passing with zero errors."*

---

## 2. Detailed Breakdown of My Contribution

When faculty ask *"What exactly was your individual contribution?"*, highlight these 6 concrete achievements:

1. **Building Footprint Segmentation Pipeline:**
   - Implemented Contrast Limited Adaptive Histogram Equalization (CLAHE) to boost rooftop contrast.
   - Designed morphological opening using square structuring elements to separate closely adjacent buildings.
   - Enforced geometric filtering: `Solidity > 0.52` (filters out irregular natural noise) and `Aspect Ratio < 3.2` (filters out elongated roads).
   - Applied the Ramer-Douglas-Peucker (RDP) algorithm (`cv2.approxPolyDP`) to convert pixelated contour masks into crisp architectural polygons.
   - Successfully extracted **264 building footprints** from the SpaceNet 7 test image.

2. **Road Network & Corridor Extraction Module:**
   - Applied multi-directional linear structuring filters ($0^\circ, 30^\circ, 45^\circ, \dots, 150^\circ$) to enhance continuous linear tracks.
   - Masked out segmented buildings to prevent building rooftops from being confused with roads.
   - Filtered road segments by requiring `Aspect Ratio >= 2.0` and `Length >= 25 px`.
   - Extracted **130 road corridors** from the satellite test scene.

3. **Spatial Resolution (GSD) Feasibility Engine & Vehicle Detector:**
   - Formulated the physical resolution relationship: a standard car is $\approx 4.5\text{ m} \times 1.8\text{ m}$. At $3.0\text{ m}$ GSD, a vehicle spans only $0.9\text{ px}^2$, making optical vehicle detection physically unreliable.
   - Suppressed detections on coarse imagery while supporting high-resolution detection ($\le 0.6\text{ m}$) via morphological top-hat filtering.

4. **Geospatial Vectorization Engine (GIS):**
   - Implemented 2D Affine coordinate transformation:
     $$x_{\text{geo}} = c + a \cdot \text{col} + b \cdot \text{row}, \quad y_{\text{geo}} = f + d \cdot \text{col} + e \cdot \text{row}$$
   - Transformed pixel coordinates into WGS84 $(\text{lon}, \text{lat})$.
   - Used Shapely to validate polygon topology, repair self-intersections (`make_valid`), calculate ground area in $\text{m}^2$, and export standard **GeoJSON FeatureCollections**.

5. **Automated Testing Suite (PyTest):**
   - Wrote 14 automated unit and integration tests in `tests/test_part2.py`.
   - Achieved a **100% pass rate** in 1.47 seconds.

6. **Inter-Module Contract & Integration:**
   - Created `run_inference(old_path, new_path)` in `run_part2_pipeline.py`, providing a direct drop-in function for Part 4's `backend/app/services/inference.py`.

---

## 3. Explanation of Datasets Used

| Dataset | Sensor / Resolution | Why It Was Selected | Role in Module |
| :--- | :--- | :--- | :--- |
| **SpaceNet 7** | PlanetScope (3.0m–4.0m GSD, 4-band NIR+RGB) | Open access on AWS S3, provides monthly multi-temporal imagery tracking urban growth over 24 months. | Building footprint segmentation and change candidate analysis. |
| **SpaceNet 5** | WorldView-3 (0.3m–0.5m GSD, RGB) | High-resolution road network benchmark with road centerlines and speed metadata. | Road corridor extraction reference. |
| **xView** | WorldView-3 (0.3m GSD, RGB) | Defense Innovation Unit benchmark containing 60 overhead object classes. | Vehicle detection benchmark and GSD cutoff reference. |
| **LEVIR-CD** | Google Earth (0.5m GSD, RGB) | Pre-aligned high-resolution before/after image pairs for building additions/demolitions. | Temporal change pair reference. |

---

## 4. Why We Chose This Model & Algorithmic Approach

Faculty may ask: *"Why didn't you train a deep learning Mask R-CNN or YOLO from scratch?"*

**Your Answer:**
> *"Training a large deep learning segmentation model like Mask R-CNN or YOLOv8 from scratch requires tens of gigabytes of labeled satellite data, specialized high-end GPUs, and multiple days of training time. Furthermore, deep neural networks function as 'black boxes' and are prone to overfitting on limited training scenes.*
>
> *Instead, I engineered a hybrid Remote Sensing and Computer Vision pipeline combining spectral indices (NDVI), adaptive contrast enhancement (CLAHE), directional linear kernels, morphological opening, and RDP polygon approximation. This approach:*
> 1. *Is fully deterministic, explainable, and transparent.*
> 2. *Executes in seconds on standard hardware without requiring GPU clusters.*
> 3. *Enforces physical geometric constraints (solidity, aspect ratio, polygon closure).*
> 4. *Can easily be swapped or augmented with a pre-trained deep learning checkpoint via our modular architecture."*

---

## 5. Crucial Scientific Requirement: Causation vs. Correlation

Faculty love to test whether students understand the limits of their system:

> **Faculty Question:** *"Does your model prove that a detected change was caused by human activity?"*  
>
> **Your Answer:**  
> *"No, sir/ma'am. It is a fundamental principle of remote sensing that **two satellite images alone show correlation and visual change, not causation**.  
> For instance:
> - A detected building might be an authorized residential construction, or an illegal encroachment, or a disaster relief shelter.
> - A cleared patch of land could be human road excavation, or it could be a natural landslide or wildfire aftermath.
> 
> My module strictly focuses on **objective physical observation**: detecting and vectorizing buildings and roads. Attributing 'cause' requires external context (land zoning laws, construction permits, historical records), which is the dedicated responsibility of **Part 3: Context Analysis**."*

---

## 6. How Part 2 Connects with Parts 1, 3, and 4

```text
Part 1 (Data Pipeline)
      │
      │ Passes: (Image_t1, Image_t2, Binary Change Mask, Candidate Regions)
      ▼
Part 2 (Object Detection & Segmentation) [MY MODULE]
      │
      │ Produces: (Classified Objects, BBoxes, Polygons, Area in m², GeoJSON)
      ▼
Part 3 (Context Analysis & Database)
      │
      │ Synthesizes: Cross-references detected objects with zoning/cadastral data
      │              to determine cause: Human / Natural / Uncertain
      ▼
Part 4 (Backend & Frontend)
      │
      │ Renders: Visualizes GeoJSON polygons on Leaflet map & displays metrics in UI
```

---

## 7. Known Limitations

1. **Spatial Resolution Limits (GSD):** Medium-resolution sensors (3.0m) cannot reliably detect passenger vehicles (~1.5 x 0.6 pixels). Very High Resolution imagery ($\le 0.6\text{m}$) is mandatory for reliable vehicle detection.
2. **Solar Shadowing:** Tall buildings cast shadows that can slightly occlude adjacent roads. Morphological dilation and aspect ratio thresholds help decouple these regions.
3. **Nadir vs. Off-Nadir Tilt:** Satellite imagery captured at oblique angles causes buildings to appear tilted (building lean), exposing their facades. Our pipeline focuses on high-reflectance rooftop planes to minimize footprint displacement.
