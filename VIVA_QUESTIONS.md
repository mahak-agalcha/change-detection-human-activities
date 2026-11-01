# VIVA QUESTIONS & ANSWERS: PART 2
## Object Detection and Segmentation in Satellite Imagery
**Student:** Mahak Agalcha | **Major Project Final Viva Prep**

---

### Q1: What is the primary objective of your assigned module (Part 2)?
**Answer:**  
My module, Part 2, is responsible for Object Detection and Segmentation. While Part 1 identifies raw pixel changes over time, my module gives semantic meaning to those changes by detecting, segmenting, and vectorizing specific human infrastructure—primarily buildings and road networks—and converting them from pixel coordinates into real-world geographic vector coordinates (GeoJSON).

---

### Q2: What is the difference between Object Detection and Semantic Segmentation?
**Answer:**  
- **Object Detection** predicts a rectangular bounding box `(x, y, width, height)` and class label for each object.
- **Semantic Segmentation** performs pixel-level classification, labeling every individual pixel that belongs to an object.
In satellite imagery, segmentation is preferred for buildings and roads because real-world structures are irregularly shaped and bounding boxes include empty background areas.

---

### Q3: What is the difference between Semantic Segmentation and Instance Segmentation?
**Answer:**  
- **Semantic Segmentation** labels all pixels of a class with the same category (e.g., all building pixels are labeled "building", without distinguishing where one building ends and another begins).
- **Instance Segmentation** detects and delineates each individual instance of an object separately (e.g., Building #1, Building #2, Building #3).

---

### Q4: Why can't we detect passenger vehicles reliably on standard satellite images like SpaceNet 7 or Sentinel-2?
**Answer:**  
This is governed by **Ground Sample Distance (GSD)**.
- A typical passenger car is approximately $4.5\text{ m} \times 1.8\text{ m}$.
- SpaceNet 7 PlanetScope imagery has a GSD of $\approx 3.0\text{ meters per pixel}$.
- Therefore, a car occupies only $1.5 \times 0.6 \text{ pixels}$, which is less than 1 single pixel ($0.9\text{ px}^2$).
Under the Nyquist-Shannon sampling theorem, an object must span multiple pixels to be distinguishable from noise. Detecting vehicles requires Very High Resolution (VHR) sensors with GSD $\le 0.5\text{ m}$ (like WorldView-3 or xView). On 3m imagery, attempting vehicle detection produces massive false positives.

---

### Q5: How does your pipeline extract building footprints?
**Answer:**  
My pipeline uses a multi-stage approach:
1. **Local Contrast Enhancement:** Uses Contrast Limited Adaptive Histogram Equalization (CLAHE) on the satellite image.
2. **Built-up Reflectance Thresholding:** Extracts high-reflectance urban rooftop candidate regions.
3. **Morphological Opening:** Disconnects thin bridges between adjacent buildings and road connections.
4. **Geometric Filtering:** Filters candidates by minimum area ($>20\text{ px}$), high solidity ($>0.52$), and compact aspect ratio ($<3.2$), since buildings are compact whereas roads are elongated.
5. **RDP Polygon Approximation:** Uses `cv2.approxPolyDP` to convert jagged pixel contours into clean vector polygons.

---

### Q6: What is the Ramer-Douglas-Peucker (RDP) algorithm and why did you use it?
**Answer:**  
The Ramer-Douglas-Peucker algorithm (`cv2.approxPolyDP`) reduces the number of vertices in a curve or contour while preserving its overall geometric shape based on an epsilon distance tolerance.
In our project, satellite raster masks have staircase-like, jagged pixel edges. RDP simplifies these jagged contours into crisp, straight-edged geometric polygons suitable for architectural building footprints.

---

### Q7: How does your road segmentation algorithm distinguish roads from buildings?
**Answer:**  
Roads and buildings have distinctly different geometric and spatial signatures:
1. **Aspect Ratio & Elongation:** Buildings are compact (aspect ratio $\approx 1:1$ to $2:1$), whereas roads are highly elongated corridors (aspect ratio $\ge 2.5:1$ and lengths $>25\text{ pixels}$).
2. **Directional Linear Filters:** Road extraction applies multi-angle line structuring kernels ($0^\circ, 30^\circ, 45^\circ, \dots, 150^\circ$) to detect continuous linear tracks.
3. **Mutual Exclusion:** Once building footprints are segmented, they are masked out so road algorithms do not misclassify building rooftops as roads.

---

### Q8: What is NDVI and how is it used in satellite image processing?
**Answer:**  
NDVI stands for **Normalized Difference Vegetation Index**:
$$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$
Chlorophyll in healthy green vegetation absorbs visible Red light and strongly reflects Near-Infrared (NIR) light. Therefore, high NDVI values ($>0.2$) indicate trees, grass, and crops. In our building segmentation pipeline, we use NDVI to suppress vegetation so that tree canopies and green fields are not misclassified as man-made urban structures.

---

### Q9: What is the difference between Raster and Vector data in GIS?
**Answer:**  
- **Raster Data:** A matrix or grid of cells/pixels containing numeric values (e.g., satellite imagery, digital elevation models). Formats: `.tif`, `.png`.
- **Vector Data:** Represents geographic features using discrete geometric primitives based on mathematical coordinates: Points, Lines, and Polygons. Formats: `.geojson`, `.shp`.
My module bridges the two: it takes raster satellite images as input and produces vector polygons as output.

---

### Q10: What is an Affine Transformation and how does your geospatial converter use it?
**Answer:**  
An Affine Transformation is a 2D linear geometric mapping that relates image pixel coordinates `(column, row)` to real-world map coordinates `(x, y)`:
$$x_{geo} = c + a \cdot col + b \cdot row$$
$$y_{geo} = f + d \cdot col + e \cdot row$$
Where:
- $a, e$ define the pixel resolution / scale in the X and Y directions.
- $b, d$ represent rotational or shearing factors.
- $c, f$ define the upper-left corner origin coordinates.
Our `GeospatialConverter` uses this transformation to convert pixel coordinates into standard WGS84 longitude and latitude.

---

### Q11: What is GeoTIFF and what metadata does it contain?
**Answer:**  
A GeoTIFF is a standard TIFF raster image format with embedded spatial metadata. It contains:
1. **Coordinate Reference System (CRS):** Defines the map projection (e.g., WGS84 / EPSG:4326 or UTM / EPSG:32643).
2. **Affine Transform Matrix:** Pixel-to-ground coordinates relationship.
3. **Bounding Box Extents:** Real-world minimum and maximum coordinates.
We use the `rasterio` Python library to read these properties.

---

### Q12: What is GeoJSON and why is it preferred for web mapping?
**Answer:**  
GeoJSON is an open, human-readable JSON format designed specifically for encoding geographic data structures. It uses standard WGS84 `[longitude, latitude]` coordinates and structures features into `Feature` and `FeatureCollection` dictionaries with geometric shapes and arbitrary property metadata.
It is the universal standard for web mapping libraries like Leaflet.js and Mapbox, making our Part 2 output directly renderable on Part 4's web dashboard.

---

### Q13: How does your module connect to Part 3 and Part 4?
**Answer:**  
We established a strict contract (`docs/api/output-contract.md` and `backend/app/services/inference.py`):
My module outputs a list of dictionaries containing:
- `object_type`: "building" | "road" | "vehicle"
- `confidence`: float between 0.0 and 1.0
- `geometry`: GeoJSON polygon dictionary with `[longitude, latitude]`
- `region_id`: Unique identifier linking with Part 1 change candidates.
Part 3 takes these classified objects to perform contextual evidence analysis, and Part 4 visualizes them on the Leaflet web map.

---

### Q14: Can two satellite images alone prove that a detected change was caused by human activity?
**Answer:**  
**No, they cannot.** Satellite images only reveal physical, visual alterations over time (correlation, not causation). For example, a cleared area could be due to authorized construction, illegal encroachment, or a natural wildfire aftermath.
Our system maintains scientific integrity: Part 2 only reports observed objects (`building`, `road`). Attributing causation requires corroborating contextual evidence (zoning permits, land tenure, historical ground records), which is handled in Part 3.

---

### Q15: What datasets did you study and evaluate for this module?
**Answer:**  
1. **SpaceNet 7:** Multi-temporal urban challenge with PlanetScope 4-band imagery (3.0m GSD) and monthly building footprints.
2. **SpaceNet 5:** High-resolution road network extraction dataset using WorldView-3 imagery (0.3m GSD) with speed and centerline metrics.
3. **xView:** DIU overhead object detection dataset with 60 classes (vehicles, maritime) at 0.3m GSD.
4. **LEVIR-CD:** Multi-temporal high-resolution change detection dataset for building additions and removals.

---

### Q16: Why did you implement automated unit tests using PyTest?
**Answer:**  
Automated tests ensure software reliability, prevent regressions, and verify that all edge cases (missing files, invalid image dimensions, coarse GSD suppression, GeoJSON polygon closure) work properly without manual intervention.
Our test suite contains 14 automated unit and integration tests with a 100% pass rate.

---

### Q17: What is the purpose of the `solidity` metric in your building filter?
**Answer:**  
$$\text{Solidity} = \frac{\text{Contour Area}}{\text{Convex Hull Area}}$$
Man-made buildings have solid, compact, rectangular geometry with minimal indentations, yielding high solidity ($\approx 0.70 - 0.95$). Irregular natural features like tree clusters or eroded terrain have jagged, concaved borders, giving low solidity ($<0.45$). We filter out non-building noise by enforcing `min_solidity = 0.52`.

---

### Q18: What is CLAHE and why did you use it instead of standard Histogram Equalization?
**Answer:**  
CLAHE stands for **Contrast Limited Adaptive Histogram Equalization**.
Standard histogram equalization operates globally on the entire image, which often over-amplifies noise in homogeneous areas (like sky or flat desert). CLAHE divides the image into small contextual tiles (e.g. $8 \times 8$), enhances local contrast within each tile, and clips extreme gradients to limit noise amplification. This is ideal for satellite imagery where urban structures have subtle rooftop contrast against the ground.

---

### Q19: What is the difference between Training and Inference in machine learning?
**Answer:**  
- **Training:** The resource-intensive offline phase where model parameters are optimized across thousands of annotated images using backpropagation and loss minimization (e.g. Adam optimizer).
- **Inference:** The operational phase where a trained model or deterministic pipeline runs on new input images to generate predictions. Our Part 2 pipeline is deployed in inference mode to process incoming satellite pairs.

---

### Q20: How do you handle images without embedded GeoTIFF coordinates?
**Answer:**  
When an input image is a standard PNG or JPEG lacking embedded spatial tags, `SatelliteImageLoader` falls back to generating a valid local affine reference grid based on the image pixel dimensions, rather than fabricating fake real-world GPS coordinates. This ensures that downstream vectorization and polygon geometries remain topologically valid and compatible with the GeoJSON specification.

---

### Q21: What is Shapely and how is it used in your code?
**Answer:**  
`Shapely` is a Python library for computational geometry based on GEOS. We use Shapely to:
1. Construct valid geometric `Polygon` objects from contour point lists.
2. Validate topological correctness (`poly.is_valid`).
3. Repair self-intersecting polygon rings using `make_valid` or `poly.buffer(0)`.
4. Export geometries directly to GeoJSON mapping dictionaries (`mapping(poly)`).

---

### Q22: What is the purpose of `.gitignore` in your repository?
**Answer:**  
`.gitignore` specifies intentionally untracked files that Git should ignore. In our computer vision and remote sensing project, it excludes virtual environments (`.venv/`), raw gigabyte-scale datasets (`data/raw/`), large model weight checkpoints (`*.pt`, `*.pth`), and OS cache files. This keeps our GitHub repository lightweight, clean, and professional.

---

### Q23: What are the main challenges in satellite object detection compared to standard smartphone images?
**Answer:**  
1. **Nadir View:** Objects are viewed strictly from top-down, lacking side profile features (e.g., cars look like small rectangles without visible wheels).
2. **Varying Resolutions (GSD):** Spatial scale varies drastically between satellites (0.3m vs 3m vs 10m vs 30m).
3. **Solar Shadows & Off-nadir Parallax:** Oblique sensor angles cause tall buildings to appear tilted, occluding streets and ground objects.
4. **Massive File Sizes:** Satellite scenes often span tens of thousands of pixels per side, requiring tiling or patch-based processing.

---

### Q24: How would you scale this module to process a massive satellite image of an entire city?
**Answer:**  
A full satellite scene can be $10,000 \times 10,000$ pixels (100 megapixels). We would use a **tiling / sliding window approach** (as demonstrated in Part 1's `predict_pair`):
1. Tile the image into overlapping patches (e.g. $512 \times 512$ with 64 px overlap).
2. Run segmentation on each tile independently.
3. Apply Non-Maximum Suppression (NMS) or morphological stitching along tile seams.
4. Convert all local tile coordinates to global geographic coordinates using the tile's geospatial offset.

---

### Q25: How do you summarize your individual contribution to the major project in 2 minutes?
**Answer:**  
*(See the 2-Minute Presentation Pitch in the project guide below)*:
"I developed Part 2: Object Detection and Segmentation. While Part 1 produces raw temporal change masks, my module gives semantic and spatial understanding to those changes by detecting building footprints, segmenting road corridors, and analyzing vehicle resolution constraints. I built a modular Python architecture using OpenCV, Shapely, and Rasterio that converts raw satellite imagery into verified GeoJSON vector polygons and annotated visual overlays. My module passed all 14 automated unit tests, complies with our team's API contract, and feeds directly into our web dashboard."
