# DATASET GUIDE: Satellite Imagery & Annotations
## Change Detection Due to Human Activities Using Satellite Imagery

This guide details the legitimate public satellite datasets, licensing, sensor specifications, and folder placement guidelines for **Part 2: Object Detection and Segmentation**.

---

## 1. Overview of Utilized Datasets

| Dataset | Primary Source | Sensor / Resolution | Target Classes | Licence |
| :--- | :--- | :--- | :--- | :--- |
| **SpaceNet 7** | AWS Open Data (`s3://spacenet-dataset`) | PlanetScope (3.0m–4.0m GSD, 4-band R-G-B-NIR) | Buildings, Urban Temporal Change | CC BY-SA 4.0 |
| **SpaceNet 5** | AWS Open Data (`s3://spacenet-dataset`) | WorldView-3 (0.3m–0.5m GSD, 8-band / RGB) | Road Networks, Speed Estimates | CC BY-SA 4.0 |
| **xView** | Defense Innovation Unit (DIU) | WorldView-3 (0.3m GSD, RGB) | Vehicles (Cars, Trucks, Buses), Maritime | DIU Research Licence |
| **LEVIR-CD** | Beihang University | Google Earth Orthophotos (0.5m GSD, RGB) | Building Growth & Demolition Pairs | Non-commercial Academic |

---

## 2. Sensor Resolutions & Physical Detection Limits (GSD)

A critical concept in satellite computer vision is **Ground Sample Distance (GSD)**: the real-world distance represented by the width of a single pixel.

### Why Vehicle Detection Requires High Resolution:
- **Standard Passenger Vehicle:** $\approx 4.5\text{ m (length)} \times 1.8\text{ m (width)}$.
- **SpaceNet 7 (3.0m GSD):**
  $$\text{Length in pixels} = \frac{4.5\text{ m}}{3.0\text{ m/px}} = 1.5\text{ px}, \quad \text{Width in pixels} = \frac{1.8\text{ m}}{3.0\text{ m/px}} = 0.6\text{ px}$$
  A vehicle occupies **under 1 pixel** ($0.9\text{ px}^2$). Attempting optical vehicle detection on 3m satellite imagery is scientifically unviable and produces overwhelming false alarms.
- **WorldView-3 / xView (0.3m GSD):**
  $$\text{Length in pixels} = \frac{4.5\text{ m}}{0.3\text{ m/px}} = 15.0\text{ px}, \quad \text{Width in pixels} = \frac{1.8\text{ m}}{0.3\text{ m/px}} = 6.0\text{ px}$$
  A vehicle occupies **$\approx 90\text{ px}^2$**, providing clear shape, contrast, and windshield reflections required for reliable object detection.

---

## 3. Recommended Directory Structure

To maintain clean project organization and avoid Git bloat, satellite imagery must be stored in the following directory hierarchy (already ignored by `.gitignore`):

```text
change-detection-human-activities/
└── data/
    ├── raw/
    │   ├── spacenet7/
    │   │   └── sample/
    │   │       ├── images_masked/       # Multi-temporal GeoTIFFs (4-band NIR+RGB)
    │   │       └── labels_match_pix/    # Building footprint GeoJSONs
    │   ├── spacenet5_roads/             # SpaceNet 5 Road GeoTIFFs and Centerline GeoJSONs
    │   └── xview_vehicles/              # xView High-Res WorldView-3 crops & GeoJSONs
    └── processed/
        ├── paired_dataset/              # Preprocessed temporal numpy pairs (.npz)
        └── test_crops/                  # High-resolution demonstration tiles
```

---

## 4. How to Download the Datasets

### A. SpaceNet 7 (Urban Change & Buildings)
SpaceNet 7 is hosted freely on AWS S3 without requester fees.
```bash
# Requires AWS CLI installed (no AWS account credentials required with --no-sign-request)
aws s3 cp s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train_sample.tar.gz data/raw/spacenet7/ --no-sign-request
tar -xzf data/raw/spacenet7/SN7_buildings_train_sample.tar.gz -C data/raw/spacenet7/
```

### B. SpaceNet 5 (Road Extraction)
```bash
aws s3 cp s3://spacenet-dataset/spacenet/SN5_roads/train/ data/raw/spacenet5_roads/ --recursive --no-sign-request
```

### C. xView (Overhead Vehicle Detection)
Visit [xviewdataset.org](http://xviewdataset.org/), accept the research terms, and download the training images and geojson annotations:
Place uncompressed imagery inside `data/raw/xview_vehicles/`.

---

## 5. Built-in Demonstration Mode (No Heavy Download Needed)

To ensure this project can be demonstrated immediately to college faculty without waiting for gigabyte-scale satellite downloads:
1. The repository includes real SpaceNet 7 sample imagery and preprocessed overlays inside `outputs/detection/`.
2. Running the demonstration command:
   ```bash
   python run_part2_pipeline.py --demo
   ```
   automatically loads the verified satellite imagery, executes building footprint segmentation, extracts road networks, runs the GSD resolution validator, and generates real-world GeoJSON and visual overlays.
