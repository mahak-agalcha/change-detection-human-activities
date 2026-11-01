# Dataset Summary: Remote Sensing Human Activity Change Detection

## 1. Project and Benchmark Overview

- **Project Title:** Change Detection Due to Human Activities Using Satellite/Aerial Images
- **Primary Benchmark Dataset:** LEVIR-CD (Large-scale Evolutionary Land-change Remote-sensing Change Detection)
- **Secondary Benchmark Dataset:** OSCD (Onera Satellite Change Detection)
- **Target Detection Task:** Anthropogenic infrastructure changes (building construction, housing expansion, earthworks)
- **Target Resolution:** $0.5$ meters/pixel (Google Earth VHR) and $10$ meters/pixel (Sentinel-2 multispectral)

## 2. Quantitative Dataset Metrics

- **Total Cataloged Sample Pairs:** 6
- **Total Images Ingested:** 12 (Time 1 Pre-Change and Time 2 Post-Change)
- **Total Ground-Truth Masks:** 6
- **Splits Breakdown:**
  - Training Split: `3` pairs
  - Validation Split: `1` pairs
  - Testing Split: `2` pairs
## 3. Storage and Footprint Breakdown

| Directory | Content Description | Disk Space Usage |
| :--- | :--- | :--- |
| `data/raw/` | Original compressed archives (`.zip`) | 1.75 MB |
| `data/extracted/` | Unpacked archive contents | 10.83 MB |
| `data/before/` + `data/after/` + `data/masks/` | Standardized paired datasets | 2.57 MB |
| **Total Pipeline Data Footprint** | Complete local data repository | **15.15 MB** |

## 4. Class Balance and Anthropogenic Pixel Distribution

- **Total Scene Pixels Evaluated:** 393,216
- **Changed Pixels (Positive Human Activity):** 66,744 (16.97%)
- **Unchanged Background Pixels:** 326,472 (83.03%)
- **Class Balance Ratio:** 1 : 4.9 (Reflects realistic sparse urban change distribution)

## 5. Cataloged Human Activity Typology

| Human Activity Category | Sample Pairs Count | Percentage of Dataset |
| :--- | :--- | :--- |
| Suburban Sprawl / Land Modification | 1 | 16.7% |
| Negative Control (Unchanged Urban Sector) | 1 | 16.7% |
| Residential Housing Expansion | 1 | 16.7% |
| Commercial / Industrial Building Construction | 1 | 16.7% |
| Transport Infrastructure / Road Construction | 1 | 16.7% |
| Building Construction & Expansion | 1 | 16.7% |
