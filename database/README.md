# Change Detection Due to Human Activities Using Satellite/Aerial Images
## Part 1: Dataset Collection, Ingestion, and Quality Assurance Pipeline

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE_OR_LICENSE_NOTES.md)
[![Dataset Health](https://img.shields.io/badge/Data%20Health-100%25-brightgreen.svg)](reports/validation_report.md)
[![Verification](https://img.shields.io/badge/QA%20Checkpoints-5%2F5%20Passed-success.svg)](reports/validation_report.md)

---

## 1. Project Background and Objective

This repository contains the verified dataset collection, automated extraction, structural pairing, and quality assurance framework for the 4th-year B.Tech Major Project: **"Change Detection Due to Human Activities Using Satellite/Aerial Images"** (Department of Computer Science & Engineering).

### 1.1 Academic Project Context
* **Project Title:** Change Detection Due to Human Activities Using Satellite/Aerial Images
* **Student Author / Data Pipeline Lead:** Palak Upadhyay (Roll No: `2310DMBCSE15401`)
* **Project Team:** Mahak Agalcha (`2310DMBCSE15387`), Krishna Kushwah (`2310DMBCSE15384`), Palak Upadhyay (`2310DMBCSE15401`), Shreeyashi Choukad (`2310DMBCSE1419`)
* **Faculty Supervisor / Guide:** Dr. Sandeep Kumar Jain

### 1.2 System Pipeline Integration
As formalized in the system contract, this repository satisfies **Part 1: Data Pipeline**:
```text
[ Multi-temporal Satellite / Aerial Images (Time 1 & Time 2) ]
                                |
                                v
               Part 1: Ingestion & Data Pipeline  <-- (This Repository)
                                |
                                v
                Standardized Bitemporal Dataset
           (data/before/, data/after/, data/masks/)
                                |
                                v
          Part 2: Change Detection & Object Segmentation
                                |
                                v
            Part 3: Context Analysis & Geospatial DB
             (Human / Natural / Uncertain Decision)
                                |
                                v
              Part 4: Web Application & Deployment
```

---

## 2. Understanding Human-Caused Change Detection

In remote sensing, surface changes between two acquisition dates $T_1$ and $T_2$ fall into two distinct categories:

1. **Anthropogenic (Human-Induced) Changes:**
   - **Building Construction & Expansion:** Conversion of bare land, cropland, or forest into residential estates, commercial warehouses, industrial parks, or urban infill structures.
   - **Transportation Infrastructure:** Road paving, highway additions, and railway corridors.
   - **Civil Earthworks & Ground Excavation:** Land clearing, quarrying, mining, and industrial leveling.
2. **Natural & Cyclic Variations (Non-Anthropogenic / Nuisance):**
   - **Phenological Variations:** Seasonal leaf budding, autumn foliage loss, vegetation greening after rainfall.
   - **Hydrological / Tidal Fluctuation:** Riverbank shifts, coastal tide levels, reservoir seasonal rise/fall.
   - **Atmospheric & Illumination Artifacts:** Sun angle disparities, differing shadow geometry, cloud haze.

**Core Scientific Premise:** An effective AI system must isolate *true anthropogenic infrastructure alterations* while remaining robust to cyclic natural illumination and phenological noise.

---

## 3. Dataset Selection & Benchmark Sources

To address this challenge rigorously, two complementary benchmarks were selected and integrated:

### 3.1 Primary Benchmark: LEVIR-CD
* **Official Source:** [LEVIR Lab, Beihang University](https://justchenhao.github.io/LEVIR/)
* **Reference Citation:** Chen, H., & Shi, Z. (2020). *A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection*. Remote Sensing, 12(10), 1662.
* **Sensor / Spatial Resolution:** Ultra-High-Resolution (VHR) Google Earth imagery at **$0.5\text{ m/pixel}$**.
* **Temporal Interval:** Bitemporal pairs spanning **5 to 14 years** across 20 distinct regions in Texas, USA.
* **Positive Class Definition:** Strictly human-induced building additions and architectural developments (over 31,333 annotated building instances across 637 full $1024 \times 1024$ image pairs).

### 3.2 Secondary Benchmark: OSCD (Onera Satellite Change Detection)
* **Official Source:** [Institut Mines-Télécom (IMT) & ONERA](https://rcdaudt.github.io/oscd/) / Hosted on IEEE DataPort.
* **Reference Citation:** Daudt, R. C., et al. (2018). *Urban Change Detection for Multispectral Earth Observation Using Convolutional Neural Networks*. IGARSS 2018.
* **Sensor / Spatial Resolution:** Sentinel-2 multispectral (13 spectral bands at $10\text{ m}$, $20\text{ m}$, $60\text{ m}$).
* **Geographical Distribution:** 24 global metropolitan areas (Paris, Las Vegas, Dubai, Mumbai, Rio de Janeiro, Hong Kong, etc.).
* **Target:** Urban land modification, new civil engineering works, and road expansion.

---

## 4. Repository Folder Structure

```text
human-activity-change-detection/
├── .gitignore                      # Git exclusion rules for large raw archives & temporary files
├── CITATION.cff                    # Standard Citation File Format (CFF v1.2.0)
├── LICENSE_OR_LICENSE_NOTES.md     # Upstream dataset usage terms and MIT repo license
├── README.md                       # Master project presentation and usage guide
├── requirements.txt                # Exact Python dependencies
│
├── data/
│   ├── raw/                        # Original compressed archives (.zip) and download manifests
│   ├── extracted/                  # Safely unpacked archives preserving upstream hierarchy
│   ├── before/                     # Standardized Time 1 pre-change imagery
│   ├── after/                      # Standardized Time 2 post-change imagery
│   ├── masks/                      # Binary ground-truth change masks (0=unchanged, 255=change)
│   └── processed/                  # Diagnostic 4-panel verification composites
│
├── scripts/
│   ├── download_dataset.py         # Automated downloader with SSL fallback and checksum validation
│   ├── extract_dataset.py          # Safe unpacker with Zip Slip mitigation and CRC32 checks
│   ├── organize_dataset.py         # Structural normalizer and composite visualizer
│   ├── validate_dataset.py         # 7-point QA verification and health score auditor
│   └── generate_inventory.py       # Cataloger generating CSV inventories and markdown summaries
│
├── reports/
│   ├── dataset_inventory.csv       # Granular row-by-row metadata inventory
│   ├── dataset_summary.md          # Statistical summary, splits, and storage footprint
│   ├── validation_report.md        # Comprehensive quality assurance audit report
│   └── validation_results.json     # Machine-readable QA metrics per sample
│
└── docs/
    ├── dataset_sources.md          # In-depth survey of remote sensing CD datasets
    └── collection_methodology.md   # Six-stage academic data engineering methodology
```

---

## 5. Software Requirements and Setup

### 5.1 Environment Prerequisites
- Python 3.10 or higher (tested with Python 3.14 on Windows 64-bit).
- Standard packages: `requests`, `pillow`, `numpy`, `pandas`, `tqdm`.

### 5.2 Installation
```bash
# Clone or navigate to the repository directory
cd database

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 6. Pipeline Execution Guide

Run each stage sequentially to reproduce the complete dataset pipeline:

### Step 1: Download the Dataset
```bash
# Option A: Ingest the verified representative benchmark suite (Fast, ~1.6 MB)
python scripts/download_dataset.py --dataset sample

# Option B: Download official OSCD benchmark labels and mirrors
python scripts/download_dataset.py --dataset oscd

# Option C: Ingest all available endpoints
python scripts/download_dataset.py --dataset all
```

### Step 2: Extract Raw Archives Safely
```bash
python scripts/extract_dataset.py
```

### Step 3: Organize and Generate Visual Diagnostics
```bash
python scripts/organize_dataset.py
```

### Step 4: Run Quality Assurance Validation
```bash
python scripts/validate_dataset.py
```

### Step 5: Generate Inventory and Statistical Reports
```bash
python scripts/generate_inventory.py
```

---

## 7. Actual Dataset Statistics (Computed from Real Files)

The statistics below are dynamically derived from actual files on disk via `scripts/generate_inventory.py`:

| Metric | Measured Value |
| :--- | :--- |
| **Cataloged Bitemporal Sample Pairs** | `6` complete triplets |
| **Total Images Ingested** | `12` images (6 Before, 6 After) |
| **Ground-Truth Change Masks** | `6` binary masks |
| **Image Resolution / Dimensions** | $256 \times 256$ pixels, 3-channel RGB (`PNG`) |
| **Ground Truth Mask Mode** | Grayscale / Binary (`L`), values $\in \{0, 255\}$ |
| **Total Evaluated Pixels** | `393,216` pixels |
| **Anthropogenic Change Pixels** | `66,744` pixels ($16.97\%$) |
| **Unchanged Background Pixels** | `326,472` pixels ($83.03\%$) |
| **Anthropogenic Imbalance Ratio** | $1 : 4.9$ (Realistic urban sparsity) |
| **Training Split** | `3` pairs ($50\%$) |
| **Validation Split** | `1` pair ($16.7\%$) |
| **Testing Split** | `2` pairs ($33.3\%$) |
| **Dataset Health Score** | **$100.0\%$** (All samples passed decodability & dimension matching) |

---

## 8. Data Limitations & Known Constraints

1. **Binary Semantic Scope:** Primary masks delineate *where* structural change occurred, but do not sub-classify into specific building types (e.g., residential vs. industrial) in the raw binary mask.
2. **Sentinel-2 Spatial Resolution:** OSCD multispectral imagery ($10\text{ m} - 60\text{ m/pixel}$) captures macroscopic urban expansion and earthworks but cannot resolve individual single-family house boundaries.
3. **Large-Scale Archive Storage:** The complete full-resolution LEVIR-CD and OSCD images require ~5.5 GB. As per GitHub best practices, large raw archives are excluded from Git commits via `.gitignore`. The included reproducible download scripts fetch them on demand.

---

## 9. Uploading to GitHub

To publish this verified repository to a new GitHub repository:

```bash
# 1. Initialize Git in the project root
git init -b main

# 2. Stage all verified scripts, reports, docs, and sample data
git add .

# 3. Verify that large archives are excluded by .gitignore
git status

# 4. Commit the initial verified data pipeline release
git commit -m "feat: complete human activity change detection dataset collection pipeline"

# 5. Link to your GitHub remote repository (replace with your personal GitHub URL)
# Example: git remote add origin https://github.com/palakupadhyay/human-activity-change-detection.git
git remote add origin <YOUR_GITHUB_REPO_URL>

# 6. Push to GitHub
git push -u origin main
```

---

## 10. Licensing and Attribution

- **Repository Tooling:** Licensed under the [MIT License](LICENSE_OR_LICENSE_NOTES.md).
- **LEVIR-CD Dataset:** Academic research and educational use only. Citation: Chen & Shi (2020), *Remote Sensing*.
- **OSCD Dataset:** Open Access under CC BY-NC-SA 4.0 / Open Data Commons. Citation: Daudt et al. (2018), *IEEE IGARSS*.
- See [LICENSE_OR_LICENSE_NOTES.md](LICENSE_OR_LICENSE_NOTES.md) for full terms.
