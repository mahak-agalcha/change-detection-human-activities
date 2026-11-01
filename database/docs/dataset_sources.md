# Remote Sensing Change Detection Dataset Sources and Comparative Analysis

## 1. Executive Context

For the B.Tech Major Project **"Change Detection Due to Human Activities Using Satellite/Aerial Images"**, the dataset selection must rigorously isolate **human-induced changes** (anthropogenic modifications such as building construction, urban sprawl, industrial development, and road earthworks) while rejecting or distinguishing **natural environmental variations** (seasonal vegetation phenology, solar illumination angle shifts, and tidal water variations).

This document details the comparative evaluation of publicly accessible remote sensing change detection datasets, justifying the selection of **LEVIR-CD** as the primary high-resolution benchmark and **OSCD** as the secondary multispectral satellite benchmark.

---

## 2. Comparative Evaluation Matrix

| Dataset | Sensor / Platform | Spatial Resolution | Image Dimensions | Total Pairs | Ground Truth Type | Human Activity Specificity | Upstream Host / Access | Academic License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LEVIR-CD** *(Primary)* | Google Earth (Aerial / VHR) | **0.5 m/pixel** | 1024 × 1024 | 637 pairs (31,333 instances) | Pixel-level binary mask (0/255) | **High**: Building appearance/expansion (Human construction) | Beihang Univ. / Hugging Face / Google Drive | Non-commercial research |
| **OSCD** *(Secondary)* | Sentinel-2 (Multispectral) | **10m, 20m, 60m** | Varied (e.g., 600×600) | 24 cities (14 train, 10 test) | Pixel-level binary mask (0/255) | **High**: Urban expansion, roadworks, earthworks | IEEE DataPort / IMT Cloud Servers | CC BY-NC-SA 4.0 / Open Access |
| **LEVIR-CD+** | Google Earth | 0.5 m/pixel | 1024 × 1024 | 985 pairs | Binary mask | **High**: Building appearance and disappearance | S2Looking / Google Drive | Academic research |
| **WHU-CD** | Aerial Imagery (Christchurch) | 0.2 m/pixel | 32,507 × 15,354 (tiled) | 1 large area (pre/post earthquake) | Binary mask | **Medium**: Building reconstruction post-earthquake | Wuhan University | Academic research |
| **SYSU-CD** | Aerial Imagery (Hong Kong) | 0.5 m/pixel | 256 × 256 | 20,000 pairs | Binary mask | **Medium-High**: Urban construction, road expansion, vegetation loss | Sun Yat-sen Univ. / GitHub | Academic research |
| **S2Looking** | GaoFen, SuperView | 0.5 – 0.8 m/pixel | 1024 × 1024 | 5,000 pairs | Binary mask | **High**: Building change detection with off-nadir side angles | Wuhan University / IEEE GRSS | Non-commercial research |
| **DSIFN-CD** | SPOT-6, GF-2 | 2.0 m/pixel | 512 × 512 | 3,914 pairs | Binary mask | **Mixed**: Urban expansion and land-cover change | Nanjing Univ. / GitHub | Academic research |

---

## 3. Selected Datasets: Detailed Rationale & Specifications

### 3.1 Primary Dataset: LEVIR-CD
* **Full Title:** Large-scale Evolutionary Land-change Remote-sensing Change Detection Dataset
* **Authors:** Hao Chen, Zhenwei Shi (Learning, Vision and Remote Sensing Lab, Beihang University)
* **Publication:** *Remote Sensing* (MDPI), 2020. DOI: `10.3390/rs12101662`
* **Official Website:** [justchenhao.github.io/LEVIR](https://justchenhao.github.io/LEVIR/)
* **Hugging Face Mirror:** [`satellite-image-deep-learning/LEVIR-CD`](https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD) and [`snchen1230/LEVIRCD`](https://huggingface.co/datasets/snchen1230/LEVIRCD)

#### Suitability for Major Project:
1. **Explicit Anthropogenic Focus:** Unlike general land-cover datasets where 80% of changes are seasonal leaf fall or crop rotations, LEVIR-CD strictly annotates **building additions and structural emergence**, representing direct human urban activity.
2. **Ultra-High Spatial Resolution ($0.5$ m/pixel):** Individual structures, residential footprints, villa complexes, and commercial high-rises are crisply resolved, facilitating Part 2 (Object Detection and Segmentation) of the major project contract.
3. **Multi-temporal Baseline:** Image pairs span a temporal baseline of **5 to 14 years** across 20 distinct regions in Texas, USA (Austin, San Antonio, Houston, Fort Worth, etc.), providing substantial architectural diversity.
4. **Clean Ground Truth:** 31,333 individual building change instances are manually annotated and double-verified by remote sensing specialists.

#### Known Limitations:
* **Binary Nature:** Focuses specifically on building changes (structural appearance/expansion); does not differentiate between industrial, residential, or commercial buildings in the raw binary mask.
* **Storage Footprint:** The complete full-resolution archive is approximately 5 to 6 GB, requiring chunked or patch-based processing for standard student hardware.

---

### 3.2 Secondary Dataset: OSCD
* **Full Title:** Onera Satellite Change Detection Dataset
* **Authors:** Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch (ONERA - The French Aerospace Lab), Yann Gousseau (LTCI, Télécom Paris)
* **Publication:** *IEEE International Geoscience and Remote Sensing Symposium (IGARSS)*, 2018. DOI: `10.1109/IGARSS.2018.8518015`
* **Official Website:** [rcdaudt.github.io/oscd](https://rcdaudt.github.io/oscd/)
* **Official Repository:** Hosted on IEEE DataPort and Institut Mines-Télécom (IMT) Open Servers

#### Direct Access Endpoints & Integrity Hashes:
* **Train Labels:**
  * URL: `https://partage.mines-telecom.fr/index.php/s/2D6n03k58ygBSpu/download`
  * Archive: `Onera Satellite Change Detection dataset - Train Labels.zip` (~1.5 MB)
  * MD5 Checksum: `4d2965af8170c705ebad3d6ee71b6990`
* **Test Labels:**
  * URL: `https://partage.imt.fr/index.php/s/gpStKn4Mpgfnr63/download`
  * Archive: `Onera Satellite Change Detection dataset - Test Labels.zip` (~1.1 MB)
  * MD5 Checksum: `8177d437793c522653c442aa4e66c617`
* **Images:**
  * URL: `https://partage.imt.fr/index.php/s/gKRaWgRnLMfwMGo/download`
  * Archive: `Onera Satellite Change Detection dataset - Images.zip` (~700 MB)
  * MD5 Checksum: `c50d4a2941da64e03a47ac4dec63d915`

#### Suitability for Major Project:
1. **Satellite Multispectral Diversity:** Captured by Sentinel-2 with 13 spectral bands, permitting advanced band arithmetic (e.g., NDVI for vegetation vs NDBI for built-up surfaces).
2. **Global Geographic Representation:** Covers 24 cities worldwide across Europe, the Americas, the Middle East, and Asia (e.g., Paris, Dubai, Las Vegas, Mumbai, Rio de Janeiro).
3. **Open Access Infrastructure:** Direct HTTPS download links are stable, unthrottled, and do not require user authentication or API keys.

#### Known Limitations:
* **Moderate Resolution:** At 10m to 60m per pixel, individual small single-family houses cannot be distinguished; changes represent macroscopic urban expansion, earthworks, and arterial road construction.

---

## 4. Academic Integrity & Non-Fabrication Notice

In accordance with strict scientific guidelines:
- We do **not** claim all detected differences in remote sensing images are human-caused; natural changes (vegetation phenology, atmospheric hazing, seasonal cloud cover) occur simultaneously.
- Datasets were selected specifically because their ground truth annotations isolate **anthropogenic modifications** (buildings and civil engineering projects) from natural cyclic phenomena.
