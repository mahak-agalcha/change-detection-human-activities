# Licensing and Attribution Notes

This repository contains data ingestion tooling, verification pipelines, and cataloging utilities developed for the B.Tech Major Project: **"Change Detection Due to Human Activities Using Satellite/Aerial Images"**.

A clear distinction is maintained between the repository code and the upstream datasets.

---

## 1. Repository Code License

The source code, scripts (`scripts/`), validation logic, and documentation tools developed in this repository are distributed under the **MIT License**.

```text
MIT License

Copyright (c) 2026 Major Project Team (Mahak Agalcha, Krishna Kushwah, Palak Upadhyay, Shreeyashi Choukad)
Guide: Dr. Sandeep Kumar Jain

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. Upstream Dataset Licenses & Usage Policies

The actual satellite and aerial images processed by this repository belong to their respective original publishers and data providers. Users downloading data through these scripts are bound by the upstream licenses:

### A. LEVIR-CD (Large-scale Evolutionary Land-change Remote-sensing Change Detection)
* **Creators:** Hao Chen and Zhenwei Shi, Learning, Vision and Remote Sensing Laboratory (LEVIR Lab), School of Astronautics, Beihang University, Beijing, China.
* **Usage Policy:** Strictly for non-commercial academic research and educational evaluation. Any redistribution or commercial use requires direct written authorization from the LEVIR Lab authors.
* **Imagery Attribution:** Bitemporal high-resolution tiles originated from Google Earth. Use of underlying imagery is subject to Google Earth Terms of Service and applicable fair-use academic research provisions.
* **Mandatory Citation:**
  > Chen, H., & Shi, Z. (2020). *A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection*. Remote Sensing, 12(10), 1662. https://doi.org/10.3390/rs12101662

### B. OSCD (Onera Satellite Change Detection)
* **Creators:** Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch (ONERA - The French Aerospace Lab), and Yann Gousseau (LTCI, Télécom Paris, Université Paris-Saclay).
* **Usage Policy:** Distributed as an Open Access scientific benchmark via IEEE DataPort and Institut Mines-Télécom (IMT). Released under academic Open Access terms (CC BY-NC-SA 4.0 / Open Data Commons Attribution License). Permitted for research, educational, and benchmarking activities; commercial re-sale is prohibited.
* **Satellite Imagery Attribution:** Copernicus Sentinel-2 multispectral observations provided by the European Space Agency (ESA) under Copernicus open access data policy.
* **Mandatory Citation:**
  > Daudt, R. C., Le Saux, B., Boulch, A., & Gousseau, Y. (2018). *Urban Change Detection for Multispectral Earth Observation Using Convolutional Neural Networks*. In IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2018), pp. 2115–2118. https://doi.org/10.1109/IGARSS.2018.8518015

---

## 3. Data Redistribution Statement

In compliance with academic dataset sharing guidelines and GitHub file size constraints:
- Raw complete archive files (multi-gigabyte `.zip` / `.tar.gz` files) are **excluded from Git version tracking** via `.gitignore`.
- Only reproducible automated scripts, verified manifests, metadata inventories, and a minimal representative validation sample are kept under version control.
- Researchers and evaluators should run `python scripts/download_dataset.py` to retrieve raw data directly from official academic mirrors.
