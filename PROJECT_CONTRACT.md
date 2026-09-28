# Project Contract
## Change Detection Due to Human Activities

**Project:** Change Detection Due to Human Activities

**Team:**
- Mahak Agalcha - 2310DMBCSE15387
- Krishna Kushwah - 2310DMBCSE15384
- Palak Upadhyay - 2310DMBCSE15401
- Shreeyashi Choukad - 2310DMBCSE1419

**Guide:** Dr. Sandeep Kumar Jain

---

## 1. Project Objective

The system detects changes between multi-temporal satellite images and analyzes whether the detected changes are likely to be caused by human activities, natural events, or insufficient evidence.

The system produces:

- Change detection results
- Changed regions
- Object-level information
- Human / Natural / Uncertain cause classification
- Confidence scores
- Geospatial evidence
- Visual results through the web application

---

# 2. System Pipeline

```text
Before Image + After Image
            |
            v
     Part 1: Data Pipeline
            |
            v
    Change Detection Output
            |
            v
   Part 2: Object Detection
       + Segmentation
            |
            v
      Object-Level Output
            |
            v
   Part 3: Context Analysis
       + Geospatial Database
            |
            v
 Human / Natural / Uncertain
            |
            v
   Part 4: Backend + Frontend
       + Deployment
            |
            v
       Final Application