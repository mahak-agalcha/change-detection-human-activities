# System Architecture: Change Detection Due to Human Activities
## Multi-Temporal Satellite Imagery System

This document outlines the overall 4-part system architecture and the internal modular design of **Part 2: Object Detection and Segmentation**.

---

## 1. Overall 4-Part System Architecture

```text
[Before Satellite Image (t1)] + [After Satellite Image (t2)]
                             │
                             ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  PART 1: Data Pipeline & Baseline Change Detection          │
 │  • Satellite pre-processing & image alignment               │
 │  • SiameseChangeCNN (PyTorch multi-temporal difference)     │
 │  • Binary change mask & candidate bounding boxes            │
 └───────────────────────────┬─────────────────────────────────┘
                             │ (Change mask + Candidate Regions)
                             ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  PART 2: Object Detection & Segmentation (THIS MODULE)      │
 │  • SatelliteImageLoader (Multi-band / NDVI / GeoTIFF CRS)   │
 │  • BuildingSegmenter (Rooftop extraction, RDP polygons)     │
 │  • RoadSegmenter (Linear directional corridor extraction)   │
 │  • VehicleDetector (GSD resolution feasibility engine)      │
 │  • GeospatialConverter (Pixel to Lon/Lat Affine -> GeoJSON) │
 └───────────────────────────┬─────────────────────────────────┘
                             │ (GeoJSON FeatureCollection & Class Labels)
                             ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  PART 3: Change Interpretation & Geospatial Database        │
 │  • Spatial overlay: Changed regions vs. Detected Objects    │
 │  • Context evidence synthesis (Zoning, historical data)     │
 │  • Causation classification: Human / Natural / Uncertain    │
 └───────────────────────────┬─────────────────────────────────┘
                             │ (Enriched Evidence & Cause Attribution)
                             ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  PART 4: Backend, Frontend & Deployment                     │
 │  • FastAPI REST API (backend/app)                           │
 │  • React + Leaflet Interactive Map UI (frontend/)           │
 │  • Docker containerization & PDF report generation          │
 └─────────────────────────────────────────────────────────────┘
```

---

## 2. Part 2 Modular Component Architecture

Inside `src/object_detection/` and `src/geospatial/`:

```text
src/
├── object_detection/
│   ├── __init__.py               # Package public API exports
│   ├── image_loader.py           # Multi-band reader (GeoTIFF, PNG, NPZ), NDVI, CRS
│   ├── building_segmentation.py  # CLAHE, rooftop threshold, morphology, RDP polygons
│   ├── road_segmentation.py      # Directional filters, aspect ratio (>2.0), corridors
│   ├── vehicle_detection.py      # GSD physics analyzer (<0.6m VHR check), contrast blobs
│   ├── bounding_boxes.py         # Connected-components candidate box utility
│   └── pipeline.py               # End-to-end multi-class orchestrator
└── geospatial/
    ├── __init__.py               # Geospatial module exports
    └── geo_converter.py          # Affine transforms, Shapely validation, GeoJSON export
```

---

## 3. Data Contract Across Modules

Our module implements and strictly satisfies `docs/api/output-contract.md` and `backend/app/services/inference.py`:

```json
{
  "type": "FeatureCollection",
  "crs": {
    "type": "name",
    "properties": { "name": "EPSG:4326" }
  },
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [[75.8601, 22.7201], [75.8615, 22.7201], [75.8615, 22.7212], [75.8601, 22.7212], [75.8601, 22.7201]]
        ]
      },
      "properties": {
        "region_id": 1,
        "object_type": "building",
        "confidence": 0.88,
        "cause_label": "human",
        "area_pixels": 450,
        "estimated_area_sq_m": 4050.0,
        "source_image": "pair18_detection_overlay.png"
      }
    }
  ]
}
```

---

## 4. Git Branching Strategy

```text
main (Production / Merged)
 │
 ├── feature/data-pipeline         (Part 1)
 ├── feature/object-detection      (Part 2 - Assigned Module)
 ├── feature/context-database      (Part 3)
 └── feature/frontend-backend      (Part 4)
```
