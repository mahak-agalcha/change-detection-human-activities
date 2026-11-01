"""
Part 3 -- Change Interpretation, Evidence & Geospatial Database
Dataset Generation Script

Generates 1,000 synthetic geospatial change interpretation records.
ALL DATA IS SYNTHETIC -- generated for development and testing purposes only.
This dataset does NOT represent real-world validated satellite observations.

Outputs:
    part3/data/synthetic/change_dataset.csv
    part3/data/synthetic/change_dataset.geojson
    part3/data/synthetic/dataset_metadata.json
    part3/data/synthetic/validation_report.json
    part3/data/synthetic/statistics.json

Usage:
    python part3/scripts/generate_dataset.py
    python part3/scripts/generate_dataset.py --seed 42 --count 1000

Author: Part 3 -- Palak Upadhyay
"""

import argparse
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURATION -- all tunable constants in one place
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42
TOTAL_RECORDS: int = 1000

# Configurable geographical study area bounding box (EPSG:4326).
# Default: South Asian Urban Corridor -- a fictional synthetic study region.
# Adjust lat/lon bounds here to target a different area.
STUDY_AREA: dict[str, Any] = {
    "name": "South Asian Urban Corridor Study Area",
    "description": (
        "Synthetic study area spanning urban, suburban, rural, coastal, "
        "agricultural, industrial, and forest zones. "
        "Coordinates are randomly generated within this bounding box."
    ),
    "crs": "EPSG:4326",
    "lat_min": 18.5,
    "lat_max": 29.5,
    "lon_min": 72.8,
    "lon_max": 88.4,
    "region_types": [
        "urban", "suburban", "rural", "coastal",
        "agricultural", "industrial", "forest",
    ],
}

# Category distribution (must sum to 1.0)
CATEGORY_DISTRIBUTION: dict[str, float] = {
    "Human-made": 0.50,   # 500 records
    "Natural":    0.30,   # 300 records
    "Uncertain":  0.20,   # 200 records
}

# Change types grouped by category
CHANGE_TYPES: dict[str, list[str]] = {
    "Human-made": [
        "new_building_construction",
        "building_demolition",
        "road_construction",
        "road_expansion",
        "vehicle_infrastructure_changes",
    ],
    "Natural": [
        "vegetation_loss",
        "vegetation_growth",
        "flooding",
        "water_body_expansion",
        "landslide",
    ],
    "Uncertain": [
        "unknown_object_change",
        "low_confidence_change",
        "ambiguous_geographical_change",
    ],
}

# Object types that object-detection would return for each change type
# (preserves the logical relationship between change_type and object_type)
OBJECT_TYPE_MAP: dict[str, list[str]] = {
    "new_building_construction":      ["building", "structure", "rooftop"],
    "building_demolition":            ["building_ruins", "bare_ground", "debris"],
    "road_construction":              ["road", "pathway", "pavement"],
    "road_expansion":                 ["road", "highway", "bridge"],
    "vehicle_infrastructure_changes": ["parking_lot", "vehicle_cluster", "transport_hub"],
    "vegetation_loss":                ["cleared_vegetation", "bare_land", "logged_area"],
    "vegetation_growth":              ["dense_vegetation", "shrubland", "regrowth_patch"],
    "flooding":                       ["flooded_area", "inundated_land", "water_body"],
    "water_body_expansion":           ["water_body", "reservoir", "lake_expansion"],
    "landslide":                      ["debris_flow", "slope_failure", "bare_land"],
    "unknown_object_change":          ["unclassified_region", "unknown_object", "unresolved_area"],
    "low_confidence_change":          ["blurred_region", "uncertain_feature", "ambiguous_patch"],
    "ambiguous_geographical_change":  ["mixed_land_use", "boundary_uncertainty", "unresolved_transition"],
}

# Region types preferred by each change type (logical environment relationship)
PREFERRED_REGIONS: dict[str, list[str]] = {
    "new_building_construction":      ["urban", "suburban", "industrial"],
    "building_demolition":            ["urban", "suburban"],
    "road_construction":              ["suburban", "rural", "industrial"],
    "road_expansion":                 ["urban", "suburban"],
    "vehicle_infrastructure_changes": ["urban", "industrial"],
    "vegetation_loss":                ["forest", "agricultural", "rural"],
    "vegetation_growth":              ["forest", "rural", "agricultural"],
    "flooding":                       ["coastal", "rural", "agricultural"],
    "water_body_expansion":           ["coastal", "rural"],
    "landslide":                      ["rural", "forest"],
    "unknown_object_change":          ["rural", "urban", "suburban", "forest"],
    "low_confidence_change":          ["rural", "coastal", "forest", "urban"],
    "ambiguous_geographical_change":  ["rural", "suburban", "agricultural"],
}

# Evidence strings per change type (logically matched to change semantics)
EVIDENCE_POOL: dict[str, list[str]] = {
    "new_building_construction": [
        "Rectangular structure boundaries detected in after-image",
        "Construction material spectral signature identified",
        "Bare land converted to built-up area between epochs",
        "Urban land-use context confirmed",
        "Rooftop geometry is geometrically regular",
        "Shadow pattern consistent with new vertical structure",
        "Road network access identified adjacent to site",
        "Ground texture change from soil to impervious surface",
    ],
    "building_demolition": [
        "Built-up area converted to bare land in after-image",
        "Structural debris spectral pattern detected",
        "Rectangular boundary loss observed between epochs",
        "Urban context with removal activity confirmed",
        "Shadow previously present in before-image, now absent",
        "Ground surface exposure and soil signature confirmed",
        "Rubble texture detected in after-image",
    ],
    "road_construction": [
        "Linear feature newly appeared in after-image",
        "Pavement spectral reflectance detected",
        "Connectivity to existing road network established",
        "Bare earth disturbance along corridor observed",
        "Geometric linearity confirmed by shape analysis",
        "Vehicle activity observed along new route",
        "Vegetation cleared along linear pathway",
    ],
    "road_expansion": [
        "Existing road width measurably increased",
        "Lane marking pattern extended in after-image",
        "Pavement area growth detected at road boundaries",
        "Adjacent land cleared for expansion",
        "Traffic infrastructure features updated",
        "Median or shoulder addition detected",
    ],
    "vehicle_infrastructure_changes": [
        "Parking lot impervious surface detected",
        "Multiple vehicle cluster signatures identified",
        "Impervious surface expansion confirmed",
        "Transport hub infrastructure features confirmed",
        "Grid-pattern surface markings detected",
        "Vehicle access lane geometry confirmed",
    ],
    "vegetation_loss": [
        "NDVI significant decrease detected between epochs",
        "Vegetation canopy removal observed in after-image",
        "Bare soil spectral signature exposed",
        "Irregular boundary pattern consistent with natural edge",
        "No construction activity associated with change",
        "Seasonal adjustment applied; change persists",
        "Tree crown density reduction measured",
    ],
    "vegetation_growth": [
        "NDVI significant increase detected between epochs",
        "Vegetation canopy expansion observed",
        "Green biomass spectral signature confirmed in after-image",
        "Irregular natural boundary pattern observed",
        "Seasonal growth correlation identified",
        "Canopy density increase measured",
    ],
    "flooding": [
        "Water spectral signature detected in after-image",
        "Inundation pattern identified across low-lying area",
        "Low-lying topographic context confirmed",
        "Temporal correlation with known rainfall season",
        "Irregular water boundary consistent with flood event",
        "Previously dry land now exhibiting water signature",
        "High moisture content detected in soil surrounding area",
    ],
    "water_body_expansion": [
        "Water body boundary expanded in after-image",
        "Water spectral reflectance area increased",
        "Shoreline shift detected between epochs",
        "Seasonal water level variation consistent with expansion",
        "Adjacent land shows inundation signature",
        "Reservoir or lake boundary growth confirmed",
    ],
    "landslide": [
        "Slope failure pattern detected on hillside",
        "Debris flow spectral signature identified",
        "Bare land exposure on sloped terrain",
        "Topographic context consistent with mass movement",
        "Irregular debris boundary observed",
        "Vegetation removal on slope without human activity",
        "Soil exposure on gradient greater than 15 degrees",
    ],
    "unknown_object_change": [
        "Change region detected but object class unresolvable",
        "Insufficient spectral resolution for classification",
        "Object boundary indeterminate at current resolution",
        "No matching object template found in classifier",
        "Manual expert review recommended",
        "Confidence below all classification thresholds",
    ],
    "low_confidence_change": [
        "Low signal-to-noise ratio in change region",
        "Possible sensor artifact or atmospheric effect present",
        "Cloud or shadow cover may affect detection reliability",
        "Detection score below confidence threshold (0.5)",
        "Temporal inconsistency observed between image epochs",
        "Secondary image confirmation required",
        "Spectral values partially saturated",
    ],
    "ambiguous_geographical_change": [
        "Mixed land-use transition detected -- classification ambiguous",
        "Boundary between change categories unclear",
        "Multiple change hypotheses plausible from available data",
        "Contextual evidence insufficient for single classification",
        "Conflicting spectral and geometric signals observed",
        "Change region overlaps multiple land-use zones",
    ],
}

# Confidence score ranges per category (min, max)
CONFIDENCE_RANGES: dict[str, tuple[float, float]] = {
    "Human-made": (0.72, 0.97),
    "Natural":    (0.65, 0.93),
    "Uncertain":  (0.18, 0.58),
}

# Detection methods from Part 1 (change detection module)
DETECTION_METHODS: list[str] = [
    "pixel_difference",
    "cnn_feature_comparison",
    "spectral_index_change",
    "texture_analysis",
    "multi_scale_comparison",
]

# Synthetic source identifiers (clearly synthetic)
SOURCES: list[str] = [
    "synthetic_levir_cd_simulation",
    "synthetic_oscd_simulation",
    "synthetic_sentinel2_simulation",
    "synthetic_landsat8_simulation",
]

# Verification statuses per category
VERIFICATION_STATUS_MAP: dict[str, list[str]] = {
    "Human-made": ["verified", "pending_review", "auto_verified"],
    "Natural":    ["verified", "pending_review", "auto_verified"],
    "Uncertain":  ["unverified", "pending_review", "requires_manual_review"],
}

# Change area ranges in square metres per change type
AREA_RANGES_SQM: dict[str, tuple[float, float]] = {
    "new_building_construction":      (200.0,   8_000.0),
    "building_demolition":            (150.0,   6_000.0),
    "road_construction":              (500.0,  50_000.0),
    "road_expansion":                 (300.0,  20_000.0),
    "vehicle_infrastructure_changes": (1_000.0, 30_000.0),
    "vegetation_loss":                (2_000.0, 500_000.0),
    "vegetation_growth":              (1_000.0, 300_000.0),
    "flooding":                       (5_000.0, 2_000_000.0),
    "water_body_expansion":           (3_000.0, 800_000.0),
    "landslide":                      (1_000.0, 100_000.0),
    "unknown_object_change":          (100.0,  50_000.0),
    "low_confidence_change":          (100.0,  30_000.0),
    "ambiguous_geographical_change":  (500.0,  100_000.0),
}

# Observation date ranges for before images
BEFORE_DATE_RANGE = ("2018-01-01", "2022-12-31")
MIN_TEMPORAL_GAP_DAYS = 90    # at least 3 months between images
MAX_TEMPORAL_GAP_DAYS = 1460  # at most 4 years


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def _utcnow_iso() -> str:
    """Return current UTC time as ISO 8601 string (no tzinfo in string)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _random_date(start: str, end: str, rng: np.random.Generator) -> str:
    """Return a random date string between start and end (inclusive)."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    delta_days = (end_dt - start_dt).days
    offset = int(rng.integers(0, delta_days + 1))
    return (start_dt + timedelta(days=offset)).strftime("%Y-%m-%d")


def _random_after_date(before_date: str, rng: np.random.Generator) -> str:
    """Return a date after before_date with a realistic temporal gap."""
    before_dt = datetime.strptime(before_date, "%Y-%m-%d")
    gap = int(rng.integers(MIN_TEMPORAL_GAP_DAYS, MAX_TEMPORAL_GAP_DAYS + 1))
    after_dt = before_dt + timedelta(days=gap)
    cap = datetime(2024, 12, 31)
    if after_dt > cap:
        after_dt = cap
    return after_dt.strftime("%Y-%m-%d")


def _random_lat_lon(rng: np.random.Generator) -> tuple[float, float]:
    """Return a (lat, lon) pair within the configured study area."""
    lat = round(float(rng.uniform(STUDY_AREA["lat_min"], STUDY_AREA["lat_max"])), 6)
    lon = round(float(rng.uniform(STUDY_AREA["lon_min"], STUDY_AREA["lon_max"])), 6)
    return lat, lon


def _meters_to_degrees(meters: float, latitude: float) -> float:
    """Approximate metres -> decimal degrees at a given latitude."""
    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = 111_320.0 * math.cos(math.radians(latitude))
    return meters / ((meters_per_degree_lat + meters_per_degree_lon) / 2)


def _build_polygon(lat: float, lon: float, area_sqm: float) -> dict:
    """
    Build a simple rectangular GeoJSON Polygon centred on (lat, lon).
    Side length is derived from area_sqm assuming a square footprint.
    """
    half_side_deg = _meters_to_degrees(math.sqrt(area_sqm) / 2, lat)
    coords = [
        [lon - half_side_deg, lat - half_side_deg],
        [lon + half_side_deg, lat - half_side_deg],
        [lon + half_side_deg, lat + half_side_deg],
        [lon - half_side_deg, lat + half_side_deg],
        [lon - half_side_deg, lat - half_side_deg],   # closed ring
    ]
    return {"type": "Polygon", "coordinates": [coords]}


def _build_point(lat: float, lon: float) -> dict:
    """Build a GeoJSON Point geometry."""
    return {"type": "Point", "coordinates": [lon, lat]}


def _select_evidence(change_type: str, rng: np.random.Generator) -> list[str]:
    """Pick 2-4 evidence strings logically matched to the change type."""
    pool = EVIDENCE_POOL.get(change_type, ["Change detected"])
    n = int(rng.integers(2, min(5, len(pool)) + 1))
    indices = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in sorted(indices)]


def _pick_region(change_type: str, rng: np.random.Generator) -> str:
    """Choose a geographically appropriate region type for the change type."""
    preferred = PREFERRED_REGIONS.get(change_type, STUDY_AREA["region_types"])
    return str(rng.choice(preferred))


# ---------------------------------------------------------------------------
# CORE GENERATOR
# ---------------------------------------------------------------------------

def generate_records(seed: int = RANDOM_SEED, count: int = TOTAL_RECORDS) -> list[dict]:
    """
    Generate `count` synthetic change interpretation records.

    Parameters
    ----------
    seed  : Random seed for full reproducibility.
    count : Total number of records to generate.

    Returns
    -------
    list[dict] -- one dict per record with all fields populated.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # Build category list matching the configured distribution
    categories: list[str] = []
    for cat, fraction in CATEGORY_DISTRIBUTION.items():
        n = round(fraction * count)
        categories.extend([cat] * n)
    while len(categories) < count:
        categories.append("Human-made")
    categories = categories[:count]
    rng.shuffle(categories)  # type: ignore[arg-type]

    records: list[dict] = []

    for i, category in enumerate(categories):
        change_id = f"CHG_{seed}_{i + 1:05d}"

        # -- Dates --
        before_date = _random_date(*BEFORE_DATE_RANGE, rng=rng)
        after_date = _random_after_date(before_date, rng=rng)
        temporal_gap_days = (
            datetime.strptime(after_date, "%Y-%m-%d")
            - datetime.strptime(before_date, "%Y-%m-%d")
        ).days

        # -- Change type (uniform within category) --
        change_type = str(rng.choice(CHANGE_TYPES[category]))

        # -- Object type (logically linked to change type) --
        object_type = str(rng.choice(OBJECT_TYPE_MAP[change_type]))

        # -- Geography --
        lat, lon = _random_lat_lon(rng)
        region_type = _pick_region(change_type, rng)

        # -- Area --
        area_min, area_max = AREA_RANGES_SQM[change_type]
        area_sqm = float(rng.uniform(area_min, area_max))

        # -- Geometry: POLYGON for 60% of Human-made and Natural; POINT otherwise --
        use_polygon = (category != "Uncertain") and (float(rng.random()) < 0.60)
        if use_polygon:
            geometry = _build_polygon(lat, lon, area_sqm)
            geometry_type = "Polygon"
        else:
            geometry = _build_point(lat, lon)
            geometry_type = "Point"

        # -- Confidence score (category-aware range) --
        conf_min, conf_max = CONFIDENCE_RANGES[category]
        confidence_score = round(
            float(np.clip(rng.uniform(conf_min, conf_max), 0.0, 1.0)), 4
        )

        # -- Evidence (logically matched to change type) --
        evidence_list = _select_evidence(change_type, rng)
        evidence_str = "; ".join(evidence_list)

        # -- Source and verification --
        source = str(rng.choice(SOURCES))
        verification_status = str(rng.choice(VERIFICATION_STATUS_MAP[category]))

        # -- Change Detection Information (Part 1 output simulation) --
        if category != "Uncertain":
            change_magnitude = round(float(rng.uniform(0.3, 1.0)), 4)
        else:
            change_magnitude = round(float(rng.uniform(0.1, 0.6)), 4)
        pixel_change_ratio = round(float(rng.uniform(0.05, 0.55)), 4)
        detection_method = str(rng.choice(DETECTION_METHODS))
        change_detection_score = round(
            float(np.clip(confidence_score + rng.uniform(-0.08, 0.08), 0.0, 1.0)), 4
        )

        # -- Object Detection Information (Part 2 output simulation) --
        if category != "Uncertain":
            od_confidence = round(float(np.clip(rng.uniform(0.55, 0.99), 0.0, 1.0)), 4)
        else:
            od_confidence = round(float(np.clip(rng.uniform(0.25, 0.65), 0.0, 1.0)), 4)
        segmentation_available = bool(
            category == "Human-made" and float(rng.random()) < 0.75
        )
        # Bounding box in pixel coords (512x512 image assumed)
        bb_x = int(rng.integers(0, 400))
        bb_y = int(rng.integers(0, 400))
        bb_w = int(rng.integers(30, 512 - bb_x))
        bb_h = int(rng.integers(30, 512 - bb_y))
        bbox = [bb_x, bb_y, bb_x + bb_w, bb_y + bb_h]

        record = {
            # Core identification
            "change_id":             change_id,
            "before_date":           before_date,
            "after_date":            after_date,
            "temporal_gap_days":     temporal_gap_days,
            # Classification
            "change_type":           change_type,
            "object_type":           object_type,
            "category":              category,
            # Scoring
            "confidence_score":      confidence_score,
            # Evidence
            "evidence":              evidence_str,
            "evidence_count":        len(evidence_list),
            # Geography
            "latitude":              lat,
            "longitude":             lon,
            "region_type":           region_type,
            "area_sqm":              round(area_sqm, 2),
            "geometry_type":         geometry_type,
            "geometry":              json.dumps(geometry),
            # Source / provenance
            "source":                source,
            "verification_status":   verification_status,
            # Change detection info (Part 1 integration)
            "cd_change_magnitude":   change_magnitude,
            "cd_pixel_change_ratio": pixel_change_ratio,
            "cd_detection_method":   detection_method,
            "cd_detection_score":    change_detection_score,
            # Object detection info (Part 2 integration)
            "od_detection_confidence":   od_confidence,
            "od_segmentation_available": segmentation_available,
            "od_bbox_x1":                bbox[0],
            "od_bbox_y1":                bbox[1],
            "od_bbox_x2":                bbox[2],
            "od_bbox_y2":                bbox[3],
            # Metadata flags
            "data_type":      "SYNTHETIC",
            "synthetic_seed": seed,
            "record_version": "1.0",
        }
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# GEOJSON BUILDER
# ---------------------------------------------------------------------------

def build_geojson(records: list[dict]) -> dict:
    """Convert records list to a GeoJSON FeatureCollection."""
    features = []
    for rec in records:
        geom = json.loads(rec["geometry"])
        props = {k: v for k, v in rec.items() if k != "geometry"}
        features.append({
            "type":       "Feature",
            "geometry":   geom,
            "properties": props,
        })
    return {
        "type": "FeatureCollection",
        "name": "change_dataset",
        "crs":  {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "metadata": {
            "data_type":      "SYNTHETIC",
            "description":    "Synthetic geospatial change interpretation dataset for Part 3 development.",
            "generated_at":   _utcnow_iso(),
            "total_features": len(features),
            "study_area":     STUDY_AREA,
        },
        "features": features,
    }


# ---------------------------------------------------------------------------
# DATASET METADATA
# ---------------------------------------------------------------------------

def build_metadata(records: list[dict], seed: int) -> dict:
    """Build a structured metadata dictionary for the dataset."""
    df = pd.DataFrame(records)
    return {
        "dataset_name": "Part 3 Synthetic Change Interpretation Dataset",
        "data_type":    "SYNTHETIC",
        "disclaimer": (
            "This dataset is entirely synthetic. It was programmatically generated "
            "for software development and testing purposes only. It does NOT represent "
            "real-world satellite observations or validated change events."
        ),
        "version":       "1.0.0",
        "generated_at":  _utcnow_iso(),
        "random_seed":   seed,
        "total_records": len(records),
        "study_area":    STUDY_AREA,
        "temporal_coverage": {
            "before_date_range": list(BEFORE_DATE_RANGE),
            "after_date_range":  [df["after_date"].min(), df["after_date"].max()],
            "temporal_gap_days": {
                "min":  int(df["temporal_gap_days"].min()),
                "max":  int(df["temporal_gap_days"].max()),
                "mean": round(float(df["temporal_gap_days"].mean()), 1),
            },
        },
        "schema": {
            "change_id":               "Unique record identifier (string)",
            "before_date":             "Observation date of before-image (YYYY-MM-DD)",
            "after_date":              "Observation date of after-image (YYYY-MM-DD)",
            "temporal_gap_days":       "Days between before and after observation",
            "change_type":             "Specific change type label",
            "object_type":             "Primary object type detected by object detection",
            "category":                "Top-level classification: Human-made | Natural | Uncertain",
            "confidence_score":        "Classification confidence [0.0, 1.0]",
            "evidence":                "Semicolon-separated list of supporting evidence strings",
            "evidence_count":          "Number of evidence items",
            "latitude":                "Centroid latitude (WGS84)",
            "longitude":               "Centroid longitude (WGS84)",
            "region_type":             "Land-use region type of the change location",
            "area_sqm":                "Estimated area of the change region in square metres",
            "geometry_type":           "GeoJSON geometry type: Point or Polygon",
            "geometry":                "GeoJSON geometry object (JSON string in CSV)",
            "source":                  "Synthetic data source identifier",
            "verification_status":     "Verification status of the record",
            "cd_change_magnitude":     "Change detection magnitude [0.0, 1.0] (Part 1)",
            "cd_pixel_change_ratio":   "Fraction of pixels that changed (Part 1)",
            "cd_detection_method":     "Change detection algorithm used (Part 1)",
            "cd_detection_score":      "Change detection model score (Part 1)",
            "od_detection_confidence": "Object detection confidence [0.0, 1.0] (Part 2)",
            "od_segmentation_available": "Whether segmentation mask is available (Part 2)",
            "od_bbox_x1":              "Object bounding box x1 in pixels (Part 2)",
            "od_bbox_y1":              "Object bounding box y1 in pixels (Part 2)",
            "od_bbox_x2":              "Object bounding box x2 in pixels (Part 2)",
            "od_bbox_y2":              "Object bounding box y2 in pixels (Part 2)",
            "data_type":               "Always SYNTHETIC -- marks this as generated data",
            "synthetic_seed":          "Random seed used for generation",
            "record_version":          "Schema version",
        },
        "category_distribution": CATEGORY_DISTRIBUTION,
        "change_types":          CHANGE_TYPES,
        "confidence_ranges":     {k: list(v) for k, v in CONFIDENCE_RANGES.items()},
    }


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Run consistency checks on the generated dataset.
    Returns a validation report dict with pass/fail results for each check.
    """
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": passed, "detail": detail})

    # 1. Record count
    check("record_count_is_1000", len(df) == 1000, f"Found {len(df)} records")

    # 2. No duplicate change_id
    dup_count = int(df["change_id"].duplicated().sum())
    check("no_duplicate_change_ids", dup_count == 0, f"{dup_count} duplicates found")

    # 3. No nulls in required columns
    required_cols = [
        "change_id", "before_date", "after_date", "change_type",
        "object_type", "category", "confidence_score", "latitude",
        "longitude", "geometry", "source", "verification_status",
    ]
    for col in required_cols:
        null_count = int(df[col].isna().sum())
        check(f"no_nulls_in_{col}", null_count == 0, f"{null_count} nulls")

    # 4. Coordinates within study area
    lat_in_range = bool(
        (df["latitude"] >= STUDY_AREA["lat_min"]).all()
        and (df["latitude"] <= STUDY_AREA["lat_max"]).all()
    )
    lon_in_range = bool(
        (df["longitude"] >= STUDY_AREA["lon_min"]).all()
        and (df["longitude"] <= STUDY_AREA["lon_max"]).all()
    )
    check(
        "lat_within_study_area", lat_in_range,
        f"lat range: [{df['latitude'].min():.4f}, {df['latitude'].max():.4f}]",
    )
    check(
        "lon_within_study_area", lon_in_range,
        f"lon range: [{df['longitude'].min():.4f}, {df['longitude'].max():.4f}]",
    )

    # 5. Confidence score in [0, 1]
    conf_ok = bool(
        (df["confidence_score"] >= 0.0).all()
        and (df["confidence_score"] <= 1.0).all()
    )
    check(
        "confidence_score_in_range", conf_ok,
        f"range: [{df['confidence_score'].min():.4f}, {df['confidence_score'].max():.4f}]",
    )

    # 6. Category values valid
    valid_cats = set(CHANGE_TYPES.keys())
    invalid_cats = set(df["category"].unique()) - valid_cats
    check("valid_category_values", len(invalid_cats) == 0, str(invalid_cats))

    # 7. Change type belongs to its category
    mismatches = int(
        df.apply(
            lambda row: row["change_type"] not in CHANGE_TYPES.get(row["category"], []),
            axis=1,
        ).sum()
    )
    check("change_type_matches_category", mismatches == 0, f"{mismatches} mismatches")

    # 8. Object type consistent with change type
    obj_mismatches = int(
        df.apply(
            lambda row: row["object_type"] not in OBJECT_TYPE_MAP.get(row["change_type"], []),
            axis=1,
        ).sum()
    )
    check("object_type_matches_change_type", obj_mismatches == 0, f"{obj_mismatches} mismatches")

    # 9. After date > before date for all records
    temporal_ok = bool(
        (pd.to_datetime(df["after_date"]) > pd.to_datetime(df["before_date"])).all()
    )
    check("after_date_after_before_date", temporal_ok, "")

    # 10. Temporal gap within configured range
    gap_ok = bool(
        (df["temporal_gap_days"] >= MIN_TEMPORAL_GAP_DAYS).all()
        and (df["temporal_gap_days"] <= MAX_TEMPORAL_GAP_DAYS).all()
    )
    check(
        "temporal_gap_in_valid_range", gap_ok,
        f"range: [{df['temporal_gap_days'].min()}, {df['temporal_gap_days'].max()}] days",
    )

    # 11. Geometry is valid GeoJSON
    invalid_geom = 0
    for geom_str in df["geometry"]:
        try:
            g = json.loads(geom_str)
            if g["type"] not in ("Point", "Polygon"):
                invalid_geom += 1
        except (json.JSONDecodeError, KeyError):
            invalid_geom += 1
    check("geometry_is_valid_geojson", invalid_geom == 0, f"{invalid_geom} invalid")

    # 12. Area is positive
    area_ok = bool((df["area_sqm"] > 0).all())
    check("area_sqm_positive", area_ok, f"min area: {df['area_sqm'].min():.2f} sqm")

    # 13. Uncertain confidence lower than Human-made (category sanity)
    uncertain_mean = float(df[df["category"] == "Uncertain"]["confidence_score"].mean())
    human_mean = float(df[df["category"] == "Human-made"]["confidence_score"].mean())
    check(
        "uncertain_confidence_lower_than_human_made",
        uncertain_mean < human_mean,
        f"Uncertain mean={uncertain_mean:.3f}, Human-made mean={human_mean:.3f}",
    )

    # 14. All records marked SYNTHETIC
    all_synthetic = bool((df["data_type"] == "SYNTHETIC").all())
    check("all_records_marked_synthetic", all_synthetic, "")

    total  = len(checks)
    passed = sum(1 for c in checks if c["passed"])
    failed = total - passed

    return {
        "validation_timestamp": _utcnow_iso(),
        "total_checks":  total,
        "passed_checks": passed,
        "failed_checks": failed,
        "all_passed":    failed == 0,
        "checks":        checks,
    }


# ---------------------------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------------------------

def compute_statistics(df: pd.DataFrame) -> dict:
    """Compute descriptive statistics over the generated dataset."""

    cat_counts = df["category"].value_counts().to_dict()
    ct_counts  = df["change_type"].value_counts().to_dict()

    conf_by_cat = (
        df.groupby("category")["confidence_score"]
        .agg(["mean", "std", "min", "max"])
        .round(4)
        .to_dict(orient="index")
    )

    vs_counts     = df["verification_status"].value_counts().to_dict()
    region_counts = df["region_type"].value_counts().to_dict()
    geom_counts   = df["geometry_type"].value_counts().to_dict()
    source_counts = df["source"].value_counts().to_dict()

    area_by_cat = (
        df.groupby("category")["area_sqm"]
        .agg(["mean", "min", "max"])
        .round(2)
        .to_dict(orient="index")
    )

    det_conf_by_cat = (
        df.groupby("category")["od_detection_confidence"]
        .agg(["mean", "min", "max"])
        .round(4)
        .to_dict(orient="index")
    )

    temporal_stats = {
        "before_date_min": df["before_date"].min(),
        "before_date_max": df["before_date"].max(),
        "after_date_min":  df["after_date"].min(),
        "after_date_max":  df["after_date"].max(),
        "temporal_gap_mean_days": round(float(df["temporal_gap_days"].mean()), 1),
        "temporal_gap_min_days":  int(df["temporal_gap_days"].min()),
        "temporal_gap_max_days":  int(df["temporal_gap_days"].max()),
    }

    seg_available = int(df["od_segmentation_available"].sum())

    return {
        "generated_at":                    _utcnow_iso(),
        "total_records":                    len(df),
        "category_distribution":            cat_counts,
        "change_type_distribution":         ct_counts,
        "confidence_score_by_category":     conf_by_cat,
        "verification_status_distribution": vs_counts,
        "region_type_distribution":         region_counts,
        "geometry_type_distribution":       geom_counts,
        "source_distribution":              source_counts,
        "area_sqm_by_category":             area_by_cat,
        "detection_confidence_by_category": det_conf_by_cat,
        "temporal_coverage":                temporal_stats,
        "segmentation_available_count":     seg_available,
        "evidence_count": {
            "mean": round(float(df["evidence_count"].mean()), 2),
            "min":  int(df["evidence_count"].min()),
            "max":  int(df["evidence_count"].max()),
        },
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Part 3 synthetic change interpretation dataset."
    )
    parser.add_argument("--seed",  type=int, default=RANDOM_SEED,   help="Random seed (default: 42)")
    parser.add_argument("--count", type=int, default=TOTAL_RECORDS, help="Number of records (default: 1000)")
    parser.add_argument("--out",   type=str, default=None,          help="Output directory override")
    args = parser.parse_args()

    # Resolve output directory
    script_dir  = Path(__file__).resolve().parent
    default_out = script_dir.parent / "data" / "synthetic"
    out_dir     = Path(args.out) if args.out else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 60
    print(f"\n{sep}")
    print("  Part 3 -- Synthetic Dataset Generator")
    print(sep)
    print(f"  Seed:       {args.seed}")
    print(f"  Records:    {args.count}")
    print(f"  Study area: {STUDY_AREA['name']}")
    print(f"  Output dir: {out_dir}")
    print(f"{sep}\n")

    # 1. Generate
    print("[1/6] Generating records...")
    records = generate_records(seed=args.seed, count=args.count)
    df = pd.DataFrame(records)
    print(f"      Generated {len(df)} records.")

    # 2. CSV
    csv_path = out_dir / "change_dataset.csv"
    print(f"[2/6] Saving CSV -> {csv_path}")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    csv_kb = csv_path.stat().st_size / 1024
    print(f"      Size: {csv_kb:.1f} KB")

    # 3. GeoJSON
    geojson_path = out_dir / "change_dataset.geojson"
    print(f"[3/6] Building GeoJSON -> {geojson_path}")
    geojson_data = build_geojson(records)
    with open(geojson_path, "w", encoding="utf-8") as fh:
        json.dump(geojson_data, fh, indent=2, ensure_ascii=False)
    geojson_kb = geojson_path.stat().st_size / 1024
    print(f"      Size: {geojson_kb:.1f} KB")

    # 4. Metadata
    metadata_path = out_dir / "dataset_metadata.json"
    print(f"[4/6] Writing metadata -> {metadata_path}")
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(build_metadata(records, args.seed), fh, indent=2, ensure_ascii=False)

    # 5. Validation
    validation_path = out_dir / "validation_report.json"
    print(f"[5/6] Running validation -> {validation_path}")
    validation = validate_dataset(df)
    with open(validation_path, "w", encoding="utf-8") as fh:
        json.dump(validation, fh, indent=2, ensure_ascii=False)

    # 6. Statistics
    stats_path = out_dir / "statistics.json"
    print(f"[6/6] Computing statistics -> {stats_path}")
    stats = compute_statistics(df)
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    # -- Summary --
    print(f"\n{sep}")
    print("  DATASET GENERATION COMPLETE")
    print(sep)

    print("\n  Category distribution:")
    for cat, cnt in stats["category_distribution"].items():
        pct = cnt / len(df) * 100
        bar = "#" * int(pct / 2)
        print(f"    {cat:<15} {cnt:>4}  ({pct:.1f}%)  {bar}")

    print("\n  Change type distribution:")
    for ct, cnt in sorted(stats["change_type_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {ct:<40} {cnt:>4}")

    print("\n  Confidence score (mean by category):")
    for cat, conf in stats["confidence_score_by_category"].items():
        print(f"    {cat:<15}  mean={conf['mean']:.4f}  "
              f"min={conf['min']:.4f}  max={conf['max']:.4f}")

    print(f"\n  Geometry types : {stats['geometry_type_distribution']}")
    print(f"  Region types   : {stats['region_type_distribution']}")

    v_str = f"{validation['passed_checks']}/{validation['total_checks']} checks passed"
    if validation["all_passed"]:
        print(f"\n  Validation : {v_str}  [ALL PASSED]")
    else:
        print(f"\n  Validation : {v_str}  [{validation['failed_checks']} FAILED]")
        for chk in validation["checks"]:
            if not chk["passed"]:
                print(f"    FAIL: {chk['check']} -- {chk['detail']}")

    print("\n  Output files:")
    print(f"    {csv_path}         ({csv_kb:.1f} KB)")
    print(f"    {geojson_path}  ({geojson_kb:.1f} KB)")
    print(f"    {metadata_path}")
    print(f"    {validation_path}")
    print(f"    {stats_path}")
    print(f"\n{sep}\n")

    if not validation["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
