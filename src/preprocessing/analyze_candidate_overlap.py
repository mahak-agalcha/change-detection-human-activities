
import csv
import json
from pathlib import Path

from shapely.geometry import box, shape

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = next((ROOT / "data/raw/spacenet7/sample").iterdir())
LABELS = SAMPLE / "labels_match_pix"
OUTPUT = ROOT / "data/processed/spacenet7"
IMAGE_BOUNDS = box(0, 0, 1024, 1024)


def load_features(month):
    code = month.replace("-", "_")
    path = next(
        LABELS.glob(
            f"global_monthly_{code}_mosaic_*_Buildings.geojson"
        )
    )
    data = json.loads(path.read_text(encoding="utf-8"))

    return {
        feature["properties"]["Id"]: shape(feature["geometry"])
        for feature in data["features"]
        if feature.get("geometry")
        and feature.get("properties", {}).get("Id") is not None
    }


def main():
    old = load_features("2018-01")
    new = load_features("2019-01")
    candidates = sorted(new.keys() - old.keys())

    rows = []

    for building_id in candidates:
        geom = new[building_id]
        clipped = geom.intersection(IMAGE_BOUNDS)

        # Check overlap with the union of all 2018 annotated footprints.
        old_geometries = [
            g.intersection(IMAGE_BOUNDS)
            for g in old.values()
            if g.intersects(clipped)
        ]

        overlap_area = sum(
            clipped.intersection(g).area
            for g in old_geometries
            if not g.is_empty
        )

        candidate_area = clipped.area
        overlap_fraction = (
            min(overlap_area / candidate_area, 1.0)
            if candidate_area > 0 else 0.0
        )

        if clipped.is_empty or candidate_area <= 0:
            status = "outside_image"
        elif not geom.is_valid:
            status = "invalid_geometry"
        elif overlap_fraction > 0:
            status = "overlaps_2018_annotation"
        else:
            status = "no_2018_annotation_overlap"

        rows.append({
            "building_id": building_id,
            "area_pixels_squared": round(geom.area, 3),
            "clipped_area_pixels_squared": round(candidate_area, 3),
            "geometry_valid": geom.is_valid,
            "within_image": IMAGE_BOUNDS.covers(geom),
            "overlap_area_pixels_squared": round(overlap_area, 3),
            "overlap_fraction": round(overlap_fraction, 4),
            "status": status,
        })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = OUTPUT / "candidate_overlap_report.csv"

    fields = list(rows[0].keys()) if rows else ["building_id"]
    with report.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("2018 annotated buildings:", len(old))
    print("2019 annotated buildings:", len(new))
    print("2019-only candidates:", len(candidates))
    print("Invalid candidate geometries:", sum(not new[i].is_valid for i in candidates))
    print("Candidates fully within image:", sum(IMAGE_BOUNDS.covers(new[i]) for i in candidates))
    print("Candidates overlapping 2018 annotations:", sum(r["overlap_fraction"] > 0 for r in rows))
    print("CSV report saved:", report)


if __name__ == "__main__":
    main()
