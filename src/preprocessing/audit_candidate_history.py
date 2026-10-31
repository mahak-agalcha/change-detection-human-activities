
import csv
import json
from pathlib import Path

from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data/raw/spacenet7/sample/L15-0506E-1204N_2027_3374_13"
LABELS = SAMPLE / "labels_match_pix"
OUTPUT = ROOT / "data/processed/spacenet7"

MONTHS = [
    f"{year}_{month:02d}"
    for year in (2018, 2019)
    for month in range(1, 13)
]


def load_buildings(month):
    path = next(
        LABELS.glob(
            f"global_monthly_{month}_mosaic_*_Buildings.geojson"
        ),
        None,
    )

    if path is None:
        print(f"WARNING: Annotation file missing for {month}")
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))

    buildings = {}

    for feature in data["features"]:
        if not feature.get("geometry"):
            continue

        properties = feature.get("properties", {})
        building_id = properties.get("Id")

        if building_id is not None:
            buildings[building_id] = shape(feature["geometry"])

    return buildings


def calculate_iou(first, second):
    union_area = first.union(second).area

    if union_area == 0:
        return 0.0

    return first.intersection(second).area / union_area


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Load annotations for all 24 months.
    monthly = {
        month: load_buildings(month)
        for month in MONTHS
    }

    # Identify buildings present in January 2019 but absent in January 2018.
    old_ids = set(monthly["2018_01"])
    new_ids = set(monthly["2019_01"])
    candidates = sorted(new_ids - old_ids)

    print(f"\nJanuary 2019-only candidates: {len(candidates)}")
    print("Auditing monthly annotation history...\n")

    rows = []

    for candidate_id in candidates:
        observed_months = [
            month for month in MONTHS
            if candidate_id in monthly[month]
        ]

        first_month = observed_months[0] if observed_months else ""
        last_month = observed_months[-1] if observed_months else ""

        geometries = [
            monthly[month][candidate_id]
            for month in observed_months
        ]

        areas = [geometry.area for geometry in geometries]

        if geometries:
            reference = geometries[0]

            identical_count = sum(
                reference.equals(geometry)
                for geometry in geometries
            )

            stable_fraction = identical_count / len(geometries)
            mean_area = sum(areas) / len(areas)
            min_area = min(areas)
            max_area = max(areas)
        else:
            stable_fraction = 0.0
            mean_area = 0.0
            min_area = 0.0
            max_area = 0.0

        # Compare with January 2018 annotations.
        best_iou = 0.0
        nearest_distance = None
        nearest_2018_id = None

        if first_month:
            candidate_geometry = monthly[first_month][candidate_id]
            candidate_centroid = candidate_geometry.centroid

            for previous_id, previous_geometry in monthly["2018_01"].items():
                iou = calculate_iou(
                    candidate_geometry,
                    previous_geometry
                )

                distance = candidate_centroid.distance(
                    previous_geometry.centroid
                )

                best_iou = max(best_iou, iou)

                if (
                    nearest_distance is None
                    or distance < nearest_distance
                ):
                    nearest_distance = distance
                    nearest_2018_id = previous_id

        rows.append({
            "candidate_id": candidate_id,
            "first_annotated_month": first_month.replace("_", "-"),
            "last_annotated_month": last_month.replace("_", "-"),
            "months_present": len(observed_months),
            "months_absent": len(MONTHS) - len(observed_months),
            "geometry_stable_fraction": round(stable_fraction, 4),
            "mean_polygon_area_px2": round(mean_area, 2),
            "min_polygon_area_px2": round(min_area, 2),
            "max_polygon_area_px2": round(max_area, 2),
            "nearest_2018_id": nearest_2018_id,
            "nearest_2018_centroid_distance_px": (
                round(nearest_distance, 2)
                if nearest_distance is not None
                else ""
            ),
            "max_iou_with_2018_buildings": round(best_iou, 4),
        })

    csv_path = OUTPUT / "candidate_change_audit.csv"

    if rows:
        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(rows)

    print(f"Candidates audited: {len(rows)}")
    print(f"CSV saved: {csv_path}")

    print("\nFirst 15 candidates:")

    for row in rows[:15]:
        print(
            f"ID {row['candidate_id']} | "
            f"first={row['first_annotated_month']} | "
            f"present={row['months_present']}/24 | "
            f"stable={row['geometry_stable_fraction']:.2f} | "
            f"nearest distance="
            f"{row['nearest_2018_centroid_distance_px']}"
        )

    print(
        "\nAudit complete. These are annotation findings, "
        "not confirmed physical construction events."
    )


if __name__ == "__main__":
    main()
