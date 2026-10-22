
import json
from pathlib import Path

from shapely.geometry import shape

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "data" / "raw" / "spacenet7" / "sample"
SAMPLE = next(
    (path for path in SAMPLE_DIR.iterdir() if path.is_dir()),
    None,
)

if SAMPLE is None:
    raise FileNotFoundError(
        f"No sample dataset directory found in {SAMPLE_DIR}"
    )

LABEL_DIR = SAMPLE / "labels_match_pix"


def load_features(month):
    code = month.replace("-", "_")
    matches = list(
        LABEL_DIR.glob(
            f"global_monthly_{code}_mosaic_*_Buildings.geojson"
        )
    )

    if not matches:
        raise FileNotFoundError(
            f"No building annotations found for {month} in {LABEL_DIR}"
        )

    data = json.loads(matches[0].read_text(encoding="utf-8"))

    return {
        feature["properties"]["Id"]: shape(feature["geometry"])
        for feature in data["features"]
        if feature.get("geometry")
        and feature.get("properties", {}).get("Id") is not None
    }


def main():
    old = load_features("2018-01")
    new = load_features("2019-01")

    new_ids = sorted(new.keys() - old.keys())

    print("2018 annotated buildings:", len(old))
    print("2019 annotated buildings:", len(new))
    print("2019-only building IDs:", len(new_ids))
    print("\nFirst 15 candidates:")

    for building_id in new_ids[:15]:
        geom = new[building_id]
        minx, miny, maxx, maxy = geom.bounds

        print(
            f"ID={building_id}, "
            f"area={geom.area:.2f} pixels^2, "
            f"width={maxx - minx:.2f}, "
            f"height={maxy - miny:.2f}, "
            f"valid={geom.is_valid}"
        )

    areas = [new[i].area for i in new_ids]
    if areas:
        sorted_areas = sorted(areas)
        print("\nCandidate polygon area summary:")
        print("Minimum:", round(min(areas), 2))
        print("Median:", round(sorted_areas[len(areas) // 2], 2))
        print("Maximum:", round(max(areas), 2))


if __name__ == "__main__":
    main()
