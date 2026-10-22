
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape
import json

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = next((ROOT / "data/raw/spacenet7/sample").iterdir())


def load_mask(month):
    code = month.replace("-", "_")
    label_path = next(
        (SAMPLE / "labels_match_pix").glob(
            f"global_monthly_{code}_mosaic_*_Buildings.geojson"
        )
    )
    image_path = next(
        (SAMPLE / "images_masked").glob(
            f"global_monthly_{code}_mosaic_*.tif"
        )
    )

    with rasterio.open(image_path) as image:
        shape_hw = (image.height, image.width)

    data = json.loads(label_path.read_text(encoding="utf-8"))
    geometries = [
        shape(feature["geometry"])
        for feature in data["features"]
        if feature.get("geometry")
    ]

    mask = rasterize(
        [(geometry, 1) for geometry in geometries],
        out_shape=shape_hw,
        fill=0,
        dtype="uint8",
    )

    return mask, len(geometries)


def main():
    mask_a, polygons_a = load_mask("2018-01")
    mask_b, polygons_b = load_mask("2019-01")

    a = mask_a.astype(bool)
    b = mask_b.astype(bool)

    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    additions = int(np.logical_and(~a, b).sum())
    removals = int(np.logical_and(a, ~b).sum())

    print("January 2018 polygons:", polygons_a)
    print("January 2019 polygons:", polygons_b)
    print("January 2018 building pixels:", int(a.sum()))
    print("January 2019 building pixels:", int(b.sum()))
    print("Intersection pixels:", intersection)
    print("Union pixels:", union)
    print("Pixel-wise IoU:", round(intersection / union, 4) if union else 0)
    print("Candidate addition pixels:", additions)
    print("Candidate removal pixels:", removals)


if __name__ == "__main__":
    main()
