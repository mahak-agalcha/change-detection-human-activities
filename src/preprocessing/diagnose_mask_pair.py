import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = next((ROOT / "data/raw/spacenet7/sample").iterdir())
LABELS = SAMPLE / "labels_match_pix"


def inspect_month(month):
    code = month.replace("-", "_")
    path = next(
        LABELS.glob(
            f"global_monthly_{code}_mosaic_*_Buildings.geojson"
        )
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    features = [
        f for f in data["features"]
        if f.get("geometry")
    ]
    geometries = [shape(f["geometry"]) for f in features]

    image_path = next(
        (SAMPLE / "images_masked").glob(
            f"global_monthly_{code}_mosaic_*.tif"
        )
    )

    with rasterio.open(image_path) as image:
        height, width = image.height, image.width

    mask = rasterize(
        [(geom, 1) for geom in geometries],
        out_shape=(height, width),
        fill=0,
        dtype="uint8",
    ).astype(bool)

    invalid = sum(not geom.is_valid for geom in geometries)
    outside = sum(
        geom.bounds[0] < 0
        or geom.bounds[1] < 0
        or geom.bounds[2] > width
        or geom.bounds[3] > height
        for geom in geometries
    )

    print(f"\n{month}")
    print("Features:", len(features))
    print("Invalid geometries:", invalid)
    print("Geometries outside image bounds:", outside)
    print("Mask pixels:", int(mask.sum()))

    return mask


a = inspect_month("2018-01")
b = inspect_month("2019-01")

print("\nCross-date comparison")
print("2018-only pixels:", int((a & ~b).sum()))
print("2019-only pixels:", int((b & ~a).sum()))
print("Identical masks:", bool(np.array_equal(a, b)))