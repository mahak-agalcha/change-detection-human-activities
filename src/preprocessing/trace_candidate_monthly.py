import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data/raw/spacenet7/sample/L15-0506E-1204N_2027_3374_13"
IMAGES = SAMPLE / "images_masked"
LABELS = SAMPLE / "labels_match_pix"
OUTPUT = ROOT / "data/processed/spacenet7"

BUILDING_ID = 2385
months = [
    f"{year}_{month:02d}"
    for year in (2018, 2019)
    for month in range(1, 13)
]

def get_path(folder, month, suffix):
    return next(folder.glob(
        f"global_monthly_{month}_mosaic_*{suffix}"
    ), None)

def read_labels(month):
    path = get_path(LABELS, month, "_Buildings.geojson")
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        f["properties"]["Id"]: shape(f["geometry"])
        for f in data["features"]
        if f.get("geometry")
        and f.get("properties", {}).get("Id") is not None
    }

history = {month: read_labels(month) for month in months}

if BUILDING_ID not in history["2019_01"]:
    raise ValueError(f"Building ID {BUILDING_ID} not found in January 2019")

reference = history["2019_01"][BUILDING_ID]
minx, miny, maxx, maxy = reference.bounds
pad = 30

x0 = max(0, int(minx) - pad)
y0 = max(0, int(miny) - pad)
x1 = min(1024, int(maxx) + pad + 1)
y1 = min(1024, int(maxy) + pad + 1)

fig, axes = plt.subplots(6, 4, figsize=(14, 18))

print(f"Monthly history for building ID {BUILDING_ID}")
print("Month    | ID present | Polygon area (pixels squared)")

for ax, month in zip(axes.flat, months):
    image_path = get_path(IMAGES, month, ".tif")
    if image_path is None:
        ax.set_title(f"{month.replace('_', '-')} - image missing")
        ax.axis("off")
        continue

    with rasterio.open(image_path) as src:
        rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
    crop = rgb[y0:y1, x0:x1]

    ax.imshow(crop)
    geom = history[month].get(BUILDING_ID)

    if geom is not None:
        polygons = [geom] if geom.geom_type == "Polygon" else [
            g for g in getattr(geom, "geoms", [])
            if g.geom_type == "Polygon"
        ]
        for polygon in polygons:
            coords = np.asarray(polygon.exterior.coords)
            ax.plot(coords[:, 0] - x0, coords[:, 1] - y0,
                    color="red", linewidth=1.2)

        print(f"{month.replace('_', '-')} | yes        | {geom.area:.2f}")
        status = "ID present"
    else:
        print(f"{month.replace('_', '-')} | no         | -")
        status = "ID absent"

    ax.set_title(f"{month.replace('_', '-')} - {status}")
    ax.axis("off")

fig.tight_layout()
OUTPUT.mkdir(parents=True, exist_ok=True)
out = OUTPUT / f"monthly_history_{BUILDING_ID}.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved visualization: {out}")
