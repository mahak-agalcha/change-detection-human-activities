
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = next((ROOT / "data/raw/spacenet7/sample").iterdir())
LABELS = SAMPLE / "labels_match_pix"
OUTPUT = ROOT / "data/processed/spacenet7"


def load_labels(month):
    code = month.replace("-", "_")
    path = next(
        LABELS.glob(
            f"global_monthly_{code}_mosaic_*_Buildings.geojson"
        )
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        f["properties"]["Id"]: shape(f["geometry"])
        for f in data["features"]
        if f.get("geometry")
        and f.get("properties", {}).get("Id") is not None
    }


def load_rgb(month):
    code = month.replace("-", "_")
    path = next(
        (SAMPLE / "images_masked").glob(
            f"global_monthly_{code}_mosaic_*.tif"
        )
    )
    with rasterio.open(path) as src:
        return np.moveaxis(src.read([1, 2, 3]), 0, -1)


def main():
    old = load_labels("2018-01")
    new = load_labels("2019-01")
    rgb18 = load_rgb("2018-01")
    rgb19 = load_rgb("2019-01")

    candidates = sorted(new.keys() - old.keys())
    # Choose a few of the largest candidates for easier visual inspection.
    candidates = sorted(candidates, key=lambda i: new[i].area, reverse=True)[:6]

    fig, axes = plt.subplots(len(candidates), 2, figsize=(10, 3 * len(candidates)))
    if len(candidates) == 1:
        axes = np.array([axes])

    for row, building_id in enumerate(candidates):
        geom = new[building_id]
        minx, miny, maxx, maxy = geom.bounds

        # Pixel coordinates in labels_match_pix use image coordinates:
        # x increases rightward and y increases downward.
        pad = 20
        x0 = max(0, int(minx) - pad)
        x1 = min(rgb19.shape[1], int(maxx) + pad + 1)
        y0 = max(0, int(miny) - pad)
        y1 = min(rgb19.shape[0], int(maxy) + pad + 1)

        for ax, image, title in [
            (axes[row, 0], rgb18, "January 2018"),
            (axes[row, 1], rgb19, "January 2019"),
        ]:
            ax.imshow(image[y0:y1, x0:x1])
            ax.set_title(f"ID {building_id} — {title}")
            ax.axis("off")

            # Plot candidate outline relative to the crop.
            try:
                polygons = (
                    [geom]
                    if geom.geom_type == "Polygon"
                    else list(geom.geoms)
                )
                for polygon in polygons:
                    coords = np.asarray(polygon.exterior.coords)
                    ax.plot(
                        coords[:, 0] - x0,
                        coords[:, 1] - y0,
                        color="red",
                        linewidth=1.5,
                    )
            except AttributeError:
                pass

    fig.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT / "candidate_building_crops.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Candidate IDs inspected:", candidates)
    print("Saved:", out)


if __name__ == "__main__":
    main()
