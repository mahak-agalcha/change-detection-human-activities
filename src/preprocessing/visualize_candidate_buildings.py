
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = next((ROOT / "data/raw/spacenet7/sample").iterdir())
OUTPUT = ROOT / "data/processed/spacenet7"
LABELS = SAMPLE / "labels_match_pix"


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
    candidate_ids = sorted(new.keys() - old.keys())
    candidate_polygons = [new[i] for i in candidate_ids]

    image_path = next(
        (SAMPLE / "images_masked").glob(
            "global_monthly_2019_01_mosaic_*.tif"
        )
    )

    with rasterio.open(image_path) as src:
        rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        height, width = src.height, src.width

    candidate_mask = rasterize(
        [(polygon, 1) for polygon in candidate_polygons],
        out_shape=(height, width),
        fill=0,
        dtype="uint8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    axes[0].imshow(rgb)
    axes[0].set_title("January 2019 satellite image")

    axes[1].imshow(rgb)
    overlay = np.ma.masked_where(candidate_mask == 0, candidate_mask)
    axes[1].imshow(
        overlay,
        cmap=ListedColormap(["red"]),
        alpha=0.85,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title(
        f"2019-only annotation candidates ({len(candidate_ids)})"
    )

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / "candidate_buildings_2019.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("Candidate buildings:", len(candidate_ids))
    print("Candidate pixels:", int(candidate_mask.sum()))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
