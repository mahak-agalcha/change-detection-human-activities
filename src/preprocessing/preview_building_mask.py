
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = PROJECT_ROOT / "data" / "raw" / "spacenet7" / "sample"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "spacenet7"

MONTHS = ["2018-01", "2019-01"]


def load_image_and_mask(sample_dir: Path, month: str):
    month_code = month.replace("-", "_")

    image_path = next(
        (sample_dir / "images_masked").glob(
            f"global_monthly_{month_code}_mosaic_*.tif"
        ),
        None,
    )
    label_path = next(
        (sample_dir / "labels_match_pix").glob(
            f"global_monthly_{month_code}_mosaic_*_Buildings.geojson"
        ),
        None,
    )

    if image_path is None or label_path is None:
        raise FileNotFoundError(f"Image or pixel labels missing for {month}")

    with rasterio.open(image_path) as src:
        rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        height, width = src.height, src.width

    data = json.loads(label_path.read_text(encoding="utf-8"))
    polygons = []

    for feature in data.get("features", []):
        geometry = feature.get("geometry")
        if geometry and geometry.get("type") in ("Polygon", "MultiPolygon"):
            polygons.append(shape(geometry))

    mask = rasterize(
        [(polygon, 1) for polygon in polygons],
        out_shape=(height, width),
        fill=0,
        dtype="uint8",
    )

    return rgb, mask, len(polygons)


def main():
    sample_dir = next(SAMPLE_DIR.iterdir())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for month in MONTHS:
        rgb, mask, polygon_count = load_image_and_mask(sample_dir, month)
        results[month] = (rgb, mask)

        mask_path = OUTPUT_DIR / f"building_mask_{month}.png"
        plt.imsave(mask_path, mask, cmap="gray", vmin=0, vmax=1)

        print(
            f"{month}: {polygon_count} polygons; "
            f"{int(mask.sum())} building pixels"
        )
        print(f"Mask saved: {mask_path}")

    rgb_2018, mask_2018 = results["2018-01"]
    rgb_2019, mask_2019 = results["2019-01"]

    if mask_2018.shape != mask_2019.shape:
        raise ValueError("The two masks have different dimensions.")

    # Preliminary pixel-wise comparison:
    # 1 = candidate addition, 2 = candidate removal, 0 = unchanged.
    change = np.zeros(mask_2018.shape, dtype="uint8")
    change[(mask_2018 == 0) & (mask_2019 == 1)] = 1
    change[(mask_2018 == 1) & (mask_2019 == 0)] = 2

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes[0, 0].imshow(rgb_2018)
    axes[0, 0].set_title("Satellite image — Jan 2018")
    axes[0, 1].imshow(mask_2018, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Building mask — Jan 2018")
    axes[1, 0].imshow(rgb_2019)
    axes[1, 0].set_title("Satellite image — Jan 2019")

    from matplotlib.colors import ListedColormap
    axes[1, 1].imshow(
        change,
        cmap=ListedColormap(["black", "lime", "red"]),
        vmin=0,
        vmax=2,
    )
    axes[1, 1].set_title("Preliminary changes: green=addition, red=removal")

    for axis in axes.flat:
        axis.axis("off")

    fig.tight_layout()
    preview_path = OUTPUT_DIR / "change_mask_preview_2018-01_2019-01.png"
    fig.savefig(preview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    change_path = OUTPUT_DIR / "change_mask_2018-01_2019-01.npy"
    np.save(change_path, change)

    print("Candidate addition pixels:", int((change == 1).sum()))
    print("Candidate removal pixels:", int((change == 2).sum()))
    print(f"Change mask saved: {change_path}")
    print(f"Preview saved: {preview_path}")


if __name__ == "__main__":
    main()
