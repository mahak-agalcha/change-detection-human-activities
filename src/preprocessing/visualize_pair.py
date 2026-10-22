
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "spacenet7"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "spacenet7"


def find_month_image(month: str) -> Path:
    matches = list(RAW_DIR.rglob(f"global_monthly_{month.replace('-', '_')}_mosaic_*.tif"))

    if not matches:
        raise FileNotFoundError(f"No image found for {month}")

    return matches[0]


def read_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.count < 3:
            raise ValueError(f"Expected at least 3 bands: {path.name}")

        # Rasterio reads bands as (bands, rows, columns).
        # Move RGB bands to the last axis for plotting.
        rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)

    # These images are 8-bit RGB. Convert to [0, 1] for plotting.
    return rgb.astype(np.float32) / 255.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    months = ["2018-01", "2019-01"]
    paths = [find_month_image(month) for month in months]
    images = [read_rgb(path) for path in paths]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for axis, month, image in zip(axes, months, images):
        axis.imshow(image)
        axis.set_title(month)
        axis.axis("off")

    fig.suptitle("SpaceNet 7 — RGB Satellite Image Comparison")
    fig.tight_layout()

    output_path = OUTPUT_DIR / "rgb_comparison_2018-01_2019-01.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved comparison: {output_path}")


if __name__ == "__main__":
    main()
