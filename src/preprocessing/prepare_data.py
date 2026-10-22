
"""Inspect and prepare SpaceNet 7 satellite imagery."""

from pathlib import Path
import re

import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "spacenet7"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "spacenet7"


def get_image_month(filename: str) -> str:
    """Extract YYYY-MM from a SpaceNet 7 monthly image filename."""
    match = re.search(r"global_monthly_(\d{4})_(\d{2})", filename)
    return f"{match.group(1)}-{match.group(2)}" if match else "Unknown"


def inspect_image(image_path: Path) -> dict:
    """Read image metadata and RGB pixel ranges."""
    with rasterio.open(image_path) as dataset:
        info = {
            "filename": image_path.name,
            "month": get_image_month(image_path.name),
            "width": dataset.width,
            "height": dataset.height,
            "bands": dataset.count,
            "crs": str(dataset.crs),
            "color_interpretation": [
                str(band) for band in dataset.colorinterp
            ],
            "bounds": tuple(dataset.bounds),
            "rgb_ranges": [],
        }

        for band_index in range(1, min(3, dataset.count) + 1):
            band = dataset.read(band_index)
            info["rgb_ranges"].append(
                (int(band.min()), int(band.max()))
            )

    return info


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data directory: {RAW_DATA_DIR}")

    if not RAW_DATA_DIR.exists():
        print("Dataset folder not found. Check the dataset location.")
        return

    image_files = sorted(
        path
        for pattern in ("*.tif", "*.tiff")
        for path in RAW_DATA_DIR.rglob(pattern)
    )

    if not image_files:
        print("No GeoTIFF images found.")
        return

    print(f"\nTotal GeoTIFF images found: {len(image_files)}")
    print("\nFirst five images:")

    for image_path in image_files[:5]:
        print(inspect_image(image_path))

    months = sorted({
        get_image_month(path.name)
        for path in image_files
    })

    print("\nAvailable months:")
    print(", ".join(months))


if __name__ == "__main__":
    main()
