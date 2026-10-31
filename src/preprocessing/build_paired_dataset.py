import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from affine import Affine

ROOT = Path.cwd()
TILE = ROOT / "data/raw/spacenet7/sample/L15-0506E-1204N_2027_3374_13"
IMAGE_DIR = TILE / "images_masked"
LABEL_DIR = TILE / "labels_match_pix"
OUT = ROOT / "data/processed/spacenet7/paired_dataset"

OUT.mkdir(parents=True, exist_ok=True)

def read_image_and_valid_mask(path):
    with rasterio.open(path) as src:
        image = src.read()
        valid = src.dataset_mask() > 0

        if image.shape[1:] != (1024, 1024):
            raise ValueError(
                f"Unexpected image dimensions: {path.name}: {image.shape}"
            )

        return np.moveaxis(image, 0, -1), valid

def read_building_mask(path):
    with open(path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    shapes = []
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry")
        if geometry and geometry.get("type") in {"Polygon", "MultiPolygon"}:
            shapes.append((geometry, 1))

    return rasterize(
        shapes,
        out_shape=(1024, 1024),
        transform=Affine.identity(),
        fill=0,
        default_value=1,
        dtype="uint8",
        all_touched=False,
    )

images = {p.stem: p for p in IMAGE_DIR.glob("*.tif")}
labels = {
    p.name.replace("_Buildings.geojson", ""): p
    for p in LABEL_DIR.glob("*.geojson")
}
dates = sorted(set(images) & set(labels))

if len(dates) < 2:
    raise RuntimeError("Fewer than two matched image/annotation dates found.")

print(f"Matched dates: {len(dates)}")
print(f"Output directory: {OUT}")

manifest = []
total_new = 0
total_removed = 0

for i in range(len(dates) - 1):
    date1, date2 = dates[i], dates[i + 1]
    print(f"[{i + 1}/{len(dates) - 1}] {date1[-20:]} -> {date2[-20:]}")

    image1, valid1 = read_image_and_valid_mask(images[date1])
    image2, valid2 = read_image_and_valid_mask(images[date2])
    mask1 = read_building_mask(labels[date1])
    mask2 = read_building_mask(labels[date2])

    if image1.shape != image2.shape:
        raise ValueError(f"Image shape mismatch: {date1}, {date2}")

    valid_mask = (valid1 & valid2).astype(np.uint8)

    change = np.zeros((1024, 1024), dtype=np.uint8)
    change[(mask1 == 0) & (mask2 == 1)] = 1
    change[(mask1 == 1) & (mask2 == 0)] = 2

    new_pixels = int(np.count_nonzero((change == 1) & (valid_mask == 1)))
    removed_pixels = int(np.count_nonzero((change == 2) & (valid_mask == 1)))
    total_new += new_pixels
    total_removed += removed_pixels

    filename = f"pair_{i + 1:02d}.npz"

    np.savez_compressed(
        OUT / filename,
        image_t1=image1,
        image_t2=image2,
        mask_t1=mask1,
        mask_t2=mask2,
        change_mask=change,
        valid_mask=valid_mask,
        date_t1=date1,
        date_t2=date2,
    )

    manifest.append({
        "pair_id": i + 1,
        "date_t1": date1,
        "date_t2": date2,
        "file": filename,
        "image_shape": list(image1.shape),
        "valid_pixels": int(valid_mask.sum()),
        "invalid_pixels": int(valid_mask.size - valid_mask.sum()),
        "building_pixels_t1": int(mask1.sum()),
        "building_pixels_t2": int(mask2.sum()),
        "new_annotation_pixels_valid": new_pixels,
        "removed_annotation_pixels_valid": removed_pixels,
    })

manifest_path = OUT / "manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("\nDATASET BUILD COMPLETE")
print(f"Image pairs created: {len(manifest)}")
print(f"Valid new annotation pixels: {total_new}")
print(f"Valid removed annotation pixels: {total_removed}")
print(f"Manifest: {manifest_path}")
