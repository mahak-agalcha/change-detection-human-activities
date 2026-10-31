from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.change_detection.baseline import load_model, predict_pair
from src.object_detection.bounding_boxes import (
    extract_change_regions,
    save_detections_csv
)


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

DATA_PATH = (
    ROOT / "data/processed/spacenet7/paired_dataset/pair_18.npz"
)

CHECKPOINT_PATH = ROOT / "outputs/baseline/best_model.pt"

OUTPUT_DIR = ROOT / "outputs/detection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.9
MIN_AREA = 2


# --------------------------------------------------
# 2. Load satellite image pair
# --------------------------------------------------

print("Loading satellite image pair...")

with np.load(DATA_PATH) as data:
    image_t1 = data["image_t1"].copy()
    image_t2 = data["image_t2"].copy()
    target = data["change_mask"].copy()

    valid_mask = (
        data["valid_mask"].copy().astype(bool)
        if "valid_mask" in data
        else np.ones(target.shape, dtype=bool)
    )

print("Image shape:", image_t1.shape)


# --------------------------------------------------
# 3. Load trained model
# --------------------------------------------------

print("\nLoading trained Siamese CNN...")

model, device = load_model(CHECKPOINT_PATH)

print("Model loaded on:", device)


# --------------------------------------------------
# 4. Predict changes
# --------------------------------------------------

print("\nRunning change detection...")

probability_map, prediction_mask = predict_pair(
    model=model,
    image_t1=image_t1,
    image_t2=image_t2,
    valid_mask=valid_mask,
    threshold=THRESHOLD,
    tile_size=128,
    device=device
)

print("Change prediction completed.")


# --------------------------------------------------
# 5. Calculate evaluation metrics
# --------------------------------------------------

actual = (target != 0) & valid_mask

tp = int((prediction_mask & actual).sum())
fp = int((prediction_mask & ~actual & valid_mask).sum())
fn = int((~prediction_mask & actual).sum())

precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)

f1 = (
    2 * precision * recall /
    max(precision + recall, 1e-8)
)

iou = tp / max(tp + fp + fn, 1)


# --------------------------------------------------
# 6. Extract bounding boxes
# --------------------------------------------------

print("\nExtracting predicted change regions...")

detections = extract_change_regions(
    probability_map,
    prediction_mask,
    min_area=MIN_AREA
)

csv_path = OUTPUT_DIR / "pair18_detections.csv"

save_detections_csv(detections, csv_path)


# --------------------------------------------------
# 7. Prepare satellite image visualization
# --------------------------------------------------

def rgb_preview(array):
    rgb = array[:, :, :3].astype(np.float32)

    low, high = np.percentile(rgb, [2, 98])

    rgb = np.clip(
        (rgb - low) / max(float(high - low), 1),
        0,
        1
    )

    return (rgb * 255).astype(np.uint8)


overlay = rgb_preview(image_t2).copy()

# Highlight predicted change pixels in red.
overlay[prediction_mask] = (
    0.45 * overlay[prediction_mask] +
    0.55 * np.array([255, 40, 40])
).astype(np.uint8)

overlay_image = Image.fromarray(overlay)
draw = ImageDraw.Draw(overlay_image)

try:
    font = ImageFont.truetype("arial.ttf", 18)
except OSError:
    font = ImageFont.load_default()


# --------------------------------------------------
# 8. Draw visible bounding boxes
# --------------------------------------------------

for detection in detections:

    x = detection["x"]
    y = detection["y"]
    width = detection["width"]
    height = detection["height"]

    confidence = detection["confidence_percent"]

    # Draw a thicker yellow bounding box.
    draw.rectangle(
        (
            x,
            y,
            x + width - 1,
            y + height - 1
        ),
        outline=(255, 255, 0),
        width=3
    )

    label = f"R{detection['region_id']} {confidence:.1f}%"

    text_box = draw.textbbox((x, max(0, y - 22)), label, font=font)

    draw.rectangle(
        text_box,
        fill=(0, 0, 0)
    )

    draw.text(
        (x, max(0, y - 22)),
        label,
        fill=(255, 255, 0),
        font=font
    )


# --------------------------------------------------
# 9. Save output images
# --------------------------------------------------

overlay_path = OUTPUT_DIR / "pair18_detection_overlay.png"

mask_path = OUTPUT_DIR / "pair18_prediction_mask.png"

overlay_image.save(overlay_path, format="PNG")

mask_image = Image.fromarray(
    (prediction_mask.astype(np.uint8) * 255)
)

mask_image.save(mask_path, format="PNG")


# --------------------------------------------------
# 10. Print final results
# --------------------------------------------------

print("\n========== DETECTION RESULTS ==========")

print("Device:", device)
print("Threshold:", THRESHOLD)
print("Valid pixels:", int(valid_mask.sum()))
print("Ground-truth change pixels:", int(actual.sum()))
print("Predicted change pixels:", int(prediction_mask.sum()))

print("Detected regions:", len(detections))

print("\nEvaluation Metrics")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"IoU:       {iou:.4f}")

print("\nOutput Files")
print("Overlay:", overlay_path)
print("Mask:", mask_path)
print("CSV:", csv_path)

print("\nDetection pipeline completed successfully.")
