
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from torch import nn
import torch.nn.functional as F

ROOT = Path.cwd()
DATA = ROOT /  "data/processed/spacenet7/paired_dataset/pair_18.npz"
OUT_DIR = ROOT / "outputs" / "baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT =  ROOT / "outputs" / "baseline" / "pair18_boxes_final.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SiameseChangeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, a, b):
        fa = self.encoder(a)
        fb = self.encoder(b)
        difference = torch.abs(fa - fb)
        difference = F.interpolate(
            difference, scale_factor=2,
            mode="bilinear", align_corners=False
        )
        return self.decoder(difference).squeeze(1)


with np.load(DATA) as d:
    a = d["image_t1"].copy()
    b = d["image_t2"].copy()
    target = d["change_mask"].copy()
    valid = (
        d["valid_mask"].copy().astype(bool)
        if "valid_mask" in d
        else np.ones(target.shape, dtype=bool)
    )

checkpoint = torch.load(
    OUT_DIR / "best_model.pt",
    map_location=DEVICE,
    weights_only=True
)
model = SiameseChangeCNN().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

h, w = target.shape
prob = np.zeros((h, w), dtype=np.float32)
tile = 128

with torch.no_grad():
    for top in range(0, h, tile):
        for left in range(0, w, tile):
            aa = a[top:top + tile, left:left + tile]
            bb = b[top:top + tile, left:left + tile]

            ta = torch.from_numpy(
                aa.transpose(2, 0, 1).copy().astype(np.float32) / 255.0
            ).unsqueeze(0).to(DEVICE)
            tb = torch.from_numpy(
                bb.transpose(2, 0, 1).copy().astype(np.float32) / 255.0
            ).unsqueeze(0).to(DEVICE)

            p = torch.sigmoid(model(ta, tb))[0].cpu().numpy()
            prob[top:top + p.shape[0], left:left + p.shape[1]] = p

threshold = 0.9
pred = (prob >= threshold) & valid
actual = (target != 0) & valid

tp = int((pred & actual).sum())
fp = int((pred & ~actual & valid).sum())
fn = int((~pred & actual).sum())
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-8)
iou = tp / max(tp + fp + fn, 1)


def rgb_preview(arr):
    rgb = arr[:, :, :3].astype(np.float32)
    lo, hi = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - lo) / max(float(hi - lo), 1), 0, 1)
    return (rgb * 255).astype(np.uint8)


def mask_preview(mask, color=(255, 255, 255)):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask] = color
    return out


# Overlay predicted changes in red on the second image.
# Create prediction overlay
overlay = rgb_preview(b).copy()

overlay[pred] = (
    0.45 * overlay[pred] + 0.55 * np.array([255, 40, 40])
).astype(np.uint8)

# Detect connected change regions
binary_mask = pred.astype(np.uint8)

num_labels, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8
    )
)

# Draw bounding boxes
overlay_image = Image.fromarray(overlay)
box_draw = ImageDraw.Draw(overlay_image)

MIN_AREA = 2
detections = []

for i in range(1, num_labels):
    x, y, bw, bh, area = stats[i]

    if area < MIN_AREA:
        continue

    component = labels == i
    confidence = float(prob[component].mean())

    detections.append({
        "x": int(x),
        "y": int(y),
        "width": int(bw),
        "height": int(bh),
        "area": int(area),
        "confidence": confidence
    })

    # Yellow bounding box
    box_draw.rectangle(
        (x, y, x + bw - 1, y + bh - 1),
        outline=(255, 255, 0),
        width=2
    )

    # Confidence label
    box_draw.text(
        (x, max(0, y - 12)),
        f"{confidence:.2f}",
        fill=(255, 255, 0)
    )

overlay = np.array(overlay_image)

print("\nBounding Box Detection Results")
print("Total detected regions:", len(detections))

for index, detection in enumerate(detections, start=1):
    print(
        f"Box {index}: "
        f"Coordinates=({detection['x']}, {detection['y']}), "
        f"Size={detection['width']}x{detection['height']}, "
        f"Area={detection['area']}, "
        f"Confidence={detection['confidence']:.4f}"
    )

panels = [
    ("Image t1", rgb_preview(a)),
    ("Image t2", rgb_preview(b)),
    ("New annotations", mask_preview((target == 1) & valid)),
    ("Removed annotations", mask_preview((target == 2) & valid)),
    ("Ground-truth change", mask_preview(actual)),
    ("Predicted change (red overlay)", overlay),
]

scale = 0.5
tw, th = int(w * scale), int(h * scale)
margin, header = 16, 30
canvas = Image.new(
    "RGB",
    (margin * 4 + tw * 3, margin * 4 + (th + header) * 2),
    "white"
)
draw = ImageDraw.Draw(canvas)

for i, (label, arr) in enumerate(panels):
    row, col = divmod(i, 3)
    x = margin + col * (tw + margin)
    y = margin + row * (th + header + margin)
    draw.text((x, y), label, fill="black")
    im = Image.fromarray(arr).resize(
        (tw, th), Image.Resampling.NEAREST
    )
    canvas.paste(im, (x, y + header))

canvas.save(OUT, format="PNG")
print("Image successfully saved:", OUT.resolve())

print("Device:", DEVICE)
print("Saved preview:", OUT)
print("Valid pixels:", int(valid.sum()))
print("Ground-truth change pixels:", int(actual.sum()))
print("Predicted change pixels:", int(pred.sum()))
print(f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f} IoU={iou:.4f}")
