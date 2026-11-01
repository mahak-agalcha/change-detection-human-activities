
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path.cwd()
DATA = ROOT / "data/processed/spacenet7/paired_dataset/pair_05.npz"
CHECKPOINT = ROOT / "outputs/baseline/best_model.pt"
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
        diff = torch.abs(self.encoder(a) - self.encoder(b))
        diff = F.interpolate(
            diff, scale_factor=2, mode="bilinear",
            align_corners=False
        )
        return self.decoder(diff).squeeze(1)


with np.load(DATA) as d:
    a = d["image_t1"].copy()
    b = d["image_t2"].copy()
    target = d["change_mask"].copy() != 0
    valid = (
        d["valid_mask"].copy().astype(bool)
        if "valid_mask" in d
        else np.ones(target.shape, dtype=bool)
    )

ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
model = SiameseChangeCNN().to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

prob = np.zeros(target.shape, dtype=np.float32)
tile = 128

with torch.no_grad():
   for top in range(0, a.shape[0], tile):
        for left in range(0, a.shape[1], tile):
            aa = a[top:min(top + tile, a.shape[0]),
                    left:min(left + tile, a.shape[1])]
            bb = b[top:min(top + tile, b.shape[0]),
                    left:min(left + tile, b.shape[1])]

            hh, ww = aa.shape[:2]

            # Pad edge tiles to a full 128 x 128.
            pad_a = np.zeros((tile, tile, 4), dtype=np.float32)
            pad_b = np.zeros((tile, tile, 4), dtype=np.float32)
            pad_a[:hh, :ww] = aa.astype(np.float32) / 255.0
            pad_b[:hh, :ww] = bb.astype(np.float32) / 255.0

            ta = torch.from_numpy(
                pad_a.transpose(2, 0, 1).copy()
            ).unsqueeze(0).to(DEVICE)
            tb = torch.from_numpy(
                pad_b.transpose(2, 0, 1).copy()
            ).unsqueeze(0).to(DEVICE)

            prediction = torch.sigmoid(model(ta, tb))[0].cpu().numpy()
            prob[top:top + hh, left:left + ww] = prediction[:hh, :ww]

p = prob[valid]
y = target[valid]

print(f"Device: {DEVICE}")
print(f"Valid pixels: {len(p)}")
print(f"Annotated changes: {int(y.sum())}")
print("Probability percentiles:")
for q in [0, 25, 50, 75, 90, 95, 99, 99.9, 100]:
    print(f"  {q:5.1f}%: {np.percentile(p, q):.6f}")

print("\nThreshold | Predicted | Precision | Recall | F1 | IoU")
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    pred = p >= threshold
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1)
    print(
        f"{threshold:9.1f} | {int(pred.sum()):9d} | "
        f"{precision:9.4f} | {recall:6.4f} | "
        f"{f1:.4f} | {iou:.4f}"
    )
