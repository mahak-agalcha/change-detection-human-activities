from pathlib import Path
import csv
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# This must match train_baseline.py exactly.
ROOT = Path.cwd()
DATA_DIR = ROOT / "data/processed/spacenet7/paired_dataset"
CHECKPOINT = ROOT / "outputs/baseline/best_model.pt"
OUT_DIR = ROOT / "outputs/baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TILE = 128
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, a, b):
        difference = torch.abs(self.encoder(a) - self.encoder(b))
        difference = F.interpolate(
            difference, scale_factor=2, mode="bilinear",
            align_corners=False
        )
        return self.decoder(difference).squeeze(1)


def infer_full_pair(model, path):
    with np.load(path) as d:
        image_a = d["image_t1"].copy()
        image_b = d["image_t2"].copy()
        target = (d["change_mask"].copy() != 0)
        valid = (
            d["valid_mask"].copy().astype(bool)
            if "valid_mask" in d else np.ones(target.shape, dtype=bool)
        )

    h, w = target.shape
    probabilities = np.zeros((h, w), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for top in range(0, h, TILE):
            for left in range(0, w, TILE):
                bottom = min(top + TILE, h)
                right = min(left + TILE, w)
                tile_h, tile_w = bottom - top, right - left

                a = np.zeros((TILE, TILE, 4), dtype=np.float32)
                b = np.zeros((TILE, TILE, 4), dtype=np.float32)
                a[:tile_h, :tile_w] = image_a[top:bottom, left:right].astype(np.float32) / 255.0
                b[:tile_h, :tile_w] = image_b[top:bottom, left:right].astype(np.float32) / 255.0

                ta = torch.from_numpy(a.transpose(2, 0, 1).copy()).unsqueeze(0).to(DEVICE)
                tb = torch.from_numpy(b.transpose(2, 0, 1).copy()).unsqueeze(0).to(DEVICE)

                logits = model(ta, tb)
                prediction = torch.sigmoid(logits)[0].cpu().numpy()
                probabilities[top:bottom, left:right] = prediction[:tile_h, :tile_w]

    return probabilities, target, valid


def scores(tp, fp, fn):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1)
    return precision, recall, f1, iou


def counts(prob, target, valid, threshold):
    predicted = (prob >= threshold) & valid
    actual = target & valid
    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual & valid).sum())
    return tp, fp, fn


def main():
    files = sorted(DATA_DIR.glob("pair_*.npz"))
    if len(files) < 3:
        raise RuntimeError(f"Expected pair_*.npz files in {DATA_DIR}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    # Reproduce the exact pair-level validation split from train_baseline.py.
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(files))
    n_val = max(1, round(len(files) * 0.2))
    val_paths = [files[i] for i in order[:n_val]]

    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    model = SiameseChangeCNN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Device: {DEVICE}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print("Validation pairs (same split as training):")
    print("  " + ", ".join(p.name for p in val_paths))
    print("\nRunning full-scene inference. This may take a few minutes...")

    pair_data = []
    for index, path in enumerate(val_paths, start=1):
        prob, target, valid = infer_full_pair(model, path)
        pair_data.append((path.name, prob[valid], target[valid]))
        print(
            f"[{index}/{len(val_paths)}] {path.name}: "
            f"valid={int(valid.sum()):,}, "
            f"annotated_changes={int((target & valid).sum()):,}"
        )

    rows = []
    print("\nAGGREGATED VALIDATION RESULTS")
    print("Threshold | Predicted | Precision | Recall | F1 | IoU")
    best = None

    for threshold in THRESHOLDS:
        total_tp = total_fp = total_fn = total_pred = 0
        for _, prob, target in pair_data:
            pred = prob >= threshold
            tp = int((pred & target).sum())
            fp = int((pred & ~target).sum())
            fn = int((~pred & target).sum())
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_pred += int(pred.sum())

        precision, recall, f1, iou = scores(total_tp, total_fp, total_fn)
        rows.append({
            "scope": "aggregate",
            "pair": "ALL_VALIDATION_PAIRS",
            "threshold": threshold,
            "predicted_pixels": total_pred,
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": precision, "recall": recall, "f1": f1, "iou": iou,
        })
        print(
            f"{threshold:9.1f} | {total_pred:9,} | "
            f"{precision:9.4f} | {recall:6.4f} | {f1:.4f} | {iou:.4f}"
        )
        if best is None or f1 > best["f1"]:
            best = rows[-1]

    print(
        f"\nBest aggregate threshold: {best['threshold']:.1f} "
        f"(F1={best['f1']:.4f}, IoU={best['iou']:.4f})"
    )

    # Also save per-pair metrics at the aggregate-selected threshold.
    threshold = best["threshold"]
    for pair_name, prob, target in pair_data:
        pred = prob >= threshold
        tp = int((pred & target).sum())
        fp = int((pred & ~target).sum())
        fn = int((~pred & target).sum())
        precision, recall, f1, iou = scores(tp, fp, fn)
        rows.append({
            "scope": "per_pair",
            "pair": pair_name,
            "threshold": threshold,
            "predicted_pixels": int(pred.sum()),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "iou": iou,
        })

    output = OUT_DIR / "validation_threshold_results.csv"
    fieldnames = [
        "scope", "pair", "threshold", "predicted_pixels",
        "tp", "fp", "fn", "precision", "recall", "f1", "iou"
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved detailed results: {output}")


if __name__ == "__main__":
    main()
