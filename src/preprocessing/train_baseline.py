
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path("data/processed/spacenet7/paired_dataset")
OUT_DIR = Path("outputs/baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
PATCH = 128
BATCH = 4
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

files = sorted(DATA_DIR.glob("pair_*.npz"))
if len(files) < 3:
    raise RuntimeError(f"Expected at least 3 pair files in {DATA_DIR}")

class ChangePatchDataset(Dataset):
    def __init__(self, paths, patches_per_pair=32):
        self.samples = []

        for file_index, path in enumerate(paths):
            with np.load(path) as d:
                a = d["image_t1"]
                b = d["image_t2"]
                y = (d["change_mask"] != 0).astype(np.float32)
                valid = d["valid_mask"].astype(np.float32) if "valid_mask" in d else np.ones_like(y, dtype=np.float32)

            h, w = y.shape
            rng = np.random.default_rng(SEED + file_index)
            positive = np.argwhere((y > 0) & (valid > 0))
            chosen = []

            # Half the patches focus on annotated changes.
            count = min(patches_per_pair // 2, len(positive))
            if count:
                indices = rng.choice(len(positive), count, replace=False)
                for idx in indices:
                    py, px = positive[idx]
                    top = int(np.clip(py - PATCH // 2, 0, h - PATCH))
                    left = int(np.clip(px - PATCH // 2, 0, w - PATCH))
                    chosen.append((top, left))

            # Remaining patches are sampled randomly.
            while len(chosen) < patches_per_pair:
                chosen.append((
                    int(rng.integers(0, h - PATCH + 1)),
                    int(rng.integers(0, w - PATCH + 1))
                ))

            for top, left in chosen:
                sl = np.s_[top:top + PATCH, left:left + PATCH]
                self.samples.append((
                    a[sl].copy(),
                    b[sl].copy(),
                    y[sl].copy(),
                    valid[sl].copy()
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        a, b, y, valid = self.samples[idx]
        a = torch.from_numpy(a.transpose(2, 0, 1).astype(np.float32) / 255.0)
        b = torch.from_numpy(b.transpose(2, 0, 1).astype(np.float32) / 255.0)
        y = torch.from_numpy(y)
        valid = torch.from_numpy(valid)
        return a, b, y, valid


class SiameseChangeCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Shared encoder extracts features from both dates.
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
        difference = nn.functional.interpolate(
            difference, scale_factor=2, mode="bilinear",
            align_corners=False
        )
        return self.decoder(difference).squeeze(1)


# Split by pair file, not by individual patches.
rng = np.random.default_rng(SEED)
order = rng.permutation(len(files))
n_val = max(1, round(len(files) * 0.2))
val_paths = [files[i] for i in order[:n_val]]
train_paths = [files[i] for i in order[n_val:]]

train_ds = ChangePatchDataset(train_paths, patches_per_pair=32)
val_ds = ChangePatchDataset(val_paths, patches_per_pair=8)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

model = SiameseChangeCNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# Estimate class weighting using valid training pixels only.
positive_count = 0.0
valid_count = 0.0
for _, _, y, valid in train_ds:
    positive_count += float((y * valid).sum())
    valid_count += float(valid.sum())

negative_count = max(valid_count - positive_count, 1.0)
pos_weight = min(30.0, max(1.0, negative_count / max(positive_count, 1.0)))
pos_weight_tensor = torch.tensor(pos_weight, device=DEVICE)

def masked_loss(logits, target, valid):
    pixel_loss = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none",
        pos_weight=pos_weight_tensor
    )
    return (pixel_loss * valid).sum() / valid.sum().clamp_min(1.0)

def evaluate():
    model.eval()
    tp = fp = fn = 0

    with torch.no_grad():
        for a, b, y, valid in val_loader:
            a, b = a.to(DEVICE), b.to(DEVICE)
            y, valid = y.to(DEVICE), valid.to(DEVICE)

            probabilities = torch.sigmoid(model(a, b))
            predicted = probabilities >= 0.5
            actual = y >= 0.5
            usable = valid >= 0.5

            tp += int((predicted & actual & usable).sum().item())
            fp += int((predicted & ~actual & usable).sum().item())
            fn += int((~predicted & actual & usable).sum().item())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1)
    return precision, recall, f1, iou


print(f"Device: {DEVICE}")
print(f"Training pairs: {len(train_paths)} | Validation pairs: {len(val_paths)}")
print(f"Training patches: {len(train_ds)} | Validation patches: {len(val_ds)}")
print(f"Valid training pixels: {valid_count:.0f}")
print(f"Positive class weight: {pos_weight:.2f}")

best_f1 = -1.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for a, b, y, valid in train_loader:
        a, b = a.to(DEVICE), b.to(DEVICE)
        y, valid = y.to(DEVICE), valid.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(a, b)
        loss = masked_loss(logits, y, valid)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    precision, recall, f1, iou = evaluate()
    mean_loss = running_loss / max(len(train_loader), 1)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | loss={mean_loss:.4f} | "
        f"precision={precision:.4f} recall={recall:.4f} "
        f"F1={f1:.4f} IoU={iou:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "validation_f1": f1,
            "patch_size": PATCH,
            "input_channels_per_date": 4,
            "architecture": "SiameseChangeCNN",
            "threshold": 0.5
        }, OUT_DIR / "best_model.pt")

print(f"Best validation F1: {best_f1:.4f}")
print(f"Saved checkpoint: {OUT_DIR / 'best_model.pt'}")
