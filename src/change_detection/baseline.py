from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SiameseChangeCNN(nn.Module):
    """Siamese CNN for binary satellite-image change prediction."""

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

    def forward(self, image_t1, image_t2):
        features_t1 = self.encoder(image_t1)
        features_t2 = self.encoder(image_t2)

        difference = torch.abs(features_t1 - features_t2)

        difference = F.interpolate(
            difference,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        return self.decoder(difference).squeeze(1)


def load_model(checkpoint_path, device=None):
    """Load the trained Siamese CNN checkpoint."""

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model = SiameseChangeCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device


def predict_pair(
    model,
    image_t1,
    image_t2,
    valid_mask=None,
    threshold=0.9,
    tile_size=128,
    device=None
):
    """
    Predict changes between two four-band satellite images.

    Images must have shape (height, width, 4).
    Returns a probability map and binary prediction mask.
    """

    image_t1 = np.asarray(image_t1)
    image_t2 = np.asarray(image_t2)

    if image_t1.shape != image_t2.shape:
        raise ValueError("Both images must have identical shapes.")

    if image_t1.ndim != 3 or image_t1.shape[2] != 4:
        raise ValueError("Expected images with shape (H, W, 4).")

    if device is None:
        device = next(model.parameters()).device

    height, width, _ = image_t1.shape
    probabilities = np.zeros((height, width), dtype=np.float32)

    model.eval()

    with torch.inference_mode():
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):

                patch_t1 = image_t1[
                    top:top + tile_size,
                    left:left + tile_size
                ]

                patch_t2 = image_t2[
                    top:top + tile_size,
                    left:left + tile_size
                ]

                tensor_t1 = torch.from_numpy(
                    patch_t1.transpose(2, 0, 1).copy().astype(np.float32) / 255.0
                ).unsqueeze(0).to(device)

                tensor_t2 = torch.from_numpy(
                    patch_t2.transpose(2, 0, 1).copy().astype(np.float32) / 255.0
                ).unsqueeze(0).to(device)

                output = torch.sigmoid(
                    model(tensor_t1, tensor_t2)
                )[0].cpu().numpy()

                probabilities[
                    top:top + output.shape[0],
                    left:left + output.shape[1]
                ] = output

    if valid_mask is None:
        valid_mask = np.ones((height, width), dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

        if valid_mask.shape != (height, width):
            raise ValueError("valid_mask dimensions do not match images.")

    prediction = (probabilities >= threshold) & valid_mask

    return probabilities, prediction
