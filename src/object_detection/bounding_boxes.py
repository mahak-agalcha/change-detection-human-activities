import csv
from pathlib import Path

import cv2
import numpy as np


def extract_change_regions(
    probability_map,
    prediction_mask,
    min_area=2
):
    """
    Extract bounding boxes around connected predicted change regions.
    """

    probability_map = np.asarray(
        probability_map, dtype=np.float32
    )
    prediction_mask = np.asarray(
        prediction_mask, dtype=bool
    )

    if probability_map.shape != prediction_mask.shape:
        raise ValueError(
            "Probability map and prediction mask must have equal dimensions."
        )

    binary_mask = prediction_mask.astype(np.uint8)

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8
        )
    )

    detections = []

    for label_id in range(1, num_labels):
        x, y, width, height, area = stats[label_id]

        if area < min_area:
            continue

        component = labels == label_id
        confidence = float(probability_map[component].mean())

        detections.append({
            "region_id": len(detections) + 1,
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
            "area": int(area),
            "confidence": confidence,
            "confidence_percent": round(confidence * 100, 2),
            "class_label": "predicted_change_region"
        })

    return detections


def save_detections_csv(detections, output_path):
    """Save extracted change regions to a CSV file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "region_id",
        "x",
        "y",
        "width",
        "height",
        "area",
        "confidence",
        "confidence_percent",
        "class_label"
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(detections)

    print(f"Saved {len(detections)} detections to: {output_path}")
