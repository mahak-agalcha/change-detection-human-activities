#!/usr/bin/env python3
"""
scripts/generate_inventory.py

Generates formal academic reporting artifacts for the dataset collection:
1. reports/dataset_inventory.csv - Granular per-sample metadata catalog.
2. reports/dataset_summary.md    - High-level executive dataset summary.

Usage:
    python scripts/generate_inventory.py
    python scripts/generate_inventory.py --data_dir data --reports_dir reports
"""

import argparse
import csv
import json
import os
import sys
import numpy as np
from PIL import Image


def infer_human_activity_category(sample_id: str) -> str:
    """Infers the specific human activity category from sample identifiers or tags."""
    sid = sample_id.lower()
    if "residential" in sid:
        return "Residential Housing Expansion"
    elif "commercial" in sid:
        return "Commercial / Industrial Building Construction"
    elif "road" in sid or "earthworks" in sid:
        return "Transport Infrastructure / Road Construction"
    elif "suburban" in sid or "sprawl" in sid:
        return "Suburban Sprawl / Land Modification"
    elif "control" in sid or "no_change" in sid:
        return "Negative Control (Unchanged Urban Sector)"
    elif "oscd" in sid:
        return "Urban Earthworks & Civil Infrastructure"
    elif "levir" in sid:
        return "Building Construction & Expansion"
    else:
        return "Anthropogenic Land-Use Change"


def get_dir_size_bytes(dir_path: str) -> int:
    """Computes total bytes of all files in a directory recursively."""
    total = 0
    if not os.path.exists(dir_path):
        return 0
    for root, _, files in os.walk(dir_path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


def format_bytes(num_bytes: int) -> str:
    """Formats bytes into human-readable string (KB, MB, GB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def main():
    parser = argparse.ArgumentParser(description="Generate dataset inventory and summary reports.")
    parser.add_argument("--data_dir", default="data", help="Root data directory")
    parser.add_argument("--reports_dir", default="reports", help="Directory for output reports")
    args = parser.parse_args()

    os.makedirs(args.reports_dir, exist_ok=True)
    before_dir = os.path.join(args.data_dir, "before")
    after_dir = os.path.join(args.data_dir, "after")
    masks_dir = os.path.join(args.data_dir, "masks")
    processed_dir = os.path.join(args.data_dir, "processed")

    print("=" * 70)
    print("REMOTE SENSING CHANGE DETECTION - INVENTORY & SUMMARY GENERATOR")
    print(f"Data Source:     {os.path.abspath(args.data_dir)}")
    print(f"Reports Output:  {os.path.abspath(args.reports_dir)}")
    print("=" * 70)

    if not os.path.exists(before_dir):
        print(f"[!] Before directory not found: {before_dir}")
        sys.exit(1)

    before_files = sorted([f for f in os.listdir(before_dir) if not f.startswith(".")])
    inventory_rows = []

    total_changed_px = 0
    total_scene_px = 0
    split_counts = {"train": 0, "val": 0, "test": 0, "other": 0}

    for bf in before_files:
        sample_id = bf.replace("_before.png", "").replace("_before.jpg", "").replace("_before.tif", "")
        af = bf.replace("_before", "_after")
        mf = bf.replace("_before", "_mask")
        comp_f = f"{sample_id}_composite.png"

        b_path = os.path.join(before_dir, bf)
        a_path = os.path.join(after_dir, af)
        m_path = os.path.join(masks_dir, mf)
        comp_path = os.path.join(processed_dir, comp_f)

        if not os.path.exists(a_path) or not os.path.exists(m_path):
            continue

        # Inspect images
        with Image.open(b_path) as b_img, Image.open(m_path) as m_img:
            w, h = b_img.size
            channels = len(b_img.getbands())
            img_format = b_img.format or os.path.splitext(bf)[1][1:].upper()

            mask_arr = np.array(m_img)
            tot_px = mask_arr.size
            ch_px = int(np.sum(mask_arr > 0))
            pct = round((ch_px / tot_px) * 100.0, 3)

            total_changed_px += ch_px
            total_scene_px += tot_px

        # Determine split
        sid_lower = sample_id.lower()
        if "_train_" in sid_lower or sid_lower.startswith("train_"):
            split = "train"
            split_counts["train"] += 1
        elif "_val_" in sid_lower or sid_lower.startswith("val_"):
            split = "val"
            split_counts["val"] += 1
        elif "_test_" in sid_lower or sid_lower.startswith("test_"):
            split = "test"
            split_counts["test"] += 1
        else:
            split = "benchmark"
            split_counts["other"] += 1

        dataset_src = "LEVIR-CD" if "levir" in sid_lower else ("OSCD" if "oscd" in sid_lower else "RemoteSensing-CD")
        activity_type = infer_human_activity_category(sample_id)
        status = "VALID"

        inventory_rows.append({
            "sample_id": sample_id,
            "dataset_source": dataset_src,
            "split": split,
            "before_image": os.path.relpath(b_path),
            "after_image": os.path.relpath(a_path),
            "mask_image": os.path.relpath(m_path),
            "composite_image": os.path.relpath(comp_path) if os.path.exists(comp_path) else "N/A",
            "width": w,
            "height": h,
            "channels": channels,
            "format": img_format,
            "changed_pixels": ch_px,
            "total_pixels": tot_px,
            "change_percentage": pct,
            "human_activity_category": activity_type,
            "validation_status": status,
        })

    # 1. Write CSV Inventory
    csv_path = os.path.join(args.reports_dir, "dataset_inventory.csv")
    fieldnames = [
        "sample_id", "dataset_source", "split", "before_image", "after_image", "mask_image",
        "composite_image", "width", "height", "channels", "format", "changed_pixels",
        "total_pixels", "change_percentage", "human_activity_category", "validation_status"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inventory_rows)

    print(f"[+] Dataset inventory CSV written: {csv_path} ({len(inventory_rows)} samples)")

    # 2. Write Markdown Summary
    raw_size = get_dir_size_bytes(os.path.join(args.data_dir, "raw"))
    extracted_size = get_dir_size_bytes(os.path.join(args.data_dir, "extracted"))
    organized_size = (
        get_dir_size_bytes(before_dir)
        + get_dir_size_bytes(after_dir)
        + get_dir_size_bytes(masks_dir)
        + get_dir_size_bytes(processed_dir)
    )

    overall_change_pct = (total_changed_px / total_scene_px * 100.0) if total_scene_px > 0 else 0.0

    summary_md_path = os.path.join(args.reports_dir, "dataset_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Dataset Summary: Remote Sensing Human Activity Change Detection\n\n")
        f.write("## 1. Project and Benchmark Overview\n\n")
        f.write("- **Project Title:** Change Detection Due to Human Activities Using Satellite/Aerial Images\n")
        f.write("- **Primary Benchmark Dataset:** LEVIR-CD (Large-scale Evolutionary Land-change Remote-sensing Change Detection)\n")
        f.write("- **Secondary Benchmark Dataset:** OSCD (Onera Satellite Change Detection)\n")
        f.write("- **Target Detection Task:** Anthropogenic infrastructure changes (building construction, housing expansion, earthworks)\n")
        f.write("- **Target Resolution:** $0.5$ meters/pixel (Google Earth VHR) and $10$ meters/pixel (Sentinel-2 multispectral)\n\n")

        f.write("## 2. Quantitative Dataset Metrics\n\n")
        f.write(f"- **Total Cataloged Sample Pairs:** {len(inventory_rows)}\n")
        f.write(f"- **Total Images Ingested:** {len(inventory_rows) * 2} (Time 1 Pre-Change and Time 2 Post-Change)\n")
        f.write(f"- **Total Ground-Truth Masks:** {len(inventory_rows)}\n")
        f.write(f"- **Splits Breakdown:**\n")
        f.write(f"  - Training Split: `{split_counts['train']}` pairs\n")
        f.write(f"  - Validation Split: `{split_counts['val']}` pairs\n")
        f.write(f"  - Testing Split: `{split_counts['test']}` pairs\n")
        if split_counts["other"] > 0:
            f.write(f"  - Benchmark Control: `{split_counts['other']}` pairs\n\n")

        f.write("## 3. Storage and Footprint Breakdown\n\n")
        f.write(f"| Directory | Content Description | Disk Space Usage |\n")
        f.write(f"| :--- | :--- | :--- |\n")
        f.write(f"| `data/raw/` | Original compressed archives (`.zip`) | {format_bytes(raw_size)} |\n")
        f.write(f"| `data/extracted/` | Unpacked archive contents | {format_bytes(extracted_size)} |\n")
        f.write(f"| `data/before/` + `data/after/` + `data/masks/` | Standardized paired datasets | {format_bytes(organized_size)} |\n")
        f.write(f"| **Total Pipeline Data Footprint** | Complete local data repository | **{format_bytes(raw_size + extracted_size + organized_size)}** |\n\n")

        f.write("## 4. Class Balance and Anthropogenic Pixel Distribution\n\n")
        f.write(f"- **Total Scene Pixels Evaluated:** {total_scene_px:,}\n")
        f.write(f"- **Changed Pixels (Positive Human Activity):** {total_changed_px:,} ({overall_change_pct:.2f}%)\n")
        f.write(f"- **Unchanged Background Pixels:** {total_scene_px - total_changed_px:,} ({100.0 - overall_change_pct:.2f}%)\n")
        f.write("- **Class Balance Ratio:** 1 : " + (f"{(total_scene_px - total_changed_px) / max(total_changed_px, 1):.1f}" if total_changed_px > 0 else "N/A") + " (Reflects realistic sparse urban change distribution)\n\n")

        f.write("## 5. Cataloged Human Activity Typology\n\n")
        activity_counts = {}
        for row in inventory_rows:
            act = row["human_activity_category"]
            activity_counts[act] = activity_counts.get(act, 0) + 1

        f.write("| Human Activity Category | Sample Pairs Count | Percentage of Dataset |\n")
        f.write("| :--- | :--- | :--- |\n")
        for act, cnt in activity_counts.items():
            pct = (cnt / len(inventory_rows)) * 100.0
            f.write(f"| {act} | {cnt} | {pct:.1f}% |\n")

    print(f"[+] Dataset summary report written: {summary_md_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
