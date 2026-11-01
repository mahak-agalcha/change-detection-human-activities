#!/usr/bin/env python3
"""
scripts/organize_dataset.py

Organizes extracted remote sensing change detection datasets into standardized directories:
    data/before/    -> Pre-change images (Time 1 / Image A)
    data/after/     -> Post-change images (Time 2 / Image B)
    data/masks/     -> Binary ground-truth change masks (0=unchanged, 255=human change)
    data/processed/ -> Visual verification composites (Before | After | Mask | Overlay)

Features:
- Automatically detects LEVIR-CD splits (train, val, test) and folder conventions (A, B, label).
- Automatically detects OSCD city directories and change masks (cm/cm.png).
- Non-destructive: copies extracted files without altering originals.
- Creates visual side-by-side composites in data/processed/ for rapid human inspection.
- Records all operations in data/organization_manifest.json.

Usage:
    python scripts/organize_dataset.py
    python scripts/organize_dataset.py --extracted_dir data/extracted --generate_composites
"""

import argparse
import json
import os
import shutil
import sys
from PIL import Image, ImageDraw


def create_verification_composite(before_path: str, after_path: str, mask_path: str, out_composite_path: str):
    """
    Creates an informative 4-panel diagnostic composite:
    [ Before (T1) ] | [ After (T2) ] | [ Ground Truth Mask ] | [ Red Overlay on T2 ]
    """
    b_img = Image.open(before_path).convert("RGB")
    a_img = Image.open(after_path).convert("RGB")
    m_img = Image.open(mask_path).convert("L")

    # Resize if needed to ensure uniform height
    w, h = b_img.size
    a_img = a_img.resize((w, h))
    m_img = m_img.resize((w, h), Image.NEAREST)

    # 4th panel: overlay change mask in bright red onto post-change image
    overlay_img = a_img.copy()
    red_layer = Image.new("RGB", (w, h), (255, 30, 30))
    # Blend using mask as alpha
    overlay_img = Image.composite(red_layer, overlay_img, m_img)

    # Create composite canvas
    composite = Image.new("RGB", (w * 4, h + 30), (30, 30, 30))
    composite.paste(b_img, (0, 30))
    composite.paste(a_img, (w, 30))
    composite.paste(m_img.convert("RGB"), (w * 2, 30))
    composite.paste(overlay_img, (w * 3, 30))

    draw = ImageDraw.Draw(composite)
    draw.text((10, 8), "Pre-Change (Time 1)", fill=(240, 240, 240))
    draw.text((w + 10, 8), "Post-Change (Time 2)", fill=(240, 240, 240))
    draw.text((w * 2 + 10, 8), "Ground Truth Mask", fill=(240, 240, 240))
    draw.text((w * 3 + 10, 8), "Change Overlay (Human Activity)", fill=(255, 100, 100))

    composite.save(out_composite_path, "PNG")


def organize_levir_structure(extracted_root: str, base_out: str, generate_composites: bool = True) -> list:
    """Discovers and organizes datasets matching LEVIR-CD convention."""
    organized = []
    # Walk to find directories named 'A' and 'B' with sibling 'label'
    for root, dirs, _ in os.walk(extracted_root):
        if "A" in dirs and "B" in dirs:
            dir_a = os.path.join(root, "A")
            dir_b = os.path.join(root, "B")
            dir_label = os.path.join(root, "label") if "label" in dirs else None

            # Determine split from parent folder name
            split = os.path.basename(root)
            if split not in ["train", "val", "test"]:
                split = "sample"

            dataset_tag = "levir"

            files_a = sorted(os.listdir(dir_a))
            for f in files_a:
                if not f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    continue

                stem = os.path.splitext(f)[0]
                src_a = os.path.join(dir_a, f)
                src_b = os.path.join(dir_b, f)
                src_label = os.path.join(dir_label, f) if dir_label and os.path.exists(os.path.join(dir_label, f)) else None

                if not os.path.exists(src_b):
                    print(f"[!] Warning: Missing pair for {src_a} in folder B")
                    continue

                sample_id = f"{dataset_tag}_{split}_{stem}"
                dst_before = os.path.join(base_out, "before", f"{sample_id}_before.png")
                dst_after = os.path.join(base_out, "after", f"{sample_id}_after.png")
                dst_mask = os.path.join(base_out, "masks", f"{sample_id}_mask.png")
                dst_composite = os.path.join(base_out, "processed", f"{sample_id}_composite.png")

                # Copy files
                shutil.copy2(src_a, dst_before)
                shutil.copy2(src_b, dst_after)

                has_mask = False
                if src_label and os.path.exists(src_label):
                    shutil.copy2(src_label, dst_mask)
                    has_mask = True
                    if generate_composites:
                        try:
                            create_verification_composite(dst_before, dst_after, dst_mask, dst_composite)
                        except Exception as e:
                            print(f"[!] Warning generating composite for {sample_id}: {e}")

                organized.append({
                    "sample_id": sample_id,
                    "dataset": "LEVIR-CD",
                    "split": split,
                    "source_id": stem,
                    "before_path": os.path.relpath(dst_before),
                    "after_path": os.path.relpath(dst_after),
                    "mask_path": os.path.relpath(dst_mask) if has_mask else None,
                    "composite_path": os.path.relpath(dst_composite) if has_mask and generate_composites else None,
                })

    return organized


def organize_oscd_structure(extracted_root: str, base_out: str) -> list:
    """Discovers and organizes datasets matching OSCD city folder convention."""
    organized = []
    # OSCD contains city directories with cm/cm.png
    for root, dirs, files in os.walk(extracted_root):
        if "cm.png" in files or "cm.tif" in files:
            city_dir = os.path.dirname(root) if os.path.basename(root) == "cm" else root
            city_name = os.path.basename(city_dir)
            mask_src = os.path.join(root, "cm.png") if "cm.png" in files else os.path.join(root, "cm.tif")

            # Look for subdirectories representing dates
            date_dirs = [d for d in os.listdir(city_dir) if os.path.isdir(os.path.join(city_dir, d)) and d != "cm"]
            date_dirs = sorted(date_dirs)

            sample_id = f"oscd_{city_name}"
            dst_mask = os.path.join(base_out, "masks", f"{sample_id}_mask.png")
            shutil.copy2(mask_src, dst_mask)

            dst_before = None
            dst_after = None
            if len(date_dirs) >= 2:
                # Pair the two dates
                t1_dir = os.path.join(city_dir, date_dirs[0])
                t2_dir = os.path.join(city_dir, date_dirs[1])
                # Check for RGB previews or band files
                t1_files = os.listdir(t1_dir)
                t2_files = os.listdir(t2_dir)
                # If there are preview or B02/B03/B04 files
                for tf in t1_files:
                    if tf.lower().endswith((".png", ".jpg", ".tif")):
                        dst_before = os.path.join(base_out, "before", f"{sample_id}_before_{tf}")
                        shutil.copy2(os.path.join(t1_dir, tf), dst_before)
                        break
                for tf in t2_files:
                    if tf.lower().endswith((".png", ".jpg", ".tif")):
                        dst_after = os.path.join(base_out, "after", f"{sample_id}_after_{tf}")
                        shutil.copy2(os.path.join(t2_dir, tf), dst_after)
                        break

            organized.append({
                "sample_id": sample_id,
                "dataset": "OSCD",
                "split": "train_or_test",
                "source_id": city_name,
                "before_path": os.path.relpath(dst_before) if dst_before else None,
                "after_path": os.path.relpath(dst_after) if dst_after else None,
                "mask_path": os.path.relpath(dst_mask),
                "composite_path": None,
            })

    return organized


def main():
    parser = argparse.ArgumentParser(description="Organize extracted change detection datasets.")
    parser.add_argument("--extracted_dir", default="data/extracted", help="Path to extracted datasets")
    parser.add_argument("--output_dir", default="data", help="Root data folder")
    parser.add_argument("--no_composites", action="store_true", help="Skip generating visual diagnostic composites")
    args = parser.parse_args()

    for d in ["before", "after", "masks", "processed"]:
        os.makedirs(os.path.join(args.output_dir, d), exist_ok=True)

    print("=" * 70)
    print("REMOTE SENSING CHANGE DETECTION - DATASET ORGANIZER")
    print(f"Source Directory: {os.path.abspath(args.extracted_dir)}")
    print(f"Target Base:     {os.path.abspath(args.output_dir)}")
    print("=" * 70)

    if not os.path.exists(args.extracted_dir):
        print(f"[!] Extracted directory not found: {args.extracted_dir}")
        sys.exit(1)

    all_records = []
    # 1. Check for LEVIR-CD structure
    levir_records = organize_levir_structure(args.extracted_dir, args.output_dir, generate_composites=not args.no_composites)
    all_records.extend(levir_records)

    # 2. Check for OSCD structure
    oscd_records = organize_oscd_structure(args.extracted_dir, args.output_dir)
    all_records.extend(oscd_records)

    manifest_path = os.path.join(args.output_dir, "organization_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[+] Organization complete.")
    print(f"    - Total paired samples cataloged: {len(all_records)}")
    print(f"    - Pre-change images stored in:     data/before/")
    print(f"    - Post-change images stored in:    data/after/")
    print(f"    - Ground-truth masks stored in:    data/masks/")
    print(f"    - Diagnostic composites stored in: data/processed/")
    print(f"    - Organization manifest logged at: {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
