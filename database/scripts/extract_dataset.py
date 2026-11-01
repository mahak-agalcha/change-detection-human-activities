#!/usr/bin/env python3
"""
scripts/extract_dataset.py

Safely unpacks raw dataset archives from data/raw/ into data/extracted/.
Features:
- Path traversal protection (Zip Slip prevention).
- Archive CRC32 integrity verification before extraction.
- Granular file count and extracted size tracking.
- Extraction manifest logging in data/extracted/extraction_manifest.json.

Usage:
    python scripts/extract_dataset.py
    python scripts/extract_dataset.py --input_dir data/raw --output_dir data/extracted
"""

import argparse
import json
import os
import sys
import tarfile
import time
import zipfile


def is_safe_path(target_dir: str, path: str) -> bool:
    """Verifies that an extracted file path does not escape the destination directory."""
    resolved_target = os.path.abspath(target_dir)
    resolved_path = os.path.abspath(os.path.join(target_dir, path))
    return os.path.commonpath([resolved_target, resolved_path]) == resolved_target


def extract_zip(archive_path: str, dest_dir: str) -> dict:
    """Safely extracts a ZIP archive with pre-validation and Zip Slip protection."""
    print(f"[*] Testing archive integrity: {os.path.basename(archive_path)}...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        # Integrity check
        bad_file = zf.testzip()
        if bad_file:
            raise RuntimeError(f"Corrupted archive detected! First corrupted entry: {bad_file}")

        file_list = zf.namelist()
        total_files = len(file_list)
        extracted_count = 0
        total_uncompressed_bytes = 0

        print(f"[*] Extracting {total_files} members to: {dest_dir}...")
        for member in file_list:
            if not is_safe_path(dest_dir, member):
                print(f"[!] Security Warning: Skipped unsafe path traversal entry: {member}")
                continue

            zf.extract(member, dest_dir)
            info = zf.getinfo(member)
            total_uncompressed_bytes += info.file_size
            extracted_count += 1

            if extracted_count % 10 == 0 or extracted_count == total_files:
                sys.stdout.write(f"\r    -> Unpacked {extracted_count}/{total_files} entries")
                sys.stdout.flush()

        print("\n    [+] Extraction complete.")
        return {
            "archive": os.path.basename(archive_path),
            "format": "zip",
            "extracted_count": extracted_count,
            "uncompressed_bytes": total_uncompressed_bytes,
            "status": "SUCCESS",
        }


def extract_tar(archive_path: str, dest_dir: str) -> dict:
    """Safely extracts a TAR / TAR.GZ archive."""
    print(f"[*] Extracting TAR archive: {os.path.basename(archive_path)}...")
    with tarfile.open(archive_path, "r:*") as tf:
        members = tf.getmembers()
        extracted_count = 0
        total_bytes = 0
        for m in members:
            if not is_safe_path(dest_dir, m.name):
                print(f"[!] Security Warning: Skipped unsafe path: {m.name}")
                continue
            tf.extract(m, dest_dir)
            total_bytes += m.size
            extracted_count += 1

        return {
            "archive": os.path.basename(archive_path),
            "format": "tar",
            "extracted_count": extracted_count,
            "uncompressed_bytes": total_bytes,
            "status": "SUCCESS",
        }


def main():
    parser = argparse.ArgumentParser(description="Extract raw dataset archives safely.")
    parser.add_argument("--input_dir", default="data/raw", help="Directory containing raw archives")
    parser.add_argument("--output_dir", default="data/extracted", help="Destination extraction directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "extraction_manifest.json")

    print("=" * 70)
    print("REMOTE SENSING CHANGE DETECTION - DATASET EXTRACTION TOOL")
    print(f"Input Directory:  {os.path.abspath(args.input_dir)}")
    print(f"Output Directory: {os.path.abspath(args.output_dir)}")
    print("=" * 70)

    if not os.path.exists(args.input_dir):
        print(f"[!] Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    archives = [
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith((".zip", ".tar.gz", ".tar", ".tgz"))
    ]

    if not archives:
        print("[!] No archives found in raw directory. Run download_dataset.py first.")
        sys.exit(0)

    extraction_results = []
    for arc in archives:
        full_arc_path = os.path.join(args.input_dir, arc)
        sub_dest = os.path.join(args.output_dir, os.path.splitext(arc)[0])
        os.makedirs(sub_dest, exist_ok=True)

        try:
            if arc.lower().endswith(".zip"):
                res = extract_zip(full_arc_path, sub_dest)
            else:
                res = extract_tar(full_arc_path, sub_dest)
            res["target_dir"] = sub_dest
            res["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            extraction_results.append(res)
        except Exception as e:
            print(f"[!] Error extracting {arc}: {e}")
            extraction_results.append({
                "archive": arc,
                "error": str(e),
                "status": "FAILED",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(extraction_results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[+] Extraction complete. Manifest logged at: {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
