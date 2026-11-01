"""
Part 2 Object Detection and Segmentation Runner.

Entry point for testing and running building segmentation, road extraction,
vehicle feasibility checks, and geospatial GeoJSON exports.

Usage:
    # Run on a single satellite image
    python run_part2_pipeline.py --image path/to/image.png --output-dir outputs/part2_object_detection

    # Run quick demonstration mode
    python run_part2_pipeline.py --demo

    # Run with forced vehicle detection on high-resolution crop
    python run_part2_pipeline.py --demo --force-vehicles
"""

import argparse
from pathlib import Path
import sys
from typing import List, Dict

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.object_detection.pipeline import ObjectDetectionPipeline


def run_inference(old_path: str, new_path: str) -> List[Dict]:
    """
    Direct replacement for backend/app/services/inference.py STUB!

    Fulfills the frontend-backend contract:
    Returns a list of dicts with keys:
      object_type: "building" | "road" | "vehicle"
      cause_label: "human" | "natural" | "uncertain"
      confidence:  float 0..1
      geometry:    GeoJSON geometry dict (lon, lat)
    """
    pipeline = ObjectDetectionPipeline()
    # Run on the post-change satellite image (new_path)
    results = pipeline.run_image(image_path=new_path)
    return results["contract_features"]


def main():
    parser = argparse.ArgumentParser(
        description="Part 2: Object Detection and Segmentation Pipeline (Buildings, Roads, Vehicles, Geospatial)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input satellite image (.png, .jpg, .tif, .npz)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "part2_object_detection"),
        help="Directory to save visual overlays, GeoJSON, and CSV results"
    )
    parser.add_argument(
        "--gsd",
        type=float,
        default=3.0,
        help="Ground Sample Distance in meters/pixel (default: 3.0 for SpaceNet 7 PlanetScope)"
    )
    parser.add_argument(
        "--force-vehicles",
        action="store_true",
        help="Forces vehicle detection on high-resolution image crop"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on existing demonstration satellite image"
    )

    args = parser.parse_args()

    # Determine input image
    input_image = args.image
    if input_image is None or args.demo:
        # Check for existing sample satellite visualizations
        candidates = [
            ROOT / "outputs" / "detection" / "pair18_detection_overlay.png",
            ROOT / "outputs" / "baseline" / "pair05_prediction_preview.png",
            ROOT / "outputs" / "baseline" / "pair18_boxes_final.png",
        ]
        for c in candidates:
            if c.exists():
                input_image = str(c)
                break

    if input_image is None or not Path(input_image).exists():
        print(f"Error: Input image not found. Please provide --image or run with --demo.")
        sys.exit(1)

    print("=" * 65)
    print("CHANGE DETECTION DUE TO HUMAN ACTIVITIES - PART 2")
    print("Object Detection & Segmentation Pipeline")
    print("=" * 65)
    print(f"Input Image: {input_image}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Ground Sample Distance (GSD): {args.gsd} m/pixel")
    print(f"Force Vehicle Detection: {args.force_vehicles}")
    print("-" * 65)

    pipeline = ObjectDetectionPipeline(default_gsd=args.gsd)

    print("\n[Step 1/5] Loading satellite image & reading metadata...")
    print("[Step 2/5] Segmenting building footprints...")
    print("[Step 3/5] Extracting road networks and corridors...")
    print("[Step 4/5] Evaluating vehicle spatial resolution feasibility...")
    print("[Step 5/5] Performing Geospatial Affine transformation & GeoJSON export...")

    results = pipeline.run_image(
        image_path=input_image,
        output_dir=args.output_dir,
        gsd_meters=args.gsd,
        force_vehicle_detection=args.force_vehicles,
    )

    print("\n" + "=" * 65)
    print("DETECTION & SEGMENTATION SUMMARY")
    print("=" * 65)
    counts = results["counts"]
    print(f"  • Buildings Detected: {counts['buildings']}")
    print(f"  • Road Segments:      {counts['roads']}")
    print(f"  • Vehicles Detected:  {counts['vehicles']}")
    print(f"  • Total Object Count: {counts['total']}")
    print("-" * 65)
    print("Vehicle Resolution Diagnostic:")
    v_diag = results["vehicle_diagnostic"]
    print(f"  Feasible: {v_diag['is_feasible']}")
    print(f"  Reason:   {v_diag['reason']}")
    print("-" * 65)
    print("Generated Artifacts Saved To:")
    for key, path in results["saved_files"].items():
        print(f"  [{key}]: {path}")
    print("=" * 65)
    print("Pipeline executed successfully!")


if __name__ == "__main__":
    main()
