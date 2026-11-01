"""
Part 2: Object Detection and Segmentation Module.

This package provides:
- Image loading and satellite band handling (image_loader.py)
- Building footprint detection and segmentation (building_segmentation.py)
- Road network extraction and corridor segmentation (road_segmentation.py)
- Vehicle detection and resolution feasibility analysis (vehicle_detection.py)
- Unified multi-class inference pipeline (pipeline.py)
"""

from src.object_detection.image_loader import SatelliteImageLoader
from src.object_detection.building_segmentation import BuildingSegmenter
from src.object_detection.road_segmentation import RoadSegmenter
from src.object_detection.vehicle_detection import VehicleDetector

__all__ = [
    "SatelliteImageLoader",
    "BuildingSegmenter",
    "RoadSegmenter",
    "VehicleDetector",
]
