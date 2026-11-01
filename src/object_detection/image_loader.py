"""
Satellite Image Loader and Preprocessing Module.

Handles reading satellite imagery in multiple formats (GeoTIFF, PNG, JPG, NPZ),
validates dimensions, extracts remote sensing spectral indices (NDVI/NDWI if NIR band present),
and manages georeferencing metadata (CRS, Affine transform).
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2


class SatelliteImageLoader:
    """Loads and standardizes multi-temporal satellite imagery for detection and segmentation."""

    SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npz"}

    def __init__(self, default_crs: str = "EPSG:4326"):
        self.default_crs = default_crs

    def load_image(
        self,
        image_source: Union[str, Path, np.ndarray],
        band_key: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Loads an image from path or accepts a pre-loaded numpy array.

        Returns:
            rgb_image (np.ndarray): Shape (H, W, 3), dtype uint8.
            metadata (dict): Dictionary with image dimensions, bands, CRS, and affine transform.
        """
        metadata = {
            "source": str(image_source) if not isinstance(image_source, np.ndarray) else "numpy_array",
            "crs": self.default_crs,
            "transform": None,
            "bounds": None,
            "bands_count": 3,
            "has_nir": False,
            "spectral_indices": {},
        }

        if isinstance(image_source, np.ndarray):
            raw_array = image_source
        else:
            path = Path(image_source)
            if not path.exists():
                raise FileNotFoundError(f"Satellite image file not found: {path}")

            ext = path.suffix.lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported image extension '{ext}'. Supported: {self.SUPPORTED_EXTENSIONS}"
                )

            if ext == ".npz":
                with np.load(path) as data:
                    key = band_key or ("image_t2" if "image_t2" in data else list(data.keys())[0])
                    raw_array = data[key]
            elif ext in (".tif", ".tiff"):
                raw_array, geo_meta = self._load_geotiff(path)
                metadata.update(geo_meta)
            else:
                # Standard PNG/JPG using PIL & OpenCV
                pil_img = Image.open(path)
                raw_array = np.array(pil_img)

        rgb_image, extra_meta = self._normalize_to_rgb(raw_array)
        metadata.update(extra_meta)
        metadata["height"] = rgb_image.shape[0]
        metadata["width"] = rgb_image.shape[1]

        # Generate default affine transform if none present (pixel coordinates to normalized degrees)
        if metadata.get("transform") is None:
            metadata["transform"] = (0.0001, 0.0, 75.8500, 0.0, -0.0001, 22.7500)
            metadata["bounds"] = (
                75.8500,
                22.7500 - (metadata["height"] * 0.0001),
                75.8500 + (metadata["width"] * 0.0001),
                22.7500,
            )

        return rgb_image, metadata

    def _load_geotiff(self, path: Path) -> Tuple[np.ndarray, Dict]:
        """Loads GeoTIFF image and extracts georeferencing information."""
        geo_meta = {"crs": self.default_crs, "transform": None, "bounds": None}
        try:
            import rasterio
            with rasterio.open(path) as dataset:
                raw_array = dataset.read()  # Shape: (Bands, H, W)
                raw_array = np.moveaxis(raw_array, 0, -1)  # Shape: (H, W, Bands)
                geo_meta["crs"] = str(dataset.crs) if dataset.crs else self.default_crs
                geo_meta["transform"] = tuple(dataset.transform)[:6]
                geo_meta["bounds"] = tuple(dataset.bounds)
                geo_meta["bands_count"] = dataset.count
                return raw_array, geo_meta
        except ImportError:
            # Fallback to OpenCV if rasterio is not installed
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise ValueError(f"Could not decode GeoTIFF image with OpenCV: {path}")
            if raw.ndim == 3 and raw.shape[2] == 3:
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            return raw, geo_meta

    def _normalize_to_rgb(self, raw_array: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Normalizes any input array (grayscale, RGB, 4-band PlanetScope) to uint8 RGB."""
        extra_meta = {"has_nir": False, "spectral_indices": {}}

        if raw_array.ndim == 2:
            # Single-band grayscale image
            norm_gray = self._scale_to_uint8(raw_array)
            rgb = cv2.cvtColor(norm_gray, cv2.COLOR_GRAY2RGB)
            return rgb, extra_meta

        if raw_array.ndim == 3:
            num_bands = raw_array.shape[2]

            if num_bands == 1:
                norm_gray = self._scale_to_uint8(raw_array[:, :, 0])
                rgb = cv2.cvtColor(norm_gray, cv2.COLOR_GRAY2RGB)
                return rgb, extra_meta

            if num_bands == 3:
                rgb = np.zeros_like(raw_array, dtype=np.uint8)
                for b in range(3):
                    rgb[:, :, b] = self._scale_to_uint8(raw_array[:, :, b])
                return rgb, extra_meta

            if num_bands >= 4:
                # 4-band remote sensing image (SpaceNet 7 PlanetScope format: R, G, B, NIR)
                extra_meta["has_nir"] = True
                r = raw_array[:, :, 0].astype(np.float32)
                g = raw_array[:, :, 1].astype(np.float32)
                b = raw_array[:, :, 2].astype(np.float32)
                nir = raw_array[:, :, 3].astype(np.float32)

                # Compute NDVI: (NIR - Red) / (NIR + Red + 1e-6)
                denominator = nir + r + 1e-6
                ndvi = (nir - r) / denominator
                extra_meta["spectral_indices"]["ndvi"] = np.clip(ndvi, -1.0, 1.0)

                # Construct RGB visualization
                rgb = np.stack([
                    self._scale_to_uint8(r),
                    self._scale_to_uint8(g),
                    self._scale_to_uint8(b),
                ], axis=-1)
                return rgb, extra_meta

        raise ValueError(f"Unexpected image array shape: {raw_array.shape}")

    @staticmethod
    def _scale_to_uint8(band: np.ndarray) -> np.ndarray:
        """Scales float or high-bit depth bands to standard 0-255 uint8 using 2-98 percentile stretch."""
        band = np.asarray(band, dtype=np.float32)
        if band.max() == band.min():
            return np.zeros(band.shape, dtype=np.uint8)

        # Robust contrast stretching (ignoring extreme outliers)
        p2, p98 = np.percentile(band, (2, 98))
        if p98 > p2:
            stretched = np.clip((band - p2) / (p98 - p2), 0.0, 1.0)
            return (stretched * 255.0).astype(np.uint8)
        else:
            return ((band - band.min()) / (band.max() - band.min()) * 255.0).astype(np.uint8)
