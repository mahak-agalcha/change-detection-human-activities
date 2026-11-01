from pathlib import Path
import rasterio

sample = next(Path("data/raw/spacenet7/sample").iterdir())
image_dir = sample / "images_masked"

image_2018 = next(image_dir.glob("*2018_01*.tif"))
image_2019 = next(image_dir.glob("*2019_01*.tif"))

with rasterio.open(image_2018) as before, rasterio.open(image_2019) as after:
    print("2018 dimensions:", before.width, before.height)
    print("2019 dimensions:", after.width, after.height)
    print("2018 CRS:", before.crs)
    print("2019 CRS:", after.crs)
    print("2018 bounds:", before.bounds)
    print("2019 bounds:", after.bounds)
    print("Same dimensions:", (before.width, before.height) == (after.width, after.height))
    print("Same CRS:", before.crs == after.crs)
    print("Same bounds:", before.bounds == after.bounds)