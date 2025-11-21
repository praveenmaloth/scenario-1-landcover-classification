import rasterio
import numpy as np
from pyproj import Transformer
from rasterio.windows import Window

def read_patch_at_latlon(raster_path, lat, lon, size_px=128, nodata=None):
    with rasterio.open(raster_path) as src:
        transformer = Transformer.from_crs(4326, src.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        col, row = src.index(x, y)
        half = size_px // 2
        w = Window(col - half, row - half, size_px, size_px)
        patch = src.read(1, window=w, boundless=True, fill_value=nodata)
        return patch

def patch_mode_and_stats(patch, nodata):
    arr = patch.flatten()
    mask = arr != nodata
    valid = arr[mask]
    total = arr.size
    nodata_frac = 1.0 - (valid.size / total)
    if valid.size == 0:
        return int(nodata if nodata is not None else -1), 0.0, nodata_frac
    counts = np.bincount(valid.astype(np.int64))
    dominant = int(np.argmax(counts))
    dominant_prop = counts[dominant] / valid.size
    return dominant, dominant_prop, nodata_frac