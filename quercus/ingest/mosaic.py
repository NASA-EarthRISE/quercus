"""
quercus.ingest.mosaic
~~~~~~~~~~~~~~~~~~~~~
Merge a list of georeferenced GeoTIFFs into a single seamless mosaic.

Uses rasterio.merge (gdal_merge equivalent) with 'first' pixel strategy
(later tiles fill gaps only).  Writes a Cloud-Optimized GeoTIFF (COG).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling


def build_mosaic(
    georef_paths: List[Path],
    output_path: str | Path = "data/outputs/mosaic.tif",
    method: str = "first",
    nodata: Optional[float] = 0,
    overview_levels: List[int] = [2, 4, 8, 16, 32],
) -> Path:
    """
    Merge georeferenced tiles into a single mosaic GeoTIFF.

    Parameters
    ----------
    georef_paths   : list of georeferenced GeoTIFF paths.
    output_path    : output mosaic path.
    method         : rasterio merge method ('first', 'last', 'min', 'max').
    nodata         : nodata value for the output.
    overview_levels: pyramid overview levels to embed (for fast display).

    Returns
    -------
    Path to the mosaic GeoTIFF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[QUERCUS] Building mosaic from {len(georef_paths)} tiles …")

    open_files = [rasterio.open(p) for p in georef_paths]
    try:
        mosaic_arr, mosaic_transform = merge(
            open_files,
            method=method,
            nodata=nodata,
        )
        profile = open_files[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic_arr.shape[1],
            width=mosaic_arr.shape[2],
            transform=mosaic_transform,
            nodata=nodata,
            compress="lzw",
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic_arr)
            # Build internal overviews for fast display
            dst.build_overviews(overview_levels, Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")

    finally:
        for f in open_files:
            f.close()

    size_mb = output_path.stat().st_size / 1e6
    print(f"[QUERCUS] Mosaic saved → {output_path}  ({size_mb:.1f} MB)")
    return output_path
