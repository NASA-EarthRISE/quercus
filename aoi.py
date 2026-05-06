"""
quercus.utils.aoi
-----------------
AOI (Area of Interest) filtering utilities.

Users can supply an AOI as:
  - A path to a GeoJSON file
  - A Shapely geometry object
  - A dict (GeoJSON geometry)
  - None (no filtering)

All coordinates in WGS84 (EPSG:4326).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union


def load_aoi(aoi) -> Optional[object]:
    """Load AOI from file path, dict, or Shapely geometry. Returns Shapely geometry or None."""
    if aoi is None:
        return None
    try:
        from shapely.geometry import shape
        from shapely import wkt
    except ImportError:
        raise ImportError("shapely required for AOI filtering: pip install shapely")

    if hasattr(aoi, "geoms") or hasattr(aoi, "exterior"):
        # Already a Shapely geometry
        return aoi

    if isinstance(aoi, dict):
        # GeoJSON geometry dict
        if aoi.get("type") == "FeatureCollection":
            from shapely.ops import unary_union
            return unary_union([shape(f["geometry"]) for f in aoi["features"]])
        if aoi.get("type") == "Feature":
            return shape(aoi["geometry"])
        return shape(aoi)

    if isinstance(aoi, (str, Path)):
        import json
        with open(aoi) as f:
            data = json.load(f)
        return load_aoi(data)

    raise TypeError(f"Cannot load AOI from {type(aoi)}. Pass a file path, GeoJSON dict, or Shapely geometry.")


def bbox_intersects_aoi(west: float, south: float, east: float, north: float,
                         aoi_geom) -> bool:
    """Return True if the bbox intersects the AOI geometry."""
    if aoi_geom is None:
        return True
    from shapely.geometry import box
    return box(west, south, east, north).intersects(aoi_geom)


def georef_bbox_wgs84(tif_path) -> tuple:
    """Return (west, south, east, north) in WGS84 for a GeoTIFF."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.crs import CRS
    with rasterio.open(tif_path) as src:
        return transform_bounds(src.crs, CRS.from_epsg(4326), *src.bounds)
