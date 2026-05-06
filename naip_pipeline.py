"""
quercus.segment.naip_pipeline
------------------------------
Fetch NAIP imagery from GEE matching each 1984 tile bbox, then run SAM.

NAIP is NOT annual. California availability: 2009,2012,2014,2016,2018,2020,2022.
AOI filtering: if aoi is provided, tiles whose bbox does not intersect
the AOI are skipped and reported.
"""
from __future__ import annotations
import gc
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
import geopandas as gpd
from tqdm import tqdm

try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False


def _get_naip_for_bbox(
    west: float, south: float, east: float, north: float,
    year_start: int, year_end: int,
    gee_project: Optional[str] = None,
) -> Tuple[Optional[object], Optional[int]]:
    """Fetch best available NAIP mosaic from GEE for a bbox and year range."""
    try:
        ee.data.getIamPolicy("projects/earthengine-public")
    except Exception:
        ee.Initialize(**({"project": gee_project} if gee_project else {}))

    region = ee.Geometry.Rectangle([west, south, east, north])
    col = (ee.ImageCollection("USDA/NAIP/DOQQ")
           .filterBounds(region)
           .filterDate(f"{year_start}-01-01", f"{year_end}-12-31")
           .filter(ee.Filter.listContains("system:band_names", "N"))
           .sort("system:time_start", False))

    count = col.size().getInfo()
    print(f"  [NAIP] Found {count} images in [{year_start}-{year_end}]")

    if count == 0:
        return None, None

    img = col.mosaic().clip(region)
    import datetime
    latest = col.first().get("system:time_start").getInfo()
    year   = datetime.datetime.fromtimestamp(latest / 1000).year
    print(f"  [NAIP] Using mosaic, most recent year: {year}")
    return img, year


def _export_naip_to_tif(
    naip_img,
    west: float, south: float, east: float, north: float,
    out_path: Path,
    scale: float = 1.0,
) -> Path:
    """Download GEE NAIP image as a local GeoTIFF."""
    import requests as req

    region = ee.Geometry.Rectangle([west, south, east, north])
    url = naip_img.getDownloadURL({
        "bands": ["R", "G", "B", "N"],
        "region": region,
        "scale": scale,
        "format": "GEO_TIFF",
        "crs": "EPSG:4326",
    })
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [NAIP] Downloading GeoTIFF ...")
    r = req.get(url, stream=True, timeout=300)
    r.raise_for_status()
    with open(out_path, "wb") as fh:
        for chunk in r.iter_content(65536):
            fh.write(chunk)
    mb = out_path.stat().st_size / 1e6
    print(f"  [NAIP] Saved -> {out_path}  ({mb:.1f} MB)")
    return out_path


def fetch_naip_and_run_sam(
    georef_paths: List[Union[str, Path]],
    output_dir: Union[str, Path] = "data/naip_outputs",
    naip_year_start: int = 2018,
    naip_year_end: int = 2022,
    naip_scale: float = 1.0,
    aoi=None,
    gee_project: Optional[str] = None,
    model_type: str = "vit_h",
    device: Optional[str] = None,
    points_per_side: int = 64,
    pred_iou_thresh: float = 0.50,
    stability_score_thresh: float = 0.50,
    crop_n_layers: int = 1,
    min_mask_region_area: int = 50,
    max_px: int = 1024,
) -> List[Tuple[gpd.GeoDataFrame, Path]]:
    """
    Fetch NAIP from GEE for each georef tile bbox, run SAM, return results.

    Parameters
    ----------
    georef_paths      : list of georeferenced 1984 GeoTIFF paths.
    output_dir        : directory for all NAIP outputs.
    naip_year_start   : start year for NAIP search.
    naip_year_end     : end year for NAIP search.
                        California availability: 2009,2012,2014,2016,2018,2020,2022.
    naip_scale        : metres per pixel (1.0 = native NAIP resolution).
    aoi               : AOI filter. Tiles outside this area are skipped.
                        GeoJSON path, Shapely geometry, or None.
    gee_project       : GEE project ID.
    model_type        : SAM model ('vit_h' or 'vit_b').
    points_per_side   : SAM prompt grid density.
    max_px            : max image dimension before SAM (RAM control).

    Returns
    -------
    List of (GeoDataFrame, geojson_path) tuples.
    """
    if not GEE_AVAILABLE:
        raise ImportError("pip install earthengine-api")

    try:
        from samgeo import SamGeo
    except ImportError:
        raise ImportError("pip install segment-geospatial")

    import torch
    from quercus.segment.sam_runner import _to_3band_uint8, _run_sam_on_tif
    from quercus.utils.aoi import load_aoi, bbox_intersects_aoi

    output_dir    = Path(output_dir)
    naip_tif_dir  = output_dir / "naip_tifs"
    naip_sam_dir  = output_dir / "sam_outputs"
    naip_ready_dir= output_dir / "sam_ready"
    for d in [naip_tif_dir, naip_sam_dir, naip_ready_dir]:
        d.mkdir(parents=True, exist_ok=True)

    aoi_geom = load_aoi(aoi)
    if aoi_geom is not None:
        print(f"[NAIP] AOI filter active: {aoi_geom.geom_type}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[NAIP] SamGeo ({model_type}) on {device}")
    sam = SamGeo(
        model_type=model_type, device=device, automatic=True,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        min_mask_region_area=min_mask_region_area,
    )

    results = []
    skipped_aoi = []

    for georef_path in tqdm(georef_paths, desc="NAIP pipeline"):
        georef_path = Path(georef_path)
        stem = georef_path.stem.replace("_georef", "")
        print(f"\n[NAIP] {georef_path.name}")

        # Get WGS84 bbox
        with rasterio.open(georef_path) as src:
            west, south, east, north = transform_bounds(
                src.crs, CRS.from_epsg(4326), *src.bounds)

        # AOI check
        if aoi_geom is not None and not bbox_intersects_aoi(
                west, south, east, north, aoi_geom):
            skipped_aoi.append(stem)
            print(f"  Skipped: outside AOI")
            continue

        # Fetch NAIP
        naip_img, naip_year = _get_naip_for_bbox(
            west, south, east, north,
            naip_year_start, naip_year_end, gee_project)

        if naip_img is None:
            print(f"  Skipped: no NAIP found in [{naip_year_start}-{naip_year_end}]")
            continue

        # Download tile
        naip_tif = naip_tif_dir / f"{stem}_naip_{naip_year}.tif"
        if not naip_tif.exists():
            _export_naip_to_tif(naip_img, west, south, east, north,
                                naip_tif, scale=naip_scale)
        else:
            print(f"  Using cached {naip_tif.name}")

        # Prepare for SAM (NAIP is colour -- CLAHE still helps for edges)
        ready_tif = _to_3band_uint8(naip_tif, naip_ready_dir, max_px=max_px)

        # Run SAM
        source_label = f"naip_{naip_year}"
        gdf, gjson   = _run_sam_on_tif(
            sam, ready_tif, naip_sam_dir, source_label, naip_tif.name)
        gdf["naip_year"] = naip_year
        if len(gdf):
            gdf.to_file(gjson, driver="GeoJSON")

        print(f"  {naip_tif.name}: {len(gdf)} segments")
        results.append((gdf, gjson))
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    if skipped_aoi:
        print(f"\n[NAIP] Skipped (outside AOI): {skipped_aoi}")

    print(f"[NAIP] Done. {len(results)} tiles processed, "
          f"{sum(len(g) for g, _ in results):,} total segments.")
    return results
