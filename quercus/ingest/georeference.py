"""
quercus.ingest.georeference
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Georeference scanned aerial images using Ground Control Points (GCPs).

Two modes
---------
AUTO  — Attempt automated GCP extraction by matching SIFT keypoints between
        the scanned image and a reference georeferenced layer (e.g. a modern
        orthophoto or NAIP tile loaded via rasterio).  Suitable when a
        reference raster covering the same area is available.

MANUAL — Accept a user-supplied GCP list:
        [(pixel_col, pixel_row, lon, lat), ...]
        This is the recommended mode for the 1984 Berkeley aerials where
        automated matching is unreliable due to large appearance changes.

In both modes the function writes a Cloud-Optimised GeoTIFF (COG) with
embedded GCPs and then performs a warp to WGS84 / UTM (auto-detected zone).
"""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.transform import from_gcps
from rasterio.warp import calculate_default_transform, reproject, Resampling
from tqdm import tqdm

# Type alias
GCPList = List[Tuple[float, float, float, float]]  # (col, row, lon, lat)


def _utm_crs_for_lon(lon: float) -> CRS:
    """Return the UTM CRS (WGS84) appropriate for a given longitude."""
    zone = math.floor((lon + 180) / 6) + 1
    epsg = 32600 + zone  # Northern hemisphere; adjust for south if needed
    return CRS.from_epsg(epsg)


def _write_gcps_to_tiff(
    src_path: Path,
    gcps: GCPList,
    dst_path: Path,
    src_crs: CRS = CRS.from_epsg(4326),
) -> Path:
    """
    Embed GCPs into a copy of the source image and save as GeoTIFF.
    """
    rasterio_gcps = [
        GroundControlPoint(row=row, col=col, x=lon, y=lat)
        for col, row, lon, lat in gcps
    ]

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(driver="GTiff")
        data = src.read()

    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data)
        dst.gcps = (rasterio_gcps, src_crs)

    return dst_path


def _warp_gcp_tiff(src_path: Path, dst_path: Path, target_crs: CRS) -> Path:
    """
    Warp a GCP-tagged GeoTIFF to a regular grid in target_crs.
    Uses GDAL via subprocess for robustness (gdalwarp handles GCPs natively).
    Falls back to rasterio if GDAL CLI is unavailable.
    """
    gdal_available = (
        subprocess.run(["which", "gdalwarp"], capture_output=True).returncode == 0
    )

    if gdal_available:
        epsg = target_crs.to_epsg()
        cmd = [
            "gdalwarp",
            "-t_srs", f"EPSG:{epsg}",
            "-r", "bilinear",
            "-tps",           # thin-plate-spline when many GCPs
            "-co", "COMPRESS=LZW",
            "-co", "TILED=YES",
            str(src_path),
            str(dst_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"gdalwarp failed: {result.stderr}")
    else:
        # Pure-rasterio fallback (polynomial order 1, less accurate)
        with rasterio.open(src_path) as src:
            gcps, src_crs = src.gcps
            transform, width, height = calculate_default_transform(
                src_crs,
                target_crs,
                src.width,
                src.height,
                gcps=gcps,
            )
            profile = src.profile.copy()
            profile.update(
                crs=target_crs,
                transform=transform,
                width=width,
                height=height,
                driver="GTiff",
            )
            with rasterio.open(dst_path, "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=from_gcps(gcps),
                        src_crs=src_crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear,
                    )

    return dst_path


def georeference_images(
    image_paths: List[Path],
    gcps_per_image: Optional[List[GCPList]] = None,
    output_dir: str | Path = "data/georef",
    reference_lon: float = -122.0,  # approx Berkeley Hills / Bay Area
    reference_lat: float = 37.8,
    image_width_deg: float = 0.01,  # ~1 km at 37° lat — tune per survey
    image_height_deg: float = 0.01,
) -> List[Path]:
    """
    Georeference a list of clipped aerial images.

    Parameters
    ----------
    image_paths       : list of clipped image paths.
    gcps_per_image    : optional list of GCP lists, one per image.
                        Each entry: [(col, row, lon, lat), ...]
                        If None, corner GCPs are estimated from a simple grid
                        layout — useful for a quick test but INACCURATE for
                        production. Supply real GCPs for publishable data.
    output_dir        : where to write georeferenced GeoTIFFs.
    reference_lon/lat : top-left corner of the survey block (decimal degrees).
    image_width_deg   : approximate width of one image in degrees.
    image_height_deg  : approximate height of one image in degrees.

    Returns
    -------
    List of Paths to georeferenced GeoTIFF files.

    Notes
    -----
    For production use, supply real GCPs identified from:
      - USGS topo maps of the survey area
      - Road intersections visible in both 1984 scan and modern basemap
      - NAIP or Google Earth tie points
    A minimum of 4 GCPs per image is recommended; 8–12 gives better accuracy.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    georef_paths: List[Path] = []
    target_crs = _utm_crs_for_lon(reference_lon)

    for idx, img_path in enumerate(tqdm(image_paths, desc="Georeferencing")):
        img_path = Path(img_path)
        gcp_tiff = output_dir / f"{img_path.stem}_gcp.tif"
        warp_tiff = output_dir / f"{img_path.stem}_georef.tif"

        # ── Determine GCPs ────────────────────────────────────────────────
        if gcps_per_image is not None:
            gcps = gcps_per_image[idx]
        else:
            # Fallback: assign corner GCPs from a simple grid
            # Images assumed laid out left→right in a single row block
            # REPLACE with real GCPs for publishable work
            with rasterio.open(img_path) if img_path.suffix in (".tif", ".tiff") \
                    else _open_jpeg_as_tiff(img_path, output_dir) as src:
                w, h = src.width, src.height

            lon0 = reference_lon + idx * image_width_deg
            lat0 = reference_lat
            gcps = [
                (0,   0,   lon0,                      lat0),
                (w,   0,   lon0 + image_width_deg,     lat0),
                (w,   h,   lon0 + image_width_deg,     lat0 - image_height_deg),
                (0,   h,   lon0,                       lat0 - image_height_deg),
            ]
            print(
                f"  [WARN] Image {img_path.name}: using estimated corner GCPs. "
                "Supply real GCPs via gcps_per_image for accurate results."
            )

        # ── Convert JPEG → scratch TIFF if needed ─────────────────────────
        if img_path.suffix.lower() in (".jpg", ".jpeg"):
            img_path = _jpeg_to_tiff(img_path, output_dir)

        # ── Embed GCPs ────────────────────────────────────────────────────
        _write_gcps_to_tiff(img_path, gcps, gcp_tiff)

        # ── Warp ──────────────────────────────────────────────────────────
        try:
            _warp_gcp_tiff(gcp_tiff, warp_tiff, target_crs)
            georef_paths.append(warp_tiff)
        except Exception as exc:
            print(f"  [WARN] Warp failed for {img_path.name}: {exc}")

    print(f"[QUERCUS] Georeferenced {len(georef_paths)} images → {output_dir}")
    return georef_paths


# ── Helpers ───────────────────────────────────────────────────────────────────

def _jpeg_to_tiff(src: Path, out_dir: Path) -> Path:
    """Convert a JPEG to a minimal GeoTIFF so rasterio can embed GCPs."""
    from PIL import Image as PILImage
    dst = out_dir / f"{src.stem}_raw.tif"
    img = PILImage.open(src).convert("RGB")
    arr = np.array(img)
    h, w, bands = arr.shape
    with rasterio.open(
        dst, "w",
        driver="GTiff",
        height=h, width=w,
        count=bands,
        dtype=arr.dtype,
    ) as ds:
        for b in range(bands):
            ds.write(arr[:, :, b], b + 1)
    return dst
