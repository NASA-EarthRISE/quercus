"""
quercus.segment.gee_ndvi
~~~~~~~~~~~~~~~~~~~~~~~~
Fetch annual cloud-free Landsat NDVI for a list of centroid coordinates
using the Google Earth Engine Python API.

Authentication
--------------
Run once before using:
    import ee
    ee.Authenticate()
    ee.Initialize(project='your-gee-project')

Or in Colab:
    ee.Authenticate()
    ee.Initialize(project='your-gee-project')

Landsat selection
-----------------
Year 1984 → Landsat 5 TM  (bands: NIR=B4, RED=B3)
Year 2000+ → Landsat 7/8/9 (auto-selected by year)

NDVI = (NIR - RED) / (NIR + RED)
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

# GEE import wrapped so the module can be imported without GEE installed
try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False


# ── Landsat collection config by year ────────────────────────────────────────
def _get_landsat_config(year: int) -> Tuple[str, str, str]:
    """Return (collection_id, nir_band, red_band) for a given year."""
    if year <= 1983:
        raise ValueError("Landsat 4 / pre-1984 not fully supported. Use year >= 1984.")
    elif year <= 2011:
        return "LANDSAT/LT05/C02/T1_L2", "SR_B4", "SR_B3"   # Landsat 5 TM
    elif year <= 2013:
        return "LANDSAT/LE07/C02/T1_L2", "SR_B4", "SR_B3"   # Landsat 7 ETM+
    else:
        return "LANDSAT/LC08/C02/T1_L2", "SR_B5", "SR_B4"   # Landsat 8/9 OLI


def _build_annual_ndvi(year: int, region: ee.Geometry) -> ee.Image:
    """Build a cloud-free annual median NDVI image."""
    coll_id, nir, red = _get_landsat_config(year)
    start = f"{year}-01-01"
    end = f"{year}-12-31"

    collection = (
        ee.ImageCollection(coll_id)
        .filterDate(start, end)
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUD_COVER", 20))
    )

    # Apply scale factors (Landsat Collection 2)
    def apply_scale(img):
        optical = img.select(["SR_B.*"]).multiply(0.0000275).add(-0.2)
        return img.addBands(optical, overwrite=True)

    collection = collection.map(apply_scale)

    nir_img = collection.select(nir).median()
    red_img = collection.select(red).median()
    ndvi = nir_img.subtract(red_img).divide(nir_img.add(red_img)).rename("NDVI")
    return ndvi


def fetch_gee_ndvi(
    centroids: List[Tuple[float, float]],  # [(lon, lat), ...]
    year: int = 1984,
    buffer_m: int = 15,                    # ~half Landsat pixel (30m)
    batch_size: int = 100,
    gee_project: Optional[str] = None,
    max_retries: int = 3,
) -> Dict[Tuple[float, float], Optional[float]]:
    """
    Fetch NDVI values from GEE for a list of centroid coordinates.

    Parameters
    ----------
    centroids   : list of (lon, lat) tuples in WGS84.
    year        : target year for the annual Landsat composite.
    buffer_m    : buffer radius around centroid in metres (≥15 for Landsat).
    batch_size  : number of centroids per GEE request (avoid timeout).
    gee_project : GEE project ID (falls back to initialized default).
    max_retries : number of retry attempts on GEE errors.

    Returns
    -------
    Dict mapping (lon, lat) → NDVI float or None if retrieval failed.
    """
    if not GEE_AVAILABLE:
        raise ImportError(
            "earthengine-api not installed. Run: pip install earthengine-api"
        )

    try:
        ee.data.getIamPolicy("projects/earthengine-public")
    except Exception:
        print("[QUERCUS] Initializing GEE …")
        kwargs = {"project": gee_project} if gee_project else {}
        ee.Initialize(**kwargs)

    ndvi_map: Dict[Tuple[float, float], Optional[float]] = {}

    # Build a bounding region from all centroids
    lons = [c[0] for c in centroids]
    lats = [c[1] for c in centroids]
    region = ee.Geometry.Rectangle([min(lons), min(lats), max(lons), max(lats)])

    print(f"[QUERCUS] Building GEE NDVI composite for {year} …")
    ndvi_img = _build_annual_ndvi(year, region)

    print(f"[QUERCUS] Sampling NDVI for {len(centroids)} centroids …")
    for batch_start in range(0, len(centroids), batch_size):
        batch = centroids[batch_start : batch_start + batch_size]

        fc = ee.FeatureCollection(
            [
                ee.Feature(
                    ee.Geometry.Point(lon, lat).buffer(buffer_m),
                    {"lon": lon, "lat": lat},
                )
                for lon, lat in batch
            ]
        )

        sampled = ndvi_img.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=30,
        )

        for attempt in range(max_retries):
            try:
                results = sampled.getInfo()
                break
            except Exception as exc:
                if attempt == max_retries - 1:
                    print(f"  [WARN] GEE batch failed after {max_retries} tries: {exc}")
                    results = None
                else:
                    time.sleep(2 ** attempt)

        if results is None:
            for lon, lat in batch:
                ndvi_map[(lon, lat)] = None
            continue

        for feat in results.get("features", []):
            props = feat.get("properties", {})
            lon = props.get("lon")
            lat = props.get("lat")
            ndvi_val = props.get("mean")
            ndvi_map[(lon, lat)] = float(ndvi_val) if ndvi_val is not None else None

    print(f"[QUERCUS] NDVI retrieved for {sum(v is not None for v in ndvi_map.values())} / {len(centroids)} centroids.")
    return ndvi_map
