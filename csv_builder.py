"""
quercus.segment.csv_builder
----------------------------
Assemble SAM results + GEE NDVI into a master CSV.

Handles UTM -> WGS84 centroid reprojection before sending to GEE
(the v6 bug where centroids in UTM metres were sent to GEE as lat/lon).
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd
from tqdm import tqdm


def build_objects_csv(
    sam_results: List[Tuple[gpd.GeoDataFrame, Path]],
    output_csv: Union[str, Path] = "data/outputs/quercus_objects.csv",
    ndvi_year: int = 1984,
    input_crs: str = "EPSG:32610",
    gee_project: Optional[str] = None,
    skip_ndvi: bool = False,
) -> Path:
    """
    Build master CSV from SAM results + GEE NDVI.

    Parameters
    ----------
    sam_results  : list of (GeoDataFrame, geojson_path) from run_sam_pipeline.
    output_csv   : output CSV path.
    ndvi_year    : year for the Landsat NDVI composite.
    input_crs    : CRS of the SAM output geometries (default UTM Zone 10N).
    gee_project  : GEE project ID.
    skip_ndvi    : skip GEE NDVI sampling (useful for offline testing).

    Returns
    -------
    Path to the output CSV.
    """
    from quercus.segment.gee_ndvi import fetch_gee_ndvi

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print("[CSV] Building master CSV ...")
    rows = []
    for gdf, _ in tqdm(sam_results, desc="Reading SAM results"):
        if gdf is None or len(gdf) == 0:
            continue
        for _, feat in gdf.iterrows():
            rows.append({
                "object_id":     feat.get("object_id", ""),
                "image":         feat.get("image", ""),
                "source":        feat.get("source", ""),
                "centroid_lon":  feat.get("centroid_lon", feat.geometry.centroid.x),
                "centroid_lat":  feat.get("centroid_lat", feat.geometry.centroid.y),
                "area_m2":       float(feat.get("area_m2", 0)),
                "stability":     float(feat.get("stability", 1.0)),
                "predicted_iou": float(feat.get("predicted_iou", 1.0)),
                "ndvi":          None,
                "geojson":       feat.geometry.wkt,
            })

    df = pd.DataFrame(rows)
    print(f"[CSV] Total objects: {len(df):,}")

    if not skip_ndvi and len(df) > 0:
        # Reproject centroids to WGS84 for GEE (they are in UTM)
        from shapely import wkt as swkt
        geoms = [swkt.loads(g) for g in df["geojson"]]
        gdf_all = gpd.GeoDataFrame(df, geometry=geoms, crs=input_crs)
        gdf_wgs = gdf_all.to_crs("EPSG:4326")
        lons = gdf_wgs.geometry.centroid.x.tolist()
        lats = gdf_wgs.geometry.centroid.y.tolist()
        centroids = list(zip(lons, lats))
        ndvi_map  = fetch_gee_ndvi(centroids, year=ndvi_year, gee_project=gee_project)
        df["ndvi"] = [ndvi_map.get((lon, lat)) for lon, lat in centroids]
        print(f"[CSV] NDVI populated for "
              f"{df['ndvi'].notna().sum():,}/{len(df):,} objects.")

    df.to_csv(output_csv, index=False)
    print(f"[CSV] -> {output_csv}  ({len(df):,} rows)")
    return output_csv
