"""
quercus.segment.sam_runner
--------------------------
Run SamGeo (segment-geospatial / opengeoai.org) on georeferenced GeoTIFFs.

Key design decisions for 1984 greyscale aerial photography:
- CLAHE contrast enhancement before SAM (amplifies local edges)
- 3-band RGB conversion (greyscale stacked R=G=B)
- points_per_side=64 for dense coverage
- Low iou/stability thresholds (0.50) for low-contrast imagery
- crop_n_layers=1 to catch small features
- area_m2 computed from polygon geometry (tiff_to_vector leaves area=0)
"""
from __future__ import annotations
import gc
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
import geopandas as gpd
from tqdm import tqdm

try:
    from samgeo import SamGeo
    SAMGEO_AVAILABLE = True
except ImportError:
    SAMGEO_AVAILABLE = False


def _to_3band_uint8(tif_path: Path, out_dir: Path, max_px: int = 1024) -> Path:
    """
    Convert GeoTIFF to 3-band uint8 RGB with CLAHE contrast enhancement.

    CLAHE is critical for 1984 greyscale aerials -- it amplifies local
    contrast so SAM can find edges that are invisible in the original.
    Resizes to max_px on longest side to fit in Colab T4 RAM.
    """
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / Path(tif_path).name

    with rasterio.open(tif_path) as src:
        scale = min(1.0, max_px / max(src.width, src.height))
        new_w = max(1, int(src.width  * scale))
        new_h = max(1, int(src.height * scale))
        data  = src.read(out_shape=(src.count, new_h, new_w),
                         resampling=Resampling.bilinear)
        t = src.transform
        new_transform = Affine(t.a / scale, t.b, t.c, t.d, t.e / scale, t.f)
        profile = src.profile.copy()

        # Normalise to uint8
        if data.dtype != np.uint8:
            mn, mx = data.min(), data.max()
            data = ((data - mn) / (mx - mn + 1e-6) * 255).astype(np.uint8)

        # CLAHE on first band (aerials are effectively greyscale)
        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(data[0])

        # Stack to 3-band RGB
        rgb = np.stack([enhanced, enhanced, enhanced], axis=0)

        profile.update(count=3, dtype="uint8", width=new_w, height=new_h,
                       transform=new_transform, compress="lzw")
        for k in ("blockxsize", "blockysize", "tiled"):
            profile.pop(k, None)

        with rasterio.open(dst, "w", **profile) as out:
            out.write(rgb)

    return dst


def _run_sam_on_tif(
    sam: "SamGeo",
    ready_tif: Path,
    output_dir: Path,
    source_label: str,
    original_name: str,
) -> Tuple[gpd.GeoDataFrame, Path]:
    """Run SAM on a prepared TIF, vectorise, compute real area. Returns (gdf, gjson_path)."""
    mask_tif  = output_dir / (ready_tif.stem + "_mask.tif")
    gjson_out = output_dir / (ready_tif.stem + "_segments.geojson")

    sam.generate(source=str(ready_tif), output=str(mask_tif),
                 batch=False, foreground=True)
    sam.tiff_to_vector(str(mask_tif), str(gjson_out))

    try:
        gdf = gpd.read_file(gjson_out)
    except Exception:
        gdf = gpd.GeoDataFrame()

    if len(gdf) > 0:
        # Compute real area (tiff_to_vector leaves area column as 0)
        try:
            utm = gdf.estimate_utm_crs()
            gdf["area_m2"] = gdf.to_crs(utm).geometry.area
        except Exception:
            gdf["area_m2"] = gdf.geometry.area

        gdf["object_id"]     = [f"{source_label}_{ready_tif.stem}_{i:06d}"
                                 for i in range(len(gdf))]
        gdf["image"]         = original_name
        gdf["source"]        = source_label
        gdf["centroid_lon"]  = gdf.geometry.centroid.x.round(8)
        gdf["centroid_lat"]  = gdf.geometry.centroid.y.round(8)
        gdf["stability"]     = gdf.get("stability_score", 1.0)
        gdf["predicted_iou"] = gdf.get("predicted_iou",   1.0)
        gdf.to_file(gjson_out, driver="GeoJSON")

    return gdf, gjson_out


def run_sam_pipeline(
    georef_paths: List[Union[str, Path]],
    output_dir: Union[str, Path] = "data/sam_outputs",
    model_type: str = "vit_h",
    device: Optional[str] = None,
    points_per_side: int = 64,
    points_per_batch: int = 128,
    pred_iou_thresh: float = 0.50,
    stability_score_thresh: float = 0.50,
    stability_score_offset: float = 0.50,
    box_nms_thresh: float = 0.50,
    crop_n_layers: int = 1,
    crop_nms_thresh: float = 0.50,
    crop_overlap_ratio: float = 0.50,
    crop_n_points_downscale_factor: int = 2,
    min_mask_region_area: int = 50,
    max_px: int = 1024,
    source_label: str = "1984_aerial",
) -> List[Tuple[gpd.GeoDataFrame, Path]]:
    """
    Run SamGeo on georeferenced GeoTIFFs.

    Parameters
    ----------
    georef_paths    : list of georeferenced GeoTIFF paths.
    output_dir      : directory for SAM outputs.
    model_type      : 'vit_h' (best) or 'vit_b' (faster, less RAM).
    points_per_side : SAM prompt grid density. 64 = 4096 points.
    pred_iou_thresh : quality threshold (0.50 = keep borderline segments).
    stability_score_thresh : stability threshold (0.50 for greyscale).
    crop_n_layers   : run SAM on image sub-crops too (finds small features).
    min_mask_region_area : minimum segment area in pixels.
    max_px          : resize to max_px before SAM (RAM control).
    source_label    : tag in object_id column ('1984_aerial' or 'naip').

    Returns
    -------
    List of (GeoDataFrame, geojson_path) tuples, one per input image.
    """
    if not SAMGEO_AVAILABLE:
        raise ImportError(
            "segment-geospatial not installed.\n"
            "Run: pip install segment-geospatial")

    import torch

    output_dir    = Path(output_dir)
    sam_ready_dir = output_dir / "sam_ready"
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[SAM] ({model_type}) on {device} "
          f"pts={points_per_side} iou={pred_iou_thresh} "
          f"stab={stability_score_thresh} crop_layers={crop_n_layers}")

    sam = SamGeo(
        model_type=model_type,
        device=device,
        automatic=True,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=stability_score_offset,
        box_nms_thresh=box_nms_thresh,
        crop_n_layers=crop_n_layers,
        crop_nms_thresh=crop_nms_thresh,
        crop_overlap_ratio=crop_overlap_ratio,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
    )

    results = []
    for img_path in tqdm(georef_paths, desc="SamGeo"):
        img_path  = Path(img_path)
        ready_tif = _to_3band_uint8(img_path, sam_ready_dir, max_px=max_px)
        gdf, gjson = _run_sam_on_tif(
            sam, ready_tif, output_dir, source_label, img_path.name)
        print(f"  {img_path.name}: {len(gdf)} segments")
        results.append((gdf, gjson))
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"[SAM] Done. {len(results)} images, "
          f"{sum(len(g) for g, _ in results):,} total segments.")
    return results
