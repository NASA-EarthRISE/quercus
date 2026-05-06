"""
quercus.ingest.download
-----------------------
Download specific frames from Berkeley Digital Collections (TIND IIIF).

AOI filtering: if aoi is provided, each frame's estimated geographic bbox is
checked against the AOI. Frames outside are skipped and reported.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import List, Optional, Union
import requests
from tqdm import tqdm

BASE           = "https://digicoll.lib.berkeley.edu"
FILES_API      = BASE + "/api/v1/record/{record_id}/files"
DEFAULT_RECORD = "312848"


def _get_all_files(record_id: str, timeout: int = 30) -> List[dict]:
    r = requests.get(FILES_API.format(record_id=record_id), timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list):
        return data
    for key in ("files", "results", "hits", "data"):
        if key in data and isinstance(data[key], list):
            return data[key]
    return []


def _jpeg_entries_in_order(all_entries: List[dict]) -> List[dict]:
    jpegs = [e for e in all_entries
             if isinstance(e, dict) and e.get("format", "").lower() in (".jpg", ".jpeg")]
    jpegs.sort(key=lambda e: int(e.get("order", 9999)))
    return jpegs


def _estimated_bbox(cv_index: int, anchor_lon: float, anchor_lat0: float,
                    anchor_lat1: float, frame_width_deg: float,
                    frame_step_deg: float) -> tuple:
    """Estimate (west, south, east, north) for a canvas index."""
    lon0 = anchor_lon + cv_index * frame_step_deg
    lon1 = lon0 + frame_width_deg
    return lon0, anchor_lat0, lon1, anchor_lat1


def fetch_iiif_images(
    page_ids: List[int],
    output_dir: Union[str, Path] = "data/raw",
    record_id: str = DEFAULT_RECORD,
    aoi=None,
    anchor_lon: float = -122.254,
    anchor_lat0: float = 37.844,
    anchor_lat1: float = 37.852,
    frame_width_deg: float = 0.008,
    frame_step_deg: float = 0.0056,
    delay: float = 0.5,
    overwrite: bool = False,
    timeout: int = 60,
) -> List[Path]:
    """
    Download specific frames by canvas index from Berkeley Digital Collections.

    Parameters
    ----------
    page_ids        : list of canvas indices (cv= values from the viewer URL).
                      e.g. [715, 716, 717, 718, 1104, 1105, 1670, 1671, 1672, 1673, 1674]
    output_dir      : local directory for downloaded files.
    record_id       : Berkeley TIND record number (default 312848 = 1984 aerials).
    aoi             : Area of Interest filter. GeoJSON file path, Shapely geometry,
                      or dict. Frames whose estimated bbox does not intersect the
                      AOI are skipped and reported. Pass None to disable.
    anchor_lon      : longitude of the western edge of the first frame (for AOI check).
    anchor_lat0/1   : south/north edges of all frames (for AOI check).
    frame_width_deg : approximate width of one frame in degrees.
    frame_step_deg  : step between frames (frame_width * overlap_factor).
    delay           : polite pause between requests (seconds).
    overwrite       : re-download existing files.
    timeout         : HTTP timeout in seconds.

    Returns
    -------
    List of Paths to downloaded files.
    """
    from quercus.utils.aoi import load_aoi, bbox_intersects_aoi

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aoi_geom = load_aoi(aoi)
    if aoi_geom is not None:
        print(f"[D1] AOI filter active: {aoi_geom.geom_type}")

    print(f"[D1] Fetching file list for record {record_id} ...")
    all_entries = _get_all_files(record_id, timeout)
    print(f"[D1] Total API entries: {len(all_entries)}")
    jpegs = _jpeg_entries_in_order(all_entries)
    print(f"[D1] JPEG frames available: {len(jpegs)}")

    if not jpegs:
        raise ValueError(f"No JPEG entries for record {record_id}.")

    # Build cv_index -> entry lookup
    cv_map = {int(e.get("order", 1)) - 1: e for e in jpegs}
    page_ids = sorted(set(page_ids))
    print(f"[D1] Requested frames: {page_ids}")

    downloaded: List[Path] = []
    skipped_aoi: List[int] = []
    skipped_missing: List[int] = []

    for cv in tqdm(page_ids, desc="Downloading"):
        # AOI check using estimated bbox
        if aoi_geom is not None:
            west, south, east, north = _estimated_bbox(
                cv, anchor_lon, anchor_lat0, anchor_lat1,
                frame_width_deg, frame_step_deg)
            if not bbox_intersects_aoi(west, south, east, north, aoi_geom):
                skipped_aoi.append(cv)
                continue

        entry = cv_map.get(cv)
        if entry is None:
            skipped_missing.append(cv)
            continue

        fn  = entry.get("name", "frame") + entry.get("format", ".jpg")
        url = entry["url"]
        dst = output_dir / ("cv%04d.jpg" % cv)

        if dst.exists() and not overwrite:
            downloaded.append(dst)
            continue

        try:
            r = requests.get(url, timeout=timeout, stream=True)
            r.raise_for_status()
            with open(dst, "wb") as fh:
                for chunk in r.iter_content(8192):
                    fh.write(chunk)
            downloaded.append(dst)
        except Exception as exc:
            print(f"  [WARN] cv={cv} ({fn}): {exc}")
        time.sleep(delay)

    print(f"[D1] Downloaded: {len(downloaded)}/{len(page_ids)}")
    if skipped_aoi:
        print(f"[D1] Skipped (outside AOI): {skipped_aoi}")
    if skipped_missing:
        print(f"[D1] Skipped (not in record): {skipped_missing}")

    return downloaded
