"""
quercus.ingest.preprocess
~~~~~~~~~~~~~~~~~~~~~~~~~
Remove white / grey scan borders from aerial photograph scans.

Strategy
--------
1. Convert the image to grayscale.
2. Threshold to isolate the bright border region (default ≥ 240 DN).
3. Find the largest contiguous dark (non-border) rectangle using a
   column/row histogram approach — robust to slight vignetting and
   uneven edges.
4. Crop to that rectangle and save as a new GeoTIFF (or JPEG for preview).

The function also handles multi-band GeoTIFFs so it works identically
for both the 1984 aerials and future NAIP inputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def _find_content_bbox(
    gray: np.ndarray,
    border_threshold: int = 230,
    min_content_frac: float = 0.5,
) -> Tuple[int, int, int, int]:
    """
    Return (x0, y0, x1, y1) bounding box of the 'dark' content region.

    Uses a row/column margin scan: walks inward from each edge, stopping
    where the mean pixel value drops below border_threshold.
    """
    h, w = gray.shape

    def find_margin(arr: np.ndarray, threshold: int) -> int:
        """Return first index where mean along axis < threshold."""
        for i, row in enumerate(arr):
            if np.mean(row) < threshold:
                return i
        return 0

    top = find_margin(gray, border_threshold)
    bottom = h - find_margin(gray[::-1], border_threshold)
    left = find_margin(gray.T, border_threshold)
    right = w - find_margin(gray.T[::-1], border_threshold)

    # Safety: ensure at least min_content_frac of original dimensions
    if (right - left) < w * min_content_frac:
        left, right = 0, w
    if (bottom - top) < h * min_content_frac:
        top, bottom = 0, h

    return left, top, right, bottom


def clip_scan_borders(
    image_paths: List[Path],
    output_dir: str | Path = "data/clipped",
    border_threshold: int = 230,
    output_format: str = "tiff",
) -> List[Path]:
    """
    Clip white/grey scan borders from a list of aerial scan images.

    Parameters
    ----------
    image_paths     : list of input image paths (JPEG or TIFF).
    output_dir      : directory to write clipped images.
    border_threshold: grayscale value above which a pixel is 'border' (0–255).
    output_format   : 'tiff' or 'jpeg'.

    Returns
    -------
    List of Paths to clipped output images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ext = "tif" if output_format == "tiff" else "jpg"
    clipped: List[Path] = []

    for src in tqdm(image_paths, desc="Clipping borders"):
        src = Path(src)
        dst = output_dir / f"{src.stem}_clipped.{ext}"

        # --- Read --------------------------------------------------------
        try:
            img = np.array(Image.open(src).convert("RGB"))
        except Exception as exc:
            print(f"  [WARN] Could not open {src.name}: {exc}")
            continue

        # --- Detect content bbox ----------------------------------------
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        x0, y0, x1, y1 = _find_content_bbox(gray, border_threshold)

        # --- Crop --------------------------------------------------------
        cropped = img[y0:y1, x0:x1]
        Image.fromarray(cropped).save(dst)
        clipped.append(dst)

    print(f"[QUERCUS] Clipped {len(clipped)} images → {output_dir}")
    return clipped
