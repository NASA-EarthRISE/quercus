"""
quercus.classify.oak_extractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Heuristically identify which K-Means cluster(s) most likely correspond to
oak trees and export them as a separate GeoPackage + map.

Heuristic rules (in priority order)
------------------------------------
1. High NDVI  (oaks are deciduous/evergreen with strong NIR reflectance)
2. Medium-to-large area  (tree canopy > shrub or grass patches)
3. High SAM stability  (clean crown boundaries vs. grass or shadow smear)

The function also generates a publication-ready map showing:
  - all objects in grey
  - candidate oak objects colour-coded by cluster
  - a simple legend
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from shapely.geometry import shape


# ── Default thresholds (tunable) ──────────────────────────────────────────────
OAK_NDVI_MIN = 0.3          # Minimum NDVI for tree/shrub
OAK_AREA_MIN_PX = 500       # Minimum area in pixels (~canopy scale)
OAK_STABILITY_MIN = 0.90    # SAM stability threshold


def _score_cluster(group: pd.DataFrame) -> float:
    """
    Compute a simple 'oak-likeness' score for a cluster.
    Score = mean(ndvi) * log(mean(area_px)) * mean(stability)
    """
    ndvi_m = group["ndvi"].median()
    area_m = np.log1p(group["area_px"].median())
    stab_m = group["stability"].median()
    if pd.isna(ndvi_m):
        ndvi_m = 0.0
    return float(ndvi_m * area_m * stab_m)


def extract_oak_clusters(
    clustered_csv: str | Path,
    output_dir: str | Path = "data/outputs",
    oak_cluster_ids: Optional[List[int]] = None,
    ndvi_min: float = OAK_NDVI_MIN,
    area_min_px: int = OAK_AREA_MIN_PX,
    stability_min: float = OAK_STABILITY_MIN,
    top_n_clusters: int = 2,
) -> Tuple[Path, Path]:
    """
    Identify and export probable oak segments.

    Parameters
    ----------
    clustered_csv    : path to the labelled CSV from run_kmeans().
    output_dir       : where to write outputs.
    oak_cluster_ids  : explicitly specify cluster IDs to keep.
                       If None, auto-select using heuristic scoring.
    ndvi_min         : minimum NDVI for oak candidates (after cluster selection).
    area_min_px      : minimum pixel area for oak candidates.
    stability_min    : minimum SAM stability for oak candidates.
    top_n_clusters   : how many top-scoring clusters to treat as oak candidates.

    Returns
    -------
    (oak_gpkg_path, map_png_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(clustered_csv)

    # ── Auto-select oak clusters ──────────────────────────────────────────
    if oak_cluster_ids is None:
        scores = df.groupby("cluster").apply(_score_cluster).sort_values(ascending=False)
        oak_cluster_ids = scores.index[:top_n_clusters].tolist()
        print(f"[QUERCUS] Auto-selected oak cluster(s): {oak_cluster_ids}")
        print(f"  Scores:\n{scores.to_string()}")

    # ── Filter to oak candidates ──────────────────────────────────────────
    oak_df = df[
        df["cluster"].isin(oak_cluster_ids)
        & (df["ndvi"].fillna(0) >= ndvi_min)
        & (df["area_px"] >= area_min_px)
        & (df["stability"] >= stability_min)
    ].copy()

    print(f"[QUERCUS] Oak candidates: {len(oak_df):,} / {len(df):,} total objects")

    # ── Write GeoPackage ──────────────────────────────────────────────────
    oak_gpkg = output_dir / "quercus_oaks.gpkg"
    try:
        geoms = [shape(json.loads(g)) for g in oak_df["geojson"]]
        gdf_oak = gpd.GeoDataFrame(
            oak_df.drop(columns=["geojson"]),
            geometry=geoms,
            crs="EPSG:4326",
        )
        gdf_oak.to_file(oak_gpkg, driver="GPKG")
        print(f"[QUERCUS] Oak GeoPackage → {oak_gpkg}")
    except Exception as exc:
        print(f"  [WARN] Could not build oak GeoPackage: {exc}")
        oak_gpkg = None

    # ── Map ───────────────────────────────────────────────────────────────
    map_path = output_dir / "quercus_oak_map.png"
    _plot_map(df, oak_df, map_path, oak_cluster_ids)

    return oak_gpkg, map_path


def _plot_map(
    all_df: pd.DataFrame,
    oak_df: pd.DataFrame,
    output_path: Path,
    oak_cluster_ids: List[int],
) -> None:
    """Generate a matplotlib map of all objects with oaks highlighted."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor("#1a1a2e")

    cmap = plt.get_cmap("YlGn")
    cluster_colors = {
        cid: cmap(0.4 + 0.5 * i / max(len(oak_cluster_ids) - 1, 1))
        for i, cid in enumerate(oak_cluster_ids)
    }

    for ax, (data, title) in zip(
        axes,
        [
            (all_df, "All SAM Objects"),
            (oak_df, "Probable Oak Candidates"),
        ],
    ):
        ax.set_facecolor("#0d1117")
        ax.set_title(title, color="white", fontsize=13, pad=10, fontfamily="monospace")
        ax.tick_params(colors="gray")
        ax.spines[:].set_color("#333")

        if len(data) == 0:
            ax.text(0.5, 0.5, "No objects", ha="center", va="center",
                    color="white", transform=ax.transAxes)
            continue

        # Background: all points
        ax.scatter(
            all_df["centroid_lon"], all_df["centroid_lat"],
            s=1, c="#3a3a5c", alpha=0.4, linewidths=0,
        )

        # Oak candidates coloured by cluster
        for cid in oak_cluster_ids:
            sub = data[data["cluster"] == cid] if "cluster" in data.columns else data
            if len(sub) == 0:
                continue
            color = cluster_colors.get(cid, "#52b788")
            ax.scatter(
                sub["centroid_lon"], sub["centroid_lat"],
                s=4,
                c=[color] * len(sub),
                alpha=0.8,
                linewidths=0,
                label=f"Cluster {cid}",
            )

        if title == "Probable Oak Candidates":
            patches = [
                mpatches.Patch(color=cluster_colors[cid], label=f"Cluster {cid}")
                for cid in oak_cluster_ids
            ]
            ax.legend(
                handles=patches,
                loc="lower right",
                facecolor="#1a1a2e",
                edgecolor="#444",
                labelcolor="white",
                fontsize=9,
            )

        ax.set_xlabel("Longitude", color="gray", fontsize=9)
        ax.set_ylabel("Latitude", color="gray", fontsize=9)

    fig.suptitle(
        "QUERCUS  ·  Oak Species Detection  ·  1984 Aerial Survey",
        color="#c8e6c9",
        fontsize=15,
        fontfamily="monospace",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[QUERCUS] Map saved → {output_path}")
