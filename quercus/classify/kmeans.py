"""
quercus.classify.kmeans
~~~~~~~~~~~~~~~~~~~~~~~
Run K-Means clustering on the master objects CSV.

Feature engineering
-------------------
From the raw CSV we derive:

  ndvi              — primary vegetation signal (higher = greener)
  area_px           — object size (large = canopy-like)
  log_area_px       — log-transform to reduce skew
  stability         — SAM confidence (proxy for edge clarity)
  predicted_iou     — SAM quality
  ndvi_x_area       — interaction term

Missing NDVI values are imputed with the column median before scaling.

Optimal k
---------
If k is not specified, the elbow method (inertia) is used to suggest an
optimal number of clusters.  The result is plotted and printed.

Outputs
-------
  - The input CSV with a new 'cluster' column
  - A GeoPackage with all objects labelled by cluster
  - An elbow plot PNG
  - A cluster summary CSV (mean features per cluster)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import shape
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "ndvi",
    "log_area_px",
    "stability",
    "predicted_iou",
]


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_area_px"] = np.log1p(df["area_px"].clip(lower=0))
    return df


def _elbow_plot(k_range: range, inertias: List[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_range), inertias, "o-", color="#2d6a4f")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster SSE)")
    ax.set_title("K-Means Elbow Method — QUERCUS")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[QUERCUS] Elbow plot saved → {output_path}")


def run_kmeans(
    objects_csv: str | Path,
    output_dir: str | Path = "data/outputs",
    k: Optional[int] = None,
    k_range: Tuple[int, int] = (2, 12),
    random_state: int = 42,
    n_init: int = 10,
) -> Tuple[Path, Path, pd.DataFrame]:
    """
    Run K-Means on the master objects CSV.

    Parameters
    ----------
    objects_csv  : path to the CSV produced by build_objects_csv().
    output_dir   : directory for all outputs.
    k            : number of clusters. If None, elbow method suggests best k.
    k_range      : (min_k, max_k) for elbow search if k is None.
    random_state : reproducibility seed.
    n_init       : number of K-Means initialisations.

    Returns
    -------
    (labelled_csv_path, geopackage_path, cluster_summary_df)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"[QUERCUS] Loading {objects_csv} …")
    df = pd.read_csv(objects_csv)
    df = _engineer_features(df)

    # ── Feature matrix ────────────────────────────────────────────────────
    X_raw = df[FEATURE_COLS].values
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_imp)

    # ── Determine k ───────────────────────────────────────────────────────
    if k is None:
        print("[QUERCUS] Running elbow method to choose k …")
        k_vals = range(k_range[0], k_range[1] + 1)
        inertias = []
        for ki in k_vals:
            km = KMeans(n_clusters=ki, random_state=random_state, n_init=n_init)
            km.fit(X)
            inertias.append(km.inertia_)
        elbow_path = output_dir / "elbow_plot.png"
        _elbow_plot(k_vals, inertias, elbow_path)

        # Simple elbow detection: largest second derivative
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        k = int(k_vals[np.argmax(diffs2) + 1])
        print(f"  → Suggested k = {k}  (review elbow plot to confirm)")

    # ── Final clustering ──────────────────────────────────────────────────
    print(f"[QUERCUS] Running K-Means with k={k} …")
    km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(X)
    df["cluster"] = labels

    # ── Cluster summary ───────────────────────────────────────────────────
    summary = (
        df.groupby("cluster")[FEATURE_COLS + ["area_px"]]
        .agg(["mean", "std", "count"])
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary_path = output_dir / "cluster_summary.csv"
    summary.to_csv(summary_path)
    print(f"[QUERCUS] Cluster summary:\n{summary.to_string()}")

    # ── Save labelled CSV ─────────────────────────────────────────────────
    labelled_csv = output_dir / "quercus_objects_clustered.csv"
    df.to_csv(labelled_csv, index=False)

    # ── Build GeoPackage ──────────────────────────────────────────────────
    gpkg_path = output_dir / "quercus_objects_clustered.gpkg"
    try:
        geoms = [shape(json.loads(g)) for g in df["geojson"]]
        gdf = gpd.GeoDataFrame(
            df.drop(columns=["geojson"]),
            geometry=geoms,
            crs="EPSG:4326",
        )
        gdf.to_file(gpkg_path, driver="GPKG")
        print(f"[QUERCUS] GeoPackage saved → {gpkg_path}")
    except Exception as exc:
        print(f"  [WARN] Could not build GeoPackage: {exc}")
        gpkg_path = None

    print(f"[QUERCUS] Labelled CSV saved → {labelled_csv}")
    return labelled_csv, gpkg_path, summary
