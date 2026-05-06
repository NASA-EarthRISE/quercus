# QUERCUS
**Quantitative Unsupervised Extraction and Remote Classification of Unstructured Scenes**

Quick start

Automated detection and classification of oak species (*Quercus* spp.) from
historical aerial imagery using SAM segmentation, Landsat/NAIP NDVI, and
K-Means clustering.

## Install

```bash
pip install git+https://github.com/MayerT1/quercus.git
pip install segment-geospatial   # SAM dependency
```

## Quick start (Google Colab)

Open `notebooks/QUERCUS_v8_demo.ipynb` or click the Colab badge in the notebook.

## Pipeline

```
1984 Aerials
  fetch_iiif_images(page_ids, aoi)  ->  clip_scan_borders  ->  georeference_images
  ->  run_sam_pipeline  ->  build_objects_csv (NDVI)  ->  build_mosaic

NAIP
  fetch_naip_and_run_sam(georef_paths, aoi, year_start, year_end)
  ->  build_objects_csv (NDVI)  ->  build_mosaic

Classify
  run_kmeans  ->  extract_oak_clusters
```

## AOI filtering

Pass a GeoJSON file path or a Shapely geometry as `aoi=` to any fetch function.
Images outside the AOI are skipped and reported to the user.

```python
from quercus.ingest import fetch_iiif_images
raw = fetch_iiif_images(
    page_ids=[715, 716, 717, 718],
    aoi="my_study_area.geojson",
)
```

## License
MIT
