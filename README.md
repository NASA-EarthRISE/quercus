# QUERCUS
**Quantitative Unsupervised Extraction and Remote Classification of Unstructured Scenes**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20059145.svg)](https://doi.org/10.5281/zenodo.20059145)
[![Python: 3.x](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![EarthRISE: Development](https://img.shields.io/badge/EarthRISE-Development-b50000?labelColor=191f4c)](https://appliedsciences.nasa.gov/what-we-do/capacity-building/develop)

Tim Mayer. (2026). NASA-EarthRISE/quercus: v1.0.1 (v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.20059145

Automated detection and classification of oak species (*Quercus* spp.) from
historical aerial imagery using SAM segmentation, Landsat/NAIP NDVI, and
K-Means clustering.

## Install

```bash
!pip install -q git+https://github.com/NASA-EarthRISE/quercus.git
!pip install -q segment-geospatial
!pip install -q earthengine-api folium
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
