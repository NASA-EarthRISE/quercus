"""QUERCUS - Quantitative Unsupervised Extraction ^& Remote Classification of US Species"""
__version__ = "0.8.0"

from quercus.ingest import fetch_iiif_images, clip_scan_borders, georeference_images, build_mosaic
from quercus.segment import run_sam_pipeline, fetch_gee_ndvi, build_objects_csv, fetch_naip_and_run_sam
from quercus.classify import run_kmeans, extract_oak_clusters
