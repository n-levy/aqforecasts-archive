"""Unchanged copies of the founding paper's spatial code.

Source: aqf/01_gather_data/07_unify_training_data/shared/spatial/ (distances.py,
neighborhoods.py, summaries.py), used by the 2025-refresh training-table builder
for the raw CAMS neighbourhood mean (radius 40 km, weights exp(-d / 20 km)).
SHA-256 of the source files: SOURCE_SHA256.txt.  Do not edit these files; the
paper's RAW feature must be computed by exactly this code.
"""
from .distances import haversine_km, normalize_longitudes
from .neighborhoods import GridNeighborhood, build_grid_neighborhood, build_station_grid_neighborhoods
from .summaries import distance_weighted_stats, summarize_grid_values
