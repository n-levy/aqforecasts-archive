"""Neighborhood selection utilities for grid-based spatial summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distances import haversine_km, normalize_longitudes


@dataclass
class GridNeighborhood:
    lat_idx: np.ndarray
    lon_idx: np.ndarray
    distances_km: np.ndarray


def _lon_target_for_grid(grid_lons: np.ndarray) -> str:
    if np.nanmax(grid_lons) > 180.0:
        return "0_360"
    return "-180_180"


def build_grid_neighborhood(
    station_lat: float,
    station_lon: float,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    radius_km: float,
) -> GridNeighborhood:
    """Return indices and distances for grid cells within radius of station."""
    lon_target = _lon_target_for_grid(grid_lons)
    station_lon_norm = float(normalize_longitudes(np.array([station_lon]), lon_target)[0])

    # Approximate bounding box to limit candidate cells.
    lat_delta = radius_km / 111.0
    cos_lat = max(np.cos(np.radians(station_lat)), 1e-6)
    lon_delta = radius_km / (111.0 * cos_lat)

    lat_min = station_lat - lat_delta
    lat_max = station_lat + lat_delta
    lon_min = station_lon_norm - lon_delta
    lon_max = station_lon_norm + lon_delta

    lat_mask = (grid_lats >= lat_min) & (grid_lats <= lat_max)
    lat_idx = np.where(lat_mask)[0]
    if lat_idx.size == 0:
        return GridNeighborhood(np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float))

    # Handle longitude wrap for 0..360 grids.
    if lon_target == "0_360":
        grid_lons_norm = grid_lons
        lon_min = lon_min % 360.0
        lon_max = lon_max % 360.0
        if lon_min <= lon_max:
            lon_mask = (grid_lons_norm >= lon_min) & (grid_lons_norm <= lon_max)
        else:
            lon_mask = (grid_lons_norm >= lon_min) | (grid_lons_norm <= lon_max)
    else:
        grid_lons_norm = normalize_longitudes(grid_lons, "-180_180")
        lon_mask = (grid_lons_norm >= lon_min) & (grid_lons_norm <= lon_max)

    lon_idx = np.where(lon_mask)[0]
    if lon_idx.size == 0:
        return GridNeighborhood(np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float))

    # Build mesh of candidate cells and compute distances.
    cand_lats = grid_lats[lat_idx]
    cand_lons = grid_lons_norm[lon_idx]
    lat_grid, lon_grid = np.meshgrid(cand_lats, cand_lons, indexing="ij")

    distances = haversine_km(station_lat, station_lon_norm, lat_grid, lon_grid)
    within = distances <= radius_km
    if not np.any(within):
        return GridNeighborhood(np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float))

    lat_indices = np.repeat(lat_idx, lon_idx.size)[within.ravel()]
    lon_indices = np.tile(lon_idx, lat_idx.size)[within.ravel()]
    distances_km = distances.ravel()[within.ravel()]
    return GridNeighborhood(lat_indices, lon_indices, distances_km)


def build_station_grid_neighborhoods(
    stations: np.ndarray,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    radius_km: float,
) -> dict[str, GridNeighborhood]:
    """Precompute grid neighborhoods for all stations.

    stations: array of (station_id, lat, lon)
    """
    neighborhoods: dict[str, GridNeighborhood] = {}
    for station_id, lat, lon in stations:
        neighborhoods[str(station_id)] = build_grid_neighborhood(
            float(lat), float(lon), grid_lats, grid_lons, radius_km
        )
    return neighborhoods
