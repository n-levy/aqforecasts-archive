"""Distance utilities for station/grid spatial summaries."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray:
    """Compute great-circle distance (km) between points.

    Supports scalar or numpy array inputs (broadcasting rules apply).
    """
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def normalize_longitudes(lons: np.ndarray, target: str) -> np.ndarray:
    """Normalize longitudes to target range.

    target:
      - "0_360": [0, 360)
      - "-180_180": [-180, 180)
    """
    if target == "0_360":
        return np.mod(lons, 360.0)
    if target == "-180_180":
        return ((lons + 180.0) % 360.0) - 180.0
    raise ValueError(f"Unknown longitude target: {target}")
