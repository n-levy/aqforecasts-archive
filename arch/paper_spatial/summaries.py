"""Spatial summary statistics for grid neighborhoods."""

from __future__ import annotations

import numpy as np

from .neighborhoods import GridNeighborhood


def distance_weighted_stats(
    values: np.ndarray,
    distances_km: np.ndarray,
    lambda_km: float,
) -> tuple[float, float, float]:
    """Compute distance-weighted mean, max, and std.

    Returns (weighted_mean, max_value, weighted_std). If values empty, returns NaNs.
    """
    if values.size == 0:
        return (np.nan, np.nan, np.nan)
    weights = np.exp(-distances_km / float(lambda_km))
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        return (np.nan, np.nan, np.nan)
    weighted_mean = float(np.sum(weights * values) / weight_sum)
    max_value = float(np.max(values))
    variance = np.sum(weights * (values - weighted_mean) ** 2) / weight_sum
    weighted_std = float(np.sqrt(variance))
    return (weighted_mean, max_value, weighted_std)


def summarize_grid_values(
    grid_values: np.ndarray,
    neighborhood: GridNeighborhood,
    lambda_km: float,
) -> tuple[float, float, float]:
    """Summarize grid values for a station neighborhood."""
    if neighborhood.lat_idx.size == 0:
        return (np.nan, np.nan, np.nan)
    values = grid_values[neighborhood.lat_idx, neighborhood.lon_idx]
    return distance_weighted_stats(values, neighborhood.distances_km, lambda_km)
