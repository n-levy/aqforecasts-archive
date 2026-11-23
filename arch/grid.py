"""Values of a regular latitude-longitude grid at station locations.

For each station: the bilinear value from the four surrounding grid points, and
the mean and maximum over the 3x3 grid points centred on the nearest one.  The
grid is regular (constant spacing) in both directions; longitudes may wrap
(for example 335 ... 359.9, 0 ... 44.95 in CAMS Europe).  Stations outside the
grid, or next to its edge for the 3x3 statistics, get NaN.
"""
from __future__ import annotations

import numpy as np


class StationIndex:
    def __init__(self, grid_lat, grid_lon, st_lat, st_lon):
        grid_lat, grid_lon = np.asarray(grid_lat, float), np.asarray(grid_lon, float)
        st_lat, st_lon = np.asarray(st_lat, float), np.asarray(st_lon, float)
        self.ny, self.nx = len(grid_lat), len(grid_lon)
        dlat = grid_lat[1] - grid_lat[0]
        dlon = (grid_lon[1] - grid_lon[0]) % 360.0
        # fractional grid indices of each station
        fy = (st_lat - grid_lat[0]) / dlat
        fx = ((st_lon - grid_lon[0]) % 360.0) / dlon
        self.inside = (fy >= 0) & (fy <= self.ny - 1) & (fx >= 0) & (fx <= self.nx - 1)
        fy, fx = np.where(self.inside, fy, 0.0), np.where(self.inside, fx, 0.0)
        self.y0 = np.minimum(np.floor(fy).astype(int), self.ny - 2)
        self.x0 = np.minimum(np.floor(fx).astype(int), self.nx - 2)
        self.wy, self.wx = fy - self.y0, fx - self.x0
        self.yn, self.xn = np.rint(fy).astype(int), np.rint(fx).astype(int)
        self.inner = self.inside & (self.yn >= 1) & (self.yn <= self.ny - 2) & (self.xn >= 1) & (self.xn <= self.nx - 2)

    def bilinear(self, field: np.ndarray) -> np.ndarray:
        f = np.asarray(field, float)
        y0, x0, wy, wx = self.y0, self.x0, self.wy, self.wx
        v = ((1 - wy) * (1 - wx) * f[y0, x0] + (1 - wy) * wx * f[y0, x0 + 1]
             + wy * (1 - wx) * f[y0 + 1, x0] + wy * wx * f[y0 + 1, x0 + 1])
        return np.where(self.inside, v, np.nan)

    def neighbourhood(self, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = np.asarray(field, float)
        yn = np.clip(self.yn, 1, self.ny - 2)
        xn = np.clip(self.xn, 1, self.nx - 2)
        block = np.stack([f[yn + dy, xn + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        mean, mx = block.mean(axis=0), block.max(axis=0)
        return np.where(self.inner, mean, np.nan), np.where(self.inner, mx, np.nan)
