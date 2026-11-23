import datetime as dt
import unittest

import numpy as np
import pandas as pd

from arch import cams_europe, grid


def cams_grid():
    lat = np.round(np.arange(69.95, 30.0, -0.1), 2)                       # descending, as CAMS Europe
    lon = np.round(np.concatenate([np.arange(335.05, 360.0, 0.1), np.arange(0.05, 45.0, 0.1)]), 2)  # as the real grid
    return lat, lon


class Grid(unittest.TestCase):
    def test_bilinear_exact_on_linear_field_across_the_wrap(self):
        lat, lon = cams_grid()
        lon_signed = np.where(lon > 180, lon - 360, lon)
        field = 2.0 * lat[:, None] + 0.5 * lon_signed[None, :]               # linear, so bilinear is exact
        st_lat, st_lon = np.array([52.52, 40.0, 60.0]), np.array([13.40, -3.7, -10.03])
        idx = grid.StationIndex(lat, lon, st_lat, st_lon)
        np.testing.assert_allclose(idx.bilinear(field), 2.0 * st_lat + 0.5 * st_lon, atol=1e-6)

    def test_outside_and_edge_give_nan(self):
        lat, lon = cams_grid()
        f = np.ones((len(lat), len(lon)))
        idx = grid.StationIndex(lat, lon, np.array([25.0, 69.95, 50.0]), np.array([10.0, 10.0, 50.0]))
        v = idx.bilinear(f)
        self.assertTrue(np.isnan(v[0]) and np.isnan(v[2]))                  # south of grid; east of grid
        self.assertEqual(v[1], 1.0)                                          # on the first row: inside
        mean, mx = idx.neighbourhood(f)
        self.assertTrue(np.isnan(mean[1]))                                   # no full 3x3 at the edge

    def test_neighbourhood_mean_and_max(self):
        lat, lon = cams_grid()
        f = np.zeros((len(lat), len(lon)))
        idx = grid.StationIndex(lat, lon, np.array([50.0]), np.array([10.0]))
        y, x = idx.yn[0], idx.xn[0]
        f[y - 1:y + 2, x - 1:x + 2] = np.arange(9).reshape(3, 3)
        mean, mx = idx.neighbourhood(f)
        self.assertAlmostEqual(mean[0], 4.0)
        self.assertEqual(mx[0], 8.0)


class Extract(unittest.TestCase):
    def test_extract_shapes_leads_and_valid_times(self):
        import xarray as xr

        lat, lon = cams_grid()
        leads = [0, 3, 16]
        data = np.full((len(leads), 1, len(lat), len(lon)), 7.0, dtype="float32")
        ds = xr.Dataset({"o3_conc": (("time", "level", "latitude", "longitude"), data, {"units": "µg/m3"}),
                         "no2_conc": (("time", "level", "latitude", "longitude"), data * 2, {"units": "µg/m3"})},
                        coords={"time": pd.to_timedelta(leads, unit="h"), "level": [0.0], "latitude": lat, "longitude": lon},
                        attrs={"source": "Data from ENSEMBLE model"})
        st = pd.DataFrame({"station_key": ["EEA:A", "EEA:B", "EEA:OUT"], "latitude": [48.2, 41.9, 20.0],
                           "longitude": [16.37, 12.5, 0.0]})
        out = cams_europe.extract(ds, st, dt.date(2024, 1, 1), "2026-11-23T00:00:00+00:00")
        self.assertEqual(list(out.columns), cams_europe.COLUMNS)
        self.assertEqual(len(out), 2 * 3 * 2)                                 # 2 species x 3 leads x 2 stations inside
        self.assertEqual(set(out["variable"]), {"o3", "no2"})
        row = out[(out["variable"] == "no2") & (out["lead_hour"] == 16)].iloc[0]
        self.assertEqual(row["valid_time_utc"], pd.Timestamp("2024-01-01T16:00Z"))
        self.assertEqual((row["value"], row["mean_3x3"], row["max_3x3"]), (14.0, 14.0, 14.0))

    def test_request_and_keys(self):
        r = cams_europe.build_request(dt.date(2024, 1, 1))
        self.assertEqual(r["date"], ["2024-01-01/2024-01-01"])
        self.assertIn("16", r["leadtime_hour"])
        self.assertIn("40", r["leadtime_hour"])
        self.assertEqual(len(r["leadtime_hour"]), 35)
        self.assertEqual(cams_europe.keys(dt.date(2024, 1, 1))[1],
                         "manifests/cams_europe/2024/01/manifest_cams_europe_2024-01-01.json")


class Quiet(unittest.TestCase):
    def test_no_new_request_during_the_lock_window(self):
        utc = dt.timezone.utc
        self.assertTrue(cams_europe.in_quiet_window(dt.datetime(2026, 11, 23, 6, 15, tzinfo=utc)))
        self.assertTrue(cams_europe.in_quiet_window(dt.datetime(2026, 11, 23, 13, 44, tzinfo=utc)))
        self.assertFalse(cams_europe.in_quiet_window(dt.datetime(2026, 11, 23, 13, 45, tzinfo=utc)))
        self.assertFalse(cams_europe.in_quiet_window(dt.datetime(2026, 11, 23, 2, 0, tzinfo=utc)))


if __name__ == "__main__":
    unittest.main()
