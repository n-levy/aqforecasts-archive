"""Backfill of the CAMS European air-quality forecast at stations (Stage 2).

Source: Atmosphere Data Store, dataset `cams-europe-air-quality-forecasts`
(0.1 degree grid, ensemble median, surface).  The ADS keeps a rolling archive of
about three years, so the oldest days must be fetched first.

One request per issue day: the 00 UTC run, six species, LEADS (every 3 h from
0 to 96 h, plus 16 and 40 h, the evaluation's slots A and B).  The grid is cut
to the stations of data/stations_v1.csv and discarded.  Per station, species
and lead the file holds the bilinear value and the mean and maximum over the
3x3 grid points around the station.

Output (private bucket, write-once):
  archive/cams_europe/YYYY/MM/cams_europe_<day>.parquet
  manifests/cams_europe/YYYY/MM/manifest_cams_europe_<day>.json   (written last)
A day whose manifest exists is skipped, so the job can be stopped and resumed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import grid, store

SOURCE = "cams_europe"
DATASET = "cams-europe-air-quality-forecasts"
SCHEMA_VERSION = 1
VARIABLES = {  # ADS request name -> NetCDF variable -> stored name
    "carbon_monoxide": ("co_conc", "co"),
    "nitrogen_dioxide": ("no2_conc", "no2"),
    "ozone": ("o3_conc", "o3"),
    "particulate_matter_10um": ("pm10_conc", "pm10"),
    "particulate_matter_2.5um": ("pm2p5_conc", "pm25"),
    "sulphur_dioxide": ("so2_conc", "so2"),
}
LEADS = sorted(set(range(0, 97, 3)) | {16, 40})
STATIONS = Path(__file__).resolve().parents[1] / "data" / "stations_v1.csv"
COLUMNS = ["station_key", "source", "variable", "unit", "issue_time_utc", "valid_time_utc", "lead_hour",
           "value", "mean_3x3", "max_3x3", "extraction", "grid_res_deg", "vintage", "retrieved_utc",
           "source_version", "schema_version"]


def build_request(day: dt.date) -> dict:
    return {
        "variable": list(VARIABLES),
        "model": ["ensemble"],
        "level": ["0"],
        "date": [f"{day}/{day}"],
        "type": ["forecast"],
        "time": ["00:00"],
        "leadtime_hour": [str(h) for h in LEADS],
        "data_format": "netcdf_zip",
    }


def keys(day: dt.date) -> tuple[str, str]:
    return (f"archive/{SOURCE}/{day:%Y/%m}/{SOURCE}_{day}.parquet",
            f"manifests/{SOURCE}/{day:%Y/%m}/manifest_{SOURCE}_{day}.json")


def load_stations(path: Path = STATIONS) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"station_key": str})


def extract(ds, stations: pd.DataFrame, day: dt.date, retrieved_utc: str) -> pd.DataFrame:
    """Station values from one day's dataset (dims time=lead, level, latitude, longitude)."""
    idx = grid.StationIndex(ds["latitude"].values, ds["longitude"].values,
                            stations["latitude"].astype(float).values, stations["longitude"].astype(float).values)
    leads = (pd.to_timedelta(ds["time"].values) / pd.Timedelta(hours=1)).round().astype(int)
    issue = pd.Timestamp(day, tz="UTC")
    parts = []
    for _, (nc_name, name) in VARIABLES.items():
        if nc_name not in ds:
            continue
        da = ds[nc_name]
        unit = da.attrs.get("units", "µg/m3")
        arr = da.values.reshape(len(leads), -1, da.shape[-2], da.shape[-1])[:, 0]
        for i, lead in enumerate(leads):
            field = arr[i]
            mean, mx = idx.neighbourhood(field)
            parts.append(pd.DataFrame({
                "station_key": stations["station_key"].values, "variable": name, "unit": unit,
                "lead_hour": int(lead), "value": idx.bilinear(field).astype("float32"),
                "mean_3x3": mean.astype("float32"), "max_3x3": mx.astype("float32")}))
    out = pd.concat(parts, ignore_index=True)
    out = out[np.isfinite(out["value"])]
    out["valid_time_utc"] = issue + pd.to_timedelta(out["lead_hour"], unit="h")
    out = out.assign(source=SOURCE, issue_time_utc=issue, extraction="bilinear; 3x3 around nearest",
                     grid_res_deg=0.1, vintage="archived", retrieved_utc=retrieved_utc,
                     source_version=ds.attrs.get("source", ""), schema_version=SCHEMA_VERSION)
    return out[COLUMNS].sort_values(["variable", "lead_hour", "station_key"], kind="mergesort").reset_index(drop=True)


def open_zip(path: Path, work: Path):
    import xarray as xr

    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith(".nc")]
        z.extractall(work, members=names)
    datasets = [xr.open_dataset(work / n) for n in names]
    return xr.merge(datasets, compat="override") if len(datasets) > 1 else datasets[0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def process_day(client, s3, bucket: str, day: dt.date, stations: pd.DataFrame, work: Path) -> dict:
    data_key, man_key = keys(day)
    if store.exists(s3, bucket, man_key):
        return {"day": str(day), "status": "already done"}
    work.mkdir(parents=True, exist_ok=True)
    zpath = work / f"{SOURCE}_{day}.zip"
    t0 = time.time()
    client.retrieve(DATASET, build_request(day), str(zpath))
    retrieved = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    ds = open_zip(zpath, work / f"nc_{day}")
    table = extract(ds, stations, day, retrieved)
    ds.close()
    ppath = work / f"{SOURCE}_{day}.parquet"
    table.to_parquet(ppath, index=False, compression="zstd")
    manifest = {"schema_version": SCHEMA_VERSION, "source": SOURCE, "dataset": DATASET, "issue_day": str(day),
                "request": build_request(day), "retrieved_utc": retrieved, "download_bytes": zpath.stat().st_size,
                "file": ppath.name, "key": data_key, "sha256": sha256(ppath), "bytes": ppath.stat().st_size,
                "rows": int(len(table)), "stations": int(table["station_key"].nunique()),
                "variables": sorted(table["variable"].unique()), "leads": sorted(int(x) for x in table["lead_hour"].unique()),
                "seconds": round(time.time() - t0, 1)}
    mpath = work / f"manifest_{SOURCE}_{day}.json"
    mpath.write_text(json.dumps(manifest, indent=1) + "\n")
    store.put(s3, bucket, data_key, ppath, overwrite=True)      # a half-written day has no manifest yet
    store.put(s3, bucket, man_key, mpath)
    for p in (zpath, ppath, mpath):
        p.unlink(missing_ok=True)
    for p in sorted((work / f"nc_{day}").glob("*")):
        p.unlink()
    (work / f"nc_{day}").rmdir()
    return {k: manifest[k] for k in ("issue_day", "rows", "stations", "bytes", "download_bytes", "seconds")} | {"status": "done"}


# The evaluation's daily lock uses the same ADS account between 07:07 and 13:37 UTC.
# No new backfill request starts in that window, so the lock never waits behind one.
QUIET_UTC = (dt.time(6, 15), dt.time(13, 45))
MAX_CONSECUTIVE_FAILURES = 3    # e.g. a wrong key: stop instead of failing every remaining day


def in_quiet_window(now: dt.datetime) -> bool:
    return QUIET_UTC[0] <= now.astimezone(dt.timezone.utc).time() < QUIET_UTC[1]


def ads_client(env=os.environ):
    import cdsapi

    return cdsapi.Client(url=env["ADS_API_URL"], key=env["ADS_API_KEY"], quiet=True, timeout=900)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="start", required=True, help="first issue day YYYY-MM-DD")
    ap.add_argument("--to", dest="end", required=True, help="last issue day YYYY-MM-DD")
    ap.add_argument("--hours", type=float, default=5.0, help="stop starting new days after this many hours")
    ap.add_argument("--work", default="work")
    a = ap.parse_args(argv)
    start, end = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    s3, bucket = store.client(), os.environ["R2_ARCHIVE_BUCKET"]
    client, stations = ads_client(), load_stations()
    t0, report, failures = time.time(), [], 0
    day = start
    while day <= end and time.time() - t0 < a.hours * 3600:
        if in_quiet_window(dt.datetime.now(dt.timezone.utc)):
            print(json.dumps({"stopped": "quiet window for the evaluation's lock (06:15-13:45 UTC)"}))
            break
        try:
            r = process_day(client, s3, bucket, day, stations, Path(a.work))
        except Exception as exc:  # one bad day must not stop the backfill
            failures += 1
            r = {"day": str(day), "status": "failed", "error": str(exc)[:500]}
        print(json.dumps(r), flush=True)
        report.append(r)
        day += dt.timedelta(days=1)
        if len(report) >= MAX_CONSECUTIVE_FAILURES and all(
                x["status"] == "failed" for x in report[-MAX_CONSECUTIVE_FAILURES:]):
            print(json.dumps({"stopped": f"{MAX_CONSECUTIVE_FAILURES} consecutive failures; "
                                         "check the ADS key and licence before the next run"}))
            day -= dt.timedelta(days=1)
            break
    done = sum(r["status"] == "done" for r in report)
    print(json.dumps({"from": a.start, "to": a.end, "days_tried": len(report), "done": done,
                      "already": sum(r["status"] == "already done" for r in report), "failed": failures,
                      "next_day": str(day) if day <= end else None}))
    return 1 if failures and not done else 0


if __name__ == "__main__":
    sys.exit(main())
