# European Air Quality Forecasts: data archive (collectors)

Code that collects air-quality and weather data at monitoring stations, for
model development and research. **This repository holds code only.** The data
are stored in a private bucket and are not published. This is separate from the
evaluation pipeline: no code or storage is shared, and a failure here can never
affect a locked forecast.

Design: `planning/STAGE2_DESIGN_PROPOSAL_2026-11-23.md` (in the project folder).

## Collectors

- `arch/cams_europe.py`: Stage 2 backfill of the CAMS European air-quality
  forecast (Atmosphere Data Store, `cams-europe-air-quality-forecasts`; 0.1°,
  ensemble median, surface) from 1 Jan 2024.
  - One request per issue day: 00 UTC run, CO, NO₂, O₃, PM10, PM2.5, SO₂, lead
    hours 0–96 every 3 h plus 16 and 40.
  - Per station: the bilinear value and the mean and maximum of the 3×3 grid
    points around it.
  - Output: one Parquet file per day plus a manifest with its SHA-256.
  - The ADS keeps about three years, so 2024 starts disappearing around
    January 2027.
  - Workflow `backfill-cams-europe.yml`.

## Stations

`data/stations_v1.csv`: 5,081 EEA stations (EoI codes) with coordinates, type
and area, from the evaluation's station table. Append-only: stations are only
added, never relabelled. `station_key` is `EEA:<EoI code>`.

## Storage layout (private bucket `aqforecasts-archive`)

    archive/<source>/YYYY/MM/<source>_<YYYY-MM-DD>.parquet
    manifests/<source>/YYYY/MM/manifest_<source>_<YYYY-MM-DD>.json   (written last)

Long format, one row per station, variable, issue time and lead:
`station_key, source, variable, unit, issue_time_utc, valid_time_utc,
lead_hour, value, mean_3x3, max_3x3, extraction, grid_res_deg, vintage,
retrieved_utc, source_version, schema_version`.

`vintage` is one of:
- `as_available`: captured at the time;
- `archived`: from a provider's archive;
- `final`: verified data.

## Secrets and variables (GitHub Actions)

- **Secrets:** `ADS_API_URL`, `ADS_API_KEY`, `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`,
  `R2_SECRET_ACCESS_KEY`. The R2 token is limited to the bucket
  `aqforecasts-archive`.
- **Variables:** `R2_ARCHIVE_BUCKET`, and `CAMS_EUROPE_FROM` / `CAMS_EUROPE_TO`
  (range for scheduled runs).

Workflows never run on pull requests from forks, and keys are never printed.

## Tests

    pip install -r requirements.txt
    python -m unittest discover -s tests

## Licences

- **Code:** MIT.
- **Data:** CAMS data © the European Union, "Generated using Copernicus
  Atmosphere Monitoring Service Information".
