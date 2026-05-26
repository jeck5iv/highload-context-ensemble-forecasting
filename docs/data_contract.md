# Data contract

Raw operational data are not stored in the repository. The pipeline expects them to be available locally.

Expected minimal layout:

```text
data/
  PASS_ALL_*.csv                    event-level passenger entries
  REF_PSG_PLACES_*.csv              place and station reference
  cache/                            generated parquet cache
```

Optional files:

```text
data/
  weather_*_hourly_openmeteo_*.csv  optional network-level weather covariates
```

The places reference must contain enough information to aggregate individual entry objects to a stable station key based on line and station codes. Non-subway objects are excluded by default through the curated metro line whitelist in `MultiStationForecastConfig`.

Generated station-place maps and bucket parquet files may be cached under `data/cache/`. These files are local artifacts and are not required in the public repository.
