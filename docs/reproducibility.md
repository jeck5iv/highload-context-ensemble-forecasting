# Reproducibility

The default random seed is fixed in `MultiStationForecastConfig.random_state`:

```text
random_state = 3228
```

The operational day is defined as `03:00-02:59`. Weekday filters, train/validation/test splits, day-lag features, and high-load proxies are based on this operational-day convention.

The recommended verification sequence is:

1. run `make test`;
2. execute `notebooks/00_repository_check.ipynb` with `RUN_FULL_EXPERIMENT = False`;
3. run a fast real experiment, for example horizon `60` and bucket `60`, without bootstrap;
4. repeat the same fast experiment and compare outputs with `scripts/compare_regression_outputs.py`;
5. enable bootstrap with a small number of resamples;
6. run the full central or grid experiment only after the fast checks pass.

Bootstrap confidence intervals are computed after predictions are fixed. They resample stations, not individual station-time rows, preserving the temporal structure of errors within each station.
