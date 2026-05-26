# Prediction artifacts

`scripts/build_prediction_artifacts.py` reads saved experiment outputs and produces report-facing CSV files, LaTeX tables, and figures. It does not retrain models.

Typical usage:

```bash
PYTHONPATH=src python scripts/build_prediction_artifacts.py \
  --megagrid-root outputs/multistation_megagrid_overnight_MAIN_RESULT \
  --out-dir outputs/prediction_artifacts
```

The builder expects machine-readable experiment outputs such as `micro_metrics.csv`, `macro_metrics.csv`, `station_bootstrap_delta_ci.csv`, `test_predictions.csv`, and `stations_used.csv`.
