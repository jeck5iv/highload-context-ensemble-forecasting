# Refactor manifest

Main public import:

```python
from metroflow.studies.metro_short_history_forecasting.entrypoint import (
    MultiStationForecastConfig,
    run_multistation_forecasting_experiment,
    save_multistation_forecasting_result,
)
```

Main scripts:

```text
scripts/run_metro_forecasting.py
scripts/run_horizon_bucket_grid.py
scripts/build_prediction_artifacts.py
scripts/compare_regression_outputs.py
```

Main notebook:

```text
notebooks/00_repository_check.ipynb
```

The previous flat experiment file was removed. The current structure separates data preparation, feature construction, model families, meta-ensembles, evaluation, and output writing.
