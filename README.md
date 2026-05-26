# Metro Passenger-Flow Forecasting

[English](README.md) | [Русский](README_RU.md)

This repository contains a reproducible research pipeline for short-history, multi-station forecasting of inbound metro passenger flow. The main experiment evaluates seasonal/statistical baselines, boosting models, lightweight sequence models, article-inspired comparators, and residual meta-ensembles under a single leakage-aware protocol.

The code is organized as a study package rather than a collection of notebooks. The public entry point is deliberately narrow:

```python
from metroflow.studies.metro_short_history_forecasting.entrypoint import (
    MultiStationForecastConfig,
    run_multistation_forecasting_experiment,
    save_multistation_forecasting_result,
)
```

## Repository layout

```text
configs/                         experiment presets and grid settings
data/                            local data placeholder, not versioned
docs/                            data contract, reproducibility notes, refactor notes
notebooks/                       short repository check and optional full run
scripts/                         command-line entry points
src/metroflow/                   Python package
  studies/metro_short_history_forecasting/
    entrypoint.py                public API for the study
    config.py                    experiment configuration
    pipeline.py                  orchestration of the full forecasting run
    data_layer/                  station mapping, aggregation, supervised frame
    models/                      baselines, boosting, sequence, article-inspired, meta models
    evaluation/                  metrics and station bootstrap
    io.py                        output writer
tests/                           quick checks and regression fixtures
```

The vertical structure is intentional: `entrypoint.py` exposes the study API, `pipeline.py` coordinates stages, and lower-level modules implement data preparation, models, evaluation, and output writing.

## Installation

For the lightweight checks:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements.txt
```

For the full model set:

```bash
python -m pip install -e '.[full]'
python -m pip install -r requirements-full.txt
```

## Data

Raw operational data are not included. The expected local layout is described in [`docs/data_contract.md`](docs/data_contract.md). A typical local setup is:

```text
data/
  PASS_ALL_*.csv
  REF_PSG_PLACES_*.csv
  cache/
```

Optional weather covariates can be supplied through `MultiStationForecastConfig.weather_csv` or by placing a matching `weather_*_hourly_openmeteo_*.csv` file in `data/`.

## Quick checks

Run the fast test suite:

```bash
make test
```

This does not train the full model set. It checks imports, metrics, deterministic station-bootstrap behavior, package structure, and compact regression fixtures.

Run the notebook check:

```bash
jupyter nbconvert --execute notebooks/00_repository_check.ipynb --to notebook --inplace
```

The notebook keeps the full experiment disabled by default:

```python
RUN_FULL_EXPERIMENT = False
```

With this value the notebook performs only a fast repository check. Setting it to `True` runs a real forecasting experiment and writes outputs to `outputs/notebook_forecasting_run`.

## Full central run

```bash
PYTHONPATH=src python scripts/run_metro_forecasting.py \
  --data-dir ./data \
  --cache-dir ./data/cache \
  --output-dir ./outputs/metro_short_history_h060_b015 \
  --n-bootstrap 1000
```

The default configuration uses:

```text
bucket_minutes = 15
horizon_minutes = 60
operational_day_start_hour = 3
random_state = 3228
```

To run a faster honest validation on local data, use a coarser bucket and disable bootstrap through the Python API or the grid script.

## Horizon-bucket grid

```bash
PYTHONPATH=src python scripts/run_horizon_bucket_grid.py \
  --profile robustness \
  --only 60:60 \
  --output-root outputs/verify_fast \
  --with-bootstrap \
  --n-bootstrap 20 \
  --rerun
```

The `60:60` setting is useful for validating the data path, station mapping, feature construction, model training, metrics, and output writing without running the full grid.

## Outputs

Each experiment directory contains machine-readable artifacts such as:

```text
micro_metrics.csv
macro_metrics.csv
station_bootstrap_delta_ci.csv
station_bootstrap_metric_ci.csv
test_predictions.csv
validation_predictions.csv
stations_used.csv
stage_timings.csv
```

The prediction artifact builder reads saved outputs and prepares report-facing CSV, LaTeX tables, and figures without retraining models:

```bash
PYTHONPATH=src python scripts/build_prediction_artifacts.py \
  --megagrid-root outputs/multistation_megagrid_overnight_MAIN_RESULT \
  --out-dir outputs/prediction_artifacts
```

## Regression comparison

After a new run, compare saved outputs with a fixture or a previous run:

```bash
PYTHONPATH=src python scripts/compare_regression_outputs.py \
  --old-dir tests/fixtures/h060m_b015m \
  --new-dir outputs/metro_short_history_h060_b015
```

For strict seed checks, compare two fresh runs with the same configuration. Bootstrap can be disabled first, then enabled with a small number of resamples.

## Reproducibility notes

The main seed is fixed at `3228`. The operational day is defined as `03:00-02:59`, and temporal splitting is based on that operational day. Station-level bootstrap uses stations as resampling units and is applied after predictions are fixed.

Private raw data are deliberately excluded from the repository. Reproducibility is provided through the code, configuration, output contract, regression fixtures, and saved experiment artifacts.
