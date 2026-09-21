# Forecasting passenger flow with a short history

My bachelor's thesis asked how to forecast busy periods when there are many related time series but very little history for each one.

I used roughly 50 million entry events from one week of the Moscow Metro. After filtering and aggregation, the study covered 260 stations. The strongest central result came from a simple structure: fit ETS separately at each station, then learn a shared correction to its errors with XGBoost.

## Main result

For the next-hour forecast on a 15-minute grid, the residual model reduced load-weighted MAE from 82.52 to 79.51, a 3.6% improvement over local ETS. Ordinary MAE fell from 57.48 to 53.97, or 6.1%.

| Model | MAE | Load-weighted MAE | High-load MAE |
| --- | ---: | ---: | ---: |
| Previous-day seasonal baseline | 58.70 | 88.82 | 104.14 |
| Local ETS | 57.48 | 82.52 | 98.30 |
| ETS with pooled linear correction | 55.58 | 79.90 | 94.59 |
| ETS with pooled XGBoost correction | 53.32 | 80.08 | 94.58 |
| ETS with load-weighted XGBoost correction | 53.97 | 79.51 | 93.75 |

These are saved thesis results, not a new training run. The public audit found and removed target-dependent fallbacks used when models fail or history is missing. Aggregate fixtures cannot establish whether these paths affected the recorded scores; a full rerun on the private data is still needed. The station-bootstrap 95% interval for the recorded 3.01-point improvement over ETS is [1.93, 4.07]. It measures variation across the observed station panel, not performance on new weeks.

The data cover 10-16 March 2025. The main comparison trains on Monday and Tuesday, uses Wednesday for model selection and residual fitting, and evaluates on Thursday. Friday and the weekend are outside that comparison. Fifty million events should not be mistaken for a long time-series sample.

[Research note](docs/research.md) covers the model, validation, unsuccessful approaches and the eight horizon/grid settings. [Saved central results](tests/fixtures/h060m_b015m/micro_metrics.csv) provide the full comparison.

## Why a local model and a shared correction?

A station's daily profile is useful even when there are few observed days. ETS retains that profile. A pooled model can then learn where the local forecast tends to miss using errors from many stations.

The correction sees base forecasts and their disagreement, time-of-day features, recent changes, rolling variability, previous-day demand and training-period station summaries. Load weighting puts more emphasis on busy intervals. Corrections are clipped using residual quantiles estimated before the test day.

Global boosting, sequence models and context-dependent blends were also evaluated. In this short-history setting, the larger models did not consistently beat local ETS. The research note keeps those results and distinguishes article-inspired adaptations from exact paper reproductions.

## Run the checks

Python 3.11 or 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python scripts/summarize_results.py
```

The tests use synthetic examples and aggregate regression fixtures. They require neither raw operational data nor a GPU. The summary command reads the saved central results; it does not retrain the models.

## Run an experiment

Raw events are not public. If you have authorized data in the format described in the [data contract](docs/data_contract.md):

```bash
pip install -e '.[full]'
python scripts/run_metro_forecasting.py \
  --data-dir ./data \
  --cache-dir ./data/cache \
  --output-dir ./outputs/metro_short_history_h060_b015 \
  --n-bootstrap 1000
```

For a smaller end-to-end check:

```bash
python scripts/run_horizon_bucket_grid.py \
  --profile robustness --only 60:60 \
  --output-root outputs/verify_fast \
  --with-bootstrap --n-bootstrap 20 --rerun
```

The seed is 3228. An operational day runs from 03:00 to 02:59. Target windows are clipped at split boundaries so a training label cannot cross into the next day.

## Repository

The study lives in `src/metroflow/studies/metro_short_history_forecasting/`. Its `entrypoint.py` exposes the configuration, runner and output writer. `models/` contains local, global and residual methods. `evaluation/` defines the metrics and station bootstrap.

[Reproducibility](docs/reproducibility.md) describes the verification sequence. [Prediction artifacts](docs/prediction_artifacts.md) documents the output format. The older [single-station notebook](https://github.com/jeck5iv/Kalinin_Diplom_AMI) is an exploratory predecessor, not a second independent project.

## Scope

The public repository contains code, configuration and aggregate results. It excludes passenger records and trained artifacts derived from the closed data. The full thesis is based on one short calendar segment. Model selection, station dependence, overlapping targets and comparisons across many candidate models limit the strength of the conclusions. The next useful test is a frozen evaluation on additional weeks.
