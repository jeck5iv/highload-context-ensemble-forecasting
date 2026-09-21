# Local structure, pooled residuals

Ivan Kalinin, bachelor's thesis, HSE University, 2026.

## Research question

When each time series is short, should a model learn every series separately or pool them?

Metro demand provides a concrete version of this question. Stations share rush-hour patterns, but their scale and daily profiles differ. A pooled model has more rows, yet those rows come from the same few calendar days. A local model preserves station structure but has little data for estimating complicated dynamics.

I tested a middle ground: forecast each station with a local statistical model and learn a correction from the panel of forecast errors.

## Data and target

The source is a closed operational extract for 10-16 March 2025, about 50 million entry events and roughly 25 GB of CSV data. Events are mapped from entry locations to station keys. Non-metro objects are excluded. The resulting panel contains 260 stations.

For station `s`, the target is the sum of entries in the next `H/B` buckets, where `H` is the horizon and `B` is the bucket size. At a 60-minute horizon on a 15-minute grid, this is the next four buckets, not the count exactly one hour later.

The eight configurations are 15-minute forecasts on 5- and 15-minute grids; 60-minute forecasts on 5-, 15-, 30- and 60-minute grids; and six-hour forecasts on 30- and 60-minute grids.

Consecutive targets overlap when the horizon is longer than the grid step. More station-time rows therefore do not mean proportionally more independent examples.

## Validation

An operational day starts at 03:00. The main experiment uses Monday and Tuesday for fitting base candidates, Wednesday for selecting configurations and fitting the residual level, and Thursday for final evaluation. Base models that support refitting use the available pre-test history. The residual model learns from validation predictions, not in-sample training predictions.

The final horizon is removed from each daily segment. This prevents a label near the end of a training day from including observations from the next segment. Station scales and high-load thresholds are estimated from the corresponding training portion.

Wednesday serves several purposes: configuration selection, selection of models for the meta-level and fitting the residual model. These are not independent validation exercises. Thursday is one final calendar day, and the thesis compares many model families on it. The best row should not be treated as a selection-adjusted estimate of future performance.

## Models

Local ETS fits a separate level, trend and seasonal structure at each station. Previous-day demand is a direct seasonal baseline. Local SARIMA provides a second classical comparison.

Global XGBoost uses lags, rolling statistics, calendar features and station context. Candidate variants change feature sets, station-scale normalization and load weighting. Line- and station-specific boosting variants test whether narrower pooling is useful.

The sequence comparisons include LSTM and a lightweight Transformer, alongside several article-inspired architectures. The suffix `lite` means an adaptation to this data and compute budget, not an exact reproduction of a published result.

For the residual approach, the local forecast is an anchor:

```text
final forecast = station ETS forecast + pooled correction
```

The correction uses base-model predictions, differences between those predictions, their dispersion, cyclical time-of-day features, recent slopes, rolling variability and station summaries. It learns the validation error across all stations. The XGBoost version uses shallow, regularized trees and clips corrections to validation residual quantiles. Linear residual correction is a useful lower-complexity comparison.

The load-weighted version gives larger weight to observations with higher demand relative to their station's training scale. This favors errors that matter during busy periods.

## Metrics

MAE measures the average absolute count error. RMSE penalizes large misses more strongly. wMAPE divides total absolute error by total actual demand.

Load-weighted MAE is the mean of absolute error multiplied by actual demand divided by the station's training mean, raised to the configured power. It is not normalized by the sum of weights. Using the observed target to score a forecast is legitimate; it must not be used to choose a prediction at runtime.

High-load MAE evaluates only outcomes above the station's training-period 80th percentile. Underprediction and asymmetric metrics separately expose misses in the costly direction.

The central table pools station-time observations. Macro results, in which each station contributes one metric, are retained in the fixtures.

## Central result

For a 60-minute horizon and 15-minute buckets:

| Model | MAE | RMSE | Load-weighted MAE | High-load MAE |
| --- | ---: | ---: | ---: | ---: |
| Previous-day baseline | 58.70 | 141.97 | 88.82 | 104.14 |
| Local ETS | 57.48 | 110.11 | 82.52 | 98.30 |
| ETS plus linear residual | 55.58 | 105.99 | 79.90 | 94.59 |
| ETS plus XGBoost residual | 53.32 | 107.84 | 80.08 | 94.58 |
| ETS plus load-weighted XGBoost residual | 53.97 | 108.14 | 79.51 | 93.75 |
| Global rich, normalized, load-weighted boosting | 61.98 | 137.07 | 96.45 | 113.04 |
| Global LSTM | 153.38 | 301.60 | 247.40 | 317.70 |
| Global Transformer-lite | 153.49 | 299.61 | 254.45 | 330.04 |

The load-weighted residual model improves on ETS by 3.01 points in load-weighted MAE, 3.51 points in MAE and 4.55 points in high-load MAE. The corresponding relative reductions are 3.6%, 6.1% and 4.6%.

The strongest model depends on the metric. The unweighted residual XGBoost has lower ordinary MAE. The linear correction has lower RMSE. The result supports a choice of objective, not a claim that one model dominates every loss.

The public central fixture contains many named rows. Some article-inspired variants fall back to another predictor and have identical scores. They should not be counted as 49 independent successful model implementations.

## Across horizons

The following values come from the final thesis tables. The central setting also has machine-readable fixtures here. Other grid cells cannot be regenerated without the private data and original runs.

| Horizon / bucket, minutes | Best residual family | Reference | LW-MAE, residual / reference | Reduction | 95% interval for absolute reduction |
| --- | --- | --- | ---: | ---: | --- |
| 15 / 5 | Weighted XGBoost over ETS | ETS | 39.82 / 42.19 | 5.6% | [1.35, 3.43] |
| 15 / 15 | Weighted XGBoost over ETS | ETS | 36.00 / 36.75 | 2.0% | [0.49, 1.00] |
| 60 / 5 | Linear residual | SARIMA | 89.11 / 89.14 | 0.0% | [-1.02, 1.06] |
| 60 / 15 | Weighted XGBoost over ETS | ETS | 79.51 / 82.52 | 3.6% | [1.93, 4.07] |
| 60 / 30 | Weighted XGBoost over ETS | ETS | 78.30 / 81.73 | 4.2% | [2.26, 4.56] |
| 60 / 60 | Weighted XGBoost over ETS | ETS | 78.38 / 83.10 | 5.7% | [3.45, 6.04] |
| 360 / 30 | XGBoost over ETS | Seasonal | 225.16 / 231.01 | 2.5% | [-11.69, 27.27] |
| 360 / 60 | Weighted XGBoost over ETS | Seasonal | 230.06 / 233.07 | 1.3% | [-13.19, 21.58] |

The intervals include zero in three settings. The six-hour result is not a statistically resolved improvement over the seasonal baseline. At the shortest horizons, ETS is a stronger anchor than the article-inspired comparators in this dataset.

## What helped, and what did not

Preserving local seasonality mattered more than adding a larger architecture. Pooling the correction rather than the whole forecasting function gave the model a narrower problem. Station-scale normalization and load weighting helped some global variants, but their central scores still lagged ETS.

Fixed and context-dependent blends did not match the best residual model. Selecting an anchor automatically on one validation day could also be unstable. Keeping ETS as an explicit anchor was a useful control.

Sequence models performed poorly here. This does not establish that they are generally poor for passenger forecasting. A week provides too few independent cycles for a strong comparison of high-capacity architectures.

## Uncertainty and limitations

The bootstrap resamples stations with their complete error histories. It preserves overlap within a station, but stations share calendar shocks and network effects. It does not resample weeks, random seeds or the entire model search. The intervals are conditional on the observed period, chosen models and station-resampling assumption.

The number of events is large; the time coverage is short. Friday, weekends, holidays and seasonal changes are not validated by the main comparison. The next step would be additional weeks and an untouched calendar holdout, with the target and model-selection procedure fixed in advance.

The source data cannot be redistributed. This repository supports code inspection, synthetic checks and verification of saved aggregate results, not an independent rerun of the full thesis from public data alone.

## Evidence and release scope

The central results and confidence intervals come from `tests/fixtures/h060m_b015m/`, committed with the original study code at `69eea0e`. The horizon comparison is transcribed from the final thesis, using the source files `table_grid_summary_with_ci.tex` and `table_central_significance_extended.tex`.

The public cleanup adds documentation and tests. It also removes target-dependent fallbacks from the seasonal anchor, classical-model failure paths and sequence-model missing predictions. These paths previously consulted evaluation-target means if historical inputs were missing. Whether they affected the recorded scores cannot be established from aggregate fixtures alone. Saved thesis scores are preserved, not certified as a rerun of the corrected code. The original code remains available in Git history.
