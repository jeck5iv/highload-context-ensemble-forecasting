from __future__ import annotations

import pandas as pd

from metroflow.studies.metro_short_history_forecasting import MultiStationForecastConfig
from metroflow.studies.metro_short_history_forecasting.entrypoint import run_multistation_forecasting_experiment as entrypoint_runner
from metroflow.studies.metro_short_history_forecasting.evaluation import evaluate_predictions, station_bootstrap_ci
from metroflow.studies.metro_short_history_forecasting.pipeline import run_multistation_forecasting_experiment as pipeline_runner


def _toy_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_key": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "y_true": [100.0, 120.0, 40.0, 60.0, 80.0, 90.0],
            "hl_actual": [0, 1, 0, 1, 0, 1],
            "station_train_mean_target": [100.0, 100.0, 50.0, 50.0, 80.0, 80.0],
            "ETS local": [98.0, 116.0, 44.0, 55.0, 75.0, 95.0],
            "Residual XGB ETS LW": [99.0, 118.0, 42.0, 58.0, 78.0, 91.0],
        }
    )


def test_entrypoint_points_to_pipeline() -> None:
    assert entrypoint_runner is pipeline_runner


def test_default_seed_is_locked() -> None:
    assert MultiStationForecastConfig().random_state == 3228


def test_metrics_and_bootstrap_are_deterministic() -> None:
    cfg = MultiStationForecastConfig(n_bootstrap=32, run_bootstrap=True)
    table = _toy_predictions()
    models = ["ETS local", "Residual XGB ETS LW"]

    micro, macro = evaluate_predictions(table, models, cfg)
    assert micro.index[0] == "Residual XGB ETS LW"
    assert macro.index[0] == "Residual XGB ETS LW"

    first = station_bootstrap_ci(table, models, cfg, baseline_cols=["ETS local"])
    second = station_bootstrap_ci(table, models, cfg, baseline_cols=["ETS local"])
    pd.testing.assert_frame_equal(first["metric_ci"], second["metric_ci"])
    pd.testing.assert_frame_equal(first["delta_ci"], second["delta_ci"])
