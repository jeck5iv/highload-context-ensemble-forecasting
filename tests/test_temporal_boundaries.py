import numpy as np
import pandas as pd

from metroflow.studies.metro_short_history_forecasting.config import MultiStationForecastConfig
from metroflow.studies.metro_short_history_forecasting.data_layer.supervised import _clip_per_station_day
from metroflow.studies.metro_short_history_forecasting.models.meta import _seasonal_anchor_series
from metroflow.studies.metro_short_history_forecasting.models import classical, sequence


def _fallback_frame():
    cfg = MultiStationForecastConfig()
    return pd.DataFrame({
        cfg.time_col: pd.date_range("2025-03-10 03:00", periods=3, freq="15min"),
        "station_key": ["s"] * 3, "count": [1.0] * 3,
        "target_daylag": [12.0, np.nan, np.nan],
        "lag_96": [10.0, 20.0, np.nan],
        "target_h": [100.0, 200.0, 300.0],
    })


def test_failed_classical_models_do_not_use_future_targets(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("synthetic model failure")
    monkeypatch.setattr(classical, "_fit_predict_station_ets", fail)
    monkeypatch.setattr(classical, "_fit_predict_station_sarima", fail)
    cfg = MultiStationForecastConfig()
    frame = _fallback_frame()
    changed = frame.assign(target_h=1e9)
    for candidate in (frame, changed):
        ets = classical._run_local_ets(cfg, frame, frame, frame, frame, candidate, candidate)
        assert ets["test_pred"].tolist() == [12.0, 20.0, 0.0]
        sarima, _ = classical._local_sarima_prediction_for_split(
            cfg, frame, frame, candidate, (1, 0, 0), (0, 0, 0),
        )
        assert sarima.tolist() == [12.0, 20.0, 0.0]


def test_missing_sequence_predictions_do_not_use_future_targets():
    cfg = MultiStationForecastConfig()
    payload = {"eval_stds": np.array([]), "eval_means": np.array([]), "eval_indices": []}
    for frame in (_fallback_frame(), _fallback_frame().assign(target_h=1e9)):
        pred = sequence._sequence_eval_pred_to_frame_order(payload, np.array([]), frame, cfg)
        assert pred.tolist() == [12.0, 20.0, 0.0]


def test_seasonal_fallback_does_not_read_targets():
    frame = pd.DataFrame({"target_daylag": [12.0, np.nan, np.nan],
                          "lag_96": [10.0, 20.0, np.nan],
                          "target_h": [100.0, 200.0, 300.0]})
    expected = _seasonal_anchor_series(frame)
    changed = frame.copy()
    changed["target_h"] = [1e8, -1e8, 1e9]
    pd.testing.assert_series_equal(expected, _seasonal_anchor_series(changed))
    assert expected.tolist() == [12.0, 20.0, 0.0]


def test_seasonal_anchor_runs_without_labels():
    frame = pd.DataFrame({"target_daylag": [12.0, np.nan]})
    assert _seasonal_anchor_series(frame).tolist() == [12.0, 0.0]


def test_training_window_does_not_cross_day_boundary():
    cfg = MultiStationForecastConfig(bucket_minutes=15, horizon_minutes=60)
    times = pd.date_range("2025-03-10 03:00", periods=96, freq="15min")
    frame = pd.DataFrame({
        cfg.time_col: times, "station_key": "synthetic", "date": times[0].date(),
    })
    clipped = _clip_per_station_day(frame, cfg)
    assert len(clipped) == 92
    assert clipped[cfg.time_col].max() + pd.Timedelta(minutes=60) <= times.max()


def test_clipping_respects_station_boundaries():
    cfg = MultiStationForecastConfig(bucket_minutes=15, horizon_minutes=60)
    a = pd.date_range("2025-03-10 03:00", periods=96, freq="15min")
    b = pd.date_range("2025-03-10 03:00", periods=80, freq="15min")
    frame = pd.concat([
        pd.DataFrame({cfg.time_col: t, "station_key": key, "date": t[0].date()})
        for key, t in [("a", a), ("b", b)]
    ])
    clipped = _clip_per_station_day(frame, cfg)
    assert clipped.groupby("station_key").size().to_dict() == {"a": 92, "b": 76}
