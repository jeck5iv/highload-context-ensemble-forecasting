from __future__ import annotations

from ..common import *
from .meta import _seasonal_anchor_series
from ..config import (
    DEFAULT_CLASSIC_METRO_LINE_CODES,
    _scope_name,
    MultiStationForecastConfig,
)
from ..evaluation.metrics import (
    calc_metrics_df,
    _calc_metric_values,
    evaluate_predictions,
    BOOTSTRAP_METRICS,
    PREDICTION_SINGLE_MODEL_CANDIDATES,
    ARTICLE_COMPARATOR_MODELS,
    _choose_bootstrap_reference_models,
    _safe_divide,
    _station_metric_component_arrays,
    station_bootstrap_ci,
)

def _future_sum_from_count_path(count_path: np.ndarray, horizon: int) -> np.ndarray:
    count_path = np.asarray(count_path, dtype=float)
    if len(count_path) <= horizon:
        return np.empty(0, dtype=float)
    return np.array([count_path[i + 1:i + horizon + 1].sum() for i in range(len(count_path) - horizon)], dtype=float)

def _fit_predict_station_ets(train_count_df: pd.DataFrame, pred_count_df: pd.DataFrame, cfg: MultiStationForecastConfig) -> np.ndarray:
    """Forecast a full count path for one station and aggregate to target_h anchors."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y_train = train_count_df["count"].astype(float).values
    if len(y_train) < 2 * cfg.day_lag or not np.isfinite(y_train).all():
        raise ValueError("Insufficient or non-finite training observations for ETS")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            y_train,
            trend="add",
            seasonal="add",
            seasonal_periods=cfg.day_lag,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True, use_brute=False)
        count_forecast = np.asarray(fit.forecast(len(pred_count_df)), dtype=float)
    count_forecast = np.maximum(count_forecast, 0.0)
    return _future_sum_from_count_path(count_forecast, horizon=cfg.horizon_steps)

def _run_local_ets(
    cfg: MultiStationForecastConfig,
    train_tune_all: pd.DataFrame,
    val_all: pd.DataFrame,
    train_full_all: pd.DataFrame,
    test_all: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Fit ETS independently per station; use seasonal day-lag fallback if ETS fails."""
    val_pred = pd.Series(index=val_df.index, dtype=float)
    test_pred = pd.Series(index=test_df.index, dtype=float)
    fallback_stations_val: list[str] = []
    fallback_stations_test: list[str] = []

    for station_key, eval_station in val_df.groupby("station_key", sort=False):
        train_station = train_tune_all[train_tune_all["station_key"] == station_key].sort_values(cfg.time_col)
        pred_station = val_all[val_all["station_key"] == station_key].sort_values(cfg.time_col)
        eval_idx = eval_station.sort_values(cfg.time_col).index
        try:
            pred = _fit_predict_station_ets(train_station, pred_station, cfg)
            if len(pred) < len(eval_idx) or not np.isfinite(pred[:len(eval_idx)]).all():
                raise ValueError("ETS produced too few/non-finite validation predictions")
            val_pred.loc[eval_idx] = pred[:len(eval_idx)]
        except Exception:
            fallback_stations_val.append(str(station_key))
            fallback = _seasonal_anchor_series(eval_station.sort_values(cfg.time_col))
            val_pred.loc[eval_idx] = fallback.values.astype(float)

    for station_key, eval_station in test_df.groupby("station_key", sort=False):
        train_station = train_full_all[train_full_all["station_key"] == station_key].sort_values(cfg.time_col)
        pred_station = test_all[test_all["station_key"] == station_key].sort_values(cfg.time_col)
        eval_idx = eval_station.sort_values(cfg.time_col).index
        try:
            pred = _fit_predict_station_ets(train_station, pred_station, cfg)
            if len(pred) < len(eval_idx) or not np.isfinite(pred[:len(eval_idx)]).all():
                raise ValueError("ETS produced too few/non-finite test predictions")
            test_pred.loc[eval_idx] = pred[:len(eval_idx)]
        except Exception:
            fallback_stations_test.append(str(station_key))
            fallback = _seasonal_anchor_series(eval_station.sort_values(cfg.time_col))
            test_pred.loc[eval_idx] = fallback.values.astype(float)

    if val_pred.isna().any() or test_pred.isna().any():
        raise ValueError("Local ETS produced missing predictions after fallback")
    return {
        "val_pred": val_pred.loc[val_df.index].values.astype(float),
        "test_pred": test_pred.loc[test_df.index].values.astype(float),
        "n_val_fallback_stations": len(fallback_stations_val),
        "n_test_fallback_stations": len(fallback_stations_test),
        "val_fallback_stations": fallback_stations_val,
        "test_fallback_stations": fallback_stations_test,
    }

def _fit_predict_station_sarima(
    train_count_df: pd.DataFrame,
    pred_count_df: pd.DataFrame,
    cfg: MultiStationForecastConfig,
    order: tuple[int, int, int],
    seasonal_order_wo_period: tuple[int, int, int],
) -> np.ndarray:
    """Forecast a station count path with a compact seasonal ARIMA candidate."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y_train = train_count_df["count"].astype(float).values
    if len(y_train) < 2 * cfg.day_lag or not np.isfinite(y_train).all():
        raise ValueError("Insufficient or non-finite training observations for SARIMA")
    seasonal_order = (*seasonal_order_wo_period, int(cfg.day_lag))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            y_train,
            order=tuple(int(x) for x in order),
            seasonal_order=tuple(int(x) for x in seasonal_order),
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False, maxiter=max(5, int(cfg.sarima_maxiter)))
        count_forecast = np.asarray(fit.forecast(len(pred_count_df)), dtype=float)
    count_forecast = np.maximum(count_forecast, 0.0)
    return _future_sum_from_count_path(count_forecast, horizon=cfg.horizon_steps)

def _local_sarima_prediction_for_split(
    cfg: MultiStationForecastConfig,
    train_all_df: pd.DataFrame,
    pred_all_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    order: tuple[int, int, int],
    seasonal_order_wo_period: tuple[int, int, int],
) -> tuple[np.ndarray, list[str]]:
    pred = pd.Series(index=eval_df.index, dtype=float)
    fallback_stations: list[str] = []
    for station_key, eval_station in eval_df.groupby("station_key", sort=False):
        train_station = train_all_df[train_all_df["station_key"] == station_key].sort_values(cfg.time_col)
        pred_station = pred_all_df[pred_all_df["station_key"] == station_key].sort_values(cfg.time_col)
        eval_idx = eval_station.sort_values(cfg.time_col).index
        try:
            station_pred = _fit_predict_station_sarima(
                train_station,
                pred_station,
                cfg,
                order=order,
                seasonal_order_wo_period=seasonal_order_wo_period,
            )
            if len(station_pred) < len(eval_idx) or not np.isfinite(station_pred[:len(eval_idx)]).all():
                raise ValueError("SARIMA produced too few/non-finite predictions")
            pred.loc[eval_idx] = station_pred[:len(eval_idx)]
        except Exception:
            fallback_stations.append(str(station_key))
            fallback = _seasonal_anchor_series(eval_station.sort_values(cfg.time_col))
            pred.loc[eval_idx] = fallback.values.astype(float)
    if pred.isna().any():
        raise ValueError("Local SARIMA produced missing predictions after fallback")
    return pred.loc[eval_df.index].values.astype(float), fallback_stations

def _run_local_sarima(
    cfg: MultiStationForecastConfig,
    train_tune_all: pd.DataFrame,
    val_all: pd.DataFrame,
    train_full_all: pd.DataFrame,
    test_all: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Validation-select a tiny local SARIMA family and refit the selected candidate for test.

    The grid is deliberately small: the goal is to check whether a classical seasonal ARIMA
    family is competitive under the same short-history station-wise protocol, not to run an
    expensive exhaustive order search on 260 short series.
    """
    specs = [
        {"name": "SARIMA(1,0,0)x(0,1,0)", "order": (1, 0, 0), "seasonal": (0, 1, 0)},
        {"name": "SARIMA(0,0,1)x(0,1,0)", "order": (0, 0, 1), "seasonal": (0, 1, 0)},
    ]
    candidates: list[dict] = []
    rows: list[dict] = []
    for spec in specs:
        val_pred, fallback_val = _local_sarima_prediction_for_split(
            cfg,
            train_tune_all,
            val_all,
            val_df,
            order=spec["order"],
            seasonal_order_wo_period=spec["seasonal"],
        )
        metrics = calc_metrics_df(val_df, val_pred, cfg, mode="micro")
        candidates.append({**spec, "val_pred": val_pred, "fallback_val": fallback_val, "metrics": metrics})
        rows.append({
            "candidate": spec["name"],
            "n_validation_fallback_stations": int(len(fallback_val)),
            **metrics,
        })
    val_table = pd.DataFrame(rows).sort_values("LW-MAE").reset_index(drop=True)
    selected_name = str(val_table.loc[0, "candidate"])
    selected = next(c for c in candidates if c["name"] == selected_name)
    test_pred, fallback_test = _local_sarima_prediction_for_split(
        cfg,
        train_full_all,
        test_all,
        test_df,
        order=selected["order"],
        seasonal_order_wo_period=selected["seasonal"],
    )
    return {
        "val_pred": np.asarray(selected["val_pred"], dtype=float),
        "test_pred": np.asarray(test_pred, dtype=float),
        "validation_table": val_table,
        "selected_candidate": selected_name,
        "selected_order": tuple(int(x) for x in selected["order"]),
        "selected_seasonal_order": (*tuple(int(x) for x in selected["seasonal"]), int(cfg.day_lag)),
        "n_val_fallback_stations": int(len(selected["fallback_val"])),
        "n_test_fallback_stations": int(len(fallback_test)),
        "val_fallback_stations": selected["fallback_val"],
        "test_fallback_stations": fallback_test,
        "test_metrics_micro": calc_metrics_df(test_df, test_pred, cfg, mode="micro"),
    }

__all__ = [
    "_future_sum_from_count_path",
    "_fit_predict_station_ets",
    "_run_local_ets",
    "_fit_predict_station_sarima",
    "_local_sarima_prediction_for_split",
    "_run_local_sarima",
]
