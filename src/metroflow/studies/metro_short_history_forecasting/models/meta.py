from __future__ import annotations

from ..common import *
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

def _seasonal_anchor_series(df: pd.DataFrame) -> pd.Series:
    seasonal = pd.to_numeric(df["target_daylag"], errors="coerce").astype(float)
    if "lag_96" in df.columns:
        seasonal = seasonal.fillna(pd.to_numeric(df["lag_96"], errors="coerce").astype(float))
    return seasonal.fillna(0.0).astype(float)

def seasonal_daylag_predictions(val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    val_pred = _seasonal_anchor_series(val_df).values.astype(float)
    test_pred = _seasonal_anchor_series(test_df).values.astype(float)
    return {"val_pred": val_pred, "test_pred": test_pred}

def _positive_blend(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    reg = LinearRegression(fit_intercept=False, positive=True)
    reg.fit(X, y)
    w = np.asarray(reg.coef_, dtype=float).clip(min=0)
    if not np.isfinite(w).all() or w.sum() <= 0:
        return np.ones(X.shape[1], dtype=float) / X.shape[1]
    return w / w.sum()

def _context_blend(val_pred_table: pd.DataFrame, test_pred_table: pd.DataFrame, model_cols: list[str]) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]:
    y = val_pred_table["y_true"].values.astype(float)
    X = val_pred_table[model_cols].values.astype(float)
    ctx = val_pred_table["hl_daylag"].values.astype(int)
    global_w = _positive_blend(X, y)
    by_ctx: dict[int, np.ndarray] = {}
    for v in [0, 1]:
        mask = ctx == v
        if mask.sum() >= max(30, len(model_cols) * 5):
            by_ctx[v] = _positive_blend(X[mask], y[mask])
        else:
            by_ctx[v] = global_w.copy()
    X_test = test_pred_table[model_cols].values.astype(float)
    ctx_test = test_pred_table["hl_daylag"].values.astype(int)
    pred = np.zeros(len(test_pred_table), dtype=float)
    for i in range(len(test_pred_table)):
        pred[i] = X_test[i] @ by_ctx[int(ctx_test[i])]
    return global_w, by_ctx, pred

def _build_meta_tables(split: dict, preds: dict[str, dict], cfg: MultiStationForecastConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_df = split["val_df"]
    test_df = split["test_df"]
    base_cols = [
        cfg.time_col, "station_key", "ST_CODE", "ST_NAME", "ST_NAME_SHORT", "LN_CODE", "LN_NAME", "LN_NAME_SHORT",
        "target_h", "hl_daylag", "hl_actual", "q80_station", "hour_sin", "hour_cos", "continuous_load_score",
        "local_slope_1", "local_slope_2", "daylag_deviation", "prev_count_ratio_daylag", "roll_std_2", "roll_std_4", "roll_std_8", "roll_std_12", "roll_std_16",
        "station_key_code", "station_id_raw", "line_id_raw", "station_name_code", "line_name_code",
        "n_place_ids", "line_n_stations", "station_order_on_line_proxy", "station_order_norm",
        "distance_to_nearest_terminal_proxy", "distance_to_nearest_terminal_norm", "is_terminal_proxy", "is_ring_line_proxy",
        "station_train_mean_target", "station_train_median_target", "station_train_q80_target", "station_train_q95_target",
        "station_train_max_target", "station_train_mean_count", "station_mean_target_vs_global",
        "station_q80_target_vs_global", "station_mean_count_vs_global", "station_code", "line_code",
    ]
    val = val_df[[c for c in base_cols if c in val_df.columns]].copy().rename(columns={"target_h": "y_true"})
    test = test_df[[c for c in base_cols if c in test_df.columns]].copy().rename(columns={"target_h": "y_true"})
    for name, payload in preds.items():
        val[name] = payload["val_pred"]
        test[name] = payload["test_pred"]
    return val, test

def _run_meta_models(cfg: MultiStationForecastConfig, val_pred_table: pd.DataFrame, test_pred_table: pd.DataFrame) -> dict:
    model_cols = [
        c for c in [
            "Seasonal daylag", "ETS local", "MegaBoost base", "MegaBoost rich",
            "Global LSTM", "Global Transformer-lite",
        ]
        if c in val_pred_table.columns
    ]
    y_val = val_pred_table["y_true"].values.astype(float)

    w_static = _positive_blend(val_pred_table[model_cols].values.astype(float), y_val)
    pred_static = test_pred_table[model_cols].values.astype(float) @ w_static

    w_global, w_by_ctx, pred_context = _context_blend(val_pred_table, test_pred_table, model_cols)

    out = {
        "model_cols": model_cols,
        "static_weights": dict(zip(model_cols, w_static)),
        "context_global_weights": dict(zip(model_cols, w_global)),
        "context_weights_by_hl_daylag": {int(k): dict(zip(model_cols, v)) for k, v in w_by_ctx.items()},
        "pred_static": pred_static,
        "pred_context": pred_context,
    }

    if cfg.run_residual_xgb:
        val_metrics = {c: calc_metrics_df(val_pred_table.rename(columns={"y_true": "target_h"}), val_pred_table[c].values, cfg, mode="micro") for c in model_cols}
        primary = pd.DataFrame(val_metrics).T.sort_values("MAE").index[0]
        resid_val = y_val - val_pred_table[primary].values.astype(float)
        X_val = _meta_feature_matrix(val_pred_table, model_cols, primary)
        X_test = _meta_feature_matrix(test_pred_table, model_cols, primary)

        ridge = Ridge(alpha=10.0)
        ridge.fit(X_val, resid_val)
        pred_resid_linear = ridge.predict(X_test)
        lo, hi = np.quantile(resid_val, [0.05, 0.95])
        pred_resid_linear = np.clip(pred_resid_linear, lo, hi)

        params = dict(objective="reg:absoluteerror", n_estimators=80, learning_rate=0.03, max_depth=2,
                      min_child_weight=20, subsample=0.85, colsample_bytree=0.85,
                      reg_lambda=20.0, reg_alpha=4.0, tree_method="hist", random_state=cfg.random_state)
        rxgb = xgb.XGBRegressor(**params)
        rxgb.fit(X_val, resid_val)
        pred_resid_xgb = rxgb.predict(X_test)
        lo, hi = np.quantile(resid_val, [0.10, 0.90])
        pred_resid_xgb = np.clip(pred_resid_xgb, lo, hi)

        primary_test = test_pred_table[primary].values.astype(float)
        out.update({
            "primary_model": primary,
            "pred_residual_linear": primary_test + pred_resid_linear,
            "pred_residual_xgb": primary_test + pred_resid_xgb,
            "residual_linear_coef": pd.DataFrame({"feature": X_val.columns, "coef": ridge.coef_}),
            "residual_xgb_importance": pd.DataFrame({"feature": X_val.columns, "importance": rxgb.feature_importances_}),
        })
    return out

def _meta_feature_matrix(df: pd.DataFrame, model_cols: list[str], primary_col: str) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in model_cols:
        X[f"pred_{c}"] = df[c].values.astype(float)
    for c in model_cols:
        if c != primary_col:
            X[f"diff_{c}_minus_{primary_col}"] = df[c].values.astype(float) - df[primary_col].values.astype(float)
            X[f"absdiff_{c}_{primary_col}"] = np.abs(df[c].values.astype(float) - df[primary_col].values.astype(float))
    mat = df[model_cols].values.astype(float)
    X["pred_mean"] = mat.mean(axis=1)
    X["pred_std"] = mat.std(axis=1)
    X["pred_range"] = mat.max(axis=1) - mat.min(axis=1)
    for c in [
        "hl_daylag", "hour_sin", "hour_cos", "continuous_load_score", "local_slope_1", "local_slope_2",
        "daylag_deviation", "prev_count_ratio_daylag", "roll_std_2", "roll_std_4", "roll_std_8", "roll_std_12", "roll_std_16", "q80_station",
        "station_train_mean_target", "station_train_median_target", "station_train_q80_target", "station_train_q95_target",
        "station_train_max_target", "station_train_mean_count", "station_mean_target_vs_global",
        "station_q80_target_vs_global", "station_mean_count_vs_global",
        "station_key_code", "station_id_raw", "line_id_raw", "station_name_code", "line_name_code",
        "n_place_ids", "line_n_stations", "station_order_on_line_proxy", "station_order_norm",
        "distance_to_nearest_terminal_proxy", "distance_to_nearest_terminal_norm", "is_terminal_proxy", "is_ring_line_proxy",
        "weather_temperature_2m", "weather_relative_humidity_2m", "weather_precipitation", "weather_rain",
        "weather_snowfall", "weather_snow_depth", "weather_weather_code", "weather_wind_speed_10m",
        "station_code", "line_code",
    ]:
        if c in df.columns:
            X[c] = df[c].values
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

__all__ = [
    "_seasonal_anchor_series",
    "seasonal_daylag_predictions",
    "_positive_blend",
    "_context_blend",
    "_build_meta_tables",
    "_run_meta_models",
    "_meta_feature_matrix",
]
