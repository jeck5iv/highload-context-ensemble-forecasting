from __future__ import annotations

from ..common import *
from ..config import (
    DEFAULT_CLASSIC_METRO_LINE_CODES,
    _scope_name,
    MultiStationForecastConfig,
)
from ..features import (
    BASE_FEATURES,
    RICH_FEATURES,
    _available_features,
    _xgb_matrix,
    _boost_lw_sample_weight,
    _aligned_sample_weight,
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

NORMALIZED_MAGNITUDE_PREFIXES = ("lag_", "roll_sum_", "roll_mean_", "roll_std_")

NORMALIZED_MAGNITUDE_COLS = {"target_daylag"}

def _station_scale_series(df: pd.DataFrame) -> pd.Series:
    """Positive station-level scale used for normalized boosting."""
    candidates = [
        "station_train_q80_target",
        "q80_station",
        "station_train_mean_target",
        "station_train_median_target",
        "station_train_mean_count",
    ]
    scale = None
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").astype(float)
            scale = s if scale is None else scale.fillna(s)
    if scale is None:
        scale = pd.Series(np.nan, index=df.index, dtype=float)
    fallback = float(np.nanmedian(scale.values)) if np.isfinite(scale.values).any() else 1.0
    scale = scale.replace([np.inf, -np.inf], np.nan).fillna(fallback).clip(lower=1.0)
    return scale.astype(float)

def _normalize_boost_frame(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Create normalized lag/rolling features and normalized target for global boosting.

    The goal is to let one global model learn profile shape instead of wasting capacity on
    station-specific passenger-flow scale.
    """
    out = df.copy()
    scale = _station_scale_series(out)
    out["__station_scale"] = scale.values
    normalized_features: list[str] = []
    for c in feature_cols:
        if c not in out.columns:
            continue
        is_magnitude = c in NORMALIZED_MAGNITUDE_COLS or c.startswith(NORMALIZED_MAGNITUDE_PREFIXES)
        if is_magnitude:
            nc = f"norm__{c}"
            out[nc] = pd.to_numeric(out[c], errors="coerce").astype(float) / scale
            normalized_features.append(nc)
        else:
            normalized_features.append(c)
    out["target_norm"] = pd.to_numeric(out["target_h"], errors="coerce").astype(float) / scale
    return out, normalized_features

def _fit_predict_xgb_normalized(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, object]:
    train_n, norm_features = _normalize_boost_frame(train_df, feature_cols)
    pred_n, _ = _normalize_boost_frame(pred_df, feature_cols)
    train_part = train_n.dropna(subset=["target_norm"]).copy()
    fit_weight = _aligned_sample_weight(train_n, train_part, sample_weight)
    model = xgb.XGBRegressor(**params)
    model.fit(
        _xgb_matrix(train_part, norm_features),
        train_part["target_norm"].astype(float),
        sample_weight=fit_weight,
    )
    pred_norm = np.asarray(model.predict(_xgb_matrix(pred_n, norm_features)), dtype=float)
    pred = np.maximum(pred_norm * pred_n["__station_scale"].values.astype(float), 0.0)
    return pred, model

def _run_global_megaboost_normalized(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    sample_weight_mode: str | None = None,
    selection_metric: str = "LW-MAE",
    param_grid: list[dict] | None = None,
    top_k: int | None = None,
    seed_offset: int = 0,
) -> dict:
    candidates = []
    train_weight = _boost_lw_sample_weight(train_df, cfg) if sample_weight_mode == "lw" else None
    train_full_weight = _boost_lw_sample_weight(train_full_df, cfg) if sample_weight_mode == "lw" else None
    grid = list(param_grid) if param_grid is not None else list(cfg.boosting_param_grid)
    keep_k = int(cfg.top_k_boost_models if top_k is None else top_k)
    for i, base_params in enumerate(grid, start=1):
        params = dict(base_params)
        params["random_state"] = cfg.random_state + 1000 + seed_offset + i
        pred_val, _ = _fit_predict_xgb_normalized(train_df, val_df, feature_cols, params, sample_weight=train_weight)
        candidates.append({
            "name": f"{label}__cfg{i}",
            "params": params,
            "val_pred": pred_val,
            "features": feature_cols,
            "metrics": calc_metrics_df(val_df, pred_val, cfg, mode="micro"),
        })
    val_table = pd.DataFrame([{"model": c["name"], **c["metrics"]} for c in candidates]).sort_values(selection_metric).reset_index(drop=True)
    selected_names = val_table.head(max(1, keep_k))["model"].tolist()
    selected = [c for c in candidates if c["name"] in selected_names]
    val_pred = np.mean(np.column_stack([c["val_pred"] for c in selected]), axis=1)

    test_preds = []
    models = []
    for c in selected:
        pred_test, model = _fit_predict_xgb_normalized(
            train_full_df,
            test_df,
            c["features"],
            c["params"],
            sample_weight=train_full_weight,
        )
        test_preds.append(pred_test)
        models.append(model)
    test_pred = np.mean(np.column_stack(test_preds), axis=1)
    return {
        "label": label,
        "features": feature_cols,
        "val_table": val_table,
        "selected_names": selected_names,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "test_metrics_micro": calc_metrics_df(test_df, test_pred, cfg, mode="micro"),
        "models": models,
        "normalized_target": True,
        "sample_weight_mode": sample_weight_mode,
        "selection_metric": selection_metric,
    }

def _fit_predict_xgb_residual(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
    anchor_train_pred: np.ndarray,
    anchor_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
    clip_quantiles: tuple[float, float] = (0.05, 0.95),
) -> tuple[np.ndarray, object]:
    train_work = train_df.copy()
    train_work["__anchor_pred"] = np.asarray(anchor_train_pred, dtype=float)
    train_work["__residual_target"] = train_work["target_h"].astype(float) - train_work["__anchor_pred"].astype(float)
    train_part = train_work.dropna(subset=["__residual_target"]).copy()
    if train_part.empty:
        raise ValueError("No residual training rows with non-missing target_h and anchor prediction")
    fit_weight = _aligned_sample_weight(train_work, train_part, sample_weight)
    model = xgb.XGBRegressor(**params)
    model.fit(
        _xgb_matrix(train_part, feature_cols),
        train_part["__residual_target"].astype(float),
        sample_weight=fit_weight,
    )
    residual_pred = np.asarray(model.predict(_xgb_matrix(pred_df, feature_cols)), dtype=float)
    lo_q, hi_q = clip_quantiles
    if 0.0 <= lo_q < hi_q <= 1.0:
        lo, hi = np.quantile(train_part["__residual_target"].astype(float).values, [lo_q, hi_q])
        residual_pred = np.clip(residual_pred, lo, hi)
    pred = np.maximum(np.asarray(anchor_pred, dtype=float) + residual_pred, 0.0)
    return pred, model

def _run_global_residual_megaboost(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    anchor_train_pred: np.ndarray,
    anchor_val_pred: np.ndarray,
    anchor_train_full_pred: np.ndarray,
    anchor_test_pred: np.ndarray,
    sample_weight_mode: str | None = None,
    selection_metric: str = "LW-MAE",
    param_grid: list[dict] | None = None,
    top_k: int | None = None,
    seed_offset: int = 0,
) -> dict:
    candidates = []
    train_weight = _boost_lw_sample_weight(train_df, cfg) if sample_weight_mode == "lw" else None
    train_full_weight = _boost_lw_sample_weight(train_full_df, cfg) if sample_weight_mode == "lw" else None
    grid = list(param_grid) if param_grid is not None else list(cfg.boosting_param_grid)
    keep_k = int(cfg.top_k_boost_models if top_k is None else top_k)
    for i, base_params in enumerate(grid, start=1):
        params = dict(base_params)
        params["random_state"] = cfg.random_state + 3000 + seed_offset + i
        pred_val, _ = _fit_predict_xgb_residual(
            train_df,
            val_df,
            feature_cols,
            params,
            anchor_train_pred=anchor_train_pred,
            anchor_pred=anchor_val_pred,
            sample_weight=train_weight,
        )
        candidates.append({
            "name": f"{label}__cfg{i}",
            "params": params,
            "val_pred": pred_val,
            "features": feature_cols,
            "metrics": calc_metrics_df(val_df, pred_val, cfg, mode="micro"),
        })
    val_table = pd.DataFrame([{"model": c["name"], **c["metrics"]} for c in candidates]).sort_values(selection_metric).reset_index(drop=True)
    selected_names = val_table.head(max(1, keep_k))["model"].tolist()
    selected = [c for c in candidates if c["name"] in selected_names]
    val_pred = np.mean(np.column_stack([c["val_pred"] for c in selected]), axis=1)

    test_preds = []
    models = []
    for c in selected:
        pred_test, model = _fit_predict_xgb_residual(
            train_full_df,
            test_df,
            c["features"],
            c["params"],
            anchor_train_pred=anchor_train_full_pred,
            anchor_pred=anchor_test_pred,
            sample_weight=train_full_weight,
        )
        test_preds.append(pred_test)
        models.append(model)
    test_pred = np.mean(np.column_stack(test_preds), axis=1)
    return {
        "label": label,
        "features": feature_cols,
        "val_table": val_table,
        "selected_names": selected_names,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "test_metrics_micro": calc_metrics_df(test_df, test_pred, cfg, mode="micro"),
        "models": models,
        "anchor_family": "seasonal",
        "sample_weight_mode": sample_weight_mode,
        "selection_metric": selection_metric,
    }

def _run_validation_residual_megaboost(
    cfg: MultiStationForecastConfig,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    anchor_val_pred: np.ndarray,
    anchor_test_pred: np.ndarray,
    sample_weight_mode: str | None = None,
) -> dict:
    """Fit a rich XGB residual correction on validation residuals and apply it to test.

    This ETS-anchored boosting block is intentionally kept outside the meta-model pool:
    under the current two-day tune split we have a clean validation residual target for
    ETS, but not a full earlier ETS residual training split.  It therefore serves as a
    validation-calibrated boosting correction, not as an input to the validation-trained
    meta-layer.
    """
    train_weight = _boost_lw_sample_weight(val_df, cfg) if sample_weight_mode == "lw" else None
    test_preds = []
    models = []
    selected_names = []
    for i, base_params in enumerate(cfg.boosting_param_grid, start=1):
        params = dict(base_params)
        params["random_state"] = cfg.random_state + 6000 + i
        pred_test, model = _fit_predict_xgb_residual(
            val_df,
            test_df,
            feature_cols,
            params,
            anchor_train_pred=anchor_val_pred,
            anchor_pred=anchor_test_pred,
            sample_weight=train_weight,
        )
        test_preds.append(pred_test)
        models.append(model)
        selected_names.append(f"{label}__cfg{i}")
    test_pred = np.mean(np.column_stack(test_preds), axis=1)
    return {
        "label": label,
        "features": feature_cols,
        "selected_names": selected_names,
        "test_pred": test_pred,
        "test_metrics_micro": calc_metrics_df(test_df, test_pred, cfg, mode="micro"),
        "models": models,
        "anchor_family": "ets",
        "validation_calibrated": True,
        "sample_weight_mode": sample_weight_mode,
    }

def _prediction_metric(df: pd.DataFrame, pred: np.ndarray, cfg: MultiStationForecastConfig, metric: str = "LW-MAE") -> float:
    return float(calc_metrics_df(df, np.asarray(pred, dtype=float), cfg, mode="micro")[metric])

def _blend_anchor_prediction(anchor_pred: np.ndarray, corrected_pred: np.ndarray, alpha: float) -> np.ndarray:
    anchor = np.asarray(anchor_pred, dtype=float)
    corrected = np.asarray(corrected_pred, dtype=float)
    return np.maximum(anchor + float(alpha) * (corrected - anchor), 0.0)

def _select_correction_alpha(
    eval_df: pd.DataFrame,
    anchor_pred: np.ndarray,
    corrected_pred: np.ndarray,
    cfg: MultiStationForecastConfig,
    metric: str = "LW-MAE",
) -> tuple[float, pd.DataFrame]:
    rows = []
    for alpha in cfg.local_correction_alpha_grid:
        pred = _blend_anchor_prediction(anchor_pred, corrected_pred, float(alpha))
        rows.append({"alpha": float(alpha), metric: _prediction_metric(eval_df, pred, cfg, metric)})
    table = pd.DataFrame(rows).sort_values([metric, "alpha"]).reset_index(drop=True)
    return float(table.loc[0, "alpha"]), table

def _line_code_column(df: pd.DataFrame) -> str:
    for c in ("line_id_raw", "LN_CODE", "line_code"):
        if c in df.columns:
            return c
    raise KeyError("No line identifier column is available for line-local MegaBoost")

def _run_line_local_normalized_megaboost(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    fallback_val_pred: np.ndarray,
    fallback_test_pred: np.ndarray,
    sample_weight_mode: str | None = "lw",
    selection_metric: str = "LW-MAE",
) -> dict:
    """Train one normalized rich booster per line and validation-gate it against the pooled fallback."""
    line_col = _line_code_column(val_df)
    val_series = pd.Series(np.asarray(fallback_val_pred, dtype=float), index=val_df.index, dtype=float)
    test_series = pd.Series(np.asarray(fallback_test_pred, dtype=float), index=test_df.index, dtype=float)
    fallback_val_series = val_series.copy()
    fallback_test_series = test_series.copy()
    rows: list[dict] = []
    payloads: dict[str, dict] = {}

    for line_i, (line_code, val_line) in enumerate(val_df.groupby(line_col, sort=True)):
        train_line = train_df[train_df[line_col].eq(line_code)].copy()
        train_full_line = train_full_df[train_full_df[line_col].eq(line_code)].copy()
        test_line = test_df[test_df[line_col].eq(line_code)].copy()
        fallback_line_pred = fallback_val_series.loc[val_line.index].values.astype(float)
        fallback_metric = _prediction_metric(val_line, fallback_line_pred, cfg, selection_metric)
        row = {
            "line_code": line_code,
            "n_train": int(len(train_line)),
            "n_val": int(len(val_line)),
            "n_train_full": int(len(train_full_line)),
            "n_test": int(len(test_line)),
            "fallback_metric": fallback_metric,
            "local_metric": np.nan,
            "selected": "fallback",
            "status": "fallback",
        }
        if len(train_line) < cfg.local_min_train_rows or len(val_line) < cfg.local_min_validation_rows or test_line.empty:
            row["status"] = "fallback_insufficient_rows"
            rows.append(row)
            continue
        try:
            local = _run_global_megaboost_normalized(
                cfg,
                train_line,
                val_line,
                train_full_line,
                test_line,
                feature_cols,
                label=f"{label} line {line_code}",
                sample_weight_mode=sample_weight_mode,
                selection_metric=selection_metric,
                param_grid=cfg.local_boosting_param_grid,
                top_k=cfg.top_k_local_boost_models,
                seed_offset=40_000 + 100 * line_i,
            )
            local_metric = _prediction_metric(val_line, local["val_pred"], cfg, selection_metric)
            use_local = bool(local_metric < fallback_metric)
            row.update({
                "local_metric": local_metric,
                "selected": "local" if use_local else "fallback",
                "status": "local_selected" if use_local else "fallback_validation_gate",
                "selected_names": "|".join(local.get("selected_names", [])),
            })
            payloads[str(line_code)] = local
            if use_local:
                val_series.loc[val_line.index] = np.asarray(local["val_pred"], dtype=float)
                test_series.loc[test_line.index] = np.asarray(local["test_pred"], dtype=float)
        except Exception as exc:  # Keep the benchmark robust on tiny or degenerate lines.
            row["status"] = f"fallback_error:{type(exc).__name__}"
        rows.append(row)

    return {
        "label": label,
        "val_pred": val_series.loc[val_df.index].values.astype(float),
        "test_pred": test_series.loc[test_df.index].values.astype(float),
        "test_metrics_micro": calc_metrics_df(test_df, test_series.loc[test_df.index].values.astype(float), cfg, mode="micro"),
        "line_selection_table": pd.DataFrame(rows),
        "line_payloads": payloads,
        "fallback_family": "global_rich_norm_lw",
        "sample_weight_mode": sample_weight_mode,
        "selection_metric": selection_metric,
    }

def _run_line_local_seasonal_residual_megaboost(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    anchor_train_pred: np.ndarray,
    anchor_val_pred: np.ndarray,
    anchor_train_full_pred: np.ndarray,
    anchor_test_pred: np.ndarray,
    sample_weight_mode: str | None = "lw",
    selection_metric: str = "LW-MAE",
) -> dict:
    """Train line-local seasonal residual boosters and shrink each correction by validation-selected alpha."""
    line_col = _line_code_column(val_df)
    train_anchor = pd.Series(np.asarray(anchor_train_pred, dtype=float), index=train_df.index, dtype=float)
    val_anchor = pd.Series(np.asarray(anchor_val_pred, dtype=float), index=val_df.index, dtype=float)
    train_full_anchor = pd.Series(np.asarray(anchor_train_full_pred, dtype=float), index=train_full_df.index, dtype=float)
    test_anchor = pd.Series(np.asarray(anchor_test_pred, dtype=float), index=test_df.index, dtype=float)
    val_series = val_anchor.copy()
    test_series = test_anchor.copy()
    rows: list[dict] = []
    payloads: dict[str, dict] = {}

    for line_i, (line_code, val_line) in enumerate(val_df.groupby(line_col, sort=True)):
        train_line = train_df[train_df[line_col].eq(line_code)].copy()
        train_full_line = train_full_df[train_full_df[line_col].eq(line_code)].copy()
        test_line = test_df[test_df[line_col].eq(line_code)].copy()
        anchor_val_line = val_anchor.loc[val_line.index].values.astype(float)
        anchor_metric = _prediction_metric(val_line, anchor_val_line, cfg, selection_metric)
        row = {
            "line_code": line_code,
            "n_train": int(len(train_line)),
            "n_val": int(len(val_line)),
            "n_train_full": int(len(train_full_line)),
            "n_test": int(len(test_line)),
            "anchor_metric": anchor_metric,
            "corrected_metric": np.nan,
            "selected_alpha": 0.0,
            "status": "anchor",
        }
        if len(train_line) < cfg.local_min_train_rows or len(val_line) < cfg.local_min_validation_rows or test_line.empty:
            row["status"] = "anchor_insufficient_rows"
            rows.append(row)
            continue
        try:
            local = _run_global_residual_megaboost(
                cfg,
                train_line,
                val_line,
                train_full_line,
                test_line,
                feature_cols,
                label=f"{label} line {line_code}",
                anchor_train_pred=train_anchor.loc[train_line.index].values.astype(float),
                anchor_val_pred=anchor_val_line,
                anchor_train_full_pred=train_full_anchor.loc[train_full_line.index].values.astype(float),
                anchor_test_pred=test_anchor.loc[test_line.index].values.astype(float),
                sample_weight_mode=sample_weight_mode,
                selection_metric=selection_metric,
                param_grid=cfg.local_boosting_param_grid,
                top_k=cfg.top_k_local_boost_models,
                seed_offset=60_000 + 100 * line_i,
            )
            alpha, alpha_table = _select_correction_alpha(
                val_line,
                anchor_val_line,
                local["val_pred"],
                cfg,
                selection_metric,
            )
            val_pred_line = _blend_anchor_prediction(anchor_val_line, local["val_pred"], alpha)
            test_pred_line = _blend_anchor_prediction(
                test_anchor.loc[test_line.index].values.astype(float),
                local["test_pred"],
                alpha,
            )
            corrected_metric = _prediction_metric(val_line, val_pred_line, cfg, selection_metric)
            row.update({
                "corrected_metric": corrected_metric,
                "selected_alpha": float(alpha),
                "status": "residual_selected" if alpha > 0 else "anchor_validation_gate",
                "selected_names": "|".join(local.get("selected_names", [])),
            })
            payloads[str(line_code)] = {**local, "alpha_table": alpha_table, "selected_alpha": alpha}
            val_series.loc[val_line.index] = val_pred_line
            test_series.loc[test_line.index] = test_pred_line
        except Exception as exc:
            row["status"] = f"anchor_error:{type(exc).__name__}"
        rows.append(row)

    return {
        "label": label,
        "val_pred": val_series.loc[val_df.index].values.astype(float),
        "test_pred": test_series.loc[test_df.index].values.astype(float),
        "test_metrics_micro": calc_metrics_df(test_df, test_series.loc[test_df.index].values.astype(float), cfg, mode="micro"),
        "line_selection_table": pd.DataFrame(rows),
        "line_payloads": payloads,
        "anchor_family": "seasonal",
        "sample_weight_mode": sample_weight_mode,
        "selection_metric": selection_metric,
    }

def _station_seasonal_fallback(eval_station: pd.DataFrame, cfg: MultiStationForecastConfig) -> np.ndarray:
    """Safe station-wise day-lag fallback used only when a local booster cannot be fitted."""
    ordered = eval_station.sort_values(cfg.time_col)
    fallback = pd.Series(np.nan, index=ordered.index, dtype=float)
    for col in ("target_daylag", f"lag_{cfg.day_lag}", "lag_96", "lag_288", "lag_48", "lag_24"):
        if col in ordered.columns:
            fallback = fallback.fillna(pd.to_numeric(ordered[col], errors="coerce").astype(float))
    target = pd.to_numeric(ordered["target_h"], errors="coerce").astype(float)
    station_level = float(target.mean()) if target.notna().any() else 0.0
    fallback = fallback.fillna(station_level).fillna(0.0)
    return fallback.values.astype(float)

def _run_station_local_normalized_megaboost(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    sample_weight_mode: str | None = None,
    selection_metric: str = "LW-MAE",
) -> dict:
    """Fit a fully station-local normalized XGBoost family and refit each selected station model on train+validation.

    Unlike ``LineBoost ...`` this is not validation-gated against a pooled global model:
    when a station-local booster can be fitted, its prediction is used. Only stations with
    insufficient rows or failed fits fall back to a station-wise seasonal day-lag predictor,
    mirroring the failure fallback strategy of ``ETS local``.
    """
    val_series = pd.Series(index=val_df.index, dtype=float)
    test_series = pd.Series(index=test_df.index, dtype=float)
    rows: list[dict] = []
    payloads: dict[str, dict] = {}

    for station_i, (station_key, val_station) in enumerate(val_df.groupby("station_key", sort=False)):
        train_station = train_df[train_df["station_key"].eq(station_key)].copy()
        train_full_station = train_full_df[train_full_df["station_key"].eq(station_key)].copy()
        test_station = test_df[test_df["station_key"].eq(station_key)].copy()

        val_station_sorted = val_station.sort_values(cfg.time_col)
        test_station_sorted = test_station.sort_values(cfg.time_col)

        fallback_val = _station_seasonal_fallback(val_station_sorted, cfg)
        fallback_test = (
            _station_seasonal_fallback(test_station_sorted, cfg)
            if not test_station_sorted.empty
            else np.empty(0, dtype=float)
        )
        val_series.loc[val_station_sorted.index] = fallback_val
        if not test_station_sorted.empty:
            test_series.loc[test_station_sorted.index] = fallback_test

        row = {
            "station_key": str(station_key),
            "n_train": int(len(train_station)),
            "n_val": int(len(val_station)),
            "n_train_full": int(len(train_full_station)),
            "n_test": int(len(test_station)),
            "fallback_metric": _prediction_metric(val_station_sorted, fallback_val, cfg, selection_metric),
            "local_metric": np.nan,
            "selected": "fallback",
            "status": "fallback",
        }
        if (
            len(train_station) < int(cfg.station_local_min_train_rows)
            or len(val_station) < int(cfg.station_local_min_validation_rows)
            or test_station.empty
        ):
            row["status"] = "fallback_insufficient_rows"
            rows.append(row)
            continue

        try:
            local = _run_global_megaboost_normalized(
                cfg,
                train_station,
                val_station,
                train_full_station,
                test_station,
                feature_cols,
                label=f"{label} station {station_key}",
                sample_weight_mode=sample_weight_mode,
                selection_metric=selection_metric,
                param_grid=cfg.station_boosting_param_grid,
                top_k=cfg.top_k_station_boost_models,
                seed_offset=80_000 + 100 * station_i,
            )
            val_pred = np.asarray(local["val_pred"], dtype=float)
            test_pred = np.asarray(local["test_pred"], dtype=float)
            if len(val_pred) != len(val_station) or len(test_pred) != len(test_station):
                raise ValueError("StationBoost produced prediction length mismatch")
            if not np.isfinite(val_pred).all() or not np.isfinite(test_pred).all():
                raise ValueError("StationBoost produced non-finite predictions")

            val_series.loc[val_station.index] = val_pred
            test_series.loc[test_station.index] = test_pred
            row.update({
                "local_metric": _prediction_metric(val_station, val_pred, cfg, selection_metric),
                "selected": "local",
                "status": "local_fit",
                "selected_names": "|".join(local.get("selected_names", [])),
            })
            payloads[str(station_key)] = local
        except Exception as exc:
            row["status"] = f"fallback_error:{type(exc).__name__}"
        rows.append(row)

    if val_series.isna().any() or test_series.isna().any():
        raise ValueError(f"{label} produced missing predictions after fallback")

    val_pred = val_series.loc[val_df.index].values.astype(float)
    test_pred = test_series.loc[test_df.index].values.astype(float)
    return {
        "label": label,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "test_metrics_micro": calc_metrics_df(test_df, test_pred, cfg, mode="micro"),
        "station_selection_table": pd.DataFrame(rows),
        "station_payloads": payloads,
        "fallback_family": "station_seasonal_daylag",
        "normalized_target": True,
        "sample_weight_mode": sample_weight_mode,
        "selection_metric": selection_metric,
    }

def _run_station_local_seasonal_residual_megaboost(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    anchor_train_pred: np.ndarray,
    anchor_val_pred: np.ndarray,
    anchor_train_full_pred: np.ndarray,
    anchor_test_pred: np.ndarray,
    sample_weight_mode: str | None = "lw",
    selection_metric: str = "LW-MAE",
) -> dict:
    """Fit one station-local seasonal residual booster per station with validation-selected shrinkage."""
    train_anchor = pd.Series(np.asarray(anchor_train_pred, dtype=float), index=train_df.index, dtype=float)
    val_anchor = pd.Series(np.asarray(anchor_val_pred, dtype=float), index=val_df.index, dtype=float)
    train_full_anchor = pd.Series(np.asarray(anchor_train_full_pred, dtype=float), index=train_full_df.index, dtype=float)
    test_anchor = pd.Series(np.asarray(anchor_test_pred, dtype=float), index=test_df.index, dtype=float)

    val_series = val_anchor.copy()
    test_series = test_anchor.copy()
    rows: list[dict] = []
    payloads: dict[str, dict] = {}

    for station_i, (station_key, val_station) in enumerate(val_df.groupby("station_key", sort=False)):
        train_station = train_df[train_df["station_key"].eq(station_key)].copy()
        train_full_station = train_full_df[train_full_df["station_key"].eq(station_key)].copy()
        test_station = test_df[test_df["station_key"].eq(station_key)].copy()
        anchor_val_station = val_anchor.loc[val_station.index].values.astype(float)

        row = {
            "station_key": str(station_key),
            "n_train": int(len(train_station)),
            "n_val": int(len(val_station)),
            "n_train_full": int(len(train_full_station)),
            "n_test": int(len(test_station)),
            "anchor_metric": _prediction_metric(val_station, anchor_val_station, cfg, selection_metric),
            "corrected_metric": np.nan,
            "selected_alpha": 0.0,
            "status": "anchor",
        }
        if (
            len(train_station) < int(cfg.station_local_min_train_rows)
            or len(val_station) < int(cfg.station_local_min_validation_rows)
            or test_station.empty
        ):
            row["status"] = "anchor_insufficient_rows"
            rows.append(row)
            continue

        try:
            local = _run_global_residual_megaboost(
                cfg,
                train_station,
                val_station,
                train_full_station,
                test_station,
                feature_cols,
                label=f"{label} station {station_key}",
                anchor_train_pred=train_anchor.loc[train_station.index].values.astype(float),
                anchor_val_pred=anchor_val_station,
                anchor_train_full_pred=train_full_anchor.loc[train_full_station.index].values.astype(float),
                anchor_test_pred=test_anchor.loc[test_station.index].values.astype(float),
                sample_weight_mode=sample_weight_mode,
                selection_metric=selection_metric,
                param_grid=cfg.station_boosting_param_grid,
                top_k=cfg.top_k_station_boost_models,
                seed_offset=100_000 + 100 * station_i,
            )
            alpha, alpha_table = _select_correction_alpha(
                val_station,
                anchor_val_station,
                local["val_pred"],
                cfg,
                selection_metric,
            )
            val_pred_station = _blend_anchor_prediction(anchor_val_station, local["val_pred"], alpha)
            test_pred_station = _blend_anchor_prediction(
                test_anchor.loc[test_station.index].values.astype(float),
                local["test_pred"],
                alpha,
            )
            if not np.isfinite(val_pred_station).all() or not np.isfinite(test_pred_station).all():
                raise ValueError("StationBoost residual produced non-finite predictions")
            corrected_metric = _prediction_metric(val_station, val_pred_station, cfg, selection_metric)
            val_series.loc[val_station.index] = val_pred_station
            test_series.loc[test_station.index] = test_pred_station
            row.update({
                "corrected_metric": corrected_metric,
                "selected_alpha": float(alpha),
                "status": "residual_selected" if alpha > 0 else "anchor_validation_gate",
                "selected_names": "|".join(local.get("selected_names", [])),
            })
            payloads[str(station_key)] = {
                **local,
                "alpha_table": alpha_table,
                "selected_alpha": float(alpha),
            }
        except Exception as exc:
            row["status"] = f"anchor_error:{type(exc).__name__}"
        rows.append(row)

    if val_series.isna().any() or test_series.isna().any():
        raise ValueError(f"{label} produced missing predictions")

    val_pred = val_series.loc[val_df.index].values.astype(float)
    test_pred = test_series.loc[test_df.index].values.astype(float)
    return {
        "label": label,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "test_metrics_micro": calc_metrics_df(test_df, test_pred, cfg, mode="micro"),
        "station_selection_table": pd.DataFrame(rows),
        "station_payloads": payloads,
        "anchor_family": "seasonal",
        "sample_weight_mode": sample_weight_mode,
        "selection_metric": selection_metric,
    }

def _station_bias_table(train_df: pd.DataFrame, anchor_pred: np.ndarray, cfg: MultiStationForecastConfig) -> tuple[pd.DataFrame, float]:
    work = train_df[["station_key", "target_h"]].copy()
    work["anchor"] = np.asarray(anchor_pred, dtype=float)
    work["residual"] = pd.to_numeric(work["target_h"], errors="coerce").astype(float) - work["anchor"]
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["residual"])
    if work.empty:
        return pd.DataFrame(columns=["station_key", "bias", "n_rows"]), 0.0
    global_bias = float(work["residual"].median())
    stats = work.groupby("station_key", sort=False)["residual"].agg(station_bias="median", n_rows="size").reset_index()
    shrink = stats["n_rows"].astype(float) / (stats["n_rows"].astype(float) + max(float(cfg.station_bias_shrinkage), 1.0))
    stats["bias"] = shrink * stats["station_bias"].astype(float) + (1.0 - shrink) * global_bias
    return stats[["station_key", "bias", "n_rows"]], global_bias

def _apply_station_bias(df: pd.DataFrame, anchor_pred: np.ndarray, bias_table: pd.DataFrame, global_bias: float, alpha: float) -> np.ndarray:
    bias_map = bias_table.set_index("station_key")["bias"] if not bias_table.empty else pd.Series(dtype=float)
    bias = df["station_key"].map(bias_map).fillna(float(global_bias)).values.astype(float)
    return np.maximum(np.asarray(anchor_pred, dtype=float) + float(alpha) * bias, 0.0)

def _run_station_bias_seasonal(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label: str,
    anchor_train_pred: np.ndarray,
    anchor_val_pred: np.ndarray,
    anchor_train_full_pred: np.ndarray,
    anchor_test_pred: np.ndarray,
    selection_metric: str = "LW-MAE",
) -> dict:
    """Station-level shrinkage bias correction around Seasonal daylag, tuned by one global validation alpha."""
    bias_tune, global_bias_tune = _station_bias_table(train_df, anchor_train_pred, cfg)
    rows = []
    for alpha in cfg.local_correction_alpha_grid:
        pred = _apply_station_bias(val_df, anchor_val_pred, bias_tune, global_bias_tune, float(alpha))
        rows.append({"alpha": float(alpha), selection_metric: _prediction_metric(val_df, pred, cfg, selection_metric)})
    alpha_table = pd.DataFrame(rows).sort_values([selection_metric, "alpha"]).reset_index(drop=True)
    selected_alpha = float(alpha_table.loc[0, "alpha"])
    val_pred = _apply_station_bias(val_df, anchor_val_pred, bias_tune, global_bias_tune, selected_alpha)

    bias_full, global_bias_full = _station_bias_table(train_full_df, anchor_train_full_pred, cfg)
    test_pred = _apply_station_bias(test_df, anchor_test_pred, bias_full, global_bias_full, selected_alpha)
    return {
        "label": label,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "test_metrics_micro": calc_metrics_df(test_df, test_pred, cfg, mode="micro"),
        "selected_alpha": selected_alpha,
        "alpha_table": alpha_table,
        "bias_tune": bias_tune,
        "bias_full": bias_full,
        "global_bias_tune": global_bias_tune,
        "global_bias_full": global_bias_full,
        "selection_metric": selection_metric,
    }

__all__ = [
    "NORMALIZED_MAGNITUDE_PREFIXES",
    "NORMALIZED_MAGNITUDE_COLS",
    "_station_scale_series",
    "_normalize_boost_frame",
    "_fit_predict_xgb_normalized",
    "_run_global_megaboost_normalized",
    "_fit_predict_xgb_residual",
    "_run_global_residual_megaboost",
    "_run_validation_residual_megaboost",
    "_prediction_metric",
    "_blend_anchor_prediction",
    "_select_correction_alpha",
    "_line_code_column",
    "_run_line_local_normalized_megaboost",
    "_run_line_local_seasonal_residual_megaboost",
    "_station_seasonal_fallback",
    "_run_station_local_normalized_megaboost",
    "_run_station_local_seasonal_residual_megaboost",
    "_station_bias_table",
    "_apply_station_bias",
    "_run_station_bias_seasonal",
]
