from __future__ import annotations

from ..common import *
from ..config import (
    DEFAULT_CLASSIC_METRO_LINE_CODES,
    _scope_name,
    MultiStationForecastConfig,
)

def calc_metrics_df(df: pd.DataFrame, y_pred: np.ndarray, cfg: MultiStationForecastConfig, mode: str = "micro") -> dict:
    work = df[["station_key", "target_h", "hl_actual", "station_train_mean_target"]].copy()
    work["pred"] = np.asarray(y_pred, dtype=float)
    if mode == "micro":
        return _calc_metric_values(work, cfg)
    if mode == "macro":
        rows = []
        for _, g in work.groupby("station_key"):
            if len(g) > 0:
                rows.append(_calc_metric_values(g, cfg))
        return pd.DataFrame(rows).mean(numeric_only=True).to_dict()
    raise ValueError("mode must be 'micro' or 'macro'")

def _calc_metric_values(work: pd.DataFrame, cfg: MultiStationForecastConfig) -> dict:
    y = work["target_h"].astype(float).values
    pred = work["pred"].astype(float).values
    ae = np.abs(y - pred)
    mae = float(np.mean(ae))
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    denom = float(np.sum(np.abs(y)))
    wmape = float(np.sum(ae) / denom) if denom > 0 else np.nan
    station_mean = work["station_train_mean_target"].astype(float).replace(0, np.nan).fillna(np.nanmean(y) if len(y) else 1.0).values
    weights = np.power(np.maximum(y, 0) / np.maximum(station_mean, 1e-9), cfg.load_weight_gamma)
    lw_mae = float(np.mean(ae * weights))
    hl_mask = work["hl_actual"].fillna(0).astype(int).values.astype(bool)
    if hl_mask.any():
        hl_mae = float(np.mean(ae[hl_mask]))
        hl_under = float(np.mean(np.maximum(y[hl_mask] - pred[hl_mask], 0)))
        hl_under_rate = float(np.mean(pred[hl_mask] < y[hl_mask]))
        hl_asym = float(np.mean(ae[hl_mask] * np.where(pred[hl_mask] < y[hl_mask], cfg.hl_underprediction_penalty, 1.0)))
    else:
        hl_mae = np.nan
        hl_under = np.nan
        hl_under_rate = np.nan
        hl_asym = np.nan
    return {
        "MAE": mae,
        "RMSE": rmse,
        "wMAPE": wmape,
        "LW-MAE": lw_mae,
        "HL-MAE": hl_mae,
        "HL-underMAE": hl_under,
        "HL-underRate": hl_under_rate,
        "HL-asymMAE": hl_asym,
    }

def evaluate_predictions(test_pred_table: pd.DataFrame, model_cols: list[str], cfg: MultiStationForecastConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_df = test_pred_table.rename(columns={"y_true": "target_h"}).copy()
    micro = {m: calc_metrics_df(eval_df, eval_df[m].values, cfg, mode="micro") for m in model_cols}
    macro = {m: calc_metrics_df(eval_df, eval_df[m].values, cfg, mode="macro") for m in model_cols}
    micro_table = pd.DataFrame(micro).T.sort_values("LW-MAE")
    macro_table = pd.DataFrame(macro).T.sort_values("LW-MAE")
    return micro_table, macro_table

BOOTSTRAP_METRICS = [
    "MAE", "RMSE", "wMAPE", "LW-MAE", "HL-MAE", "HL-underMAE", "HL-underRate", "HL-asymMAE",
]

PREDICTION_SINGLE_MODEL_CANDIDATES = [
    "Seasonal daylag", "SARIMA local", "ETS local",
    "StationBoost base norm", "StationBoost rich norm",
    "StationBoost rich norm LW", "StationBoost Seasonal residual LW",
    "MegaBoost base", "MegaBoost rich", "MegaBoost rich LW",
    "MegaBoost base norm", "MegaBoost rich norm", "MegaBoost rich norm LW",
    "CatBoost base", "CatBoost rich", "Global LSTM", "Global Transformer-lite",
]

ARTICLE_COMPARATOR_MODELS = [
    "Km-XGBoost-lite", "ET-LGBM-PSO-lite", "TP-Boost-lite",
    "ML-Fusion-SVR-lite", "VMD-LightGBM-lite", "ST-LightGBM-MIC-lite",
    "SN-LSSVM-lite", "TwoStage-FS-MSD-Hybrid-lite", "Stacking-CatBoost-lite",
    "PSR-CNN-LSTM-lite", "CNN-BiLSTM-Attention-lite",
    "MSD-Informer-BiLSTM-lite", "BiLSTM-CTPA-lite",
]

def _choose_bootstrap_reference_models(
    micro_table: pd.DataFrame,
    model_cols: list[str],
) -> tuple[list[str], pd.DataFrame]:
    """Choose prediction-facing bootstrap references for the current scenario.

    We always retain canonical seasonal/statistical references when available and
    add two data-dependent references selected on the test point estimates:
    the strongest ordinary standalone model and the strongest literature-grounded
    comparator under the primary LW-MAE metric.
    """
    available = [m for m in model_cols if m in micro_table.index]
    rows: list[dict[str, object]] = []
    refs: list[str] = []

    def add(role: str, model: str | None) -> None:
        if not model or model not in micro_table.index:
            return
        row = micro_table.loc[model]
        rows.append({
            "reference_role": role,
            "model": model,
            "LW-MAE": float(row["LW-MAE"]),
            "MAE": float(row["MAE"]),
            "RMSE": float(row["RMSE"]),
            "HL-MAE": float(row["HL-MAE"]),
        })
        if model not in refs:
            refs.append(model)

    add("canonical_seasonal", "Seasonal daylag" if "Seasonal daylag" in available else None)
    add("canonical_ets", "ETS local" if "ETS local" in available else None)

    single_present = [m for m in PREDICTION_SINGLE_MODEL_CANDIDATES if m in available]
    if single_present:
        best_single = micro_table.loc[single_present].sort_values(["LW-MAE", "MAE"]).index[0]
        add("strongest_standalone_by_LW-MAE", str(best_single))

    article_present = [m for m in ARTICLE_COMPARATOR_MODELS if m in available]
    if article_present:
        best_article = micro_table.loc[article_present].sort_values(["LW-MAE", "MAE"]).index[0]
        add("strongest_article_comparator_by_LW-MAE", str(best_article))

    return refs, pd.DataFrame(rows)

def _safe_divide(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    numer = np.asarray(numer, dtype=float)
    denom = np.asarray(denom, dtype=float)
    out = np.full(np.broadcast_shapes(numer.shape, denom.shape), np.nan, dtype=float)
    np.divide(numer, denom, out=out, where=np.asarray(denom) > 0)
    return out

def _station_metric_component_arrays(
    test_pred_table: pd.DataFrame,
    model_cols: list[str],
    cfg: MultiStationForecastConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Pre-compute station-level additive components for fast cluster bootstrap."""
    work = test_pred_table.copy()
    if "y_true" not in work.columns:
        raise KeyError("test_pred_table must contain y_true")
    stations = np.array(sorted(work["station_key"].astype(str).unique()))
    station_index = pd.Categorical(work["station_key"].astype(str), categories=stations, ordered=True).codes
    n_stations = len(stations)
    n_models = len(model_cols)

    y = pd.to_numeric(work["y_true"], errors="coerce").astype(float).values
    pred = work[model_cols].replace([np.inf, -np.inf], np.nan).astype(float).values
    if not np.isfinite(pred).all():
        raise ValueError("Predictions contain non-finite values before bootstrap")
    hl_mask = work.get("hl_actual", pd.Series(0, index=work.index)).fillna(0).astype(int).values.astype(bool)
    station_mean = pd.to_numeric(work.get("station_train_mean_target"), errors="coerce").astype(float).values
    fallback_mean = float(np.nanmean(y)) if np.isfinite(y).any() else 1.0
    station_mean = np.where(np.isfinite(station_mean) & (station_mean > 0), station_mean, fallback_mean)
    weights = np.power(np.maximum(y, 0.0) / np.maximum(station_mean, 1e-9), cfg.load_weight_gamma)
    weights = np.where(np.isfinite(weights), weights, 1.0)

    y2d = y[:, None]
    ae = np.abs(y2d - pred)
    sqe = (y2d - pred) ** 2
    lw = ae * weights[:, None]
    under = np.maximum(y2d - pred, 0.0)
    under_flag = (pred < y2d).astype(float)
    asym = ae * np.where(pred < y2d, cfg.hl_underprediction_penalty, 1.0)

    counts = np.bincount(station_index, minlength=n_stations).astype(float)
    y_abs = np.bincount(station_index, weights=np.abs(y), minlength=n_stations).astype(float)
    hl_count = np.bincount(station_index, weights=hl_mask.astype(float), minlength=n_stations).astype(float)

    additive = {
        "ae": np.zeros((n_stations, n_models), dtype=float),
        "sqe": np.zeros((n_stations, n_models), dtype=float),
        "lw": np.zeros((n_stations, n_models), dtype=float),
        "hl_ae": np.zeros((n_stations, n_models), dtype=float),
        "hl_under": np.zeros((n_stations, n_models), dtype=float),
        "hl_under_flag": np.zeros((n_stations, n_models), dtype=float),
        "hl_asym": np.zeros((n_stations, n_models), dtype=float),
    }
    hl_float = hl_mask.astype(float)
    for j in range(n_models):
        additive["ae"][:, j] = np.bincount(station_index, weights=ae[:, j], minlength=n_stations)
        additive["sqe"][:, j] = np.bincount(station_index, weights=sqe[:, j], minlength=n_stations)
        additive["lw"][:, j] = np.bincount(station_index, weights=lw[:, j], minlength=n_stations)
        additive["hl_ae"][:, j] = np.bincount(station_index, weights=ae[:, j] * hl_float, minlength=n_stations)
        additive["hl_under"][:, j] = np.bincount(station_index, weights=under[:, j] * hl_float, minlength=n_stations)
        additive["hl_under_flag"][:, j] = np.bincount(station_index, weights=under_flag[:, j] * hl_float, minlength=n_stations)
        additive["hl_asym"][:, j] = np.bincount(station_index, weights=asym[:, j] * hl_float, minlength=n_stations)

    denominators = {
        "count": counts,
        "abs_y": y_abs,
        "hl_count": hl_count,
    }

    macro_station_metrics = {
        "MAE": _safe_divide(additive["ae"], counts[:, None]),
        "RMSE": np.sqrt(_safe_divide(additive["sqe"], counts[:, None])),
        "wMAPE": _safe_divide(additive["ae"], y_abs[:, None]),
        "LW-MAE": _safe_divide(additive["lw"], counts[:, None]),
        "HL-MAE": _safe_divide(additive["hl_ae"], hl_count[:, None]),
        "HL-underMAE": _safe_divide(additive["hl_under"], hl_count[:, None]),
        "HL-underRate": _safe_divide(additive["hl_under_flag"], hl_count[:, None]),
        "HL-asymMAE": _safe_divide(additive["hl_asym"], hl_count[:, None]),
    }
    packed = {**additive, **denominators}
    return stations, packed, macro_station_metrics

def station_bootstrap_ci(
    test_pred_table: pd.DataFrame,
    model_cols: list[str],
    cfg: MultiStationForecastConfig,
    baseline_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Fast station-cluster bootstrap with prediction-facing baseline deltas.

    The previous implementation repeatedly rebuilt sampled dataframes and re-ran
    the full metric pipeline.  Here we pre-compute station-level additive metric
    contributions once and bootstrap by resampling station indices.  Macro metrics
    use the standard repeated-cluster weighting: if a station is sampled twice, it
    contributes twice to the macro replicate.
    """
    rng = np.random.default_rng(cfg.random_state)
    baseline_cols = [b for b in (baseline_cols or ["ETS local", "Seasonal daylag"]) if b in model_cols]
    point_micro, point_macro = evaluate_predictions(test_pred_table, model_cols, cfg)
    stations, packed, macro_station_metrics = _station_metric_component_arrays(test_pred_table, model_cols, cfg)
    n_boot = max(1, int(cfg.n_bootstrap))
    n_stations = len(stations)
    sampled_idx = rng.integers(0, n_stations, size=(n_boot, n_stations), endpoint=False)

    count_sample = packed["count"][sampled_idx].sum(axis=1)[:, None]
    abs_y_sample = packed["abs_y"][sampled_idx].sum(axis=1)[:, None]
    hl_count_sample = packed["hl_count"][sampled_idx].sum(axis=1)[:, None]

    def station_sum(name: str) -> np.ndarray:
        return packed[name][sampled_idx].sum(axis=1)

    micro_samples = {
        "MAE": _safe_divide(station_sum("ae"), count_sample),
        "RMSE": np.sqrt(_safe_divide(station_sum("sqe"), count_sample)),
        "wMAPE": _safe_divide(station_sum("ae"), abs_y_sample),
        "LW-MAE": _safe_divide(station_sum("lw"), count_sample),
        "HL-MAE": _safe_divide(station_sum("hl_ae"), hl_count_sample),
        "HL-underMAE": _safe_divide(station_sum("hl_under"), hl_count_sample),
        "HL-underRate": _safe_divide(station_sum("hl_under_flag"), hl_count_sample),
        "HL-asymMAE": _safe_divide(station_sum("hl_asym"), hl_count_sample),
    }

    macro_samples: dict[str, np.ndarray] = {}
    for metric, station_metric in macro_station_metrics.items():
        with np.errstate(invalid="ignore"):
            macro_samples[metric] = np.nanmean(station_metric[sampled_idx], axis=1)

    metric_rows: list[dict[str, object]] = []
    point_tables = {"micro": point_micro, "macro": point_macro}
    sample_tables = {"micro": micro_samples, "macro": macro_samples}
    for mode in ["micro", "macro"]:
        for metric in BOOTSTRAP_METRICS:
            samples = sample_tables[mode][metric]
            for j, model in enumerate(model_cols):
                arr = np.asarray(samples[:, j], dtype=float)
                arr = arr[np.isfinite(arr)]
                metric_rows.append({
                    "mode": mode,
                    "model": model,
                    "metric": metric,
                    "estimate": float(point_tables[mode].loc[model, metric]),
                    "ci_low": float(np.quantile(arr, 0.025)) if len(arr) else np.nan,
                    "ci_high": float(np.quantile(arr, 0.975)) if len(arr) else np.nan,
                })

    delta_rows: list[dict[str, object]] = []
    model_to_idx = {m: i for i, m in enumerate(model_cols)}
    for mode in ["micro", "macro"]:
        for baseline in baseline_cols:
            b_idx = model_to_idx[baseline]
            for metric in BOOTSTRAP_METRICS:
                samples = sample_tables[mode][metric]
                base_samples = samples[:, b_idx]
                for model in model_cols:
                    if model == baseline:
                        continue
                    m_idx = model_to_idx[model]
                    arr = np.asarray(base_samples - samples[:, m_idx], dtype=float)
                    arr = arr[np.isfinite(arr)]
                    estimate = float(point_tables[mode].loc[baseline, metric] - point_tables[mode].loc[model, metric])
                    low = float(np.quantile(arr, 0.025)) if len(arr) else np.nan
                    high = float(np.quantile(arr, 0.975)) if len(arr) else np.nan
                    delta_rows.append({
                        "mode": mode,
                        "model": model,
                        "baseline": baseline,
                        "metric": metric,
                        "delta_estimate": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "significant_95": bool(low > 0 or high < 0) if np.isfinite(low) and np.isfinite(high) else False,
                    })

    return {
        "metric_ci": pd.DataFrame(metric_rows).sort_values(["mode", "metric", "estimate"]),
        "delta_ci": pd.DataFrame(delta_rows).sort_values(["mode", "baseline", "metric", "delta_estimate"], ascending=[True, True, True, False]),
    }

__all__ = [
    "calc_metrics_df",
    "_calc_metric_values",
    "evaluate_predictions",
    "BOOTSTRAP_METRICS",
    "PREDICTION_SINGLE_MODEL_CANDIDATES",
    "ARTICLE_COMPARATOR_MODELS",
    "_choose_bootstrap_reference_models",
    "_safe_divide",
    "_station_metric_component_arrays",
    "station_bootstrap_ci",
]
