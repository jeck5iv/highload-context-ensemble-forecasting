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

def _fit_predict_xgb(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
    sample_weight: np.ndarray | None = None,
):
    train_part = train_df.dropna(subset=["target_h"]).copy()
    if train_part.empty:
        raise ValueError("No train rows with non-missing target_h")
    fit_weight = _aligned_sample_weight(train_df, train_part, sample_weight)
    model = xgb.XGBRegressor(**params)
    model.fit(
        _xgb_matrix(train_part, feature_cols),
        train_part["target_h"].astype(float),
        sample_weight=fit_weight,
    )
    pred = model.predict(_xgb_matrix(pred_df, feature_cols))
    return np.asarray(pred, dtype=float), model

def _run_global_megaboost(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    sample_weight_mode: str | None = None,
    selection_metric: str = "MAE",
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
        params["random_state"] = cfg.random_state + seed_offset + i
        pred_val, _ = _fit_predict_xgb(train_df, val_df, feature_cols, params, sample_weight=train_weight)
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
        pred_test, model = _fit_predict_xgb(
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
        "sample_weight_mode": sample_weight_mode,
        "selection_metric": selection_metric,
    }

__all__ = [
    "_fit_predict_xgb",
    "_run_global_megaboost",
]
