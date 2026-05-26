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

CATBOOST_CATEGORY_COLS = ["station_key", "ST_NAME_SHORT", "LN_NAME_SHORT"]

def _catboost_matrix(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    numeric_cols = [c for c in feature_cols if c in df.columns]
    categorical_cols = [c for c in CATBOOST_CATEGORY_COLS if c in df.columns]
    X = df[numeric_cols + categorical_cols].copy()
    if numeric_cols:
        X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan).astype(float)
    for c in categorical_cols:
        X[c] = X[c].fillna("__missing__").astype(str)
    return X, categorical_cols

def _fit_predict_catboost(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: list[str],
    seed_offset: int,
) -> tuple[np.ndarray, object]:
    if CatBoostRegressor is None:
        raise ModuleNotFoundError("catboost is not installed")
    train_part = train_df.dropna(subset=["target_h"]).copy()
    X_train, cat_cols = _catboost_matrix(train_part, feature_cols)
    X_pred, _ = _catboost_matrix(pred_df, feature_cols)
    model = CatBoostRegressor(
        loss_function="MAE",
        iterations=cfg.catboost_iterations,
        depth=cfg.catboost_depth,
        learning_rate=cfg.catboost_learning_rate,
        l2_leaf_reg=cfg.catboost_l2_leaf_reg,
        random_seed=cfg.random_state + seed_offset,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X_train, train_part["target_h"].astype(float), cat_features=cat_cols)
    pred = np.asarray(model.predict(X_pred), dtype=float)
    return np.maximum(pred, 0.0), model

def _run_global_catboost(
    cfg: MultiStationForecastConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    seed_offset: int,
) -> dict:
    pred_val, _ = _fit_predict_catboost(cfg, train_df, val_df, feature_cols, seed_offset)
    pred_test, model = _fit_predict_catboost(cfg, train_full_df, test_df, feature_cols, seed_offset + 100)
    val_metrics = calc_metrics_df(val_df, pred_val, cfg, mode="micro")
    return {
        "label": label,
        "features": feature_cols,
        "val_table": pd.DataFrame([{ "model": label, **val_metrics }]),
        "selected_names": [label],
        "val_pred": pred_val,
        "test_pred": pred_test,
        "test_metrics_micro": calc_metrics_df(test_df, pred_test, cfg, mode="micro"),
        "models": [model],
    }

__all__ = [
    "CATBOOST_CATEGORY_COLS",
    "_catboost_matrix",
    "_fit_predict_catboost",
    "_run_global_catboost",
]
