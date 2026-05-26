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
from .meta import (
    _seasonal_anchor_series,
    seasonal_daylag_predictions,
    _positive_blend,
    _context_blend,
    _build_meta_tables,
    _run_meta_models,
    _meta_feature_matrix,
)

def _meta_lw_sample_weight(df: pd.DataFrame, cfg: MultiStationForecastConfig) -> np.ndarray:
    y = pd.to_numeric(df["y_true"], errors="coerce").astype(float).values
    if "station_train_mean_target" in df.columns:
        denom = pd.to_numeric(df["station_train_mean_target"], errors="coerce").astype(float).values
    else:
        denom = np.full(len(df), np.nanmedian(y) if np.isfinite(y).any() else 1.0, dtype=float)
    denom = np.where(np.isfinite(denom) & (denom > 0), denom, np.nanmedian(denom[np.isfinite(denom) & (denom > 0)]) if np.any(np.isfinite(denom) & (denom > 0)) else 1.0)
    weights = np.power(np.maximum(y, 0.0) / np.maximum(denom, 1e-9), cfg.load_weight_gamma)
                                                                                             
    weights = np.clip(weights, 0.25, 8.0)
    weights = np.where(np.isfinite(weights), weights, 1.0)
    return weights.astype(float)

def _positive_blend_weighted(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> np.ndarray:
    reg = LinearRegression(fit_intercept=False, positive=True)
    if sample_weight is not None:
        reg.fit(X, y, sample_weight=sample_weight)
    else:
        reg.fit(X, y)
    w = np.asarray(reg.coef_, dtype=float).clip(min=0)
    if not np.isfinite(w).all() or w.sum() <= 0:
        return np.ones(X.shape[1], dtype=float) / X.shape[1]
    return w / w.sum()

def _context_blend_weighted(
    val_pred_table: pd.DataFrame,
    test_pred_table: pd.DataFrame,
    model_cols: list[str],
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]:
    y = val_pred_table["y_true"].values.astype(float)
    X = val_pred_table[model_cols].values.astype(float)
    ctx = val_pred_table["hl_daylag"].fillna(0).values.astype(int)
    global_w = _positive_blend_weighted(X, y, sample_weight=sample_weight)
    by_ctx: dict[int, np.ndarray] = {}
    for v in [0, 1]:
        mask = ctx == v
        if mask.sum() >= max(30, len(model_cols) * 5):
            sw = sample_weight[mask] if sample_weight is not None else None
            by_ctx[v] = _positive_blend_weighted(X[mask], y[mask], sample_weight=sw)
        else:
            by_ctx[v] = global_w.copy()
    X_test = test_pred_table[model_cols].values.astype(float)
    ctx_test = test_pred_table["hl_daylag"].fillna(0).values.astype(int)
    pred = np.zeros(len(test_pred_table), dtype=float)
    for i in range(len(test_pred_table)):
        pred[i] = X_test[i] @ by_ctx[int(ctx_test[i])]
    return global_w, by_ctx, pred

def _validation_metric_for_model(val_pred_table: pd.DataFrame, model: str, cfg: MultiStationForecastConfig, metric: str = "LW-MAE") -> float:
    tmp = val_pred_table.rename(columns={"y_true": "target_h"}).copy()
    return float(calc_metrics_df(tmp, tmp[model].values, cfg, mode="micro")[metric])

def _select_meta_model_pool(
    val_pred_table: pd.DataFrame,
    candidate_cols: list[str],
    cfg: MultiStationForecastConfig,
    baseline: str = "Seasonal daylag",
    tolerance: float = 1.05,
) -> tuple[list[str], pd.DataFrame]:
    rows = []
    for c in candidate_cols:
        if c not in val_pred_table.columns:
            continue
        rows.append({
            "model": c,
            "val_MAE": _validation_metric_for_model(val_pred_table, c, cfg, "MAE"),
            "val_LW-MAE": _validation_metric_for_model(val_pred_table, c, cfg, "LW-MAE"),
            "val_HL-MAE": _validation_metric_for_model(val_pred_table, c, cfg, "HL-MAE"),
        })
    table = pd.DataFrame(rows).sort_values("val_LW-MAE").reset_index(drop=True)
    if table.empty:
        return [], table
    base_lw = float(table.loc[table["model"].eq(baseline), "val_LW-MAE"].iloc[0]) if baseline in set(table["model"]) else float(table["val_LW-MAE"].min())
    selected = table.loc[table["val_LW-MAE"] <= tolerance * base_lw, "model"].tolist()
    if baseline in candidate_cols and baseline not in selected:
        selected.append(baseline)
                                                                                           
    if len(selected) < min(2, len(table)):
        selected = table.head(min(3, len(table)))["model"].tolist()
    return selected, table

def _fit_residual_family_for_primary(
    cfg: MultiStationForecastConfig,
    val_pred_table: pd.DataFrame,
    test_pred_table: pd.DataFrame,
    model_cols: list[str],
    primary: str,
    sw_lw: np.ndarray,
    seed_offset: int = 0,
) -> dict:
    """Fit the same residual meta-family for a fixed anchor/base model.

    The automatic residual model chooses its anchor on validation.  Fixed anchors are
    useful when we want to test whether an otherwise strong baseline (for example,
    ETS local) benefits from a learned residual correction without letting a single
    validation day re-select the anchor.
    """
    if primary not in val_pred_table.columns or primary not in test_pred_table.columns:
        raise KeyError(f"Residual anchor {primary!r} is absent from prediction tables")

    y_val = val_pred_table["y_true"].values.astype(float)
    resid_val = y_val - val_pred_table[primary].values.astype(float)

    feature_model_cols = list(dict.fromkeys([primary, *model_cols]))
    X_val = _meta_feature_matrix(val_pred_table, feature_model_cols, primary)
    X_test = _meta_feature_matrix(test_pred_table, feature_model_cols, primary)

    ridge = Ridge(alpha=10.0)
    ridge.fit(X_val, resid_val)
    pred_resid_linear = ridge.predict(X_test)
    lo, hi = np.quantile(resid_val, [0.05, 0.95])
    pred_resid_linear = np.clip(pred_resid_linear, lo, hi)

    ridge_lw = Ridge(alpha=10.0)
    ridge_lw.fit(X_val, resid_val, sample_weight=sw_lw)
    pred_resid_linear_lw = ridge_lw.predict(X_test)
    pred_resid_linear_lw = np.clip(pred_resid_linear_lw, lo, hi)

    params = dict(
        objective="reg:absoluteerror",
        n_estimators=100,
        learning_rate=0.03,
        max_depth=2,
        min_child_weight=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=20.0,
        reg_alpha=4.0,
        tree_method="hist",
        random_state=cfg.random_state + seed_offset,
    )
    rxgb = xgb.XGBRegressor(**params)
    rxgb.fit(X_val, resid_val)
    pred_resid_xgb = rxgb.predict(X_test)
    lo2, hi2 = np.quantile(resid_val, [0.10, 0.90])
    pred_resid_xgb = np.clip(pred_resid_xgb, lo2, hi2)

    rxgb_lw = xgb.XGBRegressor(**{**params, "random_state": cfg.random_state + seed_offset + 333})
    rxgb_lw.fit(X_val, resid_val, sample_weight=sw_lw)
    pred_resid_xgb_lw = rxgb_lw.predict(X_test)
    pred_resid_xgb_lw = np.clip(pred_resid_xgb_lw, lo2, hi2)

    primary_test = test_pred_table[primary].values.astype(float)
    return {
        "primary_model": primary,
        "feature_model_cols": feature_model_cols,
        "pred_residual_linear": primary_test + pred_resid_linear,
        "pred_residual_xgb": primary_test + pred_resid_xgb,
        "pred_residual_linear_lw": primary_test + pred_resid_linear_lw,
        "pred_residual_xgb_lw": primary_test + pred_resid_xgb_lw,
        "residual_linear_coef": pd.DataFrame({"feature": X_val.columns, "coef": ridge.coef_}),
        "residual_linear_lw_coef": pd.DataFrame({"feature": X_val.columns, "coef": ridge_lw.coef_}),
        "residual_xgb_importance": pd.DataFrame({"feature": X_val.columns, "importance": rxgb.feature_importances_}),
        "residual_xgb_lw_importance": pd.DataFrame({"feature": X_val.columns, "importance": rxgb_lw.feature_importances_}),
    }

def _run_meta_models_filtered_weighted(cfg: MultiStationForecastConfig, val_pred_table: pd.DataFrame, test_pred_table: pd.DataFrame) -> dict:
    preferred_order = [
        "Seasonal daylag",
        "SARIMA local",
        "ETS local",
        "StationBoost base norm",
        "StationBoost rich norm",
        "StationBoost rich norm LW",
        "StationBoost Seasonal residual LW",
        "CatBoost base",
        "CatBoost rich",
        "Km-XGBoost-lite",
        "ET-LGBM-PSO-lite",
        "TP-Boost-lite",
        "MegaBoost base norm",
        "MegaBoost rich norm",
        "MegaBoost base",
        "MegaBoost rich",
        "Global LSTM",
        "Global Transformer-lite",
        "PSR-CNN-LSTM-lite",
        "CNN-BiLSTM-Attention-lite",
        "MSD-Informer-BiLSTM-lite",
        "BiLSTM-CTPA-lite",
    ]
    candidate_cols = [c for c in preferred_order if c in val_pred_table.columns]
    model_cols, val_selection_table = _select_meta_model_pool(val_pred_table, candidate_cols, cfg, tolerance=1.05)
    if not model_cols:
        raise ValueError("No base models available for meta layer")
    y_val = val_pred_table["y_true"].values.astype(float)
    sw_lw = _meta_lw_sample_weight(val_pred_table, cfg)

                               
    w_static = _positive_blend_weighted(val_pred_table[model_cols].values.astype(float), y_val)
    pred_static = test_pred_table[model_cols].values.astype(float) @ w_static
    w_global, w_by_ctx, pred_context = _context_blend_weighted(val_pred_table, test_pred_table, model_cols)

                       
    w_static_lw = _positive_blend_weighted(val_pred_table[model_cols].values.astype(float), y_val, sample_weight=sw_lw)
    pred_static_lw = test_pred_table[model_cols].values.astype(float) @ w_static_lw
    w_global_lw, w_by_ctx_lw, pred_context_lw = _context_blend_weighted(val_pred_table, test_pred_table, model_cols, sample_weight=sw_lw)

    out = {
        "model_cols": model_cols,
        "candidate_cols": candidate_cols,
        "validation_selection_table": val_selection_table,
        "static_weights": dict(zip(model_cols, w_static)),
        "context_global_weights": dict(zip(model_cols, w_global)),
        "context_weights_by_hl_daylag": {int(k): dict(zip(model_cols, v)) for k, v in w_by_ctx.items()},
        "static_lw_weights": dict(zip(model_cols, w_static_lw)),
        "context_lw_global_weights": dict(zip(model_cols, w_global_lw)),
        "context_lw_weights_by_hl_daylag": {int(k): dict(zip(model_cols, v)) for k, v in w_by_ctx_lw.items()},
        "pred_static": pred_static,
        "pred_context": pred_context,
        "pred_static_lw": pred_static_lw,
        "pred_context_lw": pred_context_lw,
    }

    if cfg.run_residual_xgb:
        # Preserve the existing adaptive anchor: choose the validation-best base
        # model by load-weighted MAE.
        val_scores = {c: _validation_metric_for_model(val_pred_table, c, cfg, "LW-MAE") for c in model_cols}
        primary = pd.Series(val_scores).sort_values().index[0]
        auto_residual = _fit_residual_family_for_primary(
            cfg,
            val_pred_table,
            test_pred_table,
            model_cols,
            primary,
            sw_lw,
            seed_offset=0,
        )
        out.update(auto_residual)

        # Run fixed-anchor variants in parallel.  They do not replace the adaptive
        # residual model; they are additional models evaluated in the same tables.
        fixed_residuals: dict[str, dict] = {}
        fixed_anchors = [
            ("ETS", "ETS local", 10_000),
            ("Seasonal", "Seasonal daylag", 20_000),
        ]
        for label, anchor, seed_offset in fixed_anchors:
            if anchor not in val_pred_table.columns or anchor not in test_pred_table.columns:
                continue
            fixed_residuals[label] = _fit_residual_family_for_primary(
                cfg,
                val_pred_table,
                test_pred_table,
                model_cols,
                anchor,
                sw_lw,
                seed_offset=seed_offset,
            )
        out["fixed_residuals"] = fixed_residuals
    return out

__all__ = [
    "_meta_lw_sample_weight",
    "_positive_blend_weighted",
    "_context_blend_weighted",
    "_validation_metric_for_model",
    "_select_meta_model_pool",
    "_fit_residual_family_for_primary",
    "_run_meta_models_filtered_weighted",
]
