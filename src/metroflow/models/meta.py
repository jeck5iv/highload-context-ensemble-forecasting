
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge

from metroflow.config import ExperimentConfig
from metroflow.evaluation.metrics import calc_metrics


def fit_positive_blend(pred_matrix, y_true):
    reg = LinearRegression(fit_intercept=False, positive=True)
    reg.fit(pred_matrix, y_true)
    w = np.asarray(reg.coef_, dtype=float).clip(min=0)
    if w.sum() == 0:
        w = np.ones(pred_matrix.shape[1]) / pred_matrix.shape[1]
    else:
        w = w / w.sum()
    return w


def apply_blend(pred_matrix, weights):
    return np.asarray(pred_matrix) @ np.asarray(weights)


def fit_context_blend(val_df: pd.DataFrame, model_cols: list[str], context_col='hl_daylag', min_points=24):
    y = val_df['y_true'].values.astype(float)
    X = val_df[model_cols].values.astype(float)
    ctx = val_df[context_col].values.astype(int)
    w_global = fit_positive_blend(X, y)
    weights_by_ctx = {}
    for ctx_value in [0, 1]:
        mask = (ctx == ctx_value)
        if mask.sum() >= max(len(model_cols), min_points):
            w_ctx = fit_positive_blend(X[mask], y[mask])
        else:
            w_ctx = w_global.copy()
        weights_by_ctx[ctx_value] = w_ctx
    return w_global, weights_by_ctx


def apply_context_blend(test_df: pd.DataFrame, model_cols: list[str], weights_by_ctx: dict, context_col='hl_daylag'):
    X = test_df[model_cols].values.astype(float)
    ctx = test_df[context_col].values.astype(int)
    pred = np.zeros(len(test_df), dtype=float)
    for i in range(len(test_df)):
        pred[i] = X[i] @ weights_by_ctx[int(ctx[i])]
    return pred


def build_meta_features(df: pd.DataFrame, primary_col: str, model_cols: list[str], use_context=True):
    X = pd.DataFrame(index=df.index)
    for c in model_cols:
        X[f'pred_{c}'] = df[c].values.astype(float)
    for c in model_cols:
        if c != primary_col:
            X[f'diff_{c}_minus_{primary_col}'] = df[c].values.astype(float) - df[primary_col].values.astype(float)
            X[f'absdiff_{c}_{primary_col}'] = np.abs(df[c].values.astype(float) - df[primary_col].values.astype(float))
    preds_mat = df[model_cols].values.astype(float)
    X['pred_mean'] = preds_mat.mean(axis=1)
    X['pred_std'] = preds_mat.std(axis=1)
    X['pred_min'] = preds_mat.min(axis=1)
    X['pred_max'] = preds_mat.max(axis=1)
    X['pred_range'] = X['pred_max'] - X['pred_min']
    if use_context:
        for c in ['hl_daylag', 'hour_sin', 'hour_cos', 'continuous_load_score', 'local_slope_1', 'local_slope_2']:
            if c in df.columns:
                X[c] = df[c].values
    return X


def build_prediction_tables(cfg: ExperimentConfig, split: dict, preds: dict[str, dict]):
    val_df = split['val_df']
    test_df = split['test_df']
    val_table = pd.DataFrame({cfg.time_col: val_df[cfg.time_col].values, 'y_true': val_df['target_h'].values.astype(float)})
    test_table = pd.DataFrame({cfg.time_col: test_df[cfg.time_col].values, 'y_true': test_df['target_h'].values.astype(float)})
    for name, payload in preds.items():
        val_table[name] = payload['val_pred']
        test_table[name] = payload['test_pred']
    for c in ['hl_daylag', 'hour_sin', 'hour_cos', 'continuous_load_score', 'local_slope_1', 'local_slope_2']:
        if c in val_df.columns:
            val_table[c] = val_df[c].values
        if c in test_df.columns:
            test_table[c] = test_df[c].values
    return val_table, test_table


def run_linear_meta(cfg: ExperimentConfig, val_pred_table: pd.DataFrame, test_pred_table: pd.DataFrame):
    y_val = val_pred_table['y_true'].values.astype(float)
    y_test = test_pred_table['y_true'].values.astype(float)
    base_cols = [c for c in ['ETS', 'MegaBoost', 'LSTM', 'Transformer-lite'] if c in val_pred_table.columns]

    w_static = fit_positive_blend(val_pred_table[base_cols].values, y_val)
    pred_test_meta_static = apply_blend(test_pred_table[base_cols].values, w_static)
    metrics_meta_static = calc_metrics(y_test, pred_test_meta_static, gamma=cfg.features.load_weight_gamma)

    w_global_ctx, weights_by_ctx = fit_context_blend(
        val_pred_table,
        model_cols=base_cols,
        context_col='hl_daylag',
        min_points=max(24, len(base_cols) * 6),
    )
    pred_test_meta_context = apply_context_blend(test_pred_table, base_cols, weights_by_ctx, context_col='hl_daylag')
    metrics_meta_context = calc_metrics(y_test, pred_test_meta_context, gamma=cfg.features.load_weight_gamma)
    return {
        'static': {
            'weights': dict(zip(base_cols, w_static)),
            'test_pred': pred_test_meta_static,
            'metrics': metrics_meta_static,
        },
        'context': {
            'global_weights': dict(zip(base_cols, w_global_ctx)),
            'weights_by_ctx': {k: dict(zip(base_cols, v)) for k, v in weights_by_ctx.items()},
            'test_pred': pred_test_meta_context,
            'metrics': metrics_meta_context,
        }
    }


def run_residual_meta(cfg: ExperimentConfig, val_pred_table: pd.DataFrame, test_pred_table: pd.DataFrame):
    candidates = [c for c in ['ETS', 'MegaBoost', 'LSTM', 'Transformer-lite'] if c in val_pred_table.columns]
    val_base_metrics = {c: calc_metrics(val_pred_table['y_true'].values, val_pred_table[c].values, gamma=cfg.features.load_weight_gamma) for c in candidates}
    val_base_table = pd.DataFrame(val_base_metrics).T.sort_values('MAE')
    primary_model = val_base_table.index[0] if cfg.primary_mode == 'auto' else cfg.primary_mode
    meta_use_models = list(val_base_table.index[:3])
    if cfg.models.use_lstm_in_residual_meta and 'LSTM' in val_pred_table.columns and 'LSTM' not in meta_use_models:
        meta_use_models.append('LSTM')

    y_val = val_pred_table['y_true'].values.astype(float)
    y_test = test_pred_table['y_true'].values.astype(float)
    primary_val_pred = val_pred_table[primary_model].values.astype(float)
    primary_test_pred = test_pred_table[primary_model].values.astype(float)
    resid_val = y_val - primary_val_pred

    X_val_meta = build_meta_features(val_pred_table, primary_model, meta_use_models, use_context=True)
    X_test_meta = build_meta_features(test_pred_table, primary_model, meta_use_models, use_context=True)

    resid_linear = Ridge(alpha=cfg.models.residual_linear_alpha, fit_intercept=True)
    resid_linear.fit(X_val_meta, resid_val)
    resid_pred_test_linear = resid_linear.predict(X_test_meta)
    q_lo_lin, q_hi_lin = np.quantile(resid_val, cfg.models.residual_linear_clip_quantiles)
    resid_pred_test_linear = np.clip(resid_pred_test_linear, q_lo_lin, q_hi_lin)
    final_pred_linear = primary_test_pred + resid_pred_test_linear

    params = dict(cfg.models.direct_meta_xgb_params)
    params['random_state'] = cfg.models.random_state
    resid_xgb = xgb.XGBRegressor(**params)
    resid_xgb.fit(X_val_meta, resid_val)
    resid_pred_test_xgb = resid_xgb.predict(X_test_meta)
    q_lo_xgb, q_hi_xgb = np.quantile(resid_val, cfg.models.residual_xgb_clip_quantiles)
    resid_pred_test_xgb = np.clip(resid_pred_test_xgb, q_lo_xgb, q_hi_xgb)
    final_pred_xgb = primary_test_pred + resid_pred_test_xgb

    return {
        'primary_model': primary_model,
        'val_base_table': val_base_table,
        'metrics_primary': calc_metrics(y_test, primary_test_pred, gamma=cfg.features.load_weight_gamma),
        'metrics_residual_linear': calc_metrics(y_test, final_pred_linear, gamma=cfg.features.load_weight_gamma),
        'metrics_residual_xgb': calc_metrics(y_test, final_pred_xgb, gamma=cfg.features.load_weight_gamma),
        'test_pred_primary': primary_test_pred,
        'test_pred_linear': final_pred_linear,
        'test_pred_xgb': final_pred_xgb,
        'linear_coef': pd.DataFrame({'feature': X_val_meta.columns, 'coef': resid_linear.coef_}),
        'xgb_importance': pd.DataFrame({'feature': X_val_meta.columns, 'importance': resid_xgb.feature_importances_}),
    }


def run_direct_meta_xgb(cfg: ExperimentConfig, val_pred_table: pd.DataFrame, test_pred_table: pd.DataFrame):
    if not cfg.models.run_direct_meta_xgb:
        return None
    base_cols = [c for c in ['ETS', 'MegaBoost', 'Transformer-lite'] if c in val_pred_table.columns]
    context_cols = base_cols + [c for c in ['hl_daylag', 'hour_sin', 'hour_cos'] if c in val_pred_table.columns]
    y_val = val_pred_table['y_true'].values.astype(float)
    y_test = test_pred_table['y_true'].values.astype(float)
    params = dict(cfg.models.direct_meta_xgb_params)
    params['random_state'] = cfg.models.random_state

    static_model = xgb.XGBRegressor(**params)
    static_model.fit(val_pred_table[base_cols].values, y_val)
    pred_static = static_model.predict(test_pred_table[base_cols].values)

    ctx_model = xgb.XGBRegressor(**params)
    ctx_model.fit(val_pred_table[context_cols].values, y_val)
    pred_ctx = ctx_model.predict(test_pred_table[context_cols].values)

    return {
        'metrics_static': calc_metrics(y_test, pred_static, gamma=cfg.features.load_weight_gamma),
        'metrics_context': calc_metrics(y_test, pred_ctx, gamma=cfg.features.load_weight_gamma),
        'test_pred_static': pred_static,
        'test_pred_context': pred_ctx,
    }
