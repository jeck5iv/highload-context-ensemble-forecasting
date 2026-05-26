
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from metroflow.config import ExperimentConfig
from metroflow.features.builders import build_lag_buckets, build_roll_windows
from metroflow.evaluation.metrics import calc_metrics


def make_feature_sets(cfg: ExperimentConfig) -> dict[str, list[str]]:
    lags = build_lag_buckets(cfg)
    rolls = build_roll_windows(cfg)

    def take(arr, n):
        return arr[:min(n, len(arr))]

    def r(idx):
        return rolls[min(idx, len(rolls) - 1)]

    lags_short = take(lags, 3)
    lags_mid = take(lags, 5)
    lags_long = take(lags, 8)
    short_w = r(0)

                                                                         
                                                                    
                                                                       
                                                                          
                                                     
    boost_context_features = cfg.features.boosting_context_features
    if boost_context_features is None:
        boost_context_features = cfg.features.context_features

                                                            
    time_context: list[str] = []
    if 'hour_cyclical' in boost_context_features:
        time_context.extend(['hour_sin', 'hour_cos'])

                                                                               
                                           
    regime_context: list[str] = []
    if 'highload_daylag' in boost_context_features:
        regime_context.append('hl_daylag')
    if 'continuous_load_score' in boost_context_features:
        regime_context.append('continuous_load_score')
    if 'local_slope' in boost_context_features:
        regime_context.extend(['local_slope_1', 'local_slope_2'])
    if 'daylag_deviation' in boost_context_features:
        regime_context.extend(['daylag_deviation', 'prev_count_ratio_daylag'])
    if 'rolling_volatility' in boost_context_features:
        regime_context.append(f'roll_std_{short_w}')

                                                                       
    day_lag_cols = [f'lag_{cfg.day_lag}', f'lag_{cfg.day_lag + 1}']
    base_roll_cols = [f'roll_sum_{r(1)}', f'roll_mean_{r(2)}']

    fs = {
        'lags_short': [f'lag_{x}' for x in lags_short] + [f'roll_sum_{r(0)}', f'roll_mean_{r(0)}'],
        'lags_hour': [f'lag_{x}' for x in lags_mid] + [f'roll_sum_{r(1)}', f'roll_mean_{r(1)}'],
        'lags_day_time': [f'lag_{x}' for x in lags_mid] + day_lag_cols + time_context,
        'lags_day_time_ctx': [f'lag_{x}' for x in lags_mid] + day_lag_cols + time_context + regime_context,
        'lags_extended': [f'lag_{x}' for x in lags_long] + day_lag_cols + time_context + base_roll_cols,
        'lags_extended_ctx': [f'lag_{x}' for x in lags_long] + day_lag_cols + time_context + base_roll_cols + regime_context,
    }

                                                                                
                                                                           
                                                   
    deduped: dict[str, list[str]] = {}
    seen: set[tuple[str, ...]] = set()
    for name, cols in fs.items():
        unique_cols = list(dict.fromkeys(cols))
        key = tuple(sorted(unique_cols))
        if key not in seen:
            deduped[name] = unique_cols
            seen.add(key)

    return deduped


def _xgb_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
                                                                                  
                                                         
    return df[feature_cols].replace([np.inf, -np.inf], np.nan).astype(float)


def fit_predict_xgb(train_df: pd.DataFrame, pred_df: pd.DataFrame, feature_cols: list[str], params: dict):
                                                                                  
                                                                                  
                                                                                 
                                                                 
    train_part = train_df.dropna(subset=['target_h']).copy()
    if train_part.empty:
        raise ValueError('No training rows with non-missing target_h for XGBoost')

    model = xgb.XGBRegressor(**params)
    model.fit(_xgb_features(train_part, feature_cols), train_part['target_h'].astype(float))

    pred_values = model.predict(_xgb_features(pred_df, feature_cols))
    pred = pd.Series(pred_values, index=pred_df.index, dtype=float)
    if pred.isna().any():
        raise ValueError(f'XGBoost produced NaN predictions for feature set: {feature_cols}')
    return pred, model


def run_megaboost(cfg: ExperimentConfig, train_df: pd.DataFrame, val_df: pd.DataFrame, train_full_df: pd.DataFrame, test_df: pd.DataFrame):
    feature_sets = make_feature_sets(cfg)
    y_val = val_df['target_h'].values.astype(float)
    y_test = test_df['target_h'].values.astype(float)

    candidates = []
    for feat_name, feat_cols in feature_sets.items():
        for i, base_params in enumerate(cfg.models.boosting_param_grid, start=1):
            params = dict(base_params)
            params['random_state'] = cfg.models.random_state
            pred_val, _ = fit_predict_xgb(train_df, val_df, feat_cols, params)
            candidates.append({
                'name': f'{feat_name}__cfg{i}',
                'features': feat_cols,
                'params': params,
                'val_pred': pred_val.values,
                'metrics': calc_metrics(y_val, pred_val.values, gamma=cfg.features.load_weight_gamma),
            })

    val_table = pd.DataFrame([{'model': c['name'], **c['metrics']} for c in candidates]).sort_values('MAE').reset_index(drop=True)
    selected_names = val_table.head(cfg.models.top_k_boost_models)['model'].tolist()
    selected = [c for c in candidates if c['name'] in selected_names]

    val_pred = np.mean(np.column_stack([c['val_pred'] for c in selected]), axis=1)

    test_preds = {}
    for c in selected:
        pred_test, _ = fit_predict_xgb(train_full_df, test_df, c['features'], c['params'])
        test_preds[c['name']] = pred_test.values
    test_pred = np.mean(np.column_stack(list(test_preds.values())), axis=1)
    test_metrics = calc_metrics(y_test, test_pred, gamma=cfg.features.load_weight_gamma)

    return {
        'val_table': val_table,
        'selected_names': selected_names,
        'val_pred': val_pred,
        'test_pred': test_pred,
        'test_metrics': test_metrics,
    }
