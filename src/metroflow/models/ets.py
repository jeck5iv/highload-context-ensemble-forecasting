
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from metroflow.config import ExperimentConfig
from metroflow.evaluation.metrics import calc_metrics


def future_sum_from_count_path(count_path, horizon: int):
    count_path = np.asarray(count_path, dtype=float)
    return np.array([count_path[i + 1:i + horizon + 1].sum() for i in range(len(count_path) - horizon)])


def fit_predict_ets(train_count_df: pd.DataFrame, pred_count_df: pd.DataFrame, cfg: ExperimentConfig):
    y_train = train_count_df['count'].astype(float).values
    horizon_steps = len(pred_count_df)
    model = ExponentialSmoothing(
        y_train,
        trend='add',
        seasonal='add',
        seasonal_periods=cfg.day_lag,
        initialization_method='estimated'
    )
    fit = model.fit(optimized=True, use_brute=False)
    count_forecast = np.asarray(fit.forecast(horizon_steps), dtype=float)
    pred_anchor = future_sum_from_count_path(count_forecast, horizon=cfg.horizon_steps)
    anchor_index = pred_count_df.index[:len(pred_anchor)]
    pred = pd.Series(pred_anchor, index=anchor_index, dtype=float)
    return pred, fit


def run_ets(cfg: ExperimentConfig, train_tune_all: pd.DataFrame, val_all: pd.DataFrame, train_full_all: pd.DataFrame, test_all: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    y_val = val_df['target_h'].values.astype(float)
    y_test = test_df['target_h'].values.astype(float)
    pred_val, fit = fit_predict_ets(train_tune_all, val_all, cfg)
    pred_test, fit_full = fit_predict_ets(train_full_all, test_all, cfg)
    return {
        'val_pred': pred_val.loc[val_df.index].values,
        'test_pred': pred_test.loc[test_df.index].values,
        'val_metrics': calc_metrics(y_val, pred_val.loc[val_df.index].values, gamma=cfg.features.load_weight_gamma),
        'test_metrics': calc_metrics(y_test, pred_test.loc[test_df.index].values, gamma=cfg.features.load_weight_gamma),
        'fit': fit,
        'fit_full': fit_full,
    }
