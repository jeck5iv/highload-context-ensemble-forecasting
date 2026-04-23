
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calc_metrics(y_true, y_pred, gamma: float = 1.0) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
    mean_load = np.mean(y_true)
    lw_mae = np.mean(np.abs(y_true - y_pred) * (y_true / mean_load) ** gamma)
    return {
        'MAE': mae,
        'RMSE': rmse,
        'wMAPE': wmape,
        'LW-MAE': lw_mae,
    }


def result_table(metrics_by_model: dict) -> 'pd.DataFrame':
    import pandas as pd
    return pd.DataFrame(metrics_by_model).T.sort_values('MAE')
