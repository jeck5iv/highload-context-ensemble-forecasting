
from __future__ import annotations

import numpy as np
import pandas as pd
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
    return pd.DataFrame(metrics_by_model).T.sort_values('MAE')


def _block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a length-n index vector made of contiguous blocks.

    This is safer than ordinary pointwise bootstrap for rolling time-series targets,
    because adjacent anchors are autocorrelated and often overlap by construction.
    """
    if n <= 0:
        raise ValueError('Cannot bootstrap an empty test set')
    block_size = max(1, min(int(block_size), n))
    starts = np.arange(0, n - block_size + 1)
    sampled = []
    while len(sampled) < n:
        start = int(rng.choice(starts))
        sampled.extend(range(start, start + block_size))
    return np.asarray(sampled[:n], dtype=int)


def bootstrap_metric_intervals(
    pred_table: pd.DataFrame,
    model_cols: list[str],
    *,
    baseline_col: str | None = None,
    gamma: float = 1.0,
    n_boot: int = 2000,
    block_size: int = 8,
    alpha: float = 0.05,
    random_state: int = 3228,
) -> dict[str, pd.DataFrame]:
    """Paired block-bootstrap confidence intervals for test metrics.

    Returns two tables:
    - metric_ci: point estimates and 95% confidence intervals for each model/metric;
    - delta_ci: paired confidence intervals for baseline_metric - model_metric.

    Positive delta means that the candidate model improves over the baseline.
    """
    cols = ['y_true', *model_cols]
    data = pred_table[cols].dropna().reset_index(drop=True)
    y = data['y_true'].to_numpy(dtype=float)
    preds = {m: data[m].to_numpy(dtype=float) for m in model_cols}

    rng = np.random.default_rng(random_state)
    boot = {m: {metric: [] for metric in ['MAE', 'RMSE', 'wMAPE', 'LW-MAE']} for m in model_cols}
    delta_boot = None
    if baseline_col is not None:
        delta_boot = {m: {metric: [] for metric in ['MAE', 'RMSE', 'wMAPE', 'LW-MAE']} for m in model_cols if m != baseline_col}

    for _ in range(n_boot):
        idx = _block_bootstrap_indices(len(data), block_size, rng)
        sampled_metrics = {}
        for m in model_cols:
            sampled_metrics[m] = calc_metrics(y[idx], preds[m][idx], gamma=gamma)
            for metric, value in sampled_metrics[m].items():
                boot[m][metric].append(value)
        if baseline_col is not None:
            for m in delta_boot:
                for metric in delta_boot[m]:
                    delta_boot[m][metric].append(sampled_metrics[baseline_col][metric] - sampled_metrics[m][metric])

    lo_q, hi_q = alpha / 2, 1 - alpha / 2
    metric_rows = []
    for m in model_cols:
        point = calc_metrics(y, preds[m], gamma=gamma)
        for metric in ['MAE', 'RMSE', 'wMAPE', 'LW-MAE']:
            vals = np.asarray(boot[m][metric], dtype=float)
            metric_rows.append({
                'model': m,
                'metric': metric,
                'estimate': point[metric],
                'ci_low': float(np.quantile(vals, lo_q)),
                'ci_high': float(np.quantile(vals, hi_q)),
            })

    delta_rows = []
    if baseline_col is not None:
        base_point = calc_metrics(y, preds[baseline_col], gamma=gamma)
        for m in delta_boot:
            point = calc_metrics(y, preds[m], gamma=gamma)
            for metric in ['MAE', 'RMSE', 'wMAPE', 'LW-MAE']:
                vals = np.asarray(delta_boot[m][metric], dtype=float)
                delta_rows.append({
                    'baseline': baseline_col,
                    'model': m,
                    'metric': metric,
                    'delta_estimate': base_point[metric] - point[metric],
                    'ci_low': float(np.quantile(vals, lo_q)),
                    'ci_high': float(np.quantile(vals, hi_q)),
                    'significant_95': bool((np.quantile(vals, lo_q) > 0) or (np.quantile(vals, hi_q) < 0)),
                })

    return {
        'metric_ci': pd.DataFrame(metric_rows),
        'delta_ci': pd.DataFrame(delta_rows),
    }
