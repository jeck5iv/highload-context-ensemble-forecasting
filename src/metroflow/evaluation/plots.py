from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def _style_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=90)
    ax.grid(True, alpha=0.3)


def plot_forecasts(df: pd.DataFrame, time_col: str, y_col: str = 'y_true', model_cols: list[str] | None = None, span_minutes: int | None = None, title: str | None = None):
    model_cols = model_cols or [c for c in df.columns if c not in {time_col, y_col, 'hl_daylag'}]
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df[time_col], df[y_col], label='Real', linewidth=2.8)
    for c in model_cols:
        ax.plot(df[time_col], df[c], label=c, linewidth=2)
    if span_minutes and 'hl_daylag' in df.columns:
        for _, row in df.iterrows():
            if int(row['hl_daylag']) == 1:
                ax.axvspan(row[time_col], row[time_col] + pd.Timedelta(minutes=span_minutes), alpha=0.05)
    _style_time_axis(ax)
    ax.set_title(title or 'Forecast comparison')
    ax.legend(ncol=2)
    fig.tight_layout()
    return fig


def plot_abs_errors(df: pd.DataFrame, time_col: str, y_true: np.ndarray, model_cols: list[str], title: str | None = None):
    err_df = pd.DataFrame({time_col: df[time_col]})
    for c in model_cols:
        err_df[f'{c}_abs_err'] = np.abs(y_true - df[c].values)
    fig, ax = plt.subplots(figsize=(15, 5))
    for c in err_df.columns[1:]:
        ax.plot(err_df[time_col], err_df[c], label=c, linewidth=2)
    _style_time_axis(ax)
    ax.set_title(title or 'Absolute errors')
    ax.legend(ncol=2)
    fig.tight_layout()
    return fig


def plot_metric_bars(summary_table: pd.DataFrame, metric: str):
    fig, ax = plt.subplots(figsize=(11, 4))
    values = summary_table[metric].values
    labels = summary_table.index.tolist()
    ax.bar(labels, values)
    ax.set_ylabel(metric)
    ax.set_title(f'Model comparison by {metric}')
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=30, ha='right')
    fig.tight_layout()
    return fig


def plot_residual_corrections(df: pd.DataFrame, time_col: str):
    cols = [c for c in ['Linear correction', 'XGB correction'] if c in df.columns]
    fig, ax = plt.subplots(figsize=(15, 5))
    for c in cols:
        ax.plot(df[time_col], df[c], label=c, linewidth=2)
    ax.axhline(0, color='black', linewidth=1)
    _style_time_axis(ax)
    ax.set_title('Residual corrections')
    ax.legend()
    fig.tight_layout()
    return fig


def plot_importance(df: pd.DataFrame, value_col: str, title: str, top_n: int = 15):
    work = df.sort_values(value_col, ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(work['feature'][::-1], work[value_col][::-1])
    ax.set_title(title)
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    return fig


def save_core_figures(result: dict, output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    cfg = result['config']
    plot_df = result['test_pred_table'].copy()
    y_test = plot_df['y_true'].values
    main_cols = [c for c in ['ETS', 'MegaBoost', 'Transformer-lite', 'Meta static', 'Meta context', 'Residual XGB'] if c in plot_df.columns]
    fig = plot_forecasts(plot_df, cfg.time_col, model_cols=main_cols, span_minutes=cfg.bucket_minutes, title=f"Forecasts: bucket={cfg.bucket_minutes}m horizon={cfg.horizon_minutes}m")
    path = output_dir / 'forecast_comparison.png'
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    paths['forecast_comparison'] = str(path)

    fig = plot_abs_errors(plot_df, cfg.time_col, y_test, [c for c in main_cols if c in plot_df.columns], title='Absolute errors on test')
    path = output_dir / 'absolute_errors.png'
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    paths['absolute_errors'] = str(path)

    for metric in ['MAE', 'RMSE', 'wMAPE', 'LW-MAE']:
        fig = plot_metric_bars(result['summary_table'], metric)
        path = output_dir / f'metric_{metric}.png'
        fig.savefig(path, dpi=160, bbox_inches='tight')
        plt.close(fig)
        paths[f'metric_{metric}'] = str(path)

    if result.get('residual_meta') is not None:
        resid = result['residual_meta']
        fig = plot_importance(resid['linear_coef'].assign(abs_coef=resid['linear_coef']['coef'].abs()), 'abs_coef', 'Residual Linear |coef|')
        path = output_dir / 'residual_linear_coef.png'
        fig.savefig(path, dpi=160, bbox_inches='tight')
        plt.close(fig)
        paths['residual_linear_coef'] = str(path)

        fig = plot_importance(resid['xgb_importance'], 'importance', 'Residual XGB feature importance')
        path = output_dir / 'residual_xgb_importance.png'
        fig.savefig(path, dpi=160, bbox_inches='tight')
        plt.close(fig)
        paths['residual_xgb_importance'] = str(path)

        work = plot_df.copy()
        if 'Residual Linear' in work.columns and 'Primary' in work.columns:
            work['Linear correction'] = work['Residual Linear'] - work['Primary']
        if 'Residual XGB' in work.columns and 'Primary' in work.columns:
            work['XGB correction'] = work['Residual XGB'] - work['Primary']
        if any(c in work.columns for c in ['Linear correction', 'XGB correction']):
            fig = plot_residual_corrections(work, cfg.time_col)
            path = output_dir / 'residual_corrections.png'
            fig.savefig(path, dpi=160, bbox_inches='tight')
            plt.close(fig)
            paths['residual_corrections'] = str(path)
    return paths
