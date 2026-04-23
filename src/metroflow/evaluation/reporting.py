from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import json
import pandas as pd

from metroflow.config import ExperimentConfig


def _delta(a: float, b: float) -> float:
    return a - b


def build_experiment_markdown(result: dict) -> str:
    cfg: ExperimentConfig = result['config']
    summary = result['summary_table']
    split = result['split']
    residual = result.get('residual_meta')
    linear = result.get('linear_meta')
    val_base = result.get('val_base_table')

    best_model = summary.index[0]
    best_row = summary.iloc[0]
    best_base_table = summary.loc[[idx for idx in summary.index if idx in ['ETS','MegaBoost','LSTM','Transformer-lite']]].sort_values('MAE')
    best_base_name = best_base_table.index[0]
    best_base_row = best_base_table.iloc[0]

    lines = []
    lines.append(f"# Experiment report: {cfg.name}")
    lines.append('')
    lines.append('## Setup')
    lines.append(f"- bucket_minutes: {cfg.bucket_minutes}")
    lines.append(f"- horizon_minutes: {cfg.horizon_minutes}")
    lines.append(f"- primary_mode: {cfg.primary_mode}")
    lines.append(f"- place_ids: {cfg.data.place_ids}")
    lines.append(f"- context_features: {cfg.features.context_features}")
    lines.append('')
    lines.append('## Split diagnostics')
    lines.append(f"- train_tune_all rows: {len(split['train_tune_all'])}")
    lines.append(f"- val_all rows: {len(split['val_all'])}")
    lines.append(f"- train_full_all rows: {len(split['train_full_all'])}")
    lines.append(f"- test_all rows: {len(split['test_all'])}")
    lines.append(f"- train_tune anchors: {len(split['train_tune'])}")
    lines.append(f"- val anchors: {len(split['val_df'])}")
    lines.append(f"- train_full anchors: {len(split['train_full'])}")
    lines.append(f"- test anchors: {len(split['test_df'])}")
    lines.append(f"- q80_tune: {split['q80_tune']:.4f}")
    lines.append(f"- q80_full: {split['q80_full']:.4f}")
    lines.append('')
    if val_base is not None:
        lines.append('## Validation base ranking')
        lines.append(val_base.to_markdown())
        lines.append('')
    lines.append('## Test summary')
    lines.append(summary.to_markdown())
    lines.append('')
    lines.append('## Best overall model')
    lines.append(f"- best_model: {best_model}")
    lines.append(f"- MAE: {best_row['MAE']:.4f}")
    lines.append(f"- RMSE: {best_row['RMSE']:.4f}")
    lines.append(f"- wMAPE: {best_row['wMAPE']:.6f}")
    lines.append(f"- LW-MAE: {best_row['LW-MAE']:.4f}")
    lines.append('')
    lines.append('## Best single/base model on test')
    lines.append(f"- best_base_model: {best_base_name}")
    lines.append(f"- MAE: {best_base_row['MAE']:.4f}")
    lines.append(f"- LW-MAE: {best_base_row['LW-MAE']:.4f}")
    lines.append('')
    lines.append('## Meta deltas vs best base')
    for name in ['Meta static', 'Meta context', 'Residual Linear', 'Residual XGB', 'Meta XGB static', 'Meta XGB + context']:
        if name in summary.index:
            row = summary.loc[name]
            lines.append(f"- {name}: ΔMAE={_delta(best_base_row['MAE'], row['MAE']):+.4f}, ΔLW-MAE={_delta(best_base_row['LW-MAE'], row['LW-MAE']):+.4f}")
    lines.append('')
    if linear is not None:
        lines.append('## Linear meta weights')
        lines.append(f"- static weights: {linear['static']['weights']}")
        lines.append(f"- context global weights: {linear['context']['global_weights']}")
        lines.append(f"- context weights by regime: {linear['context']['weights_by_ctx']}")
        lines.append('')
    if residual is not None:
        primary = residual['primary_model']
        lines.append('## Residual meta analysis')
        lines.append(f"- primary_model: {primary}")
        lines.append(f"- primary test metrics: {residual['metrics_primary']}")
        lines.append(f"- residual linear metrics: {residual['metrics_residual_linear']}")
        lines.append(f"- residual xgb metrics: {residual['metrics_residual_xgb']}")
        lines.append(f"- residual linear ΔMAE vs primary: {_delta(residual['metrics_primary']['MAE'], residual['metrics_residual_linear']['MAE']):+.4f}")
        lines.append(f"- residual xgb ΔMAE vs primary: {_delta(residual['metrics_primary']['MAE'], residual['metrics_residual_xgb']['MAE']):+.4f}")
        lines.append(f"- residual linear ΔLW-MAE vs primary: {_delta(residual['metrics_primary']['LW-MAE'], residual['metrics_residual_linear']['LW-MAE']):+.4f}")
        lines.append(f"- residual xgb ΔLW-MAE vs primary: {_delta(residual['metrics_primary']['LW-MAE'], residual['metrics_residual_xgb']['LW-MAE']):+.4f}")
        lines.append('')
        lines.append('### Residual linear coefficients')
        lines.append(residual['linear_coef'].sort_values('coef', key=lambda s: s.abs(), ascending=False).head(20).to_markdown(index=False))
        lines.append('')
        lines.append('### Residual XGB feature importances')
        lines.append(residual['xgb_importance'].sort_values('importance', ascending=False).head(20).to_markdown(index=False))
        lines.append('')
    return "\n".join(lines)


def _cfg_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return asdict(cfg)


def save_experiment_bundle(result: dict, output_dir: str | Path) -> dict[str, str]:
    cfg: ExperimentConfig = result['config']
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    summary = result['summary_table']
    summary.to_csv(output_dir / 'summary_table.csv')
    paths['summary_table'] = str(output_dir / 'summary_table.csv')

    if 'val_base_table' in result and result['val_base_table'] is not None:
        result['val_base_table'].to_csv(output_dir / 'validation_base_ranking.csv')
        paths['validation_base_ranking'] = str(output_dir / 'validation_base_ranking.csv')

    result['test_pred_table'].to_csv(output_dir / 'test_predictions.csv', index=False)
    result['val_pred_table'].to_csv(output_dir / 'validation_predictions.csv', index=False)
    paths['test_predictions'] = str(output_dir / 'test_predictions.csv')
    paths['validation_predictions'] = str(output_dir / 'validation_predictions.csv')

    md = build_experiment_markdown(result)
    (output_dir / 'report.md').write_text(md, encoding='utf-8')
    paths['report_md'] = str(output_dir / 'report.md')

    (output_dir / 'config.json').write_text(json.dumps(_cfg_to_dict(cfg), ensure_ascii=False, indent=2), encoding='utf-8')
    paths['config_json'] = str(output_dir / 'config.json')
    return paths
