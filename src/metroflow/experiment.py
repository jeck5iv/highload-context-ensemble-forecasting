from __future__ import annotations

import pandas as pd

from metroflow.config import ExperimentConfig
from metroflow.data.access import load_pass_data, load_station_metadata
from metroflow.features.builders import make_target_frame, make_weekday_split
from metroflow.models.boosting import run_megaboost
from metroflow.models.ets import run_ets
from metroflow.models.sequence import run_sequence_models
from metroflow.models.meta import build_prediction_tables, run_linear_meta, run_residual_meta, run_direct_meta_xgb
from metroflow.evaluation.metrics import result_table, calc_metrics


def _build_metrics_by_model(ets, megaboost, seq, linear_meta, residual_meta, direct_meta):
    metrics_by_model = {
        'ETS': ets['test_metrics'],
        'MegaBoost': megaboost['test_metrics'],
    }
    for name, payload in seq.items():
        metrics_by_model[name] = payload['test_metrics']
    metrics_by_model['Meta static'] = linear_meta['static']['metrics']
    metrics_by_model['Meta context'] = linear_meta['context']['metrics']
    if residual_meta is not None:
        metrics_by_model[f"Primary ({residual_meta['primary_model']})"] = residual_meta['metrics_primary']
        metrics_by_model['Residual Linear'] = residual_meta['metrics_residual_linear']
        metrics_by_model['Residual XGB'] = residual_meta['metrics_residual_xgb']
    if direct_meta is not None:
        metrics_by_model['Meta XGB static'] = direct_meta['metrics_static']
        metrics_by_model['Meta XGB + context'] = direct_meta['metrics_context']
    return metrics_by_model


def run_experiment(cfg: ExperimentConfig, pass_df: pd.DataFrame | None = None) -> dict:
    pass_df = load_pass_data(cfg) if pass_df is None else pass_df.copy()
    station_meta = load_station_metadata(cfg)

    frame = make_target_frame(pass_df, cfg)
    split = make_weekday_split(frame, cfg)

    megaboost = run_megaboost(cfg, split['train_tune'], split['val_df'], split['train_full'], split['test_df'])
    ets = run_ets(cfg, split['train_tune_all'], split['val_all'], split['train_full_all'], split['test_all'], split['val_df'], split['test_df'])
    seq = run_sequence_models(cfg, split)

    base_preds = {
        'ETS': {'val_pred': ets['val_pred'], 'test_pred': ets['test_pred']},
        'MegaBoost': {'val_pred': megaboost['val_pred'], 'test_pred': megaboost['test_pred']},
    }
    for name, payload in seq.items():
        base_preds[name] = {'val_pred': payload['val_pred'], 'test_pred': payload['test_pred']}

    val_pred_table, test_pred_table = build_prediction_tables(cfg, split, base_preds)
    linear_meta = run_linear_meta(cfg, val_pred_table, test_pred_table)
    residual_meta = run_residual_meta(cfg, val_pred_table, test_pred_table) if cfg.models.run_residual_meta else None
    direct_meta = run_direct_meta_xgb(cfg, val_pred_table, test_pred_table)

    test_pred_table = test_pred_table.copy()
    test_pred_table['Meta static'] = linear_meta['static']['test_pred']
    test_pred_table['Meta context'] = linear_meta['context']['test_pred']
    if residual_meta is not None:
        test_pred_table['Primary'] = residual_meta['test_pred_primary']
        test_pred_table['Residual Linear'] = residual_meta['test_pred_linear']
        test_pred_table['Residual XGB'] = residual_meta['test_pred_xgb']
    if direct_meta is not None:
        test_pred_table['Meta XGB static'] = direct_meta['test_pred_static']
        test_pred_table['Meta XGB + context'] = direct_meta['test_pred_context']

    metrics_by_model = _build_metrics_by_model(ets, megaboost, seq, linear_meta, residual_meta, direct_meta)
    summary_table = result_table(metrics_by_model)

    val_base_metrics = {
        'ETS': ets['val_metrics'],
        'MegaBoost': calc_metrics(split['val_df']['target_h'].values, megaboost['val_pred'], gamma=cfg.features.load_weight_gamma),
    }
    for name, payload in seq.items():
        val_base_metrics[name] = payload['val_metrics']
    val_base_table = result_table(val_base_metrics)

    return {
        'config': cfg,
        'station_metadata': station_meta,
        'pass_df': pass_df,
        'frame': frame,
        'split': split,
        'megaboost': megaboost,
        'ets': ets,
        'sequence': seq,
        'val_pred_table': val_pred_table,
        'test_pred_table': test_pred_table,
        'linear_meta': linear_meta,
        'residual_meta': residual_meta,
        'direct_meta': direct_meta,
        'metrics_by_model': metrics_by_model,
        'summary_table': summary_table,
        'val_base_table': val_base_table,
    }
