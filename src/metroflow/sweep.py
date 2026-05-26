
from __future__ import annotations

from pathlib import Path
import pandas as pd

from metroflow.config import load_experiment_config
from metroflow.experiment import run_experiment


def run_config_paths(paths: list[str | Path]):
    rows = []
    outputs = {}
    for path in paths:
        cfg = load_experiment_config(path)
        out = run_experiment(cfg)
        outputs[cfg.name] = out
        best = out['summary_table'].iloc[0].to_dict()
        row = {'config_name': cfg.name, 'bucket_minutes': cfg.bucket_minutes, 'horizon_minutes': cfg.horizon_minutes, 'best_model': out['summary_table'].index[0]}
        for k, v in best.items():
            row[f'best_{k}'] = v
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['horizon_minutes', 'bucket_minutes']), outputs
