from __future__ import annotations

from pathlib import Path

from metroflow.config import ExperimentConfig
from metroflow.experiment import run_experiment
from metroflow.evaluation.reporting import save_experiment_bundle, build_experiment_markdown
from metroflow.evaluation.plots import save_core_figures


def default_output_dir(cfg: ExperimentConfig) -> Path:
    return Path(cfg.output.root_dir) / cfg.name


def run_and_materialize(cfg: ExperimentConfig, pass_df=None, output_dir: str | Path | None = None) -> dict:
    result = run_experiment(cfg, pass_df=pass_df)
    out_dir = Path(output_dir) if output_dir is not None else default_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = {}
    if cfg.output.save_tables or cfg.output.save_markdown or cfg.output.save_run_config:
        artifact_paths.update(save_experiment_bundle(result, out_dir))
    if cfg.output.save_plots:
        artifact_paths.update(save_core_figures(result, out_dir))

    result['artifact_paths'] = artifact_paths
    result['report_markdown'] = build_experiment_markdown(result)
    result['output_dir'] = str(out_dir)
    return result
