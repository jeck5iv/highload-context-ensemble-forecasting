from __future__ import annotations

from pathlib import Path

from metroflow.studies.metro_short_history_forecasting.entrypoint import run_multistation_forecasting_experiment
from metroflow.studies.metro_short_history_forecasting.pipeline import run_multistation_forecasting_experiment as pipeline_runner


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "src" / "metroflow" / "studies" / "metro_short_history_forecasting"


def test_public_entrypoint_is_explicit() -> None:
    assert run_multistation_forecasting_experiment is pipeline_runner
    assert run_multistation_forecasting_experiment.__module__ == "metroflow.studies.metro_short_history_forecasting.pipeline"
    assert (STUDY / "entrypoint.py").exists()


def test_no_legacy_stage_module_remains() -> None:
    assert not (ROOT / "src" / "metroflow" / "multistation").exists()
    assert not list((ROOT / "src").rglob("stage" + "1.py"))


def test_vertical_study_layout_exists() -> None:
    expected = [
        "entrypoint.py",
        "config.py",
        "pipeline.py",
        "data_layer/buckets.py",
        "data_layer/places.py",
        "data_layer/supervised.py",
        "data_layer/weather.py",
        "models/classical.py",
        "models/boosting.py",
        "models/boosting_local.py",
        "models/article_models.py",
        "models/meta.py",
        "models/meta_filtered.py",
        "evaluation/metrics.py",
        "io.py",
    ]
    missing = [name for name in expected if not (STUDY / name).exists()]
    assert not missing
