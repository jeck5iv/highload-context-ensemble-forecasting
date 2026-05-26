from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "h060m_b015m"
SCRIPT = ROOT / "scripts" / "compare_regression_outputs.py"
_spec = importlib.util.spec_from_file_location("compare_regression_outputs", SCRIPT)
_compare_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_compare_module)
compare_outputs = _compare_module.compare_outputs


def test_central_fixture_has_expected_key_values() -> None:
    micro = pd.read_csv(FIXTURE / "micro_metrics.csv").rename(columns={"Unnamed: 0": "model"})
    row = micro.loc[micro["model"] == "Residual XGB ETS LW"].iloc[0]
    assert round(float(row["LW-MAE"]), 2) == 79.51
    assert round(float(row["MAE"]), 2) == 53.97

    delta = pd.read_csv(FIXTURE / "station_bootstrap_delta_ci.csv")
    ets_lw = delta[
        (delta["mode"] == "micro")
        & (delta["model"] == "Residual XGB ETS LW")
        & (delta["baseline"] == "ETS local")
        & (delta["metric"] == "LW-MAE")
    ].iloc[0]
    assert round(float(ets_lw["delta_estimate"]), 2) == 3.01
    assert bool(ets_lw["significant_95"])


def test_output_comparator_is_fast() -> None:
    compare_outputs(FIXTURE, FIXTURE)
