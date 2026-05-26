#!/usr/bin/env python3
"""Build prediction-ready plots and LaTeX tables from multistation megagrid outputs.

The script is intentionally post-processing only: it does not retrain models.  It reads
saved prediction tables and aggregate CSV outputs, reproduces the metric definitions used
in ``metroflow.studies.metro_short_history_forecasting.entrypoint``, computes a small number of prediction-specific bootstrap
comparisons, and writes figures/tables that can be copied into the Overleaf project.

Typical usage:

    python scripts/build_prediction_artifacts.py \
        --megagrid-root outputs/multistation_megagrid_overnight \
        --out-dir outputs/prediction_artifacts

The input directory may either be the megagrid directory itself or a parent directory
containing a single subdirectory with ``grid_micro_metrics.csv``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


DEFAULT_CENTRAL_SCENARIO = "h060m_b015m"
DEFAULT_SHORT_ARTICLE_SCENARIO = "h015m_b005m"
DEFAULT_EXCLUDED_SCENARIOS = {"h360m_b015m"}
DEFAULT_BOOTSTRAP_ITERATIONS = 1000
DEFAULT_BOOTSTRAP_SEED = 20260517

# The main narrative in the prediction treats this family as the proposed family.
# Simple ETS-residual boosting remains an important internal baseline, but it is not
# counted as the final proposed meta-family in the robustness summary.
PROPOSED_MODEL_PATTERNS = (
    re.compile(r"^Residual "),
    re.compile(r"^Meta "),
)

CENTRAL_RESULT_MODELS = [
    "Residual XGB ETS LW",
    "Residual XGB ETS",
    "Residual Linear ETS",
    "MegaBoost ETS residual",
    "ETS local",
    "Seasonal daylag",
    "ET-LGBM-PSO-lite",
    "MegaBoost rich norm LW",
]

CENTRAL_SIGNIFICANCE_BASELINES = [
    "Seasonal daylag",
    "ETS local",
    "MegaBoost ETS residual",
    "ET-LGBM-PSO-lite",
]

CENTRAL_SIGNIFICANCE_METRICS = ["LW-MAE", "MAE", "HL-MAE"]

ARTICLE_MODELS_MAIN = [
    "ET-LGBM-PSO-lite",
    "VMD-LightGBM-lite",
    "ST-LightGBM-MIC-lite",
    "TwoStage-FS-MSD-Hybrid-lite",
    "PSR-CNN-LSTM-lite",
    "MSD-Informer-BiLSTM-lite",
    "CNN-BiLSTM-Attention-lite",
    "BiLSTM-CTPA-lite",
]

SHORT_ARTICLE_CASE_MODELS = [
    "Residual XGB ETS LW",
    "ET-LGBM-PSO-lite",
    "ETS local",
    "Seasonal daylag",
]

MAIN_GRID_ORDER = [
    "h015m_b005m",
    "h015m_b015m",
    "h060m_b005m",
    "h060m_b015m",
    "h060m_b030m",
    "h060m_b060m",
    "h360m_b030m",
    "h360m_b060m",
]

GRID_METRICS = ["MAE", "RMSE", "wMAPE", "LW-MAE", "HL-MAE"]
TABLE_METRICS = ["MAE", "RMSE", "wMAPE", "LW-MAE", "HL-MAE"]


@dataclass(frozen=True)
class Paths:
    root: Path
    out: Path
    figures: Path
    tables: Path
    csv: Path
    snippets: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--megagrid-root",
        type=Path,
        default=Path("outputs/multistation_megagrid_overnight"),
        help="Directory with grid_*.csv files and per-scenario subdirectories.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/prediction_artifacts"),
        help="Where to write prediction figures, LaTeX tables and CSV summaries.",
    )
    parser.add_argument(
        "--central-scenario",
        default=DEFAULT_CENTRAL_SCENARIO,
        help="Scenario used for central tables and forecast plots.",
    )
    parser.add_argument(
        "--short-article-scenario",
        default=DEFAULT_SHORT_ARTICLE_SCENARIO,
        help="Scenario used to illustrate the strong external article comparator.",
    )
    parser.add_argument(
        "--exclude-scenario",
        action="append",
        default=[],
        help="Scenario excluded from prediction-facing grid summaries; can be repeated.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=DEFAULT_BOOTSTRAP_ITERATIONS,
        help="Number of station-bootstrap repetitions for prediction-specific delta panels.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Seed for reproducible post-processing bootstrap.",
    )
    parser.add_argument(
        "--example-station",
        default=None,
        help=(
            "Optional station_key for central forecast plots. If omitted, the script selects "
            "a high-demand station with a non-degenerate high-load segment."
        ),
    )
    parser.add_argument(
        "--copy-to-overleaf",
        type=Path,
        default=None,
        help="Optional Overleaf project directory; generated figures/tables are copied into figures/ and tables/.",
    )
    return parser.parse_args()


def ensure_paths(out_dir: Path) -> Paths:
    figures = out_dir / "figures"
    tables = out_dir / "tables"
    csv_dir = out_dir / "csv"
    snippets = out_dir / "snippets"
    for path in (out_dir, figures, tables, csv_dir, snippets):
        path.mkdir(parents=True, exist_ok=True)
    return Paths(root=out_dir, out=out_dir, figures=figures, tables=tables, csv=csv_dir, snippets=snippets)


def resolve_megagrid_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "grid_micro_metrics.csv").exists():
        return path
    candidates = sorted(path.glob("**/grid_micro_metrics.csv"))
    if len(candidates) == 1:
        return candidates[0].parent
    if not candidates:
        raise FileNotFoundError(
            f"Could not find grid_micro_metrics.csv under {path}. Pass --megagrid-root explicitly."
        )
    raise FileNotFoundError(
        "More than one megagrid output directory found under "
        f"{path}: {[str(c.parent) for c in candidates]}. Pass --megagrid-root explicitly."
    )


def safe_read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def scenario_dir(root: Path, scenario: str) -> Path:
    path = root / scenario
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def scenario_predictions(root: Path, scenario: str) -> pd.DataFrame:
    df = safe_read_csv(scenario_dir(root, scenario) / "test_predictions.csv")
    time_col = infer_time_column(df)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    return df


def infer_time_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.startswith("bucket_")]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one bucket_* column, got {candidates}")
    return candidates[0]


def load_grid_frames(root: Path) -> dict[str, pd.DataFrame]:
    frames = {
        "micro": safe_read_csv(root / "grid_micro_metrics.csv"),
        "macro": safe_read_csv(root / "grid_macro_metrics.csv"),
        "top": safe_read_csv(root / "grid_top_models.csv"),
        "bootstrap_delta": safe_read_csv(root / "grid_station_bootstrap_delta_ci.csv"),
        "bootstrap_metric": safe_read_csv(root / "grid_station_bootstrap_metric_ci.csv"),
        "bootstrap_refs": safe_read_csv(root / "grid_bootstrap_reference_models.csv"),
        "significance_summary": safe_read_csv(root / "grid_significance_summary.csv", required=False),
    }
    return frames


def is_proposed_model(name: str) -> bool:
    return any(pattern.search(name) is not None for pattern in PROPOSED_MODEL_PATTERNS)


def fmt_metric(value: float, metric: str) -> str:
    if pd.isna(value):
        return "---"
    if metric == "wMAPE":
        return f"{value:.4f}"
    if metric == "HL-underRate":
        return f"{value:.3f}"
    return f"{value:.2f}"


def fmt_delta(value: float) -> str:
    if pd.isna(value):
        return "---"
    return f"{value:+.2f}"


def fmt_ci(low: float, high: float) -> str:
    if pd.isna(low) or pd.isna(high):
        return "[---; ---]"
    return f"[{low:+.2f}; {high:+.2f}]"


def model_tex(name: str) -> str:
    return r"\model{" + escape_latex_text(name) + "}"


def escape_latex_text(text: str) -> str:
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(text))


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def metrics_from_predictions(df: pd.DataFrame, model: str) -> dict[str, float]:
    y = pd.to_numeric(df["y_true"], errors="coerce").astype(float).to_numpy()
    pred = pd.to_numeric(df[model], errors="coerce").astype(float).to_numpy()
    station_mean = pd.to_numeric(df["station_train_mean_target"], errors="coerce").astype(float).to_numpy()
    finite_mean = station_mean[np.isfinite(station_mean) & (station_mean > 0)]
    fallback = float(np.nanmean(y)) if np.isfinite(y).any() else 1.0
    if finite_mean.size:
        fallback = float(np.nanmedian(finite_mean))
    station_mean = np.where(np.isfinite(station_mean) & (station_mean > 0), station_mean, fallback)
    ae = np.abs(y - pred)
    sq = (y - pred) ** 2
    hl_mask = pd.to_numeric(df["hl_actual"], errors="coerce").fillna(0).astype(int).to_numpy().astype(bool)
    weights = np.maximum(y, 0.0) / np.maximum(station_mean, 1e-9)
    result = {
        "MAE": float(np.nanmean(ae)),
        "RMSE": float(np.sqrt(np.nanmean(sq))),
        "wMAPE": float(np.nansum(ae) / np.maximum(np.nansum(np.abs(y)), 1e-9)),
        "LW-MAE": float(np.nanmean(ae * weights)),
    }
    if hl_mask.any():
        result["HL-MAE"] = float(np.nanmean(ae[hl_mask]))
        result["HL-underMAE"] = float(np.nanmean(np.maximum(y[hl_mask] - pred[hl_mask], 0.0)))
        result["HL-underRate"] = float(np.nanmean(pred[hl_mask] < y[hl_mask]))
    else:
        result["HL-MAE"] = np.nan
        result["HL-underMAE"] = np.nan
        result["HL-underRate"] = np.nan
    return result


def station_metric_components(df: pd.DataFrame, model: str) -> pd.DataFrame:
    work = df[["station_key", "y_true", "station_train_mean_target", "hl_actual"]].copy()
    work["pred"] = pd.to_numeric(df[model], errors="coerce").astype(float).to_numpy()
    work["y"] = pd.to_numeric(work["y_true"], errors="coerce").astype(float)
    work["station_mean"] = pd.to_numeric(work["station_train_mean_target"], errors="coerce").astype(float)
    finite_means = work.loc[work["station_mean"].gt(0) & work["station_mean"].notna(), "station_mean"]
    fallback = float(finite_means.median()) if not finite_means.empty else max(float(work["y"].mean()), 1.0)
    work["station_mean"] = work["station_mean"].where(work["station_mean"].gt(0) & work["station_mean"].notna(), fallback)
    work["ae"] = (work["y"] - work["pred"]).abs()
    work["sq"] = (work["y"] - work["pred"]) ** 2
    work["lw_num"] = work["ae"] * (work["y"].clip(lower=0.0) / work["station_mean"].clip(lower=1e-9))
    work["abs_y"] = work["y"].abs()
    work["hl_mask"] = pd.to_numeric(work["hl_actual"], errors="coerce").fillna(0).astype(int).astype(bool)
    work["hl_ae"] = work["ae"].where(work["hl_mask"], 0.0)
    grouped = work.groupby("station_key", sort=True).agg(
        n=("ae", "size"),
        ae_sum=("ae", "sum"),
        sq_sum=("sq", "sum"),
        lw_sum=("lw_num", "sum"),
        abs_y_sum=("abs_y", "sum"),
        hl_n=("hl_mask", "sum"),
        hl_ae_sum=("hl_ae", "sum"),
    )
    return grouped.reset_index()


def metric_from_component_sums(sample: pd.DataFrame, metric: str) -> float:
    n = max(float(sample["n"].sum()), 1.0)
    if metric == "MAE":
        return float(sample["ae_sum"].sum() / n)
    if metric == "RMSE":
        return float(math.sqrt(sample["sq_sum"].sum() / n))
    if metric == "wMAPE":
        return float(sample["ae_sum"].sum() / max(float(sample["abs_y_sum"].sum()), 1e-9))
    if metric == "LW-MAE":
        return float(sample["lw_sum"].sum() / n)
    if metric == "HL-MAE":
        hl_n = float(sample["hl_n"].sum())
        return float(sample["hl_ae_sum"].sum() / max(hl_n, 1.0))
    raise KeyError(metric)


def bootstrap_delta_table(
    df: pd.DataFrame,
    model: str,
    baselines: Sequence[str],
    metrics: Sequence[str],
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    model_components = station_metric_components(df, model)
    station_keys = model_components["station_key"].tolist()
    baseline_components = {baseline: station_metric_components(df, baseline) for baseline in baselines}
    for baseline, comp in baseline_components.items():
        if comp["station_key"].tolist() != station_keys:
            raise ValueError(f"Station order mismatch for baseline {baseline}")

    point = []
    for baseline in baselines:
        for metric in metrics:
            baseline_est = metric_from_component_sums(baseline_components[baseline], metric)
            model_est = metric_from_component_sums(model_components, metric)
            point.append({
                "model": model,
                "baseline": baseline,
                "metric": metric,
                "model_estimate": model_est,
                "baseline_estimate": baseline_est,
                "delta_estimate": baseline_est - model_est,
            })

    boot_values: dict[tuple[str, str], list[float]] = {(baseline, metric): [] for baseline in baselines for metric in metrics}
    n_stations = len(station_keys)
    # DataFrame iloc with integer indices is fast enough here: at most a handful of prediction-specific comparisons.
    for _ in range(iterations):
        idx = rng.integers(0, n_stations, size=n_stations)
        model_sample = model_components.iloc[idx]
        for baseline in baselines:
            baseline_sample = baseline_components[baseline].iloc[idx]
            for metric in metrics:
                delta = metric_from_component_sums(baseline_sample, metric) - metric_from_component_sums(model_sample, metric)
                boot_values[(baseline, metric)].append(float(delta))

    rows = []
    for row in point:
        vals = np.asarray(boot_values[(row["baseline"], row["metric"])], dtype=float)
        low, high = np.quantile(vals, [0.025, 0.975])
        rows.append({
            **row,
            "ci_low": float(low),
            "ci_high": float(high),
            "significant_95": bool((low > 0.0) or (high < 0.0)),
        })
    return pd.DataFrame(rows)


def latex_table_central_results(micro: pd.DataFrame, scenario: str, out_path: Path) -> pd.DataFrame:
    sub = micro.loc[micro["scenario"].eq(scenario) & micro["model"].isin(CENTRAL_RESULT_MODELS)].copy()
    sub["_order"] = sub["model"].map({m: i for i, m in enumerate(CENTRAL_RESULT_MODELS)})
    sub = sub.sort_values("_order")
    minima = {metric: sub[metric].min() for metric in TABLE_METRICS}
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Ключевые результаты в центральной постановке $H=60$ минут, $B=15$ минут.}",
        r"\label{tab:central_results}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Модель & MAE & RMSE & wMAPE & LW-MAE & HL-MAE \\",
        r"\midrule",
    ]
    for _, row in sub.iterrows():
        cells = [model_tex(str(row["model"]))]
        for metric in TABLE_METRICS:
            text = fmt_metric(float(row[metric]), metric)
            if math.isclose(float(row[metric]), float(minima[metric]), rel_tol=1e-12, abs_tol=1e-12):
                text = r"\textbf{" + text + "}"
            cells.append(text)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_text(out_path, "\n".join(lines))
    return sub.drop(columns=["_order"])


def latex_table_significance(delta: pd.DataFrame, out_path: Path) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Статистическая значимость улучшений финальной модели в центральной постановке. Положительная дельта означает уменьшение ошибки относительно baseline.}",
        r"\label{tab:central_ci_extended}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}Xlrrc}",
        r"\toprule",
        r"Модель & Baseline & Метрика & $\Delta$ & 95\% CI & Значимо \\",
        r"\midrule",
    ]
    for _, row in delta.iterrows():
        lines.append(
            " & ".join(
                [
                    model_tex(str(row["model"])),
                    model_tex(str(row["baseline"])),
                    escape_latex_text(str(row["metric"])),
                    fmt_delta(float(row["delta_estimate"])),
                    fmt_ci(float(row["ci_low"]), float(row["ci_high"])),
                    "да" if bool(row["significant_95"]) else "нет",
                ]
            )
            + r" \\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\normalsize", r"\end{table}"]
    write_text(out_path, "\n".join(lines))


def latex_table_article_central(micro: pd.DataFrame, scenario: str, out_path: Path) -> pd.DataFrame:
    sub = micro.loc[micro["scenario"].eq(scenario) & micro["model"].isin(ARTICLE_MODELS_MAIN)].copy()
    sub["_order"] = sub["model"].map({m: i for i, m in enumerate(ARTICLE_MODELS_MAIN)})
    sub = sub.sort_values("_order")
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Representative article-inspired comparators в центральной постановке $H=60$, $B=15$ минут.}",
        r"\label{tab:article_comparators_central}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Модель & MAE & RMSE & LW-MAE & HL-MAE \\",
        r"\midrule",
    ]
    for _, row in sub.iterrows():
        lines.append(
            " & ".join(
                [
                    model_tex(str(row["model"])),
                    fmt_metric(float(row["MAE"]), "MAE"),
                    fmt_metric(float(row["RMSE"]), "RMSE"),
                    fmt_metric(float(row["LW-MAE"]), "LW-MAE"),
                    fmt_metric(float(row["HL-MAE"]), "HL-MAE"),
                ]
            )
            + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\normalsize", r"\end{table}"]
    write_text(out_path, "\n".join(lines))
    return sub.drop(columns=["_order"])


def latex_table_short_article_case(micro: pd.DataFrame, scenario: str, out_path: Path) -> pd.DataFrame:
    sub = micro.loc[micro["scenario"].eq(scenario) & micro["model"].isin(SHORT_ARTICLE_CASE_MODELS)].copy()
    sub["_order"] = sub["model"].map({m: i for i, m in enumerate(SHORT_ARTICLE_CASE_MODELS)})
    sub = sub.sort_values("_order")
    minima = {metric: sub[metric].min() for metric in ["MAE", "RMSE", "LW-MAE", "HL-MAE"]}
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Короткий горизонт $H=15$, $B=5$ минут: внешний article comparator превосходит базовые модели, но уступает финальному ансамблю по LW-MAE.}",
        r"\label{tab:short_article_case}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Модель & MAE & RMSE & LW-MAE & HL-MAE \\",
        r"\midrule",
    ]
    for _, row in sub.iterrows():
        cells = [model_tex(str(row["model"]))]
        for metric in ["MAE", "RMSE", "LW-MAE", "HL-MAE"]:
            text = fmt_metric(float(row[metric]), metric)
            if math.isclose(float(row[metric]), float(minima[metric]), rel_tol=1e-12, abs_tol=1e-12):
                text = r"\textbf{" + text + "}"
            cells.append(text)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_text(out_path, "\n".join(lines))
    return sub.drop(columns=["_order"])


def best_proposed_vs_baseline_table(
    micro: pd.DataFrame,
    refs: pd.DataFrame,
    boot_delta: pd.DataFrame,
    excluded: set[str],
) -> pd.DataFrame:
    scenarios = [s for s in MAIN_GRID_ORDER if s in set(micro["scenario"]) and s not in excluded]
    rows = []
    for scenario in scenarios:
        scen = micro.loc[micro["scenario"].eq(scenario)].copy()
        proposed = scen.loc[scen["model"].map(is_proposed_model)].sort_values("LW-MAE").head(1)
        if proposed.empty:
            continue
        prop = proposed.iloc[0]
        ref = refs.loc[
            refs["scenario"].eq(scenario)
            & refs["reference_role"].eq("strongest_standalone_by_LW-MAE")
        ]
        if ref.empty:
            # Fall back to the best of ETS / seasonal / SARIMA if the saved reference is unavailable.
            standalone_candidates = scen.loc[scen["model"].isin(["Seasonal daylag", "ETS local", "SARIMA local"])]
            if standalone_candidates.empty:
                continue
            baseline_row = standalone_candidates.sort_values("LW-MAE").iloc[0]
            baseline = str(baseline_row["model"])
            baseline_lw = float(baseline_row["LW-MAE"])
        else:
            baseline = str(ref.iloc[0]["model"])
            baseline_lw = float(ref.iloc[0]["LW-MAE"])
        ci_row = boot_delta.loc[
            boot_delta["scenario"].eq(scenario)
            & boot_delta["mode"].eq("micro")
            & boot_delta["metric"].eq("LW-MAE")
            & boot_delta["model"].eq(str(prop["model"]))
            & boot_delta["baseline"].eq(baseline)
        ]
        if ci_row.empty:
            delta, low, high, sig = baseline_lw - float(prop["LW-MAE"]), np.nan, np.nan, False
        else:
            r = ci_row.iloc[0]
            delta, low, high, sig = float(r["delta_estimate"]), float(r["ci_low"]), float(r["ci_high"]), bool(r["significant_95"])
        rows.append(
            {
                "scenario": scenario,
                "horizon_minutes": int(prop["horizon_minutes"]),
                "bucket_minutes": int(prop["bucket_minutes"]),
                "proposed_model": str(prop["model"]),
                "proposed_lw_mae": float(prop["LW-MAE"]),
                "baseline_model": baseline,
                "baseline_lw_mae": baseline_lw,
                "delta_estimate": delta,
                "ci_low": low,
                "ci_high": high,
                "significant_95": sig,
            }
        )
    return pd.DataFrame(rows)


def latex_table_grid_summary(summary: pd.DataFrame, out_path: Path) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Устойчивость предложенного семейства моделей по горизонтам и размерам бакета. Положительная дельта означает снижение micro LW-MAE относительно сильнейшей standalone-модели.}",
        r"\label{tab:grid_summary_with_ci}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{rr>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}Xrrrrc}",
        r"\toprule",
        r"$H$ & $B$ & Лучшая proposed-модель & Сильнейший standalone & Наш LW-MAE & Baseline & $\Delta$ & 95\% CI & Sig. \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            " & ".join(
                [
                    str(int(row["horizon_minutes"])),
                    str(int(row["bucket_minutes"])),
                    model_tex(str(row["proposed_model"])),
                    model_tex(str(row["baseline_model"])),
                    fmt_metric(float(row["proposed_lw_mae"]), "LW-MAE"),
                    fmt_metric(float(row["baseline_lw_mae"]), "LW-MAE"),
                    fmt_delta(float(row["delta_estimate"])),
                    fmt_ci(float(row["ci_low"]), float(row["ci_high"])),
                    "да" if bool(row["significant_95"]) else "нет",
                ]
            )
            + r" \\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\normalsize", r"\end{table}"]
    write_text(out_path, "\n".join(lines))


def select_example_station(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred is not None:
        if preferred not in set(df["station_key"]):
            raise ValueError(f"example station {preferred!r} not found")
        return preferred
    agg = df.groupby("station_key").agg(
        total_y=("y_true", "sum"),
        highload_points=("hl_actual", "sum"),
        q80=("q80_station", "median"),
    )
    eligible = agg.loc[agg["highload_points"].ge(2)].copy()
    if eligible.empty:
        eligible = agg.copy()
    # Prefer a visibly busy station but avoid relying on a single absolute maximum only.
    threshold = eligible["total_y"].quantile(0.90)
    top = eligible.loc[eligible["total_y"].ge(threshold)].sort_values(["highload_points", "total_y"], ascending=False)
    if top.empty:
        top = eligible.sort_values("total_y", ascending=False)
    return str(top.index[0])


def plot_flow_with_highload(df: pd.DataFrame, station: str, out_path: Path) -> None:
    time_col = infer_time_column(df)
    g = df.loc[df["station_key"].eq(station)].sort_values(time_col).copy()
    if g.empty:
        return
    x = g[time_col]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.0, 1.6]})
    ax = axes[0]
    xs = np.linspace(0.0, 4.0, 200)
    ax.plot(xs, xs, linewidth=2)
    ax.set_xlabel(r"$y / \bar y_{\mathrm{train, station}}$")
    ax.set_ylabel("Вес в LW-MAE")
    ax.set_title("Весовая функция при $\\gamma=1$")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(x, g["y_true"], label="Факт", linewidth=2)
    hl = g["hl_actual"].fillna(0).astype(int).to_numpy().astype(bool)
    if hl.any():
        ymin, ymax = ax.get_ylim()
        for t in x[hl]:
            ax.axvspan(t, t + pd.Timedelta(minutes=15), alpha=0.15)
        ax.set_ylim(ymin, ymax)
    station_name = str(g["ST_NAME_SHORT"].iloc[0]) if "ST_NAME_SHORT" in g.columns else station
    ax.set_title(f"High-load интервалы на test day: {station_name}")
    ax.set_ylabel("Target, пассажиры")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    save_figure(fig, out_path)


def plot_forecast_overlay(df: pd.DataFrame, station: str, out_path: Path) -> None:
    time_col = infer_time_column(df)
    g = df.loc[df["station_key"].eq(station)].sort_values(time_col).copy()
    model_cols = ["y_true", "ETS local", "Seasonal daylag", "ET-LGBM-PSO-lite", "Residual XGB ETS LW"]
    missing = [c for c in model_cols if c not in g.columns]
    if missing:
        raise KeyError(f"Missing forecast overlay columns: {missing}")
    fig, ax = plt.subplots(figsize=(12.5, 4.5))
    labels = {
        "y_true": "Факт",
        "ETS local": "ETS local",
        "Seasonal daylag": "Seasonal daylag",
        "ET-LGBM-PSO-lite": "ET-LGBM-PSO-lite",
        "Residual XGB ETS LW": "Residual XGB ETS LW",
    }
    widths = {"y_true": 2.5, "Residual XGB ETS LW": 2.1}
    for col in model_cols:
        ax.plot(g[time_col], g[col], label=labels[col], linewidth=widths.get(col, 1.6), alpha=0.95)
    hl = g["hl_actual"].fillna(0).astype(int).to_numpy().astype(bool)
    if hl.any():
        ymin, ymax = ax.get_ylim()
        for t in g.loc[hl, time_col]:
            ax.axvspan(t, t + pd.Timedelta(minutes=15), alpha=0.10)
        ax.set_ylim(ymin, ymax)
    station_name = str(g["ST_NAME_SHORT"].iloc[0]) if "ST_NAME_SHORT" in g.columns else station
    ax.set_title(f"Центральная постановка: test-day траектории для станции {station_name}")
    ax.set_ylabel("Target, пассажиры")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    save_figure(fig, out_path)


def plot_central_ci(delta: pd.DataFrame, out_path: Path) -> None:
    sub = delta.loc[delta["metric"].eq("LW-MAE")].copy()
    sub = sub.iloc[::-1].reset_index(drop=True)
    labels = sub["baseline"].tolist()
    vals = sub["delta_estimate"].to_numpy(dtype=float)
    low = sub["ci_low"].to_numpy(dtype=float)
    high = sub["ci_high"].to_numpy(dtype=float)
    err = np.vstack([vals - low, high - vals])
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    y = np.arange(len(sub))
    ax.barh(y, vals, alpha=0.85)
    ax.errorbar(vals, y, xerr=err, fmt="none", capsize=4, linewidth=1.4)
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Улучшение по LW-MAE: baseline $-$ Residual XGB ETS LW")
    ax.set_title("95%-доверительные интервалы центрального результата")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, out_path)


def plot_grid_heatmap(summary: pd.DataFrame, out_path: Path) -> None:
    pivot = summary.pivot(index="horizon_minutes", columns="bucket_minutes", values="delta_estimate")
    horizons = sorted(pivot.index.tolist())
    buckets = sorted(pivot.columns.tolist())
    data = pivot.reindex(index=horizons, columns=buckets).to_numpy(dtype=float)
    sig_map = summary.pivot(index="horizon_minutes", columns="bucket_minutes", values="significant_95").reindex(index=horizons, columns=buckets)
    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Улучшение LW-MAE против strongest standalone")
    ax.set_xticks(np.arange(len(buckets)))
    ax.set_xticklabels([str(b) for b in buckets])
    ax.set_yticks(np.arange(len(horizons)))
    ax.set_yticklabels([str(h) for h in horizons])
    ax.set_xlabel("Размер бакета, мин")
    ax.set_ylabel("Горизонт, мин")
    ax.set_title("Robustness sweep: выигрыш лучшей proposed-модели")
    for i, h in enumerate(horizons):
        for j, b in enumerate(buckets):
            val = data[i, j]
            if not np.isfinite(val):
                text = "—"
            else:
                sig = bool(sig_map.loc[h, b]) if pd.notna(sig_map.loc[h, b]) else False
                text = f"{val:+.2f}" + ("*" if sig else "")
            ax.text(j, i, text, ha="center", va="center", fontsize=9)
    save_figure(fig, out_path)


def station_level_lw_delta(df: pd.DataFrame, model: str, baseline: str) -> pd.DataFrame:
    m = station_metric_components(df, model)
    b = station_metric_components(df, baseline)
    merged = m[["station_key", "n", "lw_sum"]].merge(
        b[["station_key", "lw_sum"]], on="station_key", suffixes=("_model", "_baseline"), validate="one_to_one"
    )
    merged["model_lw"] = merged["lw_sum_model"] / merged["n"].clip(lower=1)
    merged["baseline_lw"] = merged["lw_sum_baseline"] / merged["n"].clip(lower=1)
    merged["delta"] = merged["baseline_lw"] - merged["model_lw"]
    return merged[["station_key", "model_lw", "baseline_lw", "delta"]]


def plot_station_delta_and_highload(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    delta = station_level_lw_delta(df, "Residual XGB ETS LW", "ETS local")
    hl = df.loc[df["hl_actual"].fillna(0).astype(int).eq(1), ["y_true", "ETS local", "Residual XGB ETS LW"]].copy()
    hl["ETS abs error"] = (hl["y_true"] - hl["ETS local"]).abs()
    hl["Residual abs error"] = (hl["y_true"] - hl["Residual XGB ETS LW"]).abs()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax = axes[0]
    ax.hist(delta["delta"].to_numpy(dtype=float), bins=28)
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    improved_share = float(np.mean(delta["delta"].gt(0))) * 100.0
    ax.set_title(f"Station-level выигрыш по LW-MAE\nдоля станций с улучшением: {improved_share:.1f}%")
    ax.set_xlabel("ETS LW-MAE $-$ proposed LW-MAE")
    ax.set_ylabel("Число станций")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    plot_data = [
        hl["ETS abs error"].dropna().to_numpy(dtype=float),
        hl["Residual abs error"].dropna().to_numpy(dtype=float),
    ]
    ax.boxplot(plot_data, tick_labels=["ETS", "Residual\nmeta"], showfliers=False)
    ax.set_ylabel("Абсолютная ошибка на high-load интервалах")
    ax.set_title("High-load ошибки")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, out_path)
    return delta


def make_box(ax: plt.Axes, xy: tuple[float, float], w: float, h: float, text: str, fontsize: float = 10) -> None:
    x, y = xy
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", linewidth=1.2, edgecolor="black", facecolor="white")
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def make_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arr = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.2)
    ax.add_patch(arr)


def plot_pipeline_overview(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 3.5))
    ax.set_axis_off()
    boxes = [
        (0.02, 0.30, 0.18, 0.42, "Сырые события\nвхода"),
        (0.24, 0.30, 0.18, 0.42, "Station-time\nпанель"),
        (0.46, 0.30, 0.18, 0.42, "Baseline и\narticle models"),
        (0.68, 0.30, 0.18, 0.42, "Residual\nmeta-correction"),
        (0.88, 0.30, 0.10, 0.42, "Оценка\n+ CI"),
    ]
    for x, y, w, h, text in boxes:
        make_box(ax, (x, y), w, h, text, fontsize=9.5)
    for (x, y, w, h, _), (nx, ny, nw, nh, _) in zip(boxes[:-1], boxes[1:]):
        make_arrow(ax, (x + w, y + h / 2), (nx, ny + nh / 2))
    ax.text(0.02, 0.12, "Фиксируются: operational day 03:00–02:59, leakage-free split, load-sensitive metrics", fontsize=9)
    save_figure(fig, out_path)


def plot_data_aggregation(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 3.6))
    ax.set_axis_off()
    boxes = [
        (0.02, 0.32, 0.20, 0.38, "50 млн событий\nс timestamp и place id"),
        (0.27, 0.32, 0.18, 0.38, "Справочник\nмест прохода"),
        (0.50, 0.32, 0.18, 0.38, "Агрегация\nвходных групп"),
        (0.73, 0.32, 0.20, 0.38, "Station-level\nвременные ряды"),
    ]
    for x, y, w, h, text in boxes:
        make_box(ax, (x, y), w, h, text, fontsize=10)
    for (x, y, w, h, _), (nx, ny, nw, nh, _) in zip(boxes[:-1], boxes[1:]):
        make_arrow(ax, (x + w, y + h / 2), (nx, ny + nh / 2))
    ax.text(0.02, 0.13, "Исключаются МЦК, монорельс, речные маршруты и сервисные объекты; в финальной панели 260 станций.", fontsize=9)
    save_figure(fig, out_path)


def plot_residual_architecture(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 4.2))
    ax.set_axis_off()
    make_box(ax, (0.03, 0.60), 0.22, 0.22, "ETS anchor\n$\\hat{Y}^{ETS}$", fontsize=11)
    make_box(ax, (0.03, 0.18), 0.22, 0.22, "Гетерогенные\nbase forecasts", fontsize=10)
    make_box(ax, (0.32, 0.18), 0.22, 0.22, "Context +\ndisagreement", fontsize=10)
    make_box(ax, (0.60, 0.35), 0.20, 0.30, "Residual XGB\n$r_\\theta(Z)$", fontsize=11)
    make_box(ax, (0.87, 0.35), 0.10, 0.30, "Финальный\nпрогноз", fontsize=10)
    make_arrow(ax, (0.25, 0.71), (0.60, 0.50))
    make_arrow(ax, (0.25, 0.29), (0.60, 0.45))
    make_arrow(ax, (0.54, 0.29), (0.60, 0.45))
    make_arrow(ax, (0.80, 0.50), (0.87, 0.50))
    ax.text(0.60, 0.15, r"$\hat{Y}=\hat{Y}^{ETS}+r_\theta(Z)$", fontsize=12)
    save_figure(fig, out_path)


def plot_operational_day(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 2.7))
    ax.set_axis_off()
    x0, x1 = 0.08, 0.92
    y = 0.50
    ax.plot([x0, x1], [y, y], linewidth=2)
    ticks = [(0.08, "03:00\nstart"), (0.36, "12:00"), (0.64, "21:00"), (0.92, "02:59\nend")]
    for x, label in ticks:
        ax.plot([x, x], [y - 0.06, y + 0.06], linewidth=1.4)
        ax.text(x, y - 0.14, label, ha="center", va="top", fontsize=10)
    ax.add_patch(Rectangle((x0, y + 0.08), x1 - x0, 0.10, alpha=0.18))
    ax.text((x0 + x1) / 2, y + 0.13, "Операционный день метрополитена", ha="center", va="center", fontsize=11)
    ax.set_title("Разбиение суток: 03:00–02:59 вместо календарной полуночи")
    save_figure(fig, out_path)


def plot_station_profiles(df: pd.DataFrame, out_path: Path) -> None:
    time_col = infer_time_column(df)
    agg = df.groupby("station_key").agg(total_y=("y_true", "sum")).sort_values("total_y", ascending=False)
    chosen = agg.head(3).index.tolist()
    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    for station in chosen:
        g = df.loc[df["station_key"].eq(station)].sort_values(time_col)
        label = str(g["ST_NAME_SHORT"].iloc[0]) if "ST_NAME_SHORT" in g.columns else station
        ax.plot(g[time_col], g["y_true"], label=label, linewidth=1.8)
    ax.set_title("Test-day profiles для трёх наиболее нагруженных станций центральной постановки")
    ax.set_ylabel("Target, пассажиры")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    save_figure(fig, out_path)


def write_snippet_index(paths: Paths) -> None:
    snippet = r"""% Generated by scripts/build_prediction_artifacts.py
% Suggested includes for the next prediction editing pass.
% Figures:
% \includegraphics[width=\textwidth]{figures/fig_pipeline_overview.pdf}
% \includegraphics[width=\textwidth]{figures/fig_data_aggregation.pdf}
% \includegraphics[width=\textwidth]{figures/fig_residual_architecture.pdf}
% \includegraphics[width=\textwidth]{figures/fig_lw_weight_highload.pdf}
% \includegraphics[width=\textwidth]{figures/fig_forecast_overlay_central.pdf}
% \includegraphics[width=0.86\textwidth]{figures/fig_central_delta_ci_lwmae.pdf}
% \includegraphics[width=0.86\textwidth]{figures/fig_grid_heatmap_lwmae_gain.pdf}
% \includegraphics[width=\textwidth]{figures/fig_station_delta_distribution_central.pdf}
%
% Tables:
% \input{tables/table_central_results.tex}
% \input{tables/table_central_significance_extended.tex}
% \input{tables/table_grid_summary_with_ci.tex}
% \input{tables/table_article_comparators_central.tex}
% \input{tables/table_short_article_case.tex}
"""
    write_text(paths.snippets / "overleaf_include_suggestions.tex", snippet)


def copy_to_overleaf(paths: Paths, overleaf_dir: Path) -> None:
    overleaf_dir = overleaf_dir.expanduser().resolve()
    if not overleaf_dir.exists():
        raise FileNotFoundError(overleaf_dir)
    target_figures = overleaf_dir / "figures"
    target_tables = overleaf_dir / "tables"
    target_figures.mkdir(parents=True, exist_ok=True)
    target_tables.mkdir(parents=True, exist_ok=True)
    for src in paths.figures.glob("*.pdf"):
        shutil.copy2(src, target_figures / src.name)
    for src in paths.tables.glob("*.tex"):
        shutil.copy2(src, target_tables / src.name)
    shutil.copy2(paths.snippets / "overleaf_include_suggestions.tex", overleaf_dir / "overleaf_include_suggestions.tex")


def build_manifest(
    args: argparse.Namespace,
    megagrid_root: Path,
    paths: Paths,
    central_station: str,
    central_significance: pd.DataFrame,
    grid_summary: pd.DataFrame,
) -> None:
    manifest = {
        "megagrid_root": str(megagrid_root),
        "out_dir": str(paths.out),
        "central_scenario": args.central_scenario,
        "short_article_scenario": args.short_article_scenario,
        "excluded_scenarios": sorted(set(DEFAULT_EXCLUDED_SCENARIOS) | set(args.exclude_scenario)),
        "central_example_station": central_station,
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "bootstrap_seed": int(args.bootstrap_seed),
        "figures": sorted(p.name for p in paths.figures.glob("*.pdf")),
        "tables": sorted(p.name for p in paths.tables.glob("*.tex")),
        "csv": sorted(p.name for p in paths.csv.glob("*.csv")),
        "central_significance_rows": int(len(central_significance)),
        "grid_summary_rows": int(len(grid_summary)),
    }
    write_text(paths.root / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    megagrid_root = resolve_megagrid_root(args.megagrid_root)
    excluded = set(DEFAULT_EXCLUDED_SCENARIOS) | set(args.exclude_scenario)
    paths = ensure_paths(args.out_dir.expanduser().resolve())
    frames = load_grid_frames(megagrid_root)
    micro = frames["micro"]

    central_df = scenario_predictions(megagrid_root, args.central_scenario)
    short_df = scenario_predictions(megagrid_root, args.short_article_scenario)
    central_station = select_example_station(central_df, args.example_station)

    # Prediction-specific bootstrap comparisons for the central model against interpretable comparators.
    central_significance = bootstrap_delta_table(
        central_df,
        model="Residual XGB ETS LW",
        baselines=[b for b in CENTRAL_SIGNIFICANCE_BASELINES if b in central_df.columns],
        metrics=CENTRAL_SIGNIFICANCE_METRICS,
        iterations=int(args.bootstrap_iterations),
        seed=int(args.bootstrap_seed),
    )
    central_significance.to_csv(paths.csv / "central_significance_custom.csv", index=False)

    grid_summary = best_proposed_vs_baseline_table(
        micro=frames["micro"],
        refs=frames["bootstrap_refs"],
        boot_delta=frames["bootstrap_delta"],
        excluded=excluded,
    )
    grid_summary.to_csv(paths.csv / "grid_best_proposed_vs_best_standalone.csv", index=False)

    central_results = latex_table_central_results(
        micro, args.central_scenario, paths.tables / "table_central_results.tex"
    )
    central_results.to_csv(paths.csv / "central_results_selected_models.csv", index=False)
    latex_table_significance(central_significance, paths.tables / "table_central_significance_extended.tex")
    latex_table_grid_summary(grid_summary, paths.tables / "table_grid_summary_with_ci.tex")
    article_central = latex_table_article_central(
        micro, args.central_scenario, paths.tables / "table_article_comparators_central.tex"
    )
    article_central.to_csv(paths.csv / "article_comparators_central.csv", index=False)
    short_article = latex_table_short_article_case(
        micro, args.short_article_scenario, paths.tables / "table_short_article_case.tex"
    )
    short_article.to_csv(paths.csv / "short_article_case.csv", index=False)

    # Figures matching the placeholder list in the prediction draft.
    plot_pipeline_overview(paths.figures / "fig_pipeline_overview")
    plot_data_aggregation(paths.figures / "fig_data_aggregation")
    plot_residual_architecture(paths.figures / "fig_residual_architecture")
    plot_flow_with_highload(central_df, central_station, paths.figures / "fig_lw_weight_highload")
    plot_forecast_overlay(central_df, central_station, paths.figures / "fig_forecast_overlay_central")
    plot_central_ci(central_significance, paths.figures / "fig_central_delta_ci_lwmae")
    plot_grid_heatmap(grid_summary, paths.figures / "fig_grid_heatmap_lwmae_gain")
    station_delta = plot_station_delta_and_highload(central_df, paths.figures / "fig_station_delta_distribution_central")
    station_delta.to_csv(paths.csv / "station_delta_distribution_central.csv", index=False)

    # Optional appendix-ready illustrations.
    plot_operational_day(paths.figures / "fig_operational_day")
    plot_station_profiles(central_df, paths.figures / "fig_station_profiles")

    write_snippet_index(paths)
    build_manifest(args, megagrid_root, paths, central_station, central_significance, grid_summary)

    if args.copy_to_overleaf is not None:
        copy_to_overleaf(paths, args.copy_to_overleaf)

    print(f"Megagrid root: {megagrid_root}")
    print(f"Artifacts written to: {paths.root}")
    print(f"Central example station: {central_station}")
    print(f"Figures: {len(list(paths.figures.glob('*.pdf')))} PDF files")
    print(f"Tables: {len(list(paths.tables.glob('*.tex')))} LaTeX files")
    print("Main generated CSV summaries:")
    for name in [
        "central_significance_custom.csv",
        "grid_best_proposed_vs_best_standalone.csv",
        "central_results_selected_models.csv",
        "article_comparators_central.csv",
        "short_article_case.csv",
        "station_delta_distribution_central.csv",
    ]:
        print(f"  - {paths.csv / name}")


if __name__ == "__main__":
    main()
