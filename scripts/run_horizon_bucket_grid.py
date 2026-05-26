#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from metroflow.studies.metro_short_history_forecasting.entrypoint import (  # noqa: E402
    DEFAULT_CLASSIC_METRO_LINE_CODES,
    MultiStationForecastConfig,
    run_multistation_forecasting_experiment,
    save_multistation_forecasting_result,
)


# Target matrix requested for the robustness sweep.
# (15 min horizon, 10 min buckets) is intentionally retained in the list,
# but skipped below: the current target definition sums whole future buckets,
# and 15 is not divisible by 10.
GRID: tuple[tuple[int, int], ...] = (
    (15, 5),
    (15, 10),   # invalid under the current whole-bucket target definition
    (15, 15),
    (60, 5),
    (60, 15),
    (60, 30),
    (60, 60),
    (360, 15),
    (360, 30),
    (360, 60),
)

PRIMARY_METRICS: tuple[str, ...] = (
    "MAE",
    "RMSE",
    "wMAPE",
    "LW-MAE",
    "HL-MAE",
    "HL-underMAE",
    "HL-asymMAE",
)


def _scenario_slug(horizon_minutes: int, bucket_minutes: int) -> str:
    return f"h{int(horizon_minutes):03d}m_b{int(bucket_minutes):03d}m"


def _scenario_label(horizon_minutes: int, bucket_minutes: int) -> str:
    return f"{int(horizon_minutes)}m horizon / {int(bucket_minutes)}m buckets"


def _valid_pair(horizon_minutes: int, bucket_minutes: int) -> tuple[bool, str]:
    if horizon_minutes % bucket_minutes != 0:
        return (
            False,
            (
                "skipped: horizon_minutes must be divisible by bucket_minutes "
                "for the current whole-bucket target_h construction"
            ),
        )
    return True, "ok"


def _common_cfg(
    *,
    horizon_minutes: int,
    bucket_minutes: int,
    profile: str,
    grid_output_root: Path,
    use_weather: bool,
    n_bootstrap: int,
    run_bootstrap: bool,
) -> MultiStationForecastConfig:
    slug = _scenario_slug(horizon_minutes, bucket_minutes)
    cfg = MultiStationForecastConfig(
        name=f"multistation_grid_{slug}",
        data_dir=str(PROJECT_ROOT / "data"),
        pass_filename="PASS_ALL_202503242210.csv",
        cache_dir=str(PROJECT_ROOT / "data" / "cache"),
        output_dir=str(grid_output_root / slug),
        bucket_minutes=int(bucket_minutes),
        horizon_minutes=int(horizon_minutes),
        operational_day_start_hour=3,
        classic_metro_only=True,
        classic_metro_line_codes=DEFAULT_CLASSIC_METRO_LINE_CODES,
        line_filter=None,
        use_weather_features=bool(use_weather),
        weather_csv=str(PROJECT_ROOT / "data" / "weather_hourly_openmeteo_20250310_20250316.csv"),
        n_bootstrap=int(n_bootstrap),
        run_bootstrap=bool(run_bootstrap),
    )
    return _apply_profile(cfg, profile)


def _apply_profile(cfg: MultiStationForecastConfig, profile: str) -> MultiStationForecastConfig:
    """Choose how much of the current model zoo to run for each grid point."""
    profile = profile.strip().lower()
    if profile == "full":
        # Run the current pipeline essentially as configured in forecast defaults,
        # only preserving the caller's bootstrap choice.
        return cfg

    if profile == "robustness":
        # Default profile for the horizon/bucket sweep:
        # keep the core baselines, the main boosting family, ET-LGBM-PSO-lite,
        # ETS residual boosters, and the final residual meta-model;
        # disable expensive/diagnostic side branches already studied at 15m/60m.
        return replace(
            cfg,
            run_local_sarima=False,
            run_catboost=False,
            run_global_lstm=False,
            run_global_transformer=False,
            run_psr_cnn_lstm=False,
            run_cnn_bilstm_attention=False,
            run_msd_informer_bilstm_lite=False,
            run_bilstm_ctpa_lite=False,
            run_line_megaboost=False,
            run_station_bias_correction=False,
            run_km_xgboost_lite=False,
            run_et_lgbm_pso_lite=True,
            run_tp_boost_lite=False,
            run_ml_fusion_svr_lite=False,
            run_vmd_lightgbm_lite=False,
            run_st_lightgbm_mic_lite=False,
            run_sn_lssvm_lite=False,
            run_twostage_fs_msd_hybrid_lite=False,
        )

    if profile == "classical":
        # Fast sanity profile: seasonal / ETS / SARIMA plus the final meta/residual
        # machinery that naturally degenerates to the available strong base models.
        return replace(
            cfg,
            run_local_sarima=True,
            run_catboost=False,
            run_global_lstm=False,
            run_global_transformer=False,
            run_psr_cnn_lstm=False,
            run_cnn_bilstm_attention=False,
            run_msd_informer_bilstm_lite=False,
            run_bilstm_ctpa_lite=False,
            run_line_megaboost=False,
            run_station_bias_correction=False,
            run_km_xgboost_lite=False,
            run_et_lgbm_pso_lite=False,
            run_tp_boost_lite=False,
            run_ml_fusion_svr_lite=False,
            run_vmd_lightgbm_lite=False,
            run_st_lightgbm_mic_lite=False,
            run_sn_lssvm_lite=False,
            run_twostage_fs_msd_hybrid_lite=False,
        )

    raise ValueError(f"Unknown profile: {profile!r}; expected one of: robustness, classical, full")


def _read_saved_metrics(output_dir: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    micro_path = output_dir / "micro_metrics.csv"
    macro_path = output_dir / "macro_metrics.csv"
    timings_path = output_dir / "stage_timings.csv"
    micro = pd.read_csv(micro_path, index_col=0) if micro_path.exists() else None
    macro = pd.read_csv(macro_path, index_col=0) if macro_path.exists() else None
    timings = pd.read_csv(timings_path) if timings_path.exists() else None
    return micro, macro, timings


def _metrics_to_long(
    table: pd.DataFrame | None,
    *,
    mode: str,
    horizon_minutes: int,
    bucket_minutes: int,
    scenario: str,
) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame()
    out = table.reset_index().rename(columns={table.index.name or "index": "model"})
    if "model" not in out.columns:
        out = out.rename(columns={out.columns[0]: "model"})
    out.insert(0, "scenario", scenario)
    out.insert(1, "horizon_minutes", int(horizon_minutes))
    out.insert(2, "bucket_minutes", int(bucket_minutes))
    out.insert(3, "mode", mode)
    return out


def _top_models_for_metrics(
    micro: pd.DataFrame | None,
    macro: pd.DataFrame | None,
    *,
    horizon_minutes: int,
    bucket_minutes: int,
    scenario: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mode, table in (("micro", micro), ("macro", macro)):
        if table is None or table.empty:
            continue
        for metric in PRIMARY_METRICS:
            if metric not in table.columns:
                continue
            vals = pd.to_numeric(table[metric], errors="coerce").dropna()
            if vals.empty:
                continue
            best_model = str(vals.idxmin())
            rows.append(
                {
                    "scenario": scenario,
                    "horizon_minutes": int(horizon_minutes),
                    "bucket_minutes": int(bucket_minutes),
                    "mode": mode,
                    "metric": metric,
                    "best_model": best_model,
                    "best_value": float(vals.loc[best_model]),
                }
            )
    return pd.DataFrame(rows)


def _timings_to_long(
    timings: pd.DataFrame | None,
    *,
    horizon_minutes: int,
    bucket_minutes: int,
    scenario: str,
) -> pd.DataFrame:
    if timings is None or timings.empty:
        return pd.DataFrame()
    out = timings.copy()
    out.insert(0, "scenario", scenario)
    out.insert(1, "horizon_minutes", int(horizon_minutes))
    out.insert(2, "bucket_minutes", int(bucket_minutes))
    return out


def _write_grid_summaries(
    grid_output_root: Path,
    run_rows: list[dict[str, object]],
    micro_rows: list[pd.DataFrame],
    macro_rows: list[pd.DataFrame],
    top_rows: list[pd.DataFrame],
    timing_rows: list[pd.DataFrame],
) -> None:
    grid_output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows).to_csv(grid_output_root / "grid_runs.csv", index=False)

    micro = pd.concat([x for x in micro_rows if x is not None and not x.empty], ignore_index=True) if micro_rows else pd.DataFrame()
    macro = pd.concat([x for x in macro_rows if x is not None and not x.empty], ignore_index=True) if macro_rows else pd.DataFrame()
    top = pd.concat([x for x in top_rows if x is not None and not x.empty], ignore_index=True) if top_rows else pd.DataFrame()
    timings = pd.concat([x for x in timing_rows if x is not None and not x.empty], ignore_index=True) if timing_rows else pd.DataFrame()

    micro.to_csv(grid_output_root / "grid_micro_metrics.csv", index=False)
    macro.to_csv(grid_output_root / "grid_macro_metrics.csv", index=False)
    top.to_csv(grid_output_root / "grid_top_models.csv", index=False)
    timings.to_csv(grid_output_root / "grid_stage_timings.csv", index=False)

    summary_payload = {
        "n_grid_rows": int(len(run_rows)),
        "n_success_or_cached": int(sum(row.get("status") in {"ok", "cached"} for row in run_rows)),
        "n_skipped": int(sum(row.get("status") == "skipped" for row in run_rows)),
        "n_failed": int(sum(row.get("status") == "failed" for row in run_rows)),
        "artifacts": {
            "runs": str(grid_output_root / "grid_runs.csv"),
            "micro_metrics": str(grid_output_root / "grid_micro_metrics.csv"),
            "macro_metrics": str(grid_output_root / "grid_macro_metrics.csv"),
            "top_models": str(grid_output_root / "grid_top_models.csv"),
            "stage_timings": str(grid_output_root / "grid_stage_timings.csv"),
        },
    }
    (grid_output_root / "grid_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _selected_grid(args: argparse.Namespace) -> Iterable[tuple[int, int]]:
    if not args.only:
        return GRID
    selected: list[tuple[int, int]] = []
    for token in args.only:
        try:
            h, b = token.split(":", maxsplit=1)
            selected.append((int(h), int(b)))
        except Exception as exc:  # pragma: no cover - CLI validation
            raise ValueError(f"Invalid --only value {token!r}; expected HORIZON:BUCKET, e.g. 60:15") from exc
    return tuple(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the multi-station forecasting pipeline over the horizon/bucket robustness grid."
    )
    parser.add_argument(
        "--profile",
        choices=("robustness", "classical", "full"),
        default="robustness",
        help="How much of the model zoo to run per grid point. Default: robustness.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "multistation_horizon_bucket_grid",
        help="Directory for per-scenario outputs and aggregate grid summaries.",
    )
    parser.add_argument(
        "--force-rebuild-buckets",
        action="store_true",
        help="Rebuild cached bucket parquet files even if they already exist.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Ignore existing micro/macro metric CSVs and recompute scenarios from scratch.",
    )
    parser.add_argument(
        "--with-bootstrap",
        action="store_true",
        help="Enable station bootstrap inside each scenario. Not recommended for exploratory sweeps.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=0,
        help="Bootstrap iterations if --with-bootstrap is set. Default: 0.",
    )
    parser.add_argument(
        "--use-weather",
        action="store_true",
        help="Enable the existing weather feature path for every grid point.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="HORIZON:BUCKET",
        help="Run only selected pairs, e.g. --only 60:15 360:60.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid_output_root = args.output_root.resolve()
    grid_output_root.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    micro_rows: list[pd.DataFrame] = []
    macro_rows: list[pd.DataFrame] = []
    top_rows: list[pd.DataFrame] = []
    timing_rows: list[pd.DataFrame] = []

    print(f"[grid] project_root={PROJECT_ROOT}")
    print(f"[grid] output_root={grid_output_root}")
    print(f"[grid] profile={args.profile}; bootstrap={bool(args.with_bootstrap)}; n_bootstrap={int(args.n_bootstrap)}")

    for horizon_minutes, bucket_minutes in _selected_grid(args):
        scenario = _scenario_slug(horizon_minutes, bucket_minutes)
        label = _scenario_label(horizon_minutes, bucket_minutes)
        valid, reason = _valid_pair(horizon_minutes, bucket_minutes)
        output_dir = grid_output_root / scenario

        if not valid:
            print(f"[grid] SKIP {label}: {reason}")
            run_rows.append(
                {
                    "scenario": scenario,
                    "horizon_minutes": int(horizon_minutes),
                    "bucket_minutes": int(bucket_minutes),
                    "status": "skipped",
                    "reason": reason,
                    "output_dir": str(output_dir),
                }
            )
            _write_grid_summaries(grid_output_root, run_rows, micro_rows, macro_rows, top_rows, timing_rows)
            continue

        if not args.rerun:
            cached_micro, cached_macro, cached_timings = _read_saved_metrics(output_dir)
            if cached_micro is not None and cached_macro is not None:
                print(f"[grid] CACHE {label}: {output_dir}")
                run_rows.append(
                    {
                        "scenario": scenario,
                        "horizon_minutes": int(horizon_minutes),
                        "bucket_minutes": int(bucket_minutes),
                        "status": "cached",
                        "reason": "micro_metrics.csv and macro_metrics.csv already exist",
                        "output_dir": str(output_dir),
                    }
                )
                micro_rows.append(
                    _metrics_to_long(
                        cached_micro,
                        mode="micro",
                        horizon_minutes=horizon_minutes,
                        bucket_minutes=bucket_minutes,
                        scenario=scenario,
                    )
                )
                macro_rows.append(
                    _metrics_to_long(
                        cached_macro,
                        mode="macro",
                        horizon_minutes=horizon_minutes,
                        bucket_minutes=bucket_minutes,
                        scenario=scenario,
                    )
                )
                top_rows.append(
                    _top_models_for_metrics(
                        cached_micro,
                        cached_macro,
                        horizon_minutes=horizon_minutes,
                        bucket_minutes=bucket_minutes,
                        scenario=scenario,
                    )
                )
                timing_rows.append(
                    _timings_to_long(
                        cached_timings,
                        horizon_minutes=horizon_minutes,
                        bucket_minutes=bucket_minutes,
                        scenario=scenario,
                    )
                )
                _write_grid_summaries(grid_output_root, run_rows, micro_rows, macro_rows, top_rows, timing_rows)
                continue

        cfg = _common_cfg(
            horizon_minutes=horizon_minutes,
            bucket_minutes=bucket_minutes,
            profile=args.profile,
            grid_output_root=grid_output_root,
            use_weather=bool(args.use_weather),
            n_bootstrap=int(args.n_bootstrap),
            run_bootstrap=bool(args.with_bootstrap),
        )
        print(f"[grid] RUN  {label}: {cfg.output_dir}")
        try:
            result = run_multistation_forecasting_experiment(
                cfg,
                force_rebuild_buckets=bool(args.force_rebuild_buckets),
            )
            save_multistation_forecasting_result(result)
            micro = result["micro_table"].copy()
            macro = result["macro_table"].copy()
            timings = result.get("timing_table")
            run_rows.append(
                {
                    "scenario": scenario,
                    "horizon_minutes": int(horizon_minutes),
                    "bucket_minutes": int(bucket_minutes),
                    "status": "ok",
                    "reason": "",
                    "output_dir": str(output_dir),
                }
            )
            micro_rows.append(
                _metrics_to_long(
                    micro,
                    mode="micro",
                    horizon_minutes=horizon_minutes,
                    bucket_minutes=bucket_minutes,
                    scenario=scenario,
                )
            )
            macro_rows.append(
                _metrics_to_long(
                    macro,
                    mode="macro",
                    horizon_minutes=horizon_minutes,
                    bucket_minutes=bucket_minutes,
                    scenario=scenario,
                )
            )
            top_rows.append(
                _top_models_for_metrics(
                    micro,
                    macro,
                    horizon_minutes=horizon_minutes,
                    bucket_minutes=bucket_minutes,
                    scenario=scenario,
                )
            )
            timing_rows.append(
                _timings_to_long(
                    timings,
                    horizon_minutes=horizon_minutes,
                    bucket_minutes=bucket_minutes,
                    scenario=scenario,
                )
            )
        except Exception as exc:
            output_dir.mkdir(parents=True, exist_ok=True)
            error_path = output_dir / "grid_run_error.txt"
            error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            error_path.write_text(error_text, encoding="utf-8")
            print(f"[grid] FAIL {label}: {type(exc).__name__}: {exc}")
            print(f"[grid] error traceback saved to {error_path}")
            run_rows.append(
                {
                    "scenario": scenario,
                    "horizon_minutes": int(horizon_minutes),
                    "bucket_minutes": int(bucket_minutes),
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "output_dir": str(output_dir),
                }
            )
        finally:
            _write_grid_summaries(grid_output_root, run_rows, micro_rows, macro_rows, top_rows, timing_rows)

    print(f"[grid] done; summaries written under {grid_output_root}")


if __name__ == "__main__":
    main()
