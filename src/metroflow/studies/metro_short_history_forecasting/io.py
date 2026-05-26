from __future__ import annotations

from .common import *
from .config import (
    DEFAULT_CLASSIC_METRO_LINE_CODES,
    _scope_name,
    MultiStationForecastConfig,
)

def save_multistation_forecasting_result(result: dict, output_dir: str | Path | None = None) -> dict[str, str]:
    cfg: MultiStationForecastConfig = result["config"]
    out = Path(output_dir) if output_dir is not None else Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    result["micro_table"].to_csv(out / "micro_metrics.csv")
    result["macro_table"].to_csv(out / "macro_metrics.csv")
    result["bootstrap"]["metric_ci"].to_csv(out / "station_bootstrap_metric_ci.csv", index=False)
    result["bootstrap"]["delta_ci"].to_csv(out / "station_bootstrap_delta_ci.csv", index=False)
    result["test_pred_table"].to_csv(out / "test_predictions.csv", index=False)
    result["val_pred_table"].to_csv(out / "validation_predictions.csv", index=False)
    result["stations"].to_csv(out / "stations_used.csv", index=False)
    bootstrap_refs = result.get("bootstrap_reference_table")
    if isinstance(bootstrap_refs, pd.DataFrame) and not bootstrap_refs.empty:
        bootstrap_refs.to_csv(out / "bootstrap_reference_models.csv", index=False)
    timing_table = result.get("timing_table")
    diagnostics = [
        ("km_xgboost_lite", "candidate_table", "km_xgboost_candidate_table.csv"),
        ("km_xgboost_lite", "selected_cluster_table", "km_xgboost_selected_clusters.csv"),
        ("et_lgbm_pso_lite", "pso_trace", "et_lgbm_pso_trace.csv"),
        ("tp_boost_lite", "partition_selection_table", "tp_boost_partition_selection.csv"),
        ("ml_fusion_svr_lite", "component_table", "ml_fusion_svr_components.csv"),
        ("vmd_lightgbm_lite", "candidate_table", "vmd_lightgbm_candidates.csv"),
        ("st_lightgbm_mic_lite", "selected_feature_table", "st_lightgbm_mic_features.csv"),
        ("sn_lssvm_lite", "alpha_table", "sn_lssvm_alpha_table.csv"),
        ("twostage_fs_msd_hybrid_lite", "feature_selection_table", "twostage_fs_msd_features.csv"),
        ("twostage_fs_msd_hybrid_lite", "blend_table", "twostage_fs_msd_blend.csv"),
        ("stacking_catboost_lite", "component_validation_metrics", "stacking_catboost_components.csv"),
        ("stationboost_base_norm", "station_selection_table", "stationboost_base_norm_station_selection.csv"),
        ("stationboost_rich_norm", "station_selection_table", "stationboost_rich_norm_station_selection.csv"),
        ("stationboost_rich_norm_lw", "station_selection_table", "stationboost_rich_norm_lw_station_selection.csv"),
        ("stationboost_seasonal_residual_lw", "station_selection_table", "stationboost_seasonal_residual_lw_station_selection.csv"),
    ]
    for result_key, table_key, filename in diagnostics:
        payload = result.get(result_key)
        table = payload.get(table_key) if isinstance(payload, dict) else None
        if isinstance(table, pd.DataFrame) and not table.empty:
            table.to_csv(out / filename, index=False)
            paths[filename.removesuffix(".csv")] = str(out / filename)
    if isinstance(timing_table, pd.DataFrame) and not timing_table.empty:
        timing_table.to_csv(out / "stage_timings.csv", index=False)
        (out / "stage_timings.json").write_text(
            timing_table.to_json(orient="records", force_ascii=False, indent=2),
            encoding="utf-8",
        )
    (out / "config.json").write_text(json.dumps(cfg.__dict__, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    paths.update({
        "micro_metrics": str(out / "micro_metrics.csv"),
        "macro_metrics": str(out / "macro_metrics.csv"),
        "station_bootstrap_metric_ci": str(out / "station_bootstrap_metric_ci.csv"),
        "station_bootstrap_delta_ci": str(out / "station_bootstrap_delta_ci.csv"),
        "test_predictions": str(out / "test_predictions.csv"),
        "validation_predictions": str(out / "validation_predictions.csv"),
        "stations_used": str(out / "stations_used.csv"),
        "bootstrap_reference_models": str(out / "bootstrap_reference_models.csv"),
    })
    if isinstance(timing_table, pd.DataFrame) and not timing_table.empty:
        paths["stage_timings"] = str(out / "stage_timings.csv")
        paths["stage_timings_json"] = str(out / "stage_timings.json")
    return paths



__all__ = [
    "save_multistation_forecasting_result",
]
