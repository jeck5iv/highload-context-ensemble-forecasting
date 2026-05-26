from __future__ import annotations

from .common import *
from .config import (
    DEFAULT_CLASSIC_METRO_LINE_CODES,
    _scope_name,
    MultiStationForecastConfig,
)
from .timer import (
    _StageTimer,
)
from .data_layer.buckets import (
    _pass_source_path,
    _default_station_buckets_path,
    build_multistation_bucket_parquet,
    load_or_build_station_buckets,
    _complete_one_station,
)
from .data_layer.supervised import (
    make_multistation_supervised_frame,
    _clip_per_station_day,
    _station_q80,
    _attach_station_context,
    make_multistation_weekday_split,
)
from .features import (
    BASE_FEATURES,
    RICH_FEATURES,
    _available_features,
    _xgb_matrix,
    _boost_lw_sample_weight,
    _aligned_sample_weight,
)
from .evaluation.metrics import (
    calc_metrics_df,
    _calc_metric_values,
    evaluate_predictions,
    BOOTSTRAP_METRICS,
    PREDICTION_SINGLE_MODEL_CANDIDATES,
    ARTICLE_COMPARATOR_MODELS,
    _choose_bootstrap_reference_models,
    _safe_divide,
    _station_metric_component_arrays,
    station_bootstrap_ci,
)
from .models.boosting import (
    _fit_predict_xgb,
    _run_global_megaboost,
)
from .models.classical import (
    _future_sum_from_count_path,
    _fit_predict_station_ets,
    _run_local_ets,
    _fit_predict_station_sarima,
    _local_sarima_prediction_for_split,
    _run_local_sarima,
)
from .models.sequence import (
    _scaled_station_sequence_parts,
    _sequence_eval_pred_to_frame_order,
    _predict_scaled_sequence_model,
    _run_global_sequence_models,
    _phase_space_reconstruct_windows,
    _causal_moving_average_windows,
    _multiscale_decompose_windows,
    _predict_tensor_sequence_model,
    _train_tensor_sequence_regressor,
    _run_article_sequence_models,
)
from .models.meta import (
    _seasonal_anchor_series,
    seasonal_daylag_predictions,
    _positive_blend,
    _context_blend,
    _build_meta_tables,
    _run_meta_models,
    _meta_feature_matrix,
)
from .io import (
    save_multistation_forecasting_result,
)
from .models.boosting_local import (
    NORMALIZED_MAGNITUDE_PREFIXES,
    NORMALIZED_MAGNITUDE_COLS,
    _station_scale_series,
    _normalize_boost_frame,
    _fit_predict_xgb_normalized,
    _run_global_megaboost_normalized,
    _fit_predict_xgb_residual,
    _run_global_residual_megaboost,
    _run_validation_residual_megaboost,
    _prediction_metric,
    _blend_anchor_prediction,
    _select_correction_alpha,
    _line_code_column,
    _run_line_local_normalized_megaboost,
    _run_line_local_seasonal_residual_megaboost,
    _station_seasonal_fallback,
    _run_station_local_normalized_megaboost,
    _run_station_local_seasonal_residual_megaboost,
    _station_bias_table,
    _apply_station_bias,
    _run_station_bias_seasonal,
)
from .models.article_models import (
    _station_cluster_descriptors,
    _fit_station_kmeans_mapping,
    _with_station_cluster,
    _run_km_xgboost_lite,
    _fit_predict_extra_trees_normalized,
    _fit_predict_lgbm_normalized,
    _blend_metric,
    _optimize_pair_weight_pso,
    _run_et_lgbm_pso_lite,
    _fit_predict_lsboost_normalized,
    _time_partition_id,
    _select_pair_weight_grid,
    _run_tp_boost_lite,
    _fit_predict_adaboost_normalized,
    _run_ml_fusion_svr_lite,
    _augment_multiscale_proxy_features,
    MULTISCALE_PROXY_FEATURES,
    _fit_predict_lgbm_normalized_custom,
    _run_vmd_lightgbm_lite,
    _feature_selection_sample,
    _select_mi_features,
    _run_st_lightgbm_mic_lite,
    _fit_predict_rff_residual_correction,
    _run_sn_lssvm_lite,
    _select_correlation_features,
    _two_stage_feature_selection,
    _run_twostage_fs_msd_hybrid_lite,
    _stacking_catboost_meta_matrix,
    _run_stacking_catboost_lite,
)
from .models.catboost_models import (
    CATBOOST_CATEGORY_COLS,
    _catboost_matrix,
    _fit_predict_catboost,
    _run_global_catboost,
)
from .models.meta_filtered import (
    _meta_lw_sample_weight,
    _positive_blend_weighted,
    _context_blend_weighted,
    _validation_metric_for_model,
    _select_meta_model_pool,
    _fit_residual_family_for_primary,
    _run_meta_models_filtered_weighted,
)

def run_multistation_forecasting_experiment(cfg: MultiStationForecastConfig, force_rebuild_buckets: bool = False) -> dict:
    overall_started = perf_counter()
    timer = _StageTimer(echo=cfg.print_stage_timings)

    with timer.track("load_or_build_station_buckets"):
        buckets, stations, places = load_or_build_station_buckets(cfg, force_rebuild=force_rebuild_buckets)
    with timer.track("make_multistation_supervised_frame"):
        frame = make_multistation_supervised_frame(buckets, cfg)
    with timer.track("make_multistation_weekday_split"):
        split = make_multistation_weekday_split(frame, cfg)

    with timer.track("prepare_feature_lists_and_seasonal_anchor"):
        base_features = _available_features(split["train_tune"], BASE_FEATURES)
        rich_features = _available_features(split["train_tune"], BASE_FEATURES + RICH_FEATURES)

        seasonal = seasonal_daylag_predictions(split["val_df"], split["test_df"])
        seasonal_train_anchor = _seasonal_anchor_series(split["train_tune"]).values.astype(float)
        seasonal_val_anchor = seasonal["val_pred"]
        seasonal_train_full_anchor = _seasonal_anchor_series(split["train_full"]).values.astype(float)
        seasonal_test_anchor = seasonal["test_pred"]

    with timer.track("MegaBoost base"):
        mega_base = _run_global_megaboost(cfg, split["train_tune"], split["val_df"], split["train_full"], split["test_df"], base_features, "MegaBoost base")
    with timer.track("MegaBoost rich"):
        mega_rich = _run_global_megaboost(cfg, split["train_tune"], split["val_df"], split["train_full"], split["test_df"], rich_features, "MegaBoost rich")
    with timer.track("MegaBoost rich LW"):
        mega_rich_lw = _run_global_megaboost(
            cfg,
            split["train_tune"],
            split["val_df"],
            split["train_full"],
            split["test_df"],
            rich_features,
            "MegaBoost rich LW",
            sample_weight_mode="lw",
            selection_metric="LW-MAE",
        )
    with timer.track("MegaBoost base norm"):
        mega_base_norm = _run_global_megaboost_normalized(cfg, split["train_tune"], split["val_df"], split["train_full"], split["test_df"], base_features, "MegaBoost base norm")
    with timer.track("MegaBoost rich norm"):
        mega_rich_norm = _run_global_megaboost_normalized(cfg, split["train_tune"], split["val_df"], split["train_full"], split["test_df"], rich_features, "MegaBoost rich norm")
    with timer.track("MegaBoost rich norm LW"):
        mega_rich_norm_lw = _run_global_megaboost_normalized(
            cfg,
            split["train_tune"],
            split["val_df"],
            split["train_full"],
            split["test_df"],
            rich_features,
            "MegaBoost rich norm LW",
            sample_weight_mode="lw",
            selection_metric="LW-MAE",
        )
    with timer.track("MegaBoost Seasonal residual"):
        mega_seasonal_residual = _run_global_residual_megaboost(
            cfg,
            split["train_tune"],
            split["val_df"],
            split["train_full"],
            split["test_df"],
            rich_features,
            "MegaBoost Seasonal residual",
            anchor_train_pred=seasonal_train_anchor,
            anchor_val_pred=seasonal_val_anchor,
            anchor_train_full_pred=seasonal_train_full_anchor,
            anchor_test_pred=seasonal_test_anchor,
            selection_metric="LW-MAE",
        )
    with timer.track("MegaBoost Seasonal residual LW"):
        mega_seasonal_residual_lw = _run_global_residual_megaboost(
            cfg,
            split["train_tune"],
            split["val_df"],
            split["train_full"],
            split["test_df"],
            rich_features,
            "MegaBoost Seasonal residual LW",
            anchor_train_pred=seasonal_train_anchor,
            anchor_val_pred=seasonal_val_anchor,
            anchor_train_full_pred=seasonal_train_full_anchor,
            anchor_test_pred=seasonal_test_anchor,
            sample_weight_mode="lw",
            selection_metric="LW-MAE",
        )

    line_rich_norm_lw = None
    line_seasonal_residual_lw = None
    if cfg.run_line_megaboost:
        with timer.track("LineBoost rich norm LW"):
            line_rich_norm_lw = _run_line_local_normalized_megaboost(
                cfg,
                split["train_tune"],
                split["val_df"],
                split["train_full"],
                split["test_df"],
                rich_features,
                "LineBoost rich norm LW",
                fallback_val_pred=mega_rich_norm_lw["val_pred"],
                fallback_test_pred=mega_rich_norm_lw["test_pred"],
                sample_weight_mode="lw",
                selection_metric="LW-MAE",
            )
        with timer.track("LineBoost Seasonal residual LW"):
            line_seasonal_residual_lw = _run_line_local_seasonal_residual_megaboost(
                cfg,
                split["train_tune"],
                split["val_df"],
                split["train_full"],
                split["test_df"],
                rich_features,
                "LineBoost Seasonal residual LW",
                anchor_train_pred=seasonal_train_anchor,
                anchor_val_pred=seasonal_val_anchor,
                anchor_train_full_pred=seasonal_train_full_anchor,
                anchor_test_pred=seasonal_test_anchor,
                sample_weight_mode="lw",
                selection_metric="LW-MAE",
            )

    stationboost_base_norm = None
    stationboost_rich_norm = None
    stationboost_rich_norm_lw = None
    stationboost_seasonal_residual_lw = None
    if bool(getattr(cfg, "run_station_megaboost", False)):
        with timer.track("StationBoost base norm"):
            stationboost_base_norm = _run_station_local_normalized_megaboost(
                cfg,
                split["train_tune"],
                split["val_df"],
                split["train_full"],
                split["test_df"],
                base_features,
                "StationBoost base norm",
                selection_metric="LW-MAE",
            )
        with timer.track("StationBoost rich norm"):
            stationboost_rich_norm = _run_station_local_normalized_megaboost(
                cfg,
                split["train_tune"],
                split["val_df"],
                split["train_full"],
                split["test_df"],
                rich_features,
                "StationBoost rich norm",
                selection_metric="LW-MAE",
            )
        with timer.track("StationBoost rich norm LW"):
            stationboost_rich_norm_lw = _run_station_local_normalized_megaboost(
                cfg,
                split["train_tune"],
                split["val_df"],
                split["train_full"],
                split["test_df"],
                rich_features,
                "StationBoost rich norm LW",
                sample_weight_mode="lw",
                selection_metric="LW-MAE",
            )
        with timer.track("StationBoost Seasonal residual LW"):
            stationboost_seasonal_residual_lw = _run_station_local_seasonal_residual_megaboost(
                cfg,
                split["train_tune"],
                split["val_df"],
                split["train_full"],
                split["test_df"],
                rich_features,
                "StationBoost Seasonal residual LW",
                anchor_train_pred=seasonal_train_anchor,
                anchor_val_pred=seasonal_val_anchor,
                anchor_train_full_pred=seasonal_train_full_anchor,
                anchor_test_pred=seasonal_test_anchor,
                sample_weight_mode="lw",
                selection_metric="LW-MAE",
            )

    station_bias_seasonal_lw = None
    if cfg.run_station_bias_correction:
        with timer.track("StationBias Seasonal LW"):
            station_bias_seasonal_lw = _run_station_bias_seasonal(
                cfg,
                split["train_tune"],
                split["val_df"],
                split["train_full"],
                split["test_df"],
                "StationBias Seasonal LW",
                anchor_train_pred=seasonal_train_anchor,
                anchor_val_pred=seasonal_val_anchor,
                anchor_train_full_pred=seasonal_train_full_anchor,
                anchor_test_pred=seasonal_test_anchor,
                selection_metric="LW-MAE",
            )

    catboost_base = None
    catboost_rich = None
    if cfg.run_catboost:
        if CatBoostRegressor is None:
            warnings.warn("CatBoost requested but catboost is not installed; CatBoost models will be skipped")
        else:
            with timer.track("CatBoost base"):
                catboost_base = _run_global_catboost(cfg, split["train_tune"], split["val_df"], split["train_full"], split["test_df"], base_features, "CatBoost base", 7000)
            with timer.track("CatBoost rich"):
                catboost_rich = _run_global_catboost(cfg, split["train_tune"], split["val_df"], split["train_full"], split["test_df"], rich_features, "CatBoost rich", 8000)

    km_xgboost_lite = None
    et_lgbm_pso_lite = None
    tp_boost_lite = None
    if getattr(cfg, "run_km_xgboost_lite", False):
        with timer.track("Km-XGBoost-lite"):
            km_xgboost_lite = _run_km_xgboost_lite(
                cfg,
                split["train_tune"],
                split["val_df"],
                split["train_full"],
                split["test_df"],
                rich_features,
                "Km-XGBoost-lite",
                fallback_val_pred=mega_rich_norm_lw["val_pred"],
                fallback_test_pred=mega_rich_norm_lw["test_pred"],
                selection_metric="LW-MAE",
            )
    if getattr(cfg, "run_et_lgbm_pso_lite", False):
        if LGBMRegressor is None:
            warnings.warn("ET-LGBM-PSO-lite requested but lightgbm is not installed; this comparator will be skipped")
        else:
            with timer.track("ET-LGBM-PSO-lite"):
                et_lgbm_pso_lite = _run_et_lgbm_pso_lite(
                    cfg,
                    split["train_tune"],
                    split["val_df"],
                    split["train_full"],
                    split["test_df"],
                    rich_features,
                    "ET-LGBM-PSO-lite",
                    selection_metric="LW-MAE",
                )
    if getattr(cfg, "run_tp_boost_lite", False):
        with timer.track("TP-Boost-lite"):
            tp_boost_lite = _run_tp_boost_lite(
                cfg,
                split["train_tune"],
                split["val_df"],
                split["train_full"],
                split["test_df"],
                rich_features,
                "TP-Boost-lite",
                fallback_val_pred=mega_rich_norm_lw["val_pred"],
                fallback_test_pred=mega_rich_norm_lw["test_pred"],
                selection_metric="LW-MAE",
            )

    ml_fusion_svr_lite = None
    vmd_lightgbm_lite = None
    st_lightgbm_mic_lite = None
    sn_lssvm_lite = None
    twostage_fs_msd_hybrid_lite = None
    stacking_catboost_lite = None
    if getattr(cfg, "run_ml_fusion_svr_lite", False):
        if LGBMRegressor is None:
            warnings.warn("ML-Fusion-SVR-lite requested but lightgbm is not installed; this comparator will be skipped")
        else:
            with timer.track("ML-Fusion-SVR-lite"):
                ml_fusion_svr_lite = _run_ml_fusion_svr_lite(
                    cfg,
                    split["train_tune"],
                    split["val_df"],
                    split["train_full"],
                    split["test_df"],
                    rich_features,
                    "ML-Fusion-SVR-lite",
                    xgb_val_pred=mega_rich_norm_lw["val_pred"],
                    xgb_test_pred=mega_rich_norm_lw["test_pred"],
                )
    if getattr(cfg, "run_vmd_lightgbm_lite", False):
        if LGBMRegressor is None:
            warnings.warn("VMD-LightGBM-lite requested but lightgbm is not installed; this comparator will be skipped")
        else:
            with timer.track("VMD-LightGBM-lite"):
                vmd_lightgbm_lite = _run_vmd_lightgbm_lite(
                    cfg,
                    split["train_tune"],
                    split["val_df"],
                    split["train_full"],
                    split["test_df"],
                    rich_features,
                    "VMD-LightGBM-lite",
                    selection_metric="MAE",
                )
    if getattr(cfg, "run_st_lightgbm_mic_lite", False):
        if LGBMRegressor is None:
            warnings.warn("ST-LightGBM-MIC-lite requested but lightgbm is not installed; this comparator will be skipped")
        else:
            with timer.track("ST-LightGBM-MIC-lite"):
                st_lightgbm_mic_lite = _run_st_lightgbm_mic_lite(
                    cfg,
                    split["train_tune"],
                    split["val_df"],
                    split["train_full"],
                    split["test_df"],
                    rich_features,
                    "ST-LightGBM-MIC-lite",
                )
    if getattr(cfg, "run_sn_lssvm_lite", False):
        with timer.track("SN-LSSVM-lite"):
            sn_lssvm_lite = _run_sn_lssvm_lite(
                cfg,
                split["train_tune"],
                split["val_df"],
                split["train_full"],
                split["test_df"],
                rich_features,
                "SN-LSSVM-lite",
                anchor_train_pred=seasonal_train_anchor,
                anchor_val_pred=seasonal_val_anchor,
                anchor_train_full_pred=seasonal_train_full_anchor,
                anchor_test_pred=seasonal_test_anchor,
                selection_metric="MAE",
            )
    if getattr(cfg, "run_twostage_fs_msd_hybrid_lite", False):
        if LGBMRegressor is None:
            warnings.warn("TwoStage-FS-MSD-Hybrid-lite requested but lightgbm is not installed; this comparator will be skipped")
        else:
            with timer.track("TwoStage-FS-MSD-Hybrid-lite"):
                twostage_fs_msd_hybrid_lite = _run_twostage_fs_msd_hybrid_lite(
                    cfg,
                    split["train_tune"],
                    split["val_df"],
                    split["train_full"],
                    split["test_df"],
                    rich_features,
                    "TwoStage-FS-MSD-Hybrid-lite",
                    selection_metric="MAE",
                )

    if bool(getattr(cfg, "run_stacking_catboost_lite", False)):
        if CatBoostRegressor is None or LGBMRegressor is None:
            warnings.warn("Stacking-CatBoost-lite requested but catboost or lightgbm is not installed; this comparator will be skipped")
        else:
            with timer.track("Stacking-CatBoost-lite"):
                stacking_catboost_lite = _run_stacking_catboost_lite(
                    cfg,
                    split["train_tune"],
                    split["val_df"],
                    split["train_full"],
                    split["test_df"],
                    rich_features,
                    "Stacking-CatBoost-lite",
                )

    with timer.track("assemble_initial_prediction_pool"):
        preds = {
            "Seasonal daylag": seasonal,
            "MegaBoost base": mega_base,
            "MegaBoost rich": mega_rich,
            "MegaBoost rich LW": mega_rich_lw,
            "MegaBoost base norm": mega_base_norm,
            "MegaBoost rich norm": mega_rich_norm,
            "MegaBoost rich norm LW": mega_rich_norm_lw,
            "MegaBoost Seasonal residual": mega_seasonal_residual,
            "MegaBoost Seasonal residual LW": mega_seasonal_residual_lw,
        }
        if line_rich_norm_lw is not None:
            preds["LineBoost rich norm LW"] = line_rich_norm_lw
        if line_seasonal_residual_lw is not None:
            preds["LineBoost Seasonal residual LW"] = line_seasonal_residual_lw
        if stationboost_base_norm is not None:
            preds["StationBoost base norm"] = stationboost_base_norm
        if stationboost_rich_norm is not None:
            preds["StationBoost rich norm"] = stationboost_rich_norm
        if stationboost_rich_norm_lw is not None:
            preds["StationBoost rich norm LW"] = stationboost_rich_norm_lw
        if stationboost_seasonal_residual_lw is not None:
            preds["StationBoost Seasonal residual LW"] = stationboost_seasonal_residual_lw
        if station_bias_seasonal_lw is not None:
            preds["StationBias Seasonal LW"] = station_bias_seasonal_lw
        if catboost_base is not None:
            preds["CatBoost base"] = catboost_base
        if catboost_rich is not None:
            preds["CatBoost rich"] = catboost_rich
        if km_xgboost_lite is not None:
            preds["Km-XGBoost-lite"] = km_xgboost_lite
        if et_lgbm_pso_lite is not None:
            preds["ET-LGBM-PSO-lite"] = et_lgbm_pso_lite
        if tp_boost_lite is not None:
            preds["TP-Boost-lite"] = tp_boost_lite
        if ml_fusion_svr_lite is not None:
            preds["ML-Fusion-SVR-lite"] = ml_fusion_svr_lite
        if vmd_lightgbm_lite is not None:
            preds["VMD-LightGBM-lite"] = vmd_lightgbm_lite
        if st_lightgbm_mic_lite is not None:
            preds["ST-LightGBM-MIC-lite"] = st_lightgbm_mic_lite
        if sn_lssvm_lite is not None:
            preds["SN-LSSVM-lite"] = sn_lssvm_lite
        if twostage_fs_msd_hybrid_lite is not None:
            preds["TwoStage-FS-MSD-Hybrid-lite"] = twostage_fs_msd_hybrid_lite
        if stacking_catboost_lite is not None:
            preds["Stacking-CatBoost-lite"] = stacking_catboost_lite

    sarima_result = None
    if getattr(cfg, "run_local_sarima", False):
        with timer.track("SARIMA local"):
            sarima_result = _run_local_sarima(
                cfg,
                split["train_tune_all"],
                split["val_all"],
                split["train_full_all"],
                split["test_all"],
                split["val_df"],
                split["test_df"],
            )
            preds["SARIMA local"] = sarima_result

    ets_result = None
    mega_ets_residual = None
    mega_ets_residual_lw = None
    if getattr(cfg, "run_local_ets", False):
        with timer.track("ETS local"):
            ets_result = _run_local_ets(cfg, split["train_tune_all"], split["val_all"], split["train_full_all"], split["test_all"], split["val_df"], split["test_df"])
            preds["ETS local"] = ets_result
        with timer.track("MegaBoost ETS residual"):
            mega_ets_residual = _run_validation_residual_megaboost(
                cfg,
                split["val_df"],
                split["test_df"],
                rich_features,
                "MegaBoost ETS residual",
                anchor_val_pred=ets_result["val_pred"],
                anchor_test_pred=ets_result["test_pred"],
            )
        with timer.track("MegaBoost ETS residual LW"):
            mega_ets_residual_lw = _run_validation_residual_megaboost(
                cfg,
                split["val_df"],
                split["test_df"],
                rich_features,
                "MegaBoost ETS residual LW",
                anchor_val_pred=ets_result["val_pred"],
                anchor_test_pred=ets_result["test_pred"],
                sample_weight_mode="lw",
            )

    seq_results: dict[str, dict] = {}
    if getattr(cfg, "run_global_lstm", False) or getattr(cfg, "run_global_transformer", False):
        with timer.track("Global sequence models"):
            seq_all = _run_global_sequence_models(cfg, split)
        with timer.track("register_global_sequence_predictions"):
            if getattr(cfg, "run_global_lstm", False) and "Global LSTM" in seq_all:
                preds["Global LSTM"] = seq_all["Global LSTM"]
                seq_results["Global LSTM"] = seq_all["Global LSTM"]
            if getattr(cfg, "run_global_transformer", False) and "Global Transformer-lite" in seq_all:
                preds["Global Transformer-lite"] = seq_all["Global Transformer-lite"]
                seq_results["Global Transformer-lite"] = seq_all["Global Transformer-lite"]

    article_sequence_results: dict[str, dict] = {}
    if (
        getattr(cfg, "run_psr_cnn_lstm", False)
        or getattr(cfg, "run_cnn_bilstm_attention", False)
        or getattr(cfg, "run_msd_informer_bilstm_lite", False)
        or getattr(cfg, "run_bilstm_ctpa_lite", False)
    ):
        with timer.track("Article-grounded sequence models"):
            article_sequence_results = _run_article_sequence_models(cfg, split)
        with timer.track("register_article_sequence_predictions"):
            for name, payload in article_sequence_results.items():
                preds[name] = payload

    with timer.track("build_meta_prediction_tables"):
        val_pred_table, test_pred_table = _build_meta_tables(split, preds, cfg)
    with timer.track("fit_meta_models"):
        meta = _run_meta_models_filtered_weighted(cfg, val_pred_table, test_pred_table)
    with timer.track("attach_meta_predictions"):
        test_pred_table["Meta static"] = meta["pred_static"]
        test_pred_table["Meta context"] = meta["pred_context"]
        test_pred_table["Meta static LW"] = meta["pred_static_lw"]
        test_pred_table["Meta context LW"] = meta["pred_context_lw"]
        if cfg.run_residual_xgb:
            test_pred_table["Residual Linear"] = meta["pred_residual_linear"]
            test_pred_table["Residual XGB"] = meta["pred_residual_xgb"]
            test_pred_table["Residual Linear LW"] = meta["pred_residual_linear_lw"]
            test_pred_table["Residual XGB LW"] = meta["pred_residual_xgb_lw"]

            for anchor_label, block in meta.get("fixed_residuals", {}).items():
                test_pred_table[f"Residual Linear {anchor_label}"] = block["pred_residual_linear"]
                test_pred_table[f"Residual XGB {anchor_label}"] = block["pred_residual_xgb"]
                test_pred_table[f"Residual Linear {anchor_label} LW"] = block["pred_residual_linear_lw"]
                test_pred_table[f"Residual XGB {anchor_label} LW"] = block["pred_residual_xgb_lw"]

        if mega_ets_residual is not None:
            test_pred_table["MegaBoost ETS residual"] = mega_ets_residual["test_pred"]
        if mega_ets_residual_lw is not None:
            test_pred_table["MegaBoost ETS residual LW"] = mega_ets_residual_lw["test_pred"]

    with timer.track("evaluate_point_metrics"):
        model_order = [
            "Seasonal daylag", "SARIMA local", "ETS local",
            "StationBoost base norm", "StationBoost rich norm", "StationBoost rich norm LW", "StationBoost Seasonal residual LW",
            "MegaBoost base", "MegaBoost rich", "MegaBoost rich LW",
            "MegaBoost base norm", "MegaBoost rich norm", "MegaBoost rich norm LW",
            "MegaBoost Seasonal residual", "MegaBoost Seasonal residual LW",
            "LineBoost rich norm LW", "LineBoost Seasonal residual LW", "StationBias Seasonal LW",
            "MegaBoost ETS residual", "MegaBoost ETS residual LW",
            "CatBoost base", "CatBoost rich", "Km-XGBoost-lite", "ET-LGBM-PSO-lite", "TP-Boost-lite",
            "ML-Fusion-SVR-lite", "VMD-LightGBM-lite", "ST-LightGBM-MIC-lite", "SN-LSSVM-lite", "TwoStage-FS-MSD-Hybrid-lite", "Stacking-CatBoost-lite",
            "Global LSTM", "Global Transformer-lite",
            "PSR-CNN-LSTM-lite", "CNN-BiLSTM-Attention-lite", "MSD-Informer-BiLSTM-lite", "BiLSTM-CTPA-lite",
            "Meta static", "Meta context", "Meta static LW", "Meta context LW",
            "Residual Linear", "Residual XGB", "Residual Linear LW", "Residual XGB LW",
            "Residual Linear ETS", "Residual XGB ETS", "Residual Linear ETS LW", "Residual XGB ETS LW",
            "Residual Linear Seasonal", "Residual XGB Seasonal", "Residual Linear Seasonal LW", "Residual XGB Seasonal LW",
        ]
        model_cols = [c for c in model_order if c in test_pred_table.columns]
        micro_table, macro_table = evaluate_predictions(test_pred_table, model_cols, cfg)
        baseline_cols, bootstrap_reference_table = _choose_bootstrap_reference_models(micro_table, model_cols)
    if bool(getattr(cfg, "run_bootstrap", True)) and int(getattr(cfg, "n_bootstrap", 0)) > 0:
        with timer.track("station_bootstrap_ci"):
            bootstrap = station_bootstrap_ci(test_pred_table, model_cols, cfg, baseline_cols=baseline_cols)
    else:
        timer.record("station_bootstrap_ci", 0.0, status="skipped")
        bootstrap = {"metric_ci": pd.DataFrame(), "delta_ci": pd.DataFrame()}

    timer.record("TOTAL run_multistation_forecasting_experiment", perf_counter() - overall_started)
    timing_table = timer.to_frame()

    return {
        "config": cfg,
        "buckets": buckets,
        "stations": stations,
        "places": places,
        "frame": frame,
        "split": split,
        "base_features": base_features,
        "rich_features": rich_features,
        "seasonal": seasonal,
        "sarima_local": sarima_result,
        "ets_local": ets_result,
        "megaboost_base": mega_base,
        "megaboost_rich": mega_rich,
        "megaboost_rich_lw": mega_rich_lw,
        "megaboost_base_norm": mega_base_norm,
        "megaboost_rich_norm": mega_rich_norm,
        "megaboost_rich_norm_lw": mega_rich_norm_lw,
        "megaboost_seasonal_residual": mega_seasonal_residual,
        "megaboost_seasonal_residual_lw": mega_seasonal_residual_lw,
        "lineboost_rich_norm_lw": line_rich_norm_lw,
        "lineboost_seasonal_residual_lw": line_seasonal_residual_lw,
        "stationboost_base_norm": stationboost_base_norm,
        "stationboost_rich_norm": stationboost_rich_norm,
        "stationboost_rich_norm_lw": stationboost_rich_norm_lw,
        "stationboost_seasonal_residual_lw": stationboost_seasonal_residual_lw,
        "station_bias_seasonal_lw": station_bias_seasonal_lw,
        "megaboost_ets_residual": mega_ets_residual,
        "megaboost_ets_residual_lw": mega_ets_residual_lw,
        "catboost_base": catboost_base,
        "catboost_rich": catboost_rich,
        "km_xgboost_lite": km_xgboost_lite,
        "et_lgbm_pso_lite": et_lgbm_pso_lite,
        "tp_boost_lite": tp_boost_lite,
        "ml_fusion_svr_lite": ml_fusion_svr_lite,
        "vmd_lightgbm_lite": vmd_lightgbm_lite,
        "st_lightgbm_mic_lite": st_lightgbm_mic_lite,
        "sn_lssvm_lite": sn_lssvm_lite,
        "twostage_fs_msd_hybrid_lite": twostage_fs_msd_hybrid_lite,
        "stacking_catboost_lite": stacking_catboost_lite,
        "sequence": seq_results,
        "article_sequence": article_sequence_results,
        "val_pred_table": val_pred_table,
        "test_pred_table": test_pred_table,
        "meta": meta,
        "model_cols": model_cols,
        "micro_table": micro_table,
        "macro_table": macro_table,
        "bootstrap": bootstrap,
        "bootstrap_reference_table": bootstrap_reference_table,
        "timing_table": timing_table,
    }

def run_station_local_boosting_only(
    cfg: MultiStationForecastConfig,
    force_rebuild_buckets: bool = False,
) -> dict:
    """Run only the station-local boosting ablation, skipping the full model zoo.

    This fast path reuses the same data preparation, operational-day split and
    feature construction as :func:`run_multistation_forecasting_experiment`, but it does **not**
    retrain global MegaBoost, ETS, SARIMA, deep/article comparators or meta-models.
    It is intended for the station-local XGBoost ablation after the main megagrid
    has already been computed and saved to disk.
    """
    overall_started = perf_counter()
    timer = _StageTimer(echo=cfg.print_stage_timings)

    with timer.track("load_or_build_station_buckets"):
        buckets, stations, places = load_or_build_station_buckets(
            cfg,
            force_rebuild=force_rebuild_buckets,
        )
    with timer.track("make_multistation_supervised_frame"):
        frame = make_multistation_supervised_frame(buckets, cfg)
    with timer.track("make_multistation_weekday_split"):
        split = make_multistation_weekday_split(frame, cfg)

    with timer.track("prepare_feature_lists_and_seasonal_anchor"):
        base_features = _available_features(split["train_tune"], BASE_FEATURES)
        rich_features = _available_features(split["train_tune"], BASE_FEATURES + RICH_FEATURES)
        seasonal = seasonal_daylag_predictions(split["val_df"], split["test_df"])
        seasonal_train_anchor = _seasonal_anchor_series(split["train_tune"]).values.astype(float)
        seasonal_val_anchor = seasonal["val_pred"]
        seasonal_train_full_anchor = _seasonal_anchor_series(split["train_full"]).values.astype(float)
        seasonal_test_anchor = seasonal["test_pred"]

    with timer.track("StationBoost base norm"):
        stationboost_base_norm = _run_station_local_normalized_megaboost(
            cfg,
            split["train_tune"],
            split["val_df"],
            split["train_full"],
            split["test_df"],
            base_features,
            "StationBoost base norm",
            selection_metric="LW-MAE",
        )

    with timer.track("StationBoost rich norm"):
        stationboost_rich_norm = _run_station_local_normalized_megaboost(
            cfg,
            split["train_tune"],
            split["val_df"],
            split["train_full"],
            split["test_df"],
            rich_features,
            "StationBoost rich norm",
            selection_metric="LW-MAE",
        )

    with timer.track("StationBoost rich norm LW"):
        stationboost_rich_norm_lw = _run_station_local_normalized_megaboost(
            cfg,
            split["train_tune"],
            split["val_df"],
            split["train_full"],
            split["test_df"],
            rich_features,
            "StationBoost rich norm LW",
            sample_weight_mode="lw",
            selection_metric="LW-MAE",
        )

    with timer.track("StationBoost Seasonal residual LW"):
        stationboost_seasonal_residual_lw = _run_station_local_seasonal_residual_megaboost(
            cfg,
            split["train_tune"],
            split["val_df"],
            split["train_full"],
            split["test_df"],
            rich_features,
            "StationBoost Seasonal residual LW",
            anchor_train_pred=seasonal_train_anchor,
            anchor_val_pred=seasonal_val_anchor,
            anchor_train_full_pred=seasonal_train_full_anchor,
            anchor_test_pred=seasonal_test_anchor,
            sample_weight_mode="lw",
            selection_metric="LW-MAE",
        )

    with timer.track("assemble_stationboost_prediction_tables"):
        preds = {
            "StationBoost base norm": stationboost_base_norm,
            "StationBoost rich norm": stationboost_rich_norm,
            "StationBoost rich norm LW": stationboost_rich_norm_lw,
            "StationBoost Seasonal residual LW": stationboost_seasonal_residual_lw,
        }
        val_pred_table, test_pred_table = _build_meta_tables(split, preds, cfg)

    with timer.track("evaluate_stationboost_point_metrics"):
        model_cols = list(preds)
        micro_table, macro_table = evaluate_predictions(test_pred_table, model_cols, cfg)

    timer.record("station_bootstrap_ci", 0.0, status="skipped")
    timer.record("TOTAL run_station_local_boosting_only", perf_counter() - overall_started)
    timing_table = timer.to_frame()

    return {
        "config": cfg,
        "buckets": buckets,
        "stations": stations,
        "places": places,
        "frame": frame,
        "split": split,
        "base_features": base_features,
        "rich_features": rich_features,
        "seasonal": seasonal,
        "stationboost_base_norm": stationboost_base_norm,
        "stationboost_rich_norm": stationboost_rich_norm,
        "stationboost_rich_norm_lw": stationboost_rich_norm_lw,
        "stationboost_seasonal_residual_lw": stationboost_seasonal_residual_lw,
        "val_pred_table": val_pred_table,
        "test_pred_table": test_pred_table,
        "meta": {},
        "model_cols": model_cols,
        "micro_table": micro_table,
        "macro_table": macro_table,
        "bootstrap": {"metric_ci": pd.DataFrame(), "delta_ci": pd.DataFrame()},
        "bootstrap_reference_table": pd.DataFrame(),
        "timing_table": timing_table,
    }


__all__ = [
    "run_multistation_forecasting_experiment",
    "run_station_local_boosting_only",
]
