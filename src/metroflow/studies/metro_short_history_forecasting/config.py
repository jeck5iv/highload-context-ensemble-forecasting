from __future__ import annotations

from .common import *

DEFAULT_CLASSIC_METRO_LINE_CODES: tuple[int, ...] = (
    8,                      
    9,                       
    12,                          
    13,                 
    14,                  
    15,                         
    16,                                 
    17,                                  
    18,                                                                    
    609,               
    1500,              
    1509,                                                     
    1691,                 
    1749,                            
    1779,                 
    2162,              
)

def _scope_name(cfg: "MultiStationForecastConfig") -> str:
    if cfg.line_filter:
        return re.sub(r"[^a-zA-Z0-9а-яА-Я]+", "_", cfg.line_filter).strip("_")
    return "classic_metro" if cfg.classic_metro_only else "metro_type1"

@dataclass
class MultiStationForecastConfig:
    """Configuration for the multi-station forecasting experiment.

    The experiment keeps the same temporal protocol as the single-station setup,
    but changes the unit of observation from time to (station, time).
    """

    name: str = "multistation_forecasting_all_metro_15m_60m_opday3"
    data_dir: str = "./data"
    pass_filename: str = "PASS_ALL_202503242210.csv"
    pass_sep: str = ";"
    ref_places_glob: str = "REF_PSG_PLACES*.csv"
    cache_dir: str = "./data/cache"
    output_dir: str = "./outputs/multistation_forecasting_all_metro_15m_60m_opday3"
    bucket_minutes: int = 15
    horizon_minutes: int = 60
    operational_day_start_hour: int = 3
    transport_type_id: int = 1
    place_type_id: int = 1
    classic_metro_only: bool = True
    classic_metro_line_codes: tuple[int, ...] = DEFAULT_CLASSIC_METRO_LINE_CODES
    line_filter: str | None = None
                                                                                                              
    use_weather_features: bool = False
    weather_csv: str | None = None
    station_places_csv: str | None = None
    station_buckets_parquet: str | None = None
    weekday_filter: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    highload_quantile: float = 0.8
    load_weight_gamma: float = 1.0
    hl_underprediction_penalty: float = 2.0
    top_k_boost_models: int = 3
    top_k_local_boost_models: int = 2
    # Per-station boosters are substantially more expensive than pooled/line-local models,
    # so they are opt-in and use their own compact candidate grid.
    run_station_megaboost: bool = False
    top_k_station_boost_models: int = 1
    station_local_min_train_rows: int = 48
    station_local_min_validation_rows: int = 12
    local_min_train_rows: int = 240
    local_min_validation_rows: int = 72
    run_line_megaboost: bool = True
    run_station_bias_correction: bool = True
    local_correction_alpha_grid: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
    station_bias_shrinkage: float = 96.0
    random_state: int = 3228
    # Exploratory runs do not need 1000 bootstrap resamples; set to 100 by default
    # and switch to 1000/2000 only for final reporting.
    n_bootstrap: int = 100
    run_bootstrap: bool = False
    print_stage_timings: bool = True
    run_residual_xgb: bool = True
    run_catboost: bool = True
    catboost_iterations: int = 600
    catboost_depth: int = 8
    catboost_learning_rate: float = 0.05
    catboost_l2_leaf_reg: float = 8.0
                                                                                 
                                                                                                     
                                                                                                   
    run_local_ets: bool = True
    run_local_sarima: bool = True
    sarima_maxiter: int = 40
    run_global_lstm: bool = True
    run_global_transformer: bool = True
    # Article-grounded lightweight neural comparators.
    run_psr_cnn_lstm: bool = True
    run_cnn_bilstm_attention: bool = True
    run_msd_informer_bilstm_lite: bool = True
    run_bilstm_ctpa_lite: bool = True
    # Recent article-inspired tabular/ensemble comparators.
    run_km_xgboost_lite: bool = True
    run_et_lgbm_pso_lite: bool = True
    run_tp_boost_lite: bool = True
    # Additional literature-grounded tabular / hybrid comparators kept out of the meta-pool.
    run_ml_fusion_svr_lite: bool = True
    run_vmd_lightgbm_lite: bool = True
    run_st_lightgbm_mic_lite: bool = True
    run_sn_lssvm_lite: bool = True
    run_twostage_fs_msd_hybrid_lite: bool = True
    # Additional article-inspired stack kept outside the prediction meta-ensemble.
    run_stacking_catboost_lite: bool = True
    km_xgb_cluster_grid: tuple[int, ...] = (3, 4, 5)
    km_xgb_min_train_rows: int = 900
    km_xgb_min_validation_rows: int = 180
    et_lgbm_extra_trees_estimators: int = 320
    et_lgbm_extra_trees_min_samples_leaf: int = 2
    et_lgbm_num_leaves: int = 31
    et_lgbm_n_estimators: int = 480
    et_lgbm_learning_rate: float = 0.035
    et_lgbm_pso_particles: int = 10
    et_lgbm_pso_iters: int = 24
    tp_boost_min_train_rows: int = 1400
    tp_boost_min_validation_rows: int = 300
    tp_boost_ls_estimators: int = 260
    tp_boost_ls_learning_rate: float = 0.035
    ml_fusion_ada_estimators: int = 180
    ml_fusion_ada_learning_rate: float = 0.045
    ml_fusion_ada_tree_depth: int = 3
    ml_fusion_svr_max_rows: int = 6000
    ml_fusion_svr_c: float = 25.0
    ml_fusion_svr_epsilon: float = 0.05
    vmd_lgbm_candidate_leaves: tuple[int, ...] = (15, 31, 47)
    vmd_lgbm_candidate_estimators: tuple[int, ...] = (360, 480, 560)
    st_mic_top_k_features: int = 28
    feature_selection_max_rows: int = 15000
    sn_lssvm_rbf_components: int = 256
    sn_lssvm_rbf_gamma: float = 0.08
    sn_lssvm_ridge_alpha: float = 12.0
    twostage_corr_top_k_features: int = 36
    twostage_final_top_k_features: int = 20
    stacking_catboost_iterations: int = 360
    stacking_catboost_depth: int = 5
    stacking_catboost_learning_rate: float = 0.045
    stacking_catboost_l2_leaf_reg: float = 10.0
    seq_window: int = 24
    seq_batch_size: int = 512
    lstm_max_epochs: int = 35
    transformer_max_epochs: int = 25
    seq_patience: int = 5
    article_seq_max_epochs: int = 22
    article_seq_patience: int = 4
    article_seq_learning_rate: float = 1e-3
    article_seq_weight_decay: float = 1e-4
    psr_embedding_dim: int = 4
    psr_delay: int = 2
    msd_short_window: int = 3
    msd_long_window: int = 8
    boosting_param_grid: list[dict] = field(default_factory=lambda: [
        # Conservative shallow trees: useful when two train days are not enough for deep partitions.
        dict(objective="reg:absoluteerror", n_estimators=180, learning_rate=0.055, max_depth=2,
             min_child_weight=40, subsample=0.90, colsample_bytree=0.90,
             reg_lambda=30.0, reg_alpha=6.0, tree_method="hist"),
        dict(objective="reg:absoluteerror", n_estimators=240, learning_rate=0.050, max_depth=3,
             min_child_weight=30, subsample=0.90, colsample_bytree=0.90,
             reg_lambda=24.0, reg_alpha=5.0, tree_method="hist"),
        dict(objective="reg:absoluteerror", n_estimators=320, learning_rate=0.045, max_depth=3,
             min_child_weight=22, subsample=0.88, colsample_bytree=0.88,
             reg_lambda=18.0, reg_alpha=4.0, tree_method="hist"),
        # Mid-capacity models: keep the earlier successful family in the search space.
        dict(objective="reg:absoluteerror", n_estimators=250, learning_rate=0.050, max_depth=4,
             min_child_weight=18, subsample=0.85, colsample_bytree=0.85,
             reg_lambda=14.0, reg_alpha=2.0, tree_method="hist"),
        dict(objective="reg:absoluteerror", n_estimators=400, learning_rate=0.040, max_depth=5,
             min_child_weight=14, subsample=0.85, colsample_bytree=0.85,
             reg_lambda=12.0, reg_alpha=1.5, tree_method="hist"),
        dict(objective="reg:absoluteerror", n_estimators=520, learning_rate=0.035, max_depth=5,
             min_child_weight=12, subsample=0.88, colsample_bytree=0.88,
             reg_lambda=10.0, reg_alpha=1.0, tree_method="hist"),
        # Higher-capacity alternatives remain available, but only validation-selected members enter MegaBoost.
        dict(objective="reg:absoluteerror", n_estimators=600, learning_rate=0.030, max_depth=6,
             min_child_weight=10, subsample=0.90, colsample_bytree=0.90,
             reg_lambda=8.0, reg_alpha=0.5, tree_method="hist"),
        dict(objective="reg:absoluteerror", n_estimators=720, learning_rate=0.025, max_depth=4,
             min_child_weight=26, subsample=0.92, colsample_bytree=0.92,
             reg_lambda=20.0, reg_alpha=3.0, tree_method="hist"),
        dict(objective="reg:absoluteerror", n_estimators=840, learning_rate=0.020, max_depth=3,
             min_child_weight=34, subsample=0.92, colsample_bytree=0.92,
             reg_lambda=26.0, reg_alpha=4.0, tree_method="hist"),
    ])
    local_boosting_param_grid: list[dict] = field(default_factory=lambda: [
        # Line-local models see far fewer rows than the pooled global model, so keep them deliberately small.
        dict(objective="reg:absoluteerror", n_estimators=120, learning_rate=0.055, max_depth=2,
             min_child_weight=14, subsample=0.90, colsample_bytree=0.90,
             reg_lambda=28.0, reg_alpha=6.0, tree_method="hist"),
        dict(objective="reg:absoluteerror", n_estimators=180, learning_rate=0.045, max_depth=3,
             min_child_weight=18, subsample=0.90, colsample_bytree=0.90,
             reg_lambda=24.0, reg_alpha=5.0, tree_method="hist"),
        dict(objective="reg:absoluteerror", n_estimators=240, learning_rate=0.035, max_depth=3,
             min_child_weight=12, subsample=0.88, colsample_bytree=0.88,
             reg_lambda=20.0, reg_alpha=4.0, tree_method="hist"),
    ])
    station_boosting_param_grid: list[dict] = field(default_factory=lambda: [
        # Station-local train splits are short.  These candidates deliberately trade
        # capacity for stability and are validation-selected independently per station.
        dict(objective="reg:absoluteerror", n_estimators=72, learning_rate=0.060, max_depth=1,
             min_child_weight=4, subsample=0.95, colsample_bytree=0.95,
             reg_lambda=36.0, reg_alpha=8.0, tree_method="hist"),
        dict(objective="reg:absoluteerror", n_estimators=120, learning_rate=0.045, max_depth=2,
             min_child_weight=6, subsample=0.92, colsample_bytree=0.92,
             reg_lambda=28.0, reg_alpha=6.0, tree_method="hist"),
        dict(objective="reg:absoluteerror", n_estimators=180, learning_rate=0.035, max_depth=2,
             min_child_weight=8, subsample=0.90, colsample_bytree=0.90,
             reg_lambda=22.0, reg_alpha=5.0, tree_method="hist"),
    ])

    @property
    def freq(self) -> str:
        return f"{self.bucket_minutes}min"

    @property
    def time_col(self) -> str:
        return f"bucket_{self.bucket_minutes}m"

    @property
    def day_lag(self) -> int:
        return 24 * 60 // self.bucket_minutes

    @property
    def horizon_steps(self) -> int:
        if self.horizon_minutes % self.bucket_minutes != 0:
            raise ValueError("horizon_minutes must be divisible by bucket_minutes")
        return self.horizon_minutes // self.bucket_minutes


__all__ = [
    "DEFAULT_CLASSIC_METRO_LINE_CODES",
    "_scope_name",
    "MultiStationForecastConfig",
]
