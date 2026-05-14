from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


@dataclass
class DataConfig:
    data_dir: str = './data'
    pass_filename: str = 'PASS_ALL_202503242210.csv'
    pass_sep: str = ';'
    cache_dir: str = './data/cache'
    transport_type_id: int | None = 1
    place_ids: list[int] | None = field(default_factory=lambda: [1623, 1624])
    validation_modes: list[int] | None = None
    station_metadata_path: str | None = None


@dataclass
class FeatureConfig:
    lag_minutes: list[int] = field(default_factory=lambda: [15, 30, 45, 60, 120, 180, 240, 360, 1440, 1455, 2880])
    rolling_minutes: list[int] = field(default_factory=lambda: [30, 60, 120, 180, 240])
    context_features: list[str] = field(default_factory=lambda: [
        'highload_daylag',
        'hour_cyclical',
    ])
                                                                  
                                                                        
                                                                     
                                                                   
    boosting_context_features: list[str] | None = None
                                                       
                                                                   
                                              
    operational_day_start_hour: int = 0
    highload_quantile: float = 0.8
    load_weight_gamma: float = 1.0
    weekday_filter: list[int] = field(default_factory=lambda: [0, 1, 2, 3])


@dataclass
class ModelConfig:
    random_state: int = 3228
    top_k_boost_models: int = 3
    run_lstm: bool = True
    run_transformer: bool = True
    run_direct_meta_xgb: bool = False
    run_residual_meta: bool = True
    use_lstm_in_residual_meta: bool = False
    boosting_param_grid: list[dict] = field(default_factory=lambda: [
        dict(objective='reg:absoluteerror', n_estimators=200, learning_rate=0.05, max_depth=4,
             subsample=0.8, colsample_bytree=0.8, tree_method='hist'),
        dict(objective='reg:absoluteerror', n_estimators=300, learning_rate=0.05, max_depth=5,
             subsample=0.8, colsample_bytree=0.8, tree_method='hist'),
        dict(objective='reg:absoluteerror', n_estimators=500, learning_rate=0.03, max_depth=6,
             subsample=0.85, colsample_bytree=0.85, tree_method='hist'),
    ])
    lstm_max_epochs: int = 100
    transformer_max_epochs: int = 120
    seq_patience: int = 12
    direct_meta_xgb_params: dict = field(default_factory=lambda: dict(
        objective='reg:absoluteerror', n_estimators=20, learning_rate=0.03, max_depth=1,
        min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=15.0, reg_alpha=3.0, tree_method='hist'
    ))
    residual_linear_alpha: float = 10.0
    residual_linear_clip_quantiles: tuple[float, float] = (0.05, 0.95)
    residual_xgb_clip_quantiles: tuple[float, float] = (0.10, 0.90)


@dataclass
class OutputConfig:
    root_dir: str = './outputs'
    save_tables: bool = True
    save_plots: bool = True
    save_markdown: bool = True
    save_run_config: bool = True


@dataclass
class ExperimentConfig:
    name: str = 'sportivnaya_experiment'
    bucket_minutes: int = 15
    horizon_minutes: int = 60
    primary_mode: str = 'auto'
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def freq(self) -> str:
        return f'{self.bucket_minutes}min'

    @property
    def horizon_steps(self) -> int:
        assert self.horizon_minutes % self.bucket_minutes == 0
        return self.horizon_minutes // self.bucket_minutes

    @property
    def day_lag(self) -> int:
        return 24 * 60 // self.bucket_minutes

    @property
    def time_col(self) -> str:
        return f'bucket_{self.bucket_minutes}m'

    @property
    def seq_window(self) -> int:
        return self.day_lag


def _merge_dataclass(dc_cls, payload: dict | None):
    payload = payload or {}
    return dc_cls(**payload)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open('rb') as f:
        raw = tomllib.load(f)

    return ExperimentConfig(
        name=raw.get('name', path.stem),
        bucket_minutes=raw.get('bucket_minutes', 15),
        horizon_minutes=raw.get('horizon_minutes', 60),
        primary_mode=raw.get('primary_mode', 'auto'),
        data=_merge_dataclass(DataConfig, raw.get('data')),
        features=_merge_dataclass(FeatureConfig, raw.get('features')),
        models=_merge_dataclass(ModelConfig, raw.get('models')),
        output=_merge_dataclass(OutputConfig, raw.get('output')),
    )
