from __future__ import annotations

from .common import *
from .config import (
    DEFAULT_CLASSIC_METRO_LINE_CODES,
    _scope_name,
    MultiStationForecastConfig,
)

BASE_FEATURES = [
    # Static station descriptors.
    "station_key_code", "station_id_raw", "line_id_raw", "station_name_code", "line_name_code",
    "n_place_ids", "line_n_stations", "station_order_on_line_proxy", "station_order_norm",
    "distance_to_nearest_terminal_proxy", "distance_to_nearest_terminal_norm", "is_terminal_proxy",
    "is_ring_line_proxy",
    # Local dynamics and the complete lag/rolling grid already computed upstream.
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_8", "lag_12", "lag_16", "lag_24",
    "lag_96", "lag_97", "lag_192",
    "roll_sum_2", "roll_sum_4", "roll_sum_8", "roll_sum_12", "roll_sum_16",
    "roll_mean_2", "roll_mean_4", "roll_mean_8", "roll_mean_12", "roll_mean_16",
    "hour_sin", "hour_cos",
]

RICH_FEATURES = [
    # Give rich boosting the exact seasonal regressor used by the day-lag baseline.
    "target_daylag",
    "hl_daylag", "continuous_load_score", "local_slope_1", "local_slope_2",
    "daylag_deviation", "prev_count_ratio_daylag", "roll_std_2", "roll_std_4", "roll_std_8", "roll_std_12", "roll_std_16",
    # Station-level context available at fit time.
    "q80_station", "station_train_mean_target", "station_train_median_target", "station_train_q80_target",
    "station_train_q95_target", "station_train_max_target", "station_train_mean_count",
    "station_mean_target_vs_global", "station_q80_target_vs_global", "station_mean_count_vs_global",
    # Optional exogenous context.
    "weather_temperature_2m", "weather_relative_humidity_2m", "weather_precipitation", "weather_rain",
    "weather_snowfall", "weather_snow_depth", "weather_weather_code", "weather_wind_speed_10m",
]

def _available_features(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

def _xgb_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].replace([np.inf, -np.inf], np.nan).astype(float)

def _boost_lw_sample_weight(df: pd.DataFrame, cfg: MultiStationForecastConfig) -> np.ndarray:
    """Load-sensitive sample weights aligned with the reported LW-MAE metric."""
    y = pd.to_numeric(df["target_h"], errors="coerce").astype(float).values
    if "station_train_mean_target" in df.columns:
        denom = pd.to_numeric(df["station_train_mean_target"], errors="coerce").astype(float).values
    else:
        fallback = np.nanmedian(y) if np.isfinite(y).any() else 1.0
        denom = np.full(len(df), fallback, dtype=float)
    valid_denom = denom[np.isfinite(denom) & (denom > 0)]
    fallback_denom = float(np.nanmedian(valid_denom)) if len(valid_denom) else 1.0
    denom = np.where(np.isfinite(denom) & (denom > 0), denom, fallback_denom)
    weights = np.power(np.maximum(y, 0.0) / np.maximum(denom, 1e-9), cfg.load_weight_gamma)
    weights = np.clip(weights, 0.25, 8.0)
    weights = np.where(np.isfinite(weights), weights, 1.0)
    return weights.astype(float)

def _aligned_sample_weight(df: pd.DataFrame, train_part: pd.DataFrame, sample_weight: np.ndarray | None) -> np.ndarray | None:
    if sample_weight is None:
        return None
    sw = np.asarray(sample_weight, dtype=float).reshape(-1)
    if len(sw) == len(train_part):
        return sw
    if len(sw) != len(df):
        raise ValueError(f"sample_weight length {len(sw)} does not match {len(df)} input rows or {len(train_part)} fit rows")
    return pd.Series(sw, index=df.index).loc[train_part.index].values.astype(float)

__all__ = [
    "BASE_FEATURES",
    "RICH_FEATURES",
    "_available_features",
    "_xgb_matrix",
    "_boost_lw_sample_weight",
    "_aligned_sample_weight",
]
