from __future__ import annotations

from ..common import *
from ..config import (
    DEFAULT_CLASSIC_METRO_LINE_CODES,
    _scope_name,
    MultiStationForecastConfig,
)
from .buckets import (
    _pass_source_path,
    _default_station_buckets_path,
    build_multistation_bucket_parquet,
    load_or_build_station_buckets,
    _complete_one_station,
)
from .weather import (
    WEATHER_FEATURES,
    _weather_csv_path,
    _load_weather_hourly,
    _attach_weather_features,
)

def make_multistation_supervised_frame(bucket_df: pd.DataFrame, cfg: MultiStationForecastConfig) -> pd.DataFrame:
    parts = [_complete_one_station(g, cfg) for _, g in bucket_df.groupby("station_key", sort=False)]
    df = pd.concat(parts, ignore_index=True).sort_values(["station_key", cfg.time_col]).reset_index(drop=True)

    g = df.groupby("station_key", sort=False)["count"]
    future_parts = [g.shift(-k) for k in range(1, cfg.horizon_steps + 1)]
    df["target_h"] = pd.concat(future_parts, axis=1).sum(axis=1, min_count=cfg.horizon_steps)

    lag_list = sorted(set([1, 2, 3, 4, 8, 12, 16, 24, cfg.day_lag, cfg.day_lag + 1, 2 * cfg.day_lag]))
    for lag in lag_list:
        df[f"lag_{lag}"] = g.shift(lag)

    for w in [2, 4, 8, 12, 16]:
        df[f"roll_sum_{w}"] = g.shift(1).rolling(w).sum().reset_index(level=0, drop=True)
        df[f"roll_mean_{w}"] = g.shift(1).rolling(w).mean().reset_index(level=0, drop=True)
    for w in [2, 4, 8, 12, 16]:
        df[f"roll_std_{w}"] = g.shift(1).rolling(w).std().reset_index(level=0, drop=True)

    op_time = df[cfg.time_col] - pd.Timedelta(hours=cfg.operational_day_start_hour)
    df["operational_date"] = op_time.dt.date
    df["date"] = df["operational_date"]
    df["day_of_week"] = op_time.dt.dayofweek
    df["slot_in_day"] = (
        op_time.dt.hour * (60 // cfg.bucket_minutes) + op_time.dt.minute // cfg.bucket_minutes
    ).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["slot_in_day"] / cfg.day_lag)
    df["hour_cos"] = np.cos(2 * np.pi * df["slot_in_day"] / cfg.day_lag)

                                                                        
    df = _attach_weather_features(df, cfg)

    df["target_daylag"] = df.groupby("station_key", sort=False)["target_h"].shift(cfg.day_lag)
    df["local_slope_1"] = df["lag_1"] - df["lag_2"]
    df["local_slope_2"] = df["lag_1"] - df["lag_3"]
    df["daylag_deviation"] = df["lag_1"] - df[f"lag_{cfg.day_lag}"]
    denom = df[f"lag_{cfg.day_lag}"].replace(0, np.nan)
    df["prev_count_ratio_daylag"] = (df["lag_1"] / denom).replace([np.inf, -np.inf], np.nan)

    df["station_key_code"] = pd.Categorical(df["station_key"].astype(str)).codes.astype(int)
    df["station_name_code"] = pd.Categorical(df.get("ST_NAME_SHORT", df.get("ST_NAME", "")).astype(str)).codes.astype(int)
    df["line_name_code"] = pd.Categorical(df.get("LN_NAME_SHORT", df.get("LN_NAME", "")).astype(str)).codes.astype(int)
    df["station_id_raw"] = pd.to_numeric(df.get("ST_CODE"), errors="coerce").fillna(-1).astype(int)
    df["line_id_raw"] = pd.to_numeric(df.get("LN_CODE"), errors="coerce").fillna(-1).astype(int)

                                                                    
    df["station_code"] = df["station_key_code"]
    df["line_code"] = df["line_id_raw"]
    return df

def _clip_per_station_day(seg: pd.DataFrame, cfg: MultiStationForecastConfig) -> pd.DataFrame:
    if seg.empty:
        return seg.copy()
    parts = []
    for (_, _date), g in seg.groupby(["station_key", "date"], sort=False):
        cutoff = g[cfg.time_col].max() - pd.Timedelta(minutes=cfg.horizon_minutes)
        parts.append(g[g[cfg.time_col] <= cutoff].copy())
    return pd.concat(parts, ignore_index=True) if parts else seg.iloc[0:0].copy()

def _station_q80(train_df: pd.DataFrame, cfg: MultiStationForecastConfig) -> pd.Series:
    q = train_df.dropna(subset=["target_h"]).groupby("station_key")["target_h"].quantile(cfg.highload_quantile)
    global_q = train_df["target_h"].quantile(cfg.highload_quantile)
    if q.empty:
        return pd.Series(dtype=float)
    return q.fillna(global_q)

def _attach_station_context(train_df: pd.DataFrame, pred_df: pd.DataFrame, cfg: MultiStationForecastConfig):
    train_df = train_df.copy()
    pred_df = pred_df.copy()
    q80 = _station_q80(train_df, cfg)
    train_q = train_df["station_key"].map(q80)
    pred_q = pred_df["station_key"].map(q80)
    fallback_q = train_df["target_h"].quantile(cfg.highload_quantile)
    train_q = train_q.fillna(fallback_q).replace(0, np.nan)
    pred_q = pred_q.fillna(fallback_q).replace(0, np.nan)

    train_df["q80_station"] = train_q.values
    pred_df["q80_station"] = pred_q.values
    train_df["hl_daylag"] = (train_df["target_daylag"] > train_df["q80_station"]).astype(int)
    pred_df["hl_daylag"] = (pred_df["target_daylag"] > pred_df["q80_station"]).astype(int)
    train_df["hl_actual"] = (train_df["target_h"] > train_df["q80_station"]).astype(int)
    pred_df["hl_actual"] = (pred_df["target_h"] > pred_df["q80_station"]).astype(int)
    train_df["continuous_load_score"] = (train_df["target_daylag"] / train_df["q80_station"]).replace([np.inf, -np.inf], np.nan)
    pred_df["continuous_load_score"] = (pred_df["target_daylag"] / pred_df["q80_station"]).replace([np.inf, -np.inf], np.nan)

    target_train = train_df.dropna(subset=["target_h"]).copy()
    if target_train.empty:
        raise ValueError("No target_h observations available for station context statistics")

    station_stats = target_train.groupby("station_key")["target_h"].agg(
        station_train_mean_target="mean",
        station_train_median_target="median",
        station_train_q95_target=lambda s: s.quantile(0.95),
        station_train_max_target="max",
    )
    station_stats["station_train_q80_target"] = q80
    count_stats = train_df.groupby("station_key")["count"].mean().rename("station_train_mean_count")
    station_stats = station_stats.join(count_stats, how="left")

    global_stats = {
        "station_train_mean_target": float(target_train["target_h"].mean()),
        "station_train_median_target": float(target_train["target_h"].median()),
        "station_train_q80_target": float(target_train["target_h"].quantile(cfg.highload_quantile)),
        "station_train_q95_target": float(target_train["target_h"].quantile(0.95)),
        "station_train_max_target": float(target_train["target_h"].max()),
        "station_train_mean_count": float(train_df["count"].mean()),
    }

    for frame in (train_df, pred_df):
        for col, fallback in global_stats.items():
            frame[col] = frame["station_key"].map(station_stats[col]).fillna(fallback)
        frame["station_mean_target_vs_global"] = frame["station_train_mean_target"] / max(global_stats["station_train_mean_target"], 1e-9)
        frame["station_q80_target_vs_global"] = frame["station_train_q80_target"] / max(global_stats["station_train_q80_target"], 1e-9)
        frame["station_mean_count_vs_global"] = frame["station_train_mean_count"] / max(global_stats["station_train_mean_count"], 1e-9)

    return train_df, pred_df, q80

def make_multistation_weekday_split(frame: pd.DataFrame, cfg: MultiStationForecastConfig) -> dict:
    frame = frame[frame["day_of_week"].isin(cfg.weekday_filter)].copy()
    unique_dates = sorted(frame["date"].unique())
    if len(unique_dates) < 4:
        raise ValueError(f"Need at least 4 operational weekdays, got {len(unique_dates)}: {unique_dates}")
    mon, tue, wed, thu = unique_dates[:4]

    train_tune_all = frame[frame["date"].isin([mon, tue])].copy()
    val_all = frame[frame["date"] == wed].copy()
    train_full_all = frame[frame["date"].isin([mon, tue, wed])].copy()
    test_all = frame[frame["date"] == thu].copy()

    train_tune = _clip_per_station_day(train_tune_all, cfg)
    val_df = _clip_per_station_day(val_all, cfg)
    train_full = _clip_per_station_day(train_full_all, cfg)
    test_df = _clip_per_station_day(test_all, cfg)

    train_tune, val_df, q80_tune = _attach_station_context(train_tune, val_df, cfg)
    train_full, test_df, q80_full = _attach_station_context(train_full, test_df, cfg)

    return {
        "dates": {"train_tune": [mon, tue], "validation": wed, "train_full": [mon, tue, wed], "test": thu},
        "train_tune_all": train_tune_all,
        "val_all": val_all,
        "train_full_all": train_full_all,
        "test_all": test_all,
        "train_tune": train_tune,
        "val_df": val_df,
        "train_full": train_full,
        "test_df": test_df,
        "q80_tune": q80_tune,
        "q80_full": q80_full,
    }

__all__ = [
    "make_multistation_supervised_frame",
    "_clip_per_station_day",
    "_station_q80",
    "_attach_station_context",
    "make_multistation_weekday_split",
]
