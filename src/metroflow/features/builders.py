
from __future__ import annotations

import numpy as np
import pandas as pd

from metroflow.config import ExperimentConfig


def build_lag_buckets(cfg: ExperimentConfig) -> list[int]:
    vals = sorted(set(max(1, int(round(m / cfg.bucket_minutes))) for m in cfg.features.lag_minutes))
    return vals


def build_roll_windows(cfg: ExperimentConfig) -> list[int]:
    vals = sorted(set(max(2, int(round(m / cfg.bucket_minutes))) for m in cfg.features.rolling_minutes))
    return vals


def make_target_frame(pass_df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    df = pass_df[['TRAN_DATE']].copy()
    df[cfg.time_col] = df['TRAN_DATE'].dt.floor(cfg.freq)
    df = (
        df.groupby(cfg.time_col)
          .size()
          .reset_index(name='count')
          .sort_values(cfg.time_col)
    )
    df = df.set_index(cfg.time_col).asfreq(cfg.freq)
    df['count'] = df['count'].fillna(0.0)
    df = df.reset_index()

    future_parts = [df['count'].shift(-k) for k in range(1, cfg.horizon_steps + 1)]
    df['target_h'] = pd.concat(future_parts, axis=1).sum(axis=1, min_count=cfg.horizon_steps)

    lag_buckets = build_lag_buckets(cfg)
    roll_windows = build_roll_windows(cfg)

    for lag in lag_buckets:
        df[f'lag_{lag}'] = df['count'].shift(lag)

    for w in roll_windows:
        df[f'roll_sum_{w}'] = df['count'].shift(1).rolling(w).sum()
        df[f'roll_mean_{w}'] = df['count'].shift(1).rolling(w).mean()

    slot_in_day = (
        df[cfg.time_col].dt.hour * (60 // cfg.bucket_minutes)
        + df[cfg.time_col].dt.minute // cfg.bucket_minutes
    )
    df['slot_in_day'] = slot_in_day.astype(int)
    df['day_of_week'] = df[cfg.time_col].dt.dayofweek
    df['date'] = df[cfg.time_col].dt.date

    if 'hour_cyclical' in cfg.features.context_features:
        df['hour_sin'] = np.sin(2 * np.pi * df['slot_in_day'] / cfg.day_lag)
        df['hour_cos'] = np.cos(2 * np.pi * df['slot_in_day'] / cfg.day_lag)

    df['target_daylag'] = df['target_h'].shift(cfg.day_lag)

    if 'continuous_load_score' in cfg.features.context_features:
        baseline = df['target_daylag'].replace(0, np.nan)
        df['continuous_load_score'] = (df['target_h'].shift(1) / baseline).replace([np.inf, -np.inf], np.nan)

    if 'local_slope' in cfg.features.context_features:
        df['local_slope_1'] = df['count'].shift(1) - df['count'].shift(2)
        df['local_slope_2'] = df['count'].shift(1) - df['count'].shift(3)

    if 'rolling_volatility' in cfg.features.context_features:
        short_w = min(build_roll_windows(cfg))
        df[f'roll_std_{short_w}'] = df['count'].shift(1).rolling(short_w).std()

    return df


def clip_segment_for_horizon(seg_df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    seg_df = seg_df.sort_values(cfg.time_col).copy()
    if seg_df.empty:
        return seg_df
    cutoff = seg_df[cfg.time_col].max() - pd.Timedelta(minutes=cfg.horizon_minutes)
    return seg_df[seg_df[cfg.time_col] <= cutoff].copy()


def attach_context_feature(train_df: pd.DataFrame, pred_df: pd.DataFrame, cfg: ExperimentConfig):
    train_df = train_df.copy()
    pred_df = pred_df.copy()
    q = cfg.features.highload_quantile
    q_train = train_df['target_h'].quantile(q)

    if 'highload_daylag' in cfg.features.context_features:
        train_df['hl_daylag'] = (train_df['target_daylag'] > q_train).astype(int)
        pred_df['hl_daylag'] = (pred_df['target_daylag'] > q_train).astype(int)
    else:
        train_df['hl_daylag'] = 0
        pred_df['hl_daylag'] = 0

    return train_df, pred_df, q_train


def make_weekday_split(frame: pd.DataFrame, cfg: ExperimentConfig):
    frame = frame[frame['day_of_week'].isin(cfg.features.weekday_filter)].copy()
    unique_dates = sorted(frame['date'].unique())
    if len(unique_dates) < 4:
        raise ValueError('Need at least 4 dates for Mon/Tue/Wed/Thu split')

    mon, tue, wed, thu = unique_dates[:4]

    train_tune_all = frame[frame['date'].isin([mon, tue])].copy()
    val_all = frame[frame['date'] == wed].copy()
    train_full_all = frame[frame['date'].isin([mon, tue, wed])].copy()
    test_all = frame[frame['date'] == thu].copy()

    train_tune = clip_segment_for_horizon(train_tune_all, cfg)
    val_df = clip_segment_for_horizon(val_all, cfg)
    train_full = clip_segment_for_horizon(train_full_all, cfg)
    test_df = clip_segment_for_horizon(test_all, cfg)

    train_tune, val_df, q80_tune = attach_context_feature(train_tune, val_df, cfg)
    train_full, test_df, q80_full = attach_context_feature(train_full, test_df, cfg)

    return {
        'train_tune_all': train_tune_all,
        'val_all': val_all,
        'train_full_all': train_full_all,
        'test_all': test_all,
        'train_tune': train_tune,
        'val_df': val_df,
        'train_full': train_full,
        'test_df': test_df,
        'q80_tune': q80_tune,
        'q80_full': q80_full,
    }
