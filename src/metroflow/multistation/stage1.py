from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import json
import re
import warnings

import numpy as np
import pandas as pd

try:
    import duckdb
except ModuleNotFoundError:                    
    duckdb = None

import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


                                                                                  
                                                                                         
                                                                                    
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


def _scope_name(cfg: "MultiStationStage1Config") -> str:
    if cfg.line_filter:
        return re.sub(r"[^a-zA-Z0-9а-яА-Я]+", "_", cfg.line_filter).strip("_")
    return "classic_metro" if cfg.classic_metro_only else "metro_type1"


@dataclass
class MultiStationStage1Config:
    """Configuration for the first multi-station forecasting experiment.

    The experiment keeps the same temporal protocol as the single-station setup,
    but changes the unit of observation from time to (station, time).
    """

    name: str = "multistation_stage1_all_metro_15m_60m_opday3"
    data_dir: str = "./data"
    pass_filename: str = "PASS_ALL_202503242210.csv"
    pass_sep: str = ";"
    ref_places_glob: str = "REF_PSG_PLACES*.csv"
    cache_dir: str = "./data/cache"
    output_dir: str = "./outputs/multistation_stage1_all_metro_15m_60m_opday3"
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
    top_k_boost_models: int = 3
    random_state: int = 3228
    n_bootstrap: int = 1000
    run_residual_xgb: bool = True
                                                                                 
                                                                                                     
                                                                                                   
    run_local_ets: bool = True
    run_global_lstm: bool = True
    run_global_transformer: bool = True
    seq_window: int = 24
    seq_batch_size: int = 512
    lstm_max_epochs: int = 35
    transformer_max_epochs: int = 25
    seq_patience: int = 5
    boosting_param_grid: list[dict] = field(default_factory=lambda: [
        dict(objective="reg:absoluteerror", n_estimators=250, learning_rate=0.05, max_depth=4,
             subsample=0.85, colsample_bytree=0.85, tree_method="hist"),
        dict(objective="reg:absoluteerror", n_estimators=400, learning_rate=0.04, max_depth=5,
             subsample=0.85, colsample_bytree=0.85, tree_method="hist"),
        dict(objective="reg:absoluteerror", n_estimators=600, learning_rate=0.03, max_depth=6,
             subsample=0.90, colsample_bytree=0.90, tree_method="hist"),
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


def _find_ref_places(data_dir: Path, pattern: str) -> Path:
    matches = sorted(data_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No reference file matched {pattern!r} in {data_dir}")
    return matches[-1]


def _clean_col_name(c: object) -> str:
    return str(c).strip().upper().replace(" ", "_")


def read_places_reference(path: str | Path, sep: str = ";") -> pd.DataFrame:
    """Read REF_PSG_PLACES with or without header and normalize useful columns."""
    path = Path(path)
    raw = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    raw.columns = [_clean_col_name(c) for c in raw.columns]

    expected = {"PLACE_ID", "TYPE_ID", "ST_CODE", "ST_NAME", "LN_CODE"}
    if not expected.issubset(set(raw.columns)):
        raw = pd.read_csv(path, sep=sep, encoding="utf-8-sig", header=None)
        raw.columns = [
            "PLACE_ID", "PLACE_NUM", "PLACE_NAME", "TYPE_ID", "COL_4", "COL_5",
            "ST_CODE", "ST_NAME", "ST_NAME_SHORT", "LN_CODE", "LN_NAME", "LN_NAME_SHORT",
        ][: raw.shape[1]]
        raw.columns = [_clean_col_name(c) for c in raw.columns]

    aliases = {
        "ID": "PLACE_ID",
        "PSG_PLACE_ID": "PLACE_ID",
        "TRANSPORT_TYPE_ID": "TYPE_ID",
        "STATION_ID": "ST_CODE",
        "STATION_NAME": "ST_NAME",
        "LINE_ID": "LN_CODE",
        "LINE_NAME": "LN_NAME",
    }
    for old, new in aliases.items():
        if old in raw.columns and new not in raw.columns:
            raw[new] = raw[old]

    for c in ["PLACE_ID", "TYPE_ID", "ST_CODE", "LN_CODE"]:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce").astype("Int64")

    for c in ["PLACE_NAME", "ST_NAME", "ST_NAME_SHORT", "LN_NAME", "LN_NAME_SHORT"]:
        if c in raw.columns:
            raw[c] = raw[c].astype(str).str.strip()

    needed = ["PLACE_ID", "TYPE_ID", "ST_CODE", "ST_NAME", "LN_CODE"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"Reference file {path} does not contain required columns: {missing}")

    if "ST_NAME_SHORT" not in raw.columns:
        raw["ST_NAME_SHORT"] = raw["ST_NAME"]
    if "LN_NAME" not in raw.columns:
        raw["LN_NAME"] = raw["LN_CODE"].astype(str)
    if "LN_NAME_SHORT" not in raw.columns:
        raw["LN_NAME_SHORT"] = raw["LN_NAME"]
    if "PLACE_NAME" not in raw.columns:
        raw["PLACE_NAME"] = raw["ST_NAME"]

    return raw


def _station_key(line_code: int, station_code: int) -> str:
    return f"ln{int(line_code):05d}_st{int(station_code):05d}"


def _add_station_structure_features(stations: pd.DataFrame) -> pd.DataFrame:
    """Add static station descriptors used by the global multi-station models.

    Ordering on a line is a proxy obtained by sorting (LN_CODE, ST_CODE, station_key).
    For the classic Koltsevaya line, distance-to-terminal is set to zero by design.
    """
    if stations.empty:
        return stations.copy()

    out = stations.copy()
    out["ST_CODE"] = pd.to_numeric(out["ST_CODE"], errors="coerce").fillna(-1).astype(int)
    out["LN_CODE"] = pd.to_numeric(out["LN_CODE"], errors="coerce").fillna(-1).astype(int)
    out["n_place_ids"] = pd.to_numeric(out.get("n_place_ids"), errors="coerce").fillna(0).astype(int)

    order_df = out.sort_values(["LN_CODE", "ST_CODE", "station_key"]).copy()
    order_df["station_order_on_line_proxy"] = order_df.groupby("LN_CODE", sort=False).cumcount() + 1
    order_df["line_n_stations"] = order_df.groupby("LN_CODE", sort=False)["station_key"].transform("nunique").astype(int)
    order_df["station_order_norm"] = np.where(
        order_df["line_n_stations"] > 1,
        (order_df["station_order_on_line_proxy"] - 1) / (order_df["line_n_stations"] - 1),
        0.0,
    )

    line_text = (
        order_df.get("LN_NAME_SHORT", pd.Series("", index=order_df.index)).astype(str)
        + " "
        + order_df.get("LN_NAME", pd.Series("", index=order_df.index)).astype(str)
    ).str.lower()
    order_df["is_ring_line_proxy"] = ((order_df["LN_CODE"] == 609) | line_text.str.contains("кольцев", regex=False)).astype(int)

    left_distance = order_df["station_order_on_line_proxy"] - 1
    right_distance = order_df["line_n_stations"] - order_df["station_order_on_line_proxy"]
    distance = np.minimum(left_distance, right_distance).astype(float)
    distance = np.where(order_df["is_ring_line_proxy"].eq(1), 0.0, distance)
    order_df["distance_to_nearest_terminal_proxy"] = distance
    order_df["distance_to_nearest_terminal_norm"] = np.where(
        order_df["line_n_stations"] > 1,
        order_df["distance_to_nearest_terminal_proxy"] / (order_df["line_n_stations"] - 1),
        0.0,
    )
    order_df["is_terminal_proxy"] = (
        order_df["is_ring_line_proxy"].eq(0)
        & (
            order_df["station_order_on_line_proxy"].eq(1)
            | order_df["station_order_on_line_proxy"].eq(order_df["line_n_stations"])
        )
    ).astype(int)

    feature_cols = [
        "station_key", "station_order_on_line_proxy", "line_n_stations", "station_order_norm",
        "is_ring_line_proxy", "distance_to_nearest_terminal_proxy", "distance_to_nearest_terminal_norm",
        "is_terminal_proxy",
    ]
    out = out.drop(columns=[c for c in feature_cols[1:] if c in out.columns], errors="ignore")
    out = out.merge(order_df[feature_cols], on="station_key", how="left")
    return out.sort_values(["LN_CODE", "ST_CODE", "station_key"]).reset_index(drop=True)


def build_station_place_maps(
    cfg: MultiStationStage1Config,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build station-level and place-level maps from REF_PSG_PLACES.

    station_key is based on (LN_CODE, ST_CODE), not only on station name.
    This avoids collisions between stations with identical names on different lines.
    """
    data_dir = Path(cfg.data_dir)
    ref_path = _find_ref_places(data_dir, cfg.ref_places_glob)
    places = read_places_reference(ref_path, sep=cfg.pass_sep)
    places = places[places["TYPE_ID"].astype("Int64") == cfg.place_type_id].copy()
    if cfg.line_filter:
        needle = cfg.line_filter.lower()
        line_text = places.get("LN_NAME_SHORT", places["LN_NAME"]).astype(str).str.lower()
        places = places[line_text.str.contains(needle, regex=False)].copy()

    if cfg.classic_metro_only:
        allowed_lines = {int(x) for x in cfg.classic_metro_line_codes}
        line_code = pd.to_numeric(places["LN_CODE"], errors="coerce")
        places = places[line_code.isin(allowed_lines)].copy()

    places = places.dropna(subset=["PLACE_ID", "ST_CODE", "LN_CODE"]).copy()
    places["PLACE_ID"] = places["PLACE_ID"].astype(int)
    places["ST_CODE"] = places["ST_CODE"].astype(int)
    places["LN_CODE"] = places["LN_CODE"].astype(int)
    places["station_key"] = [
        _station_key(line_code, station_code)
        for line_code, station_code in zip(places["LN_CODE"], places["ST_CODE"])
    ]

    keep_place_cols = [
        "PLACE_ID", "PLACE_NAME", "TYPE_ID", "ST_CODE", "ST_NAME", "ST_NAME_SHORT",
        "LN_CODE", "LN_NAME", "LN_NAME_SHORT", "station_key",
    ]
    places = places[[c for c in keep_place_cols if c in places.columns]].sort_values(
        ["LN_CODE", "ST_CODE", "PLACE_ID"]
    ).reset_index(drop=True)

    station_rows = []
    for station_key, g in places.groupby("station_key", sort=True):
        first = g.iloc[0]
        station_rows.append({
            "station_key": station_key,
            "ST_CODE": int(first["ST_CODE"]),
            "ST_NAME": first["ST_NAME"],
            "ST_NAME_SHORT": first.get("ST_NAME_SHORT", first["ST_NAME"]),
            "LN_CODE": int(first["LN_CODE"]),
            "LN_NAME": first.get("LN_NAME", ""),
            "LN_NAME_SHORT": first.get("LN_NAME_SHORT", first.get("LN_NAME", "")),
            "n_place_ids": int(g["PLACE_ID"].nunique()),
            "place_ids": ",".join(str(int(x)) for x in sorted(g["PLACE_ID"].unique())),
        })
    stations = pd.DataFrame(station_rows).sort_values(["LN_CODE", "ST_CODE"]).reset_index(drop=True)
    stations = _add_station_structure_features(stations)

    if save:
        out_dir = data_dir
        suffix = _scope_name(cfg)
        places.to_csv(out_dir / f"station_place_map_{suffix}_places.csv", index=False)
        stations.to_csv(out_dir / f"station_place_map_{suffix}_stations.csv", index=False)

    return stations, places


def _pass_source_path(cfg: MultiStationStage1Config) -> Path:
    data_dir = Path(cfg.data_dir)
    return data_dir / cfg.pass_filename


def _default_station_buckets_path(cfg: MultiStationStage1Config) -> Path:
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cfg.station_buckets_parquet:
        return Path(cfg.station_buckets_parquet)
    scope = _scope_name(cfg)
    return cache_dir / f"multistation_{scope}_{cfg.bucket_minutes}m.parquet"


def build_multistation_bucket_parquet(
    cfg: MultiStationStage1Config,
    station_places: pd.DataFrame,
    output_parquet: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Aggregate PASS_ALL to station-level time buckets and cache as parquet."""
    output_path = Path(output_parquet) if output_parquet else _default_station_buckets_path(cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        return output_path

    pass_path = _pass_source_path(cfg)
    if not pass_path.exists():
        raise FileNotFoundError(
            f"PASS_ALL file is missing: {pass_path}. If parquet already exists, set cfg.station_buckets_parquet to it."
        )

    places_path = output_path.parent / f"{output_path.stem}_places_for_join.csv"
    station_places[["PLACE_ID", "station_key", "ST_CODE", "ST_NAME", "ST_NAME_SHORT", "LN_CODE", "LN_NAME", "LN_NAME_SHORT"]].to_csv(
        places_path, index=False
    )

    if duckdb is not None:
        con = duckdb.connect()
        con.execute(f"""
            COPY (
                SELECT
                    p.station_key,
                    p.ST_CODE,
                    p.ST_NAME,
                    p.ST_NAME_SHORT,
                    p.LN_CODE,
                    p.LN_NAME,
                    p.LN_NAME_SHORT,
                    time_bucket(INTERVAL '{cfg.bucket_minutes} minutes', CAST(t.TRAN_DATE AS TIMESTAMP)) AS {cfg.time_col},
                    COUNT(*)::DOUBLE AS count
                FROM read_csv_auto('{str(pass_path)}', delim='{cfg.pass_sep}') AS t
                INNER JOIN read_csv_auto('{str(places_path)}') AS p
                    ON CAST(t.PLACE_ID AS BIGINT) = CAST(p.PLACE_ID AS BIGINT)
                WHERE CAST(t.TRANSPORT_TYPE_ID AS BIGINT) = {int(cfg.transport_type_id)}
                GROUP BY 1,2,3,4,5,6,7,8
                ORDER BY 5,2,1,8
            ) TO '{str(output_path)}' (FORMAT PARQUET);
        """)
        con.close()
    else:                                                       
        usecols = ["TRAN_DATE", "TRANSPORT_TYPE_ID", "PLACE_ID"]
        chunks = []
        place_ids = set(station_places["PLACE_ID"].astype(int))
        for chunk in pd.read_csv(pass_path, sep=cfg.pass_sep, usecols=usecols, chunksize=2_000_000):
            chunk = chunk[chunk["TRANSPORT_TYPE_ID"].eq(cfg.transport_type_id) & chunk["PLACE_ID"].isin(place_ids)].copy()
            if chunk.empty:
                continue
            chunk["TRAN_DATE"] = pd.to_datetime(chunk["TRAN_DATE"], errors="coerce")
            chunk = chunk.dropna(subset=["TRAN_DATE"])
            chunk[cfg.time_col] = chunk["TRAN_DATE"].dt.floor(cfg.freq)
            chunks.append(chunk[["PLACE_ID", cfg.time_col]])
        if not chunks:
            raise ValueError("No passenger rows matched selected station places")
        events = pd.concat(chunks, ignore_index=True)
        events = events.merge(station_places, on="PLACE_ID", how="inner")
        bucketed = (
            events.groupby(["station_key", "ST_CODE", "ST_NAME", "ST_NAME_SHORT", "LN_CODE", "LN_NAME", "LN_NAME_SHORT", cfg.time_col])
                  .size()
                  .reset_index(name="count")
        )
        bucketed["count"] = bucketed["count"].astype(float)
        bucketed.to_parquet(output_path, index=False)

    return output_path


def load_or_build_station_buckets(cfg: MultiStationStage1Config, force_rebuild: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stations, places = build_station_place_maps(cfg, save=True)
    parquet_path = _default_station_buckets_path(cfg)
    if cfg.station_buckets_parquet:
        parquet_path = Path(cfg.station_buckets_parquet)
    if (not parquet_path.exists()) or force_rebuild:
        parquet_path = build_multistation_bucket_parquet(cfg, places, output_parquet=parquet_path, force=force_rebuild)
    buckets = pd.read_parquet(parquet_path)
    buckets[cfg.time_col] = pd.to_datetime(buckets[cfg.time_col], errors="coerce")
    buckets = buckets.dropna(subset=[cfg.time_col]).sort_values(["station_key", cfg.time_col]).reset_index(drop=True)

    static_station_cols = [
        "station_key", "n_place_ids", "station_order_on_line_proxy", "line_n_stations", "station_order_norm",
        "is_ring_line_proxy", "distance_to_nearest_terminal_proxy", "distance_to_nearest_terminal_norm",
        "is_terminal_proxy",
    ]
    static_station_cols = [c for c in static_station_cols if c in stations.columns]
    if len(static_station_cols) > 1:
        buckets = buckets.drop(columns=[c for c in static_station_cols[1:] if c in buckets.columns], errors="ignore")
        buckets = buckets.merge(stations[static_station_cols], on="station_key", how="left")
    return buckets, stations, places


def _complete_one_station(g: pd.DataFrame, cfg: MultiStationStage1Config) -> pd.DataFrame:
    g = g.sort_values(cfg.time_col).copy()
    full_idx = pd.date_range(g[cfg.time_col].min(), g[cfg.time_col].max(), freq=cfg.freq)
    meta_cols = [
        "station_key", "ST_CODE", "ST_NAME", "ST_NAME_SHORT", "LN_CODE", "LN_NAME", "LN_NAME_SHORT",
        "n_place_ids", "station_order_on_line_proxy", "line_n_stations", "station_order_norm",
        "is_ring_line_proxy", "distance_to_nearest_terminal_proxy", "distance_to_nearest_terminal_norm",
        "is_terminal_proxy",
    ]
    first = g.iloc[0]
    out = pd.DataFrame({cfg.time_col: full_idx})
    out = out.merge(g[[cfg.time_col, "count"]], on=cfg.time_col, how="left")
    out["count"] = out["count"].fillna(0.0).astype(float)
    for c in meta_cols:
        if c in g.columns:
            out[c] = first[c]
    return out



WEATHER_FEATURES = [
    "weather_temperature_2m",
    "weather_relative_humidity_2m",
    "weather_precipitation",
    "weather_rain",
    "weather_snowfall",
    "weather_snow_depth",
    "weather_weather_code",
    "weather_wind_speed_10m",
]


def _weather_csv_path(cfg: "MultiStationStage1Config") -> Path:
    if cfg.weather_csv:
        return Path(cfg.weather_csv)
    return Path(cfg.data_dir) / "weather_moscow_hourly_openmeteo_20250310_20250316.csv"


def _load_weather_hourly(cfg: "MultiStationStage1Config") -> pd.DataFrame | None:
    """Load a single Moscow-wide hourly weather table.

    Weather is treated as an exogenous city-level contextual covariate:
    all stations receive the same weather values for a given forecast anchor hour.
    The loader expects local Moscow timestamps compatible with the passenger data.
    """
    if not cfg.use_weather_features:
        return None

    path = _weather_csv_path(cfg)
    if not path.exists():
        warnings.warn(
            f"Weather features requested, but CSV was not found: {path}. "
            "The run will continue without weather features."
        )
        return None

    weather = pd.read_csv(path)
    time_col = None
    for candidate in ("time", "timestamp", "datetime", "weather_time"):
        if candidate in weather.columns:
            time_col = candidate
            break
    if time_col is None:
        raise ValueError(
            f"Weather CSV must contain one of time/timestamp/datetime/weather_time columns, got {list(weather.columns)}"
        )

    weather = weather.copy()
    weather["weather_hour"] = pd.to_datetime(weather[time_col], errors="coerce")
    if weather["weather_hour"].isna().all():
        raise ValueError(f"Could not parse weather timestamps from column {time_col!r} in {path}")
    weather["weather_hour"] = weather["weather_hour"].dt.floor("h")

    source_to_target = {
        "temperature_2m": "weather_temperature_2m",
        "relative_humidity_2m": "weather_relative_humidity_2m",
        "precipitation": "weather_precipitation",
        "rain": "weather_rain",
        "snowfall": "weather_snowfall",
        "snow_depth": "weather_snow_depth",
        "weather_code": "weather_weather_code",
        "wind_speed_10m": "weather_wind_speed_10m",
    }
    keep_cols = ["weather_hour"]
    for source, target in source_to_target.items():
        if target in weather.columns:
            keep_cols.append(target)
        elif source in weather.columns:
            weather[target] = pd.to_numeric(weather[source], errors="coerce")
            keep_cols.append(target)

    available = [c for c in WEATHER_FEATURES if c in keep_cols]
    if not available:
        warnings.warn(
            f"Weather CSV {path} was read, but none of the expected weather variables were found. "
            "The run will continue without weather features."
        )
        return None

    weather = weather[["weather_hour", *available]].copy()
    weather = weather.dropna(subset=["weather_hour"]).drop_duplicates("weather_hour", keep="last")
    weather = weather.sort_values("weather_hour").reset_index(drop=True)
    return weather


def _attach_weather_features(df: pd.DataFrame, cfg: "MultiStationStage1Config") -> pd.DataFrame:
    weather = _load_weather_hourly(cfg)
    if weather is None or weather.empty:
        return df

    out = df.copy()
    out["weather_hour"] = out[cfg.time_col].dt.floor("h")
    out = out.merge(weather, on="weather_hour", how="left", validate="many_to_one")
    return out


def make_multistation_supervised_frame(bucket_df: pd.DataFrame, cfg: MultiStationStage1Config) -> pd.DataFrame:
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
    df["roll_std_2"] = g.shift(1).rolling(2).std().reset_index(level=0, drop=True)
    df["roll_std_4"] = g.shift(1).rolling(4).std().reset_index(level=0, drop=True)

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


def _clip_per_station_day(seg: pd.DataFrame, cfg: MultiStationStage1Config) -> pd.DataFrame:
    if seg.empty:
        return seg.copy()
    parts = []
    for (_, _date), g in seg.groupby(["station_key", "date"], sort=False):
        cutoff = g[cfg.time_col].max() - pd.Timedelta(minutes=cfg.horizon_minutes)
        parts.append(g[g[cfg.time_col] <= cutoff].copy())
    return pd.concat(parts, ignore_index=True) if parts else seg.iloc[0:0].copy()


def _station_q80(train_df: pd.DataFrame, cfg: MultiStationStage1Config) -> pd.Series:
    q = train_df.dropna(subset=["target_h"]).groupby("station_key")["target_h"].quantile(cfg.highload_quantile)
    global_q = train_df["target_h"].quantile(cfg.highload_quantile)
    if q.empty:
        return pd.Series(dtype=float)
    return q.fillna(global_q)


def _attach_station_context(train_df: pd.DataFrame, pred_df: pd.DataFrame, cfg: MultiStationStage1Config):
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


def make_multistation_weekday_split(frame: pd.DataFrame, cfg: MultiStationStage1Config) -> dict:
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


BASE_FEATURES = [
                                                       
    "station_key_code", "station_id_raw", "line_id_raw", "station_name_code", "line_name_code",
    "n_place_ids", "line_n_stations", "station_order_on_line_proxy", "station_order_norm",
    "distance_to_nearest_terminal_proxy", "distance_to_nearest_terminal_norm", "is_terminal_proxy",
    "is_ring_line_proxy",
                                        
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_8", "lag_12", "lag_16", "lag_24",
    "lag_96", "lag_97", "roll_sum_4", "roll_mean_8", "hour_sin", "hour_cos",
]

RICH_FEATURES = [
                              
    "hl_daylag", "continuous_load_score", "local_slope_1", "local_slope_2",
    "daylag_deviation", "prev_count_ratio_daylag", "roll_std_2", "roll_std_4",
                                                                         
    "q80_station", "station_train_mean_target", "station_train_median_target", "station_train_q80_target",
    "station_train_q95_target", "station_train_max_target", "station_train_mean_count",
    "station_mean_target_vs_global", "station_q80_target_vs_global", "station_mean_count_vs_global",
                                             
    "weather_temperature_2m", "weather_relative_humidity_2m", "weather_precipitation", "weather_rain",
    "weather_snowfall", "weather_snow_depth", "weather_weather_code", "weather_wind_speed_10m",
]


def _available_features(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _xgb_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].replace([np.inf, -np.inf], np.nan).astype(float)


def _fit_predict_xgb(train_df: pd.DataFrame, pred_df: pd.DataFrame, feature_cols: list[str], params: dict):
    train_part = train_df.dropna(subset=["target_h"]).copy()
    if train_part.empty:
        raise ValueError("No train rows with non-missing target_h")
    model = xgb.XGBRegressor(**params)
    model.fit(_xgb_matrix(train_part, feature_cols), train_part["target_h"].astype(float))
    pred = model.predict(_xgb_matrix(pred_df, feature_cols))
    return np.asarray(pred, dtype=float), model


def _run_global_megaboost(
    cfg: MultiStationStage1Config,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
) -> dict:
    candidates = []
    y_val = val_df["target_h"].values.astype(float)
    for i, base_params in enumerate(cfg.boosting_param_grid, start=1):
        params = dict(base_params)
        params["random_state"] = cfg.random_state + i
        pred_val, _ = _fit_predict_xgb(train_df, val_df, feature_cols, params)
        candidates.append({
            "name": f"{label}__cfg{i}",
            "params": params,
            "val_pred": pred_val,
            "features": feature_cols,
            "metrics": calc_metrics_df(val_df, pred_val, cfg, mode="micro"),
        })
    val_table = pd.DataFrame([{"model": c["name"], **c["metrics"]} for c in candidates]).sort_values("MAE").reset_index(drop=True)
    selected_names = val_table.head(cfg.top_k_boost_models)["model"].tolist()
    selected = [c for c in candidates if c["name"] in selected_names]
    val_pred = np.mean(np.column_stack([c["val_pred"] for c in selected]), axis=1)

    test_preds = []
    models = []
    for c in selected:
        pred_test, model = _fit_predict_xgb(train_full_df, test_df, c["features"], c["params"])
        test_preds.append(pred_test)
        models.append(model)
    test_pred = np.mean(np.column_stack(test_preds), axis=1)
    return {
        "label": label,
        "features": feature_cols,
        "val_table": val_table,
        "selected_names": selected_names,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "test_metrics_micro": calc_metrics_df(test_df, test_pred, cfg, mode="micro"),
        "models": models,
    }



                                     

def _future_sum_from_count_path(count_path: np.ndarray, horizon: int) -> np.ndarray:
    count_path = np.asarray(count_path, dtype=float)
    if len(count_path) <= horizon:
        return np.empty(0, dtype=float)
    return np.array([count_path[i + 1:i + horizon + 1].sum() for i in range(len(count_path) - horizon)], dtype=float)


def _fit_predict_station_ets(train_count_df: pd.DataFrame, pred_count_df: pd.DataFrame, cfg: MultiStationStage1Config) -> np.ndarray:
    """Forecast a full count path for one station and aggregate to target_h anchors."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y_train = train_count_df["count"].astype(float).values
    if len(y_train) < 2 * cfg.day_lag or not np.isfinite(y_train).all():
        raise ValueError("Insufficient or non-finite training observations for ETS")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            y_train,
            trend="add",
            seasonal="add",
            seasonal_periods=cfg.day_lag,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True, use_brute=False)
        count_forecast = np.asarray(fit.forecast(len(pred_count_df)), dtype=float)
    count_forecast = np.maximum(count_forecast, 0.0)
    return _future_sum_from_count_path(count_forecast, horizon=cfg.horizon_steps)


def _run_local_ets(
    cfg: MultiStationStage1Config,
    train_tune_all: pd.DataFrame,
    val_all: pd.DataFrame,
    train_full_all: pd.DataFrame,
    test_all: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Fit ETS independently per station; use seasonal day-lag fallback if ETS fails."""
    val_pred = pd.Series(index=val_df.index, dtype=float)
    test_pred = pd.Series(index=test_df.index, dtype=float)
    fallback_stations_val: list[str] = []
    fallback_stations_test: list[str] = []

    for station_key, eval_station in val_df.groupby("station_key", sort=False):
        train_station = train_tune_all[train_tune_all["station_key"] == station_key].sort_values(cfg.time_col)
        pred_station = val_all[val_all["station_key"] == station_key].sort_values(cfg.time_col)
        eval_idx = eval_station.sort_values(cfg.time_col).index
        try:
            pred = _fit_predict_station_ets(train_station, pred_station, cfg)
            if len(pred) < len(eval_idx) or not np.isfinite(pred[:len(eval_idx)]).all():
                raise ValueError("ETS produced too few/non-finite validation predictions")
            val_pred.loc[eval_idx] = pred[:len(eval_idx)]
        except Exception:
            fallback_stations_val.append(str(station_key))
            fallback = eval_station.sort_values(cfg.time_col)["target_daylag"].fillna(eval_station["lag_96"]).fillna(eval_station["target_h"].mean())
            val_pred.loc[eval_idx] = fallback.values.astype(float)

    for station_key, eval_station in test_df.groupby("station_key", sort=False):
        train_station = train_full_all[train_full_all["station_key"] == station_key].sort_values(cfg.time_col)
        pred_station = test_all[test_all["station_key"] == station_key].sort_values(cfg.time_col)
        eval_idx = eval_station.sort_values(cfg.time_col).index
        try:
            pred = _fit_predict_station_ets(train_station, pred_station, cfg)
            if len(pred) < len(eval_idx) or not np.isfinite(pred[:len(eval_idx)]).all():
                raise ValueError("ETS produced too few/non-finite test predictions")
            test_pred.loc[eval_idx] = pred[:len(eval_idx)]
        except Exception:
            fallback_stations_test.append(str(station_key))
            fallback = eval_station.sort_values(cfg.time_col)["target_daylag"].fillna(eval_station["lag_96"]).fillna(eval_station["target_h"].mean())
            test_pred.loc[eval_idx] = fallback.values.astype(float)

    if val_pred.isna().any() or test_pred.isna().any():
        raise ValueError("Local ETS produced missing predictions after fallback")
    return {
        "val_pred": val_pred.loc[val_df.index].values.astype(float),
        "test_pred": test_pred.loc[test_df.index].values.astype(float),
        "n_val_fallback_stations": len(fallback_stations_val),
        "n_test_fallback_stations": len(fallback_stations_test),
        "val_fallback_stations": fallback_stations_val,
        "test_fallback_stations": fallback_stations_test,
    }


def _scaled_station_sequence_parts(
    train_all_df: pd.DataFrame,
    pred_all_df: pd.DataFrame,
    pred_anchor_df: pd.DataFrame,
    cfg: MultiStationStage1Config,
) -> dict:
    """Create pooled sequence windows with per-station standardization."""
    window = int(cfg.seq_window)
    h = int(cfg.horizon_steps)
    X_train_parts: list[np.ndarray] = []
    y_train_parts: list[np.ndarray] = []
    X_eval_parts: list[np.ndarray] = []
    y_eval_parts: list[np.ndarray] = []
    eval_indices: list[int] = []
    eval_means: list[float] = []
    eval_stds: list[float] = []

    train_groups = {k: g.sort_values(cfg.time_col) for k, g in train_all_df.groupby("station_key", sort=False)}
    pred_groups = {k: g.sort_values(cfg.time_col) for k, g in pred_all_df.groupby("station_key", sort=False)}

    for station_key, anchor_g in pred_anchor_df.groupby("station_key", sort=False):
        train_g = train_groups.get(station_key)
        pred_g = pred_groups.get(station_key)
        if train_g is None or pred_g is None:
            continue
        train_values = train_g["count"].astype(float).values
        pred_values = pred_g["count"].astype(float).values
        if len(train_values) < window + h + 1 or len(pred_values) < h + 1:
            continue
        mean_ = float(np.mean(train_values))
        std_ = float(np.std(train_values) + 1e-8)
        train_scaled = (train_values - mean_) / std_
        pred_scaled = (pred_values - mean_) / std_

        X_train_local = []
        y_train_local = []
        for anchor in range(window - 1, len(train_scaled) - h):
            X_train_local.append(train_scaled[anchor - window + 1: anchor + 1])
            y_train_local.append(float(train_scaled[anchor + 1: anchor + h + 1].sum()))
        if X_train_local:
            X_train_parts.append(np.asarray(X_train_local, dtype=np.float32))
            y_train_parts.append(np.asarray(y_train_local, dtype=np.float32))

        concat_scaled = np.concatenate([train_scaled, pred_scaled])
        pred_positions = {pd.Timestamp(ts): i for i, ts in enumerate(pd.to_datetime(pred_g[cfg.time_col]).tolist())}
        for idx, row in anchor_g.sort_values(cfg.time_col).iterrows():
            ts = pd.Timestamp(row[cfg.time_col])
            pred_pos = pred_positions.get(ts)
            if pred_pos is None:
                continue
            anchor = len(train_scaled) + int(pred_pos)
            if anchor - window + 1 < 0 or anchor + h >= len(concat_scaled):
                continue
            x = concat_scaled[anchor - window + 1: anchor + 1]
            y_scaled = float(concat_scaled[anchor + 1: anchor + h + 1].sum())
            X_eval_parts.append(np.asarray(x, dtype=np.float32))
            y_eval_parts.append(np.asarray(y_scaled, dtype=np.float32))
            eval_indices.append(int(idx))
            eval_means.append(mean_)
            eval_stds.append(std_)

    if not X_train_parts or not X_eval_parts:
        raise ValueError("No sequence windows were constructed")
    return {
        "X_train": np.concatenate(X_train_parts, axis=0),
        "y_train": np.concatenate(y_train_parts, axis=0),
        "X_eval": np.stack(X_eval_parts).astype(np.float32),
        "y_eval": np.asarray(y_eval_parts, dtype=np.float32),
        "eval_indices": np.asarray(eval_indices, dtype=int),
        "eval_means": np.asarray(eval_means, dtype=float),
        "eval_stds": np.asarray(eval_stds, dtype=float),
    }


def _sequence_eval_pred_to_frame_order(payload: dict, raw_pred: np.ndarray, eval_df: pd.DataFrame, cfg: MultiStationStage1Config) -> np.ndarray:
    pred_unscaled = raw_pred * payload["eval_stds"] + cfg.horizon_steps * payload["eval_means"]
    pred_unscaled = np.maximum(pred_unscaled, 0.0)
    series = pd.Series(pred_unscaled, index=payload["eval_indices"], dtype=float)
    missing = eval_df.index.difference(series.index)
    if len(missing):
                                                                                               
        fallback = eval_df.loc[missing, "target_daylag"].fillna(eval_df.loc[missing, "lag_96"]).fillna(eval_df["target_h"].mean())
        series.loc[missing] = fallback.values.astype(float)
    return series.loc[eval_df.index].values.astype(float)


def _predict_scaled_sequence_model(model, X_scaled: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    import torch
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_scaled), max(1, int(batch_size))):
            xb = torch.tensor(X_scaled[start:start + int(batch_size)], dtype=torch.float32).unsqueeze(-1).to(device)
            preds.append(model(xb).detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds, axis=0).astype(float) if preds else np.empty(0, dtype=float)


def _run_global_sequence_models(cfg: MultiStationStage1Config, split: dict) -> dict[str, dict]:
    """Fit pooled global sequence experts on per-station standardized windows."""
    import torch
    from torch.utils.data import DataLoader
    from metroflow.models.sequence import SeqDataset, LSTMRegressor, TransformerLiteRegressor, train_seq_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    tune = _scaled_station_sequence_parts(split["train_tune_all"], split["val_all"], split["val_df"], cfg)
    full = _scaled_station_sequence_parts(split["train_full_all"], split["test_all"], split["test_df"], cfg)

    train_loader = DataLoader(
        SeqDataset(tune["X_train"], tune["y_train"]),
        batch_size=min(int(cfg.seq_batch_size), max(1, len(tune["X_train"]))),
        shuffle=True,
    )
    val_loader = DataLoader(
        SeqDataset(tune["X_eval"], tune["y_eval"]),
        batch_size=min(int(cfg.seq_batch_size), max(1, len(tune["X_eval"]))),
        shuffle=False,
    )
    full_loader = DataLoader(
        SeqDataset(full["X_train"], full["y_train"]),
        batch_size=min(int(cfg.seq_batch_size), max(1, len(full["X_train"]))),
        shuffle=True,
    )

    out: dict[str, dict] = {}

    if cfg.run_global_lstm:
        lstm = LSTMRegressor(hidden_size=32, num_layers=1, dropout=0.0)
        lstm, val_loss, best_epoch = train_seq_model(
            lstm, train_loader, device=device, val_loader=val_loader,
            epochs=int(cfg.lstm_max_epochs), patience=int(cfg.seq_patience),
        )
        val_raw = _predict_scaled_sequence_model(lstm, tune["X_eval"], cfg.seq_batch_size, device)
        val_pred = _sequence_eval_pred_to_frame_order(tune, val_raw, split["val_df"], cfg)

        lstm_full = LSTMRegressor(hidden_size=32, num_layers=1, dropout=0.0)
        lstm_full, _, _ = train_seq_model(
            lstm_full, full_loader, device=device, val_loader=None,
            epochs=max(1, int(best_epoch)), patience=int(cfg.seq_patience),
        )
        test_raw = _predict_scaled_sequence_model(lstm_full, full["X_eval"], cfg.seq_batch_size, device)
        test_pred = _sequence_eval_pred_to_frame_order(full, test_raw, split["test_df"], cfg)
        out["Global LSTM"] = {
            "val_pred": val_pred,
            "test_pred": test_pred,
            "best_epoch": int(best_epoch),
            "val_loss_scaled": float(val_loss) if np.isfinite(val_loss) else np.nan,
            "n_train_windows_tune": int(len(tune["X_train"])),
            "n_train_windows_full": int(len(full["X_train"])),
        }

    if cfg.run_global_transformer:
        tf = TransformerLiteRegressor(d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1)
        tf, val_loss, best_epoch = train_seq_model(
            tf, train_loader, device=device, val_loader=val_loader,
            epochs=int(cfg.transformer_max_epochs), patience=int(cfg.seq_patience),
        )
        val_raw = _predict_scaled_sequence_model(tf, tune["X_eval"], cfg.seq_batch_size, device)
        val_pred = _sequence_eval_pred_to_frame_order(tune, val_raw, split["val_df"], cfg)

        tf_full = TransformerLiteRegressor(d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1)
        tf_full, _, _ = train_seq_model(
            tf_full, full_loader, device=device, val_loader=None,
            epochs=max(1, int(best_epoch)), patience=int(cfg.seq_patience),
        )
        test_raw = _predict_scaled_sequence_model(tf_full, full["X_eval"], cfg.seq_batch_size, device)
        test_pred = _sequence_eval_pred_to_frame_order(full, test_raw, split["test_df"], cfg)
        out["Global Transformer-lite"] = {
            "val_pred": val_pred,
            "test_pred": test_pred,
            "best_epoch": int(best_epoch),
            "val_loss_scaled": float(val_loss) if np.isfinite(val_loss) else np.nan,
            "n_train_windows_tune": int(len(tune["X_train"])),
            "n_train_windows_full": int(len(full["X_train"])),
        }

    return out

def seasonal_daylag_predictions(val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    val_pred = val_df["target_daylag"].fillna(val_df["lag_96"]).fillna(val_df["target_h"].mean()).values.astype(float)
    test_pred = test_df["target_daylag"].fillna(test_df["lag_96"]).fillna(test_df["target_h"].mean()).values.astype(float)
    return {"val_pred": val_pred, "test_pred": test_pred}


def _positive_blend(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    reg = LinearRegression(fit_intercept=False, positive=True)
    reg.fit(X, y)
    w = np.asarray(reg.coef_, dtype=float).clip(min=0)
    if not np.isfinite(w).all() or w.sum() <= 0:
        return np.ones(X.shape[1], dtype=float) / X.shape[1]
    return w / w.sum()


def _context_blend(val_pred_table: pd.DataFrame, test_pred_table: pd.DataFrame, model_cols: list[str]) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]:
    y = val_pred_table["y_true"].values.astype(float)
    X = val_pred_table[model_cols].values.astype(float)
    ctx = val_pred_table["hl_daylag"].values.astype(int)
    global_w = _positive_blend(X, y)
    by_ctx: dict[int, np.ndarray] = {}
    for v in [0, 1]:
        mask = ctx == v
        if mask.sum() >= max(30, len(model_cols) * 5):
            by_ctx[v] = _positive_blend(X[mask], y[mask])
        else:
            by_ctx[v] = global_w.copy()
    X_test = test_pred_table[model_cols].values.astype(float)
    ctx_test = test_pred_table["hl_daylag"].values.astype(int)
    pred = np.zeros(len(test_pred_table), dtype=float)
    for i in range(len(test_pred_table)):
        pred[i] = X_test[i] @ by_ctx[int(ctx_test[i])]
    return global_w, by_ctx, pred


def _build_meta_tables(split: dict, preds: dict[str, dict], cfg: MultiStationStage1Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_df = split["val_df"]
    test_df = split["test_df"]
    base_cols = [
        cfg.time_col, "station_key", "ST_CODE", "ST_NAME", "ST_NAME_SHORT", "LN_CODE", "LN_NAME", "LN_NAME_SHORT",
        "target_h", "hl_daylag", "hl_actual", "q80_station", "hour_sin", "hour_cos", "continuous_load_score",
        "local_slope_1", "local_slope_2", "daylag_deviation", "prev_count_ratio_daylag", "roll_std_2", "roll_std_4",
        "station_key_code", "station_id_raw", "line_id_raw", "station_name_code", "line_name_code",
        "n_place_ids", "line_n_stations", "station_order_on_line_proxy", "station_order_norm",
        "distance_to_nearest_terminal_proxy", "distance_to_nearest_terminal_norm", "is_terminal_proxy", "is_ring_line_proxy",
        "station_train_mean_target", "station_train_median_target", "station_train_q80_target", "station_train_q95_target",
        "station_train_max_target", "station_train_mean_count", "station_mean_target_vs_global",
        "station_q80_target_vs_global", "station_mean_count_vs_global", "station_code", "line_code",
    ]
    val = val_df[[c for c in base_cols if c in val_df.columns]].copy().rename(columns={"target_h": "y_true"})
    test = test_df[[c for c in base_cols if c in test_df.columns]].copy().rename(columns={"target_h": "y_true"})
    for name, payload in preds.items():
        val[name] = payload["val_pred"]
        test[name] = payload["test_pred"]
    return val, test


def _run_meta_models(cfg: MultiStationStage1Config, val_pred_table: pd.DataFrame, test_pred_table: pd.DataFrame) -> dict:
    model_cols = [
        c for c in [
            "Seasonal daylag", "ETS local", "MegaBoost base", "MegaBoost rich",
            "Global LSTM", "Global Transformer-lite",
        ]
        if c in val_pred_table.columns
    ]
    y_val = val_pred_table["y_true"].values.astype(float)

    w_static = _positive_blend(val_pred_table[model_cols].values.astype(float), y_val)
    pred_static = test_pred_table[model_cols].values.astype(float) @ w_static

    w_global, w_by_ctx, pred_context = _context_blend(val_pred_table, test_pred_table, model_cols)

    out = {
        "model_cols": model_cols,
        "static_weights": dict(zip(model_cols, w_static)),
        "context_global_weights": dict(zip(model_cols, w_global)),
        "context_weights_by_hl_daylag": {int(k): dict(zip(model_cols, v)) for k, v in w_by_ctx.items()},
        "pred_static": pred_static,
        "pred_context": pred_context,
    }

    if cfg.run_residual_xgb:
        val_metrics = {c: calc_metrics_df(val_pred_table.rename(columns={"y_true": "target_h"}), val_pred_table[c].values, cfg, mode="micro") for c in model_cols}
        primary = pd.DataFrame(val_metrics).T.sort_values("MAE").index[0]
        resid_val = y_val - val_pred_table[primary].values.astype(float)
        X_val = _meta_feature_matrix(val_pred_table, model_cols, primary)
        X_test = _meta_feature_matrix(test_pred_table, model_cols, primary)

        ridge = Ridge(alpha=10.0)
        ridge.fit(X_val, resid_val)
        pred_resid_linear = ridge.predict(X_test)
        lo, hi = np.quantile(resid_val, [0.05, 0.95])
        pred_resid_linear = np.clip(pred_resid_linear, lo, hi)

        params = dict(objective="reg:absoluteerror", n_estimators=80, learning_rate=0.03, max_depth=2,
                      min_child_weight=20, subsample=0.85, colsample_bytree=0.85,
                      reg_lambda=20.0, reg_alpha=4.0, tree_method="hist", random_state=cfg.random_state)
        rxgb = xgb.XGBRegressor(**params)
        rxgb.fit(X_val, resid_val)
        pred_resid_xgb = rxgb.predict(X_test)
        lo, hi = np.quantile(resid_val, [0.10, 0.90])
        pred_resid_xgb = np.clip(pred_resid_xgb, lo, hi)

        primary_test = test_pred_table[primary].values.astype(float)
        out.update({
            "primary_model": primary,
            "pred_residual_linear": primary_test + pred_resid_linear,
            "pred_residual_xgb": primary_test + pred_resid_xgb,
            "residual_linear_coef": pd.DataFrame({"feature": X_val.columns, "coef": ridge.coef_}),
            "residual_xgb_importance": pd.DataFrame({"feature": X_val.columns, "importance": rxgb.feature_importances_}),
        })
    return out


def _meta_feature_matrix(df: pd.DataFrame, model_cols: list[str], primary_col: str) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in model_cols:
        X[f"pred_{c}"] = df[c].values.astype(float)
    for c in model_cols:
        if c != primary_col:
            X[f"diff_{c}_minus_{primary_col}"] = df[c].values.astype(float) - df[primary_col].values.astype(float)
            X[f"absdiff_{c}_{primary_col}"] = np.abs(df[c].values.astype(float) - df[primary_col].values.astype(float))
    mat = df[model_cols].values.astype(float)
    X["pred_mean"] = mat.mean(axis=1)
    X["pred_std"] = mat.std(axis=1)
    X["pred_range"] = mat.max(axis=1) - mat.min(axis=1)
    for c in [
        "hl_daylag", "hour_sin", "hour_cos", "continuous_load_score", "local_slope_1", "local_slope_2",
        "daylag_deviation", "prev_count_ratio_daylag", "roll_std_2", "roll_std_4", "q80_station",
        "station_train_mean_target", "station_train_median_target", "station_train_q80_target", "station_train_q95_target",
        "station_train_max_target", "station_train_mean_count", "station_mean_target_vs_global",
        "station_q80_target_vs_global", "station_mean_count_vs_global",
        "station_key_code", "station_id_raw", "line_id_raw", "station_name_code", "line_name_code",
        "n_place_ids", "line_n_stations", "station_order_on_line_proxy", "station_order_norm",
        "distance_to_nearest_terminal_proxy", "distance_to_nearest_terminal_norm", "is_terminal_proxy", "is_ring_line_proxy",
        "weather_temperature_2m", "weather_relative_humidity_2m", "weather_precipitation", "weather_rain",
        "weather_snowfall", "weather_snow_depth", "weather_weather_code", "weather_wind_speed_10m",
        "station_code", "line_code",
    ]:
        if c in df.columns:
            X[c] = df[c].values
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def calc_metrics_df(df: pd.DataFrame, y_pred: np.ndarray, cfg: MultiStationStage1Config, mode: str = "micro") -> dict:
    work = df[["station_key", "target_h", "hl_actual", "station_train_mean_target"]].copy()
    work["pred"] = np.asarray(y_pred, dtype=float)
    if mode == "micro":
        return _calc_metric_values(work, cfg)
    if mode == "macro":
        rows = []
        for _, g in work.groupby("station_key"):
            if len(g) > 0:
                rows.append(_calc_metric_values(g, cfg))
        return pd.DataFrame(rows).mean(numeric_only=True).to_dict()
    raise ValueError("mode must be 'micro' or 'macro'")


def _calc_metric_values(work: pd.DataFrame, cfg: MultiStationStage1Config) -> dict:
    y = work["target_h"].astype(float).values
    pred = work["pred"].astype(float).values
    ae = np.abs(y - pred)
    mae = float(np.mean(ae))
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    denom = float(np.sum(np.abs(y)))
    wmape = float(np.sum(ae) / denom) if denom > 0 else np.nan
    station_mean = work["station_train_mean_target"].astype(float).replace(0, np.nan).fillna(np.nanmean(y) if len(y) else 1.0).values
    weights = np.power(np.maximum(y, 0) / np.maximum(station_mean, 1e-9), cfg.load_weight_gamma)
    lw_mae = float(np.mean(ae * weights))
    hl_mask = work["hl_actual"].fillna(0).astype(int).values.astype(bool)
    if hl_mask.any():
        hl_mae = float(np.mean(ae[hl_mask]))
        hl_under = float(np.mean(np.maximum(y[hl_mask] - pred[hl_mask], 0)))
        hl_under_rate = float(np.mean(pred[hl_mask] < y[hl_mask]))
    else:
        hl_mae = np.nan
        hl_under = np.nan
        hl_under_rate = np.nan
    return {
        "MAE": mae,
        "RMSE": rmse,
        "wMAPE": wmape,
        "LW-MAE": lw_mae,
        "HL-MAE": hl_mae,
        "HL-underMAE": hl_under,
        "HL-underRate": hl_under_rate,
    }


def evaluate_predictions(test_pred_table: pd.DataFrame, model_cols: list[str], cfg: MultiStationStage1Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_df = test_pred_table.rename(columns={"y_true": "target_h"}).copy()
    micro = {m: calc_metrics_df(eval_df, eval_df[m].values, cfg, mode="micro") for m in model_cols}
    macro = {m: calc_metrics_df(eval_df, eval_df[m].values, cfg, mode="macro") for m in model_cols}
    micro_table = pd.DataFrame(micro).T.sort_values("LW-MAE")
    macro_table = pd.DataFrame(macro).T.sort_values("LW-MAE")
    return micro_table, macro_table


def station_bootstrap_ci(
    test_pred_table: pd.DataFrame,
    model_cols: list[str],
    cfg: MultiStationStage1Config,
    baseline_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.random_state)
    stations = np.array(sorted(test_pred_table["station_key"].unique()))
    baseline_cols = baseline_cols or ["MegaBoost base", "MegaBoost rich"]
    metrics = ["MAE", "RMSE", "wMAPE", "LW-MAE", "HL-MAE", "HL-underMAE", "HL-underRate"]

    point_micro, point_macro = evaluate_predictions(test_pred_table, model_cols, cfg)
    metric_samples = {("micro", model, metric): [] for model in model_cols for metric in metrics}
    metric_samples.update({("macro", model, metric): [] for model in model_cols for metric in metrics})
    delta_samples: dict[tuple[str, str, str, str], list[float]] = {}
    for mode in ["micro", "macro"]:
        for baseline in baseline_cols:
            if baseline not in model_cols:
                continue
            for model in model_cols:
                if model == baseline:
                    continue
                for metric in metrics:
                    delta_samples[(mode, model, baseline, metric)] = []

    for _ in range(cfg.n_bootstrap):
        sampled = rng.choice(stations, size=len(stations), replace=True)
        sample_df = pd.concat([test_pred_table[test_pred_table["station_key"] == s] for s in sampled], ignore_index=True)
        micro, macro = evaluate_predictions(sample_df, model_cols, cfg)
        for mode, table in [("micro", micro), ("macro", macro)]:
            for model in model_cols:
                for metric in metrics:
                    metric_samples[(mode, model, metric)].append(float(table.loc[model, metric]))
            for baseline in baseline_cols:
                if baseline not in model_cols:
                    continue
                for model in model_cols:
                    if model == baseline:
                        continue
                    for metric in metrics:
                                                                                                   
                        delta_samples[(mode, model, baseline, metric)].append(float(table.loc[baseline, metric] - table.loc[model, metric]))

    metric_rows = []
    point_tables = {"micro": point_micro, "macro": point_macro}
    for (mode, model, metric), vals in metric_samples.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        metric_rows.append({
            "mode": mode,
            "model": model,
            "metric": metric,
            "estimate": float(point_tables[mode].loc[model, metric]),
            "ci_low": float(np.quantile(arr, 0.025)) if len(arr) else np.nan,
            "ci_high": float(np.quantile(arr, 0.975)) if len(arr) else np.nan,
        })

    delta_rows = []
    for (mode, model, baseline, metric), vals in delta_samples.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        estimate = float(point_tables[mode].loc[baseline, metric] - point_tables[mode].loc[model, metric])
        low = float(np.quantile(arr, 0.025)) if len(arr) else np.nan
        high = float(np.quantile(arr, 0.975)) if len(arr) else np.nan
        delta_rows.append({
            "mode": mode,
            "model": model,
            "baseline": baseline,
            "metric": metric,
            "delta_estimate": estimate,
            "ci_low": low,
            "ci_high": high,
            "significant_95": bool(low > 0 or high < 0) if np.isfinite(low) and np.isfinite(high) else False,
        })
    return {
        "metric_ci": pd.DataFrame(metric_rows).sort_values(["mode", "metric", "estimate"]),
        "delta_ci": pd.DataFrame(delta_rows).sort_values(["mode", "baseline", "metric", "delta_estimate"], ascending=[True, True, True, False]),
    }


def run_multistation_stage1(cfg: MultiStationStage1Config, force_rebuild_buckets: bool = False) -> dict:
    buckets, stations, places = load_or_build_station_buckets(cfg, force_rebuild=force_rebuild_buckets)
    frame = make_multistation_supervised_frame(buckets, cfg)
    split = make_multistation_weekday_split(frame, cfg)

    base_features = _available_features(split["train_tune"], BASE_FEATURES)
    rich_features = _available_features(split["train_tune"], BASE_FEATURES + RICH_FEATURES)

    seasonal = seasonal_daylag_predictions(split["val_df"], split["test_df"])
    mega_base = _run_global_megaboost(cfg, split["train_tune"], split["val_df"], split["train_full"], split["test_df"], base_features, "MegaBoost base")
    mega_rich = _run_global_megaboost(cfg, split["train_tune"], split["val_df"], split["train_full"], split["test_df"], rich_features, "MegaBoost rich")

    preds = {
        "Seasonal daylag": seasonal,
        "MegaBoost base": mega_base,
        "MegaBoost rich": mega_rich,
    }
    local_ets = None
    if cfg.run_local_ets:
        local_ets = _run_local_ets(
            cfg, split["train_tune_all"], split["val_all"], split["train_full_all"], split["test_all"],
            split["val_df"], split["test_df"],
        )
        preds["ETS local"] = local_ets

    sequence_models: dict[str, dict] = {}
    if cfg.run_global_lstm or cfg.run_global_transformer:
        sequence_models = _run_global_sequence_models(cfg, split)
        preds.update(sequence_models)
    val_pred_table, test_pred_table = _build_meta_tables(split, preds, cfg)
    meta = _run_meta_models(cfg, val_pred_table, test_pred_table)
    test_pred_table["Meta static"] = meta["pred_static"]
    test_pred_table["Meta context"] = meta["pred_context"]
    if cfg.run_residual_xgb:
        test_pred_table["Residual Linear"] = meta["pred_residual_linear"]
        test_pred_table["Residual XGB"] = meta["pred_residual_xgb"]

    model_cols = [
        c for c in [
            "Seasonal daylag", "ETS local", "MegaBoost base", "MegaBoost rich",
            "Global LSTM", "Global Transformer-lite",
            "Meta static", "Meta context", "Residual Linear", "Residual XGB",
        ]
        if c in test_pred_table.columns
    ]
    micro_table, macro_table = evaluate_predictions(test_pred_table, model_cols, cfg)
    bootstrap = station_bootstrap_ci(test_pred_table, model_cols, cfg, baseline_cols=["Seasonal daylag", "ETS local", "MegaBoost base", "MegaBoost rich"])

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
        "local_ets": local_ets,
        "sequence_models": sequence_models,
        "megaboost_base": mega_base,
        "megaboost_rich": mega_rich,
        "val_pred_table": val_pred_table,
        "test_pred_table": test_pred_table,
        "meta": meta,
        "model_cols": model_cols,
        "micro_table": micro_table,
        "macro_table": macro_table,
        "bootstrap": bootstrap,
    }


def save_multistation_stage1_result(result: dict, output_dir: str | Path | None = None) -> dict[str, str]:
    cfg: MultiStationStage1Config = result["config"]
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
    (out / "config.json").write_text(json.dumps(cfg.__dict__, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    paths.update({
        "micro_metrics": str(out / "micro_metrics.csv"),
        "macro_metrics": str(out / "macro_metrics.csv"),
        "station_bootstrap_metric_ci": str(out / "station_bootstrap_metric_ci.csv"),
        "station_bootstrap_delta_ci": str(out / "station_bootstrap_delta_ci.csv"),
        "test_predictions": str(out / "test_predictions.csv"),
        "validation_predictions": str(out / "validation_predictions.csv"),
        "stations_used": str(out / "stations_used.csv"),
    })
    return paths


                                              
                                                                                 
                                                                                 
                                

NORMALIZED_MAGNITUDE_PREFIXES = ("lag_", "roll_sum_", "roll_mean_")
NORMALIZED_MAGNITUDE_COLS = {"target_daylag"}


def _station_scale_series(df: pd.DataFrame) -> pd.Series:
    """Positive station-level scale used for normalized boosting."""
    candidates = [
        "station_train_q80_target",
        "q80_station",
        "station_train_mean_target",
        "station_train_median_target",
        "station_train_mean_count",
    ]
    scale = None
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").astype(float)
            scale = s if scale is None else scale.fillna(s)
    if scale is None:
        scale = pd.Series(np.nan, index=df.index, dtype=float)
    fallback = float(np.nanmedian(scale.values)) if np.isfinite(scale.values).any() else 1.0
    scale = scale.replace([np.inf, -np.inf], np.nan).fillna(fallback).clip(lower=1.0)
    return scale.astype(float)


def _normalize_boost_frame(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Create normalized lag/rolling features and normalized target for global boosting.

    The goal is to let one global model learn profile shape instead of wasting capacity on
    station-specific passenger-flow scale.
    """
    out = df.copy()
    scale = _station_scale_series(out)
    out["__station_scale"] = scale.values
    normalized_features: list[str] = []
    for c in feature_cols:
        if c not in out.columns:
            continue
        is_magnitude = c in NORMALIZED_MAGNITUDE_COLS or c.startswith(NORMALIZED_MAGNITUDE_PREFIXES)
        if is_magnitude:
            nc = f"norm__{c}"
            out[nc] = pd.to_numeric(out[c], errors="coerce").astype(float) / scale
            normalized_features.append(nc)
        else:
            normalized_features.append(c)
    out["target_norm"] = pd.to_numeric(out["target_h"], errors="coerce").astype(float) / scale
    return out, normalized_features


def _fit_predict_xgb_normalized(train_df: pd.DataFrame, pred_df: pd.DataFrame, feature_cols: list[str], params: dict) -> tuple[np.ndarray, object]:
    train_n, norm_features = _normalize_boost_frame(train_df, feature_cols)
    pred_n, _ = _normalize_boost_frame(pred_df, feature_cols)
    train_part = train_n.dropna(subset=["target_norm"]).copy()
    model = xgb.XGBRegressor(**params)
    model.fit(_xgb_matrix(train_part, norm_features), train_part["target_norm"].astype(float))
    pred_norm = np.asarray(model.predict(_xgb_matrix(pred_n, norm_features)), dtype=float)
    pred = np.maximum(pred_norm * pred_n["__station_scale"].values.astype(float), 0.0)
    return pred, model


def _run_global_megaboost_normalized(
    cfg: MultiStationStage1Config,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
) -> dict:
    candidates = []
    for i, base_params in enumerate(cfg.boosting_param_grid, start=1):
        params = dict(base_params)
        params["random_state"] = cfg.random_state + 1000 + i
        pred_val, _ = _fit_predict_xgb_normalized(train_df, val_df, feature_cols, params)
        candidates.append({
            "name": f"{label}__cfg{i}",
            "params": params,
            "val_pred": pred_val,
            "features": feature_cols,
            "metrics": calc_metrics_df(val_df, pred_val, cfg, mode="micro"),
        })
    val_table = pd.DataFrame([{"model": c["name"], **c["metrics"]} for c in candidates]).sort_values("LW-MAE").reset_index(drop=True)
    selected_names = val_table.head(cfg.top_k_boost_models)["model"].tolist()
    selected = [c for c in candidates if c["name"] in selected_names]
    val_pred = np.mean(np.column_stack([c["val_pred"] for c in selected]), axis=1)

    test_preds = []
    models = []
    for c in selected:
        pred_test, model = _fit_predict_xgb_normalized(train_full_df, test_df, c["features"], c["params"])
        test_preds.append(pred_test)
        models.append(model)
    test_pred = np.mean(np.column_stack(test_preds), axis=1)
    return {
        "label": label,
        "features": feature_cols,
        "val_table": val_table,
        "selected_names": selected_names,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "test_metrics_micro": calc_metrics_df(test_df, test_pred, cfg, mode="micro"),
        "models": models,
        "normalized_target": True,
    }


def _meta_lw_sample_weight(df: pd.DataFrame, cfg: MultiStationStage1Config) -> np.ndarray:
    y = pd.to_numeric(df["y_true"], errors="coerce").astype(float).values
    if "station_train_mean_target" in df.columns:
        denom = pd.to_numeric(df["station_train_mean_target"], errors="coerce").astype(float).values
    else:
        denom = np.full(len(df), np.nanmedian(y) if np.isfinite(y).any() else 1.0, dtype=float)
    denom = np.where(np.isfinite(denom) & (denom > 0), denom, np.nanmedian(denom[np.isfinite(denom) & (denom > 0)]) if np.any(np.isfinite(denom) & (denom > 0)) else 1.0)
    weights = np.power(np.maximum(y, 0.0) / np.maximum(denom, 1e-9), cfg.load_weight_gamma)
                                                                                             
    weights = np.clip(weights, 0.25, 8.0)
    weights = np.where(np.isfinite(weights), weights, 1.0)
    return weights.astype(float)


def _positive_blend_weighted(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> np.ndarray:
    reg = LinearRegression(fit_intercept=False, positive=True)
    if sample_weight is not None:
        reg.fit(X, y, sample_weight=sample_weight)
    else:
        reg.fit(X, y)
    w = np.asarray(reg.coef_, dtype=float).clip(min=0)
    if not np.isfinite(w).all() or w.sum() <= 0:
        return np.ones(X.shape[1], dtype=float) / X.shape[1]
    return w / w.sum()


def _context_blend_weighted(
    val_pred_table: pd.DataFrame,
    test_pred_table: pd.DataFrame,
    model_cols: list[str],
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]:
    y = val_pred_table["y_true"].values.astype(float)
    X = val_pred_table[model_cols].values.astype(float)
    ctx = val_pred_table["hl_daylag"].fillna(0).values.astype(int)
    global_w = _positive_blend_weighted(X, y, sample_weight=sample_weight)
    by_ctx: dict[int, np.ndarray] = {}
    for v in [0, 1]:
        mask = ctx == v
        if mask.sum() >= max(30, len(model_cols) * 5):
            sw = sample_weight[mask] if sample_weight is not None else None
            by_ctx[v] = _positive_blend_weighted(X[mask], y[mask], sample_weight=sw)
        else:
            by_ctx[v] = global_w.copy()
    X_test = test_pred_table[model_cols].values.astype(float)
    ctx_test = test_pred_table["hl_daylag"].fillna(0).values.astype(int)
    pred = np.zeros(len(test_pred_table), dtype=float)
    for i in range(len(test_pred_table)):
        pred[i] = X_test[i] @ by_ctx[int(ctx_test[i])]
    return global_w, by_ctx, pred


def _validation_metric_for_model(val_pred_table: pd.DataFrame, model: str, cfg: MultiStationStage1Config, metric: str = "LW-MAE") -> float:
    tmp = val_pred_table.rename(columns={"y_true": "target_h"}).copy()
    return float(calc_metrics_df(tmp, tmp[model].values, cfg, mode="micro")[metric])


def _select_meta_model_pool(
    val_pred_table: pd.DataFrame,
    candidate_cols: list[str],
    cfg: MultiStationStage1Config,
    baseline: str = "Seasonal daylag",
    tolerance: float = 1.05,
) -> tuple[list[str], pd.DataFrame]:
    rows = []
    for c in candidate_cols:
        if c not in val_pred_table.columns:
            continue
        rows.append({
            "model": c,
            "val_MAE": _validation_metric_for_model(val_pred_table, c, cfg, "MAE"),
            "val_LW-MAE": _validation_metric_for_model(val_pred_table, c, cfg, "LW-MAE"),
            "val_HL-MAE": _validation_metric_for_model(val_pred_table, c, cfg, "HL-MAE"),
        })
    table = pd.DataFrame(rows).sort_values("val_LW-MAE").reset_index(drop=True)
    if table.empty:
        return [], table
    base_lw = float(table.loc[table["model"].eq(baseline), "val_LW-MAE"].iloc[0]) if baseline in set(table["model"]) else float(table["val_LW-MAE"].min())
    selected = table.loc[table["val_LW-MAE"] <= tolerance * base_lw, "model"].tolist()
    if baseline in candidate_cols and baseline not in selected:
        selected.append(baseline)
                                                                                           
    if len(selected) < min(2, len(table)):
        selected = table.head(min(3, len(table)))["model"].tolist()
    return selected, table


def _run_meta_models_filtered_weighted(cfg: MultiStationStage1Config, val_pred_table: pd.DataFrame, test_pred_table: pd.DataFrame) -> dict:
    preferred_order = [
        "Seasonal daylag",
        "ETS local",
        "MegaBoost base norm",
        "MegaBoost rich norm",
        "MegaBoost base",
        "MegaBoost rich",
        "Global LSTM",
        "Global Transformer-lite",
    ]
    candidate_cols = [c for c in preferred_order if c in val_pred_table.columns]
    model_cols, val_selection_table = _select_meta_model_pool(val_pred_table, candidate_cols, cfg, tolerance=1.05)
    if not model_cols:
        raise ValueError("No base models available for meta layer")
    y_val = val_pred_table["y_true"].values.astype(float)
    sw_lw = _meta_lw_sample_weight(val_pred_table, cfg)

                               
    w_static = _positive_blend_weighted(val_pred_table[model_cols].values.astype(float), y_val)
    pred_static = test_pred_table[model_cols].values.astype(float) @ w_static
    w_global, w_by_ctx, pred_context = _context_blend_weighted(val_pred_table, test_pred_table, model_cols)

                       
    w_static_lw = _positive_blend_weighted(val_pred_table[model_cols].values.astype(float), y_val, sample_weight=sw_lw)
    pred_static_lw = test_pred_table[model_cols].values.astype(float) @ w_static_lw
    w_global_lw, w_by_ctx_lw, pred_context_lw = _context_blend_weighted(val_pred_table, test_pred_table, model_cols, sample_weight=sw_lw)

    out = {
        "model_cols": model_cols,
        "candidate_cols": candidate_cols,
        "validation_selection_table": val_selection_table,
        "static_weights": dict(zip(model_cols, w_static)),
        "context_global_weights": dict(zip(model_cols, w_global)),
        "context_weights_by_hl_daylag": {int(k): dict(zip(model_cols, v)) for k, v in w_by_ctx.items()},
        "static_lw_weights": dict(zip(model_cols, w_static_lw)),
        "context_lw_global_weights": dict(zip(model_cols, w_global_lw)),
        "context_lw_weights_by_hl_daylag": {int(k): dict(zip(model_cols, v)) for k, v in w_by_ctx_lw.items()},
        "pred_static": pred_static,
        "pred_context": pred_context,
        "pred_static_lw": pred_static_lw,
        "pred_context_lw": pred_context_lw,
    }

    if cfg.run_residual_xgb:
                                                                                                                         
        val_scores = {c: _validation_metric_for_model(val_pred_table, c, cfg, "LW-MAE") for c in model_cols}
        primary = pd.Series(val_scores).sort_values().index[0]
        resid_val = y_val - val_pred_table[primary].values.astype(float)
        X_val = _meta_feature_matrix(val_pred_table, model_cols, primary)
        X_test = _meta_feature_matrix(test_pred_table, model_cols, primary)

        ridge = Ridge(alpha=10.0)
        ridge.fit(X_val, resid_val)
        pred_resid_linear = ridge.predict(X_test)
        lo, hi = np.quantile(resid_val, [0.05, 0.95])
        pred_resid_linear = np.clip(pred_resid_linear, lo, hi)

        ridge_lw = Ridge(alpha=10.0)
        ridge_lw.fit(X_val, resid_val, sample_weight=sw_lw)
        pred_resid_linear_lw = ridge_lw.predict(X_test)
        pred_resid_linear_lw = np.clip(pred_resid_linear_lw, lo, hi)

        params = dict(objective="reg:absoluteerror", n_estimators=100, learning_rate=0.03, max_depth=2,
                      min_child_weight=20, subsample=0.85, colsample_bytree=0.85,
                      reg_lambda=20.0, reg_alpha=4.0, tree_method="hist", random_state=cfg.random_state)
        rxgb = xgb.XGBRegressor(**params)
        rxgb.fit(X_val, resid_val)
        pred_resid_xgb = rxgb.predict(X_test)
        lo2, hi2 = np.quantile(resid_val, [0.10, 0.90])
        pred_resid_xgb = np.clip(pred_resid_xgb, lo2, hi2)

        rxgb_lw = xgb.XGBRegressor(**{**params, "random_state": cfg.random_state + 333})
        rxgb_lw.fit(X_val, resid_val, sample_weight=sw_lw)
        pred_resid_xgb_lw = rxgb_lw.predict(X_test)
        pred_resid_xgb_lw = np.clip(pred_resid_xgb_lw, lo2, hi2)

        primary_test = test_pred_table[primary].values.astype(float)
        out.update({
            "primary_model": primary,
            "pred_residual_linear": primary_test + pred_resid_linear,
            "pred_residual_xgb": primary_test + pred_resid_xgb,
            "pred_residual_linear_lw": primary_test + pred_resid_linear_lw,
            "pred_residual_xgb_lw": primary_test + pred_resid_xgb_lw,
            "residual_linear_coef": pd.DataFrame({"feature": X_val.columns, "coef": ridge.coef_}),
            "residual_linear_lw_coef": pd.DataFrame({"feature": X_val.columns, "coef": ridge_lw.coef_}),
            "residual_xgb_importance": pd.DataFrame({"feature": X_val.columns, "importance": rxgb.feature_importances_}),
            "residual_xgb_lw_importance": pd.DataFrame({"feature": X_val.columns, "importance": rxgb_lw.feature_importances_}),
        })
    return out


def run_multistation_stage1(cfg: MultiStationStage1Config, force_rebuild_buckets: bool = False) -> dict:
    buckets, stations, places = load_or_build_station_buckets(cfg, force_rebuild=force_rebuild_buckets)
    frame = make_multistation_supervised_frame(buckets, cfg)
    split = make_multistation_weekday_split(frame, cfg)

    base_features = _available_features(split["train_tune"], BASE_FEATURES)
    rich_features = _available_features(split["train_tune"], BASE_FEATURES + RICH_FEATURES)

    seasonal = seasonal_daylag_predictions(split["val_df"], split["test_df"])
    mega_base = _run_global_megaboost(cfg, split["train_tune"], split["val_df"], split["train_full"], split["test_df"], base_features, "MegaBoost base")
    mega_rich = _run_global_megaboost(cfg, split["train_tune"], split["val_df"], split["train_full"], split["test_df"], rich_features, "MegaBoost rich")
    mega_base_norm = _run_global_megaboost_normalized(cfg, split["train_tune"], split["val_df"], split["train_full"], split["test_df"], base_features, "MegaBoost base norm")
    mega_rich_norm = _run_global_megaboost_normalized(cfg, split["train_tune"], split["val_df"], split["train_full"], split["test_df"], rich_features, "MegaBoost rich norm")

    preds = {
        "Seasonal daylag": seasonal,
        "MegaBoost base": mega_base,
        "MegaBoost rich": mega_rich,
        "MegaBoost base norm": mega_base_norm,
        "MegaBoost rich norm": mega_rich_norm,
    }

    ets_result = None
    if getattr(cfg, "run_local_ets", False):
        ets_result = _run_local_ets(cfg, split["train_tune_all"], split["val_all"], split["train_full_all"], split["test_all"], split["val_df"], split["test_df"])
        preds["ETS local"] = ets_result

    seq_results: dict[str, dict] = {}
    if getattr(cfg, "run_global_lstm", False) or getattr(cfg, "run_global_transformer", False):
        seq_all = _run_global_sequence_models(cfg, split)
        if getattr(cfg, "run_global_lstm", False) and "Global LSTM" in seq_all:
            preds["Global LSTM"] = seq_all["Global LSTM"]
            seq_results["Global LSTM"] = seq_all["Global LSTM"]
        if getattr(cfg, "run_global_transformer", False) and "Global Transformer-lite" in seq_all:
            preds["Global Transformer-lite"] = seq_all["Global Transformer-lite"]
            seq_results["Global Transformer-lite"] = seq_all["Global Transformer-lite"]

    val_pred_table, test_pred_table = _build_meta_tables(split, preds, cfg)
    meta = _run_meta_models_filtered_weighted(cfg, val_pred_table, test_pred_table)
    test_pred_table["Meta static"] = meta["pred_static"]
    test_pred_table["Meta context"] = meta["pred_context"]
    test_pred_table["Meta static LW"] = meta["pred_static_lw"]
    test_pred_table["Meta context LW"] = meta["pred_context_lw"]
    if cfg.run_residual_xgb:
        test_pred_table["Residual Linear"] = meta["pred_residual_linear"]
        test_pred_table["Residual XGB"] = meta["pred_residual_xgb"]
        test_pred_table["Residual Linear LW"] = meta["pred_residual_linear_lw"]
        test_pred_table["Residual XGB LW"] = meta["pred_residual_xgb_lw"]

    model_order = [
        "Seasonal daylag", "ETS local", "MegaBoost base", "MegaBoost rich",
        "MegaBoost base norm", "MegaBoost rich norm", "Global LSTM", "Global Transformer-lite",
        "Meta static", "Meta context", "Meta static LW", "Meta context LW",
        "Residual Linear", "Residual XGB", "Residual Linear LW", "Residual XGB LW",
    ]
    model_cols = [c for c in model_order if c in test_pred_table.columns]
    micro_table, macro_table = evaluate_predictions(test_pred_table, model_cols, cfg)
    baseline_cols = [c for c in ["Seasonal daylag", "ETS local", "MegaBoost base", "MegaBoost rich", "MegaBoost base norm", "MegaBoost rich norm"] if c in model_cols]
    bootstrap = station_bootstrap_ci(test_pred_table, model_cols, cfg, baseline_cols=baseline_cols)

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
        "ets_local": ets_result,
        "megaboost_base": mega_base,
        "megaboost_rich": mega_rich,
        "megaboost_base_norm": mega_base_norm,
        "megaboost_rich_norm": mega_rich_norm,
        "sequence": seq_results,
        "val_pred_table": val_pred_table,
        "test_pred_table": test_pred_table,
        "meta": meta,
        "model_cols": model_cols,
        "micro_table": micro_table,
        "macro_table": macro_table,
        "bootstrap": bootstrap,
    }
