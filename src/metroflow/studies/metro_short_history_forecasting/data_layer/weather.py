from __future__ import annotations

from ..common import *

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

def _weather_csv_path(cfg: "MultiStationForecastConfig") -> Path:
    if cfg.weather_csv:
        return Path(cfg.weather_csv)
    data_dir = Path(cfg.data_dir)
    preferred = data_dir / "weather_hourly_openmeteo_20250310_20250316.csv"
    if preferred.exists():
        return preferred
    matches = sorted(data_dir.glob("weather_*_hourly_openmeteo_*.csv"))
    if matches:
        return matches[-1]
    return preferred

def _load_weather_hourly(cfg: "MultiStationForecastConfig") -> pd.DataFrame | None:
    """Load a single network-wide hourly weather table.

    Weather is treated as an exogenous network-level contextual covariate:
    all stations receive the same weather values for a given forecast anchor hour.
    The loader expects local timestamps compatible with the passenger data.
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

def _attach_weather_features(df: pd.DataFrame, cfg: "MultiStationForecastConfig") -> pd.DataFrame:
    weather = _load_weather_hourly(cfg)
    if weather is None or weather.empty:
        return df

    out = df.copy()
    out["weather_hour"] = out[cfg.time_col].dt.floor("h")
    out = out.merge(weather, on="weather_hour", how="left", validate="many_to_one")
    return out

__all__ = [
    "WEATHER_FEATURES",
    "_weather_csv_path",
    "_load_weather_hourly",
    "_attach_weather_features",
]
