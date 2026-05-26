from __future__ import annotations

from ..common import *
from ..config import (
    DEFAULT_CLASSIC_METRO_LINE_CODES,
    _scope_name,
    MultiStationForecastConfig,
)
from .places import (
    _find_ref_places,
    _clean_col_name,
    read_places_reference,
    _station_key,
    _add_station_structure_features,
    build_station_place_maps,
)

def _pass_source_path(cfg: MultiStationForecastConfig) -> Path:
    data_dir = Path(cfg.data_dir)
    return data_dir / cfg.pass_filename

def _default_station_buckets_path(cfg: MultiStationForecastConfig) -> Path:
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cfg.station_buckets_parquet:
        return Path(cfg.station_buckets_parquet)
    scope = _scope_name(cfg)
    return cache_dir / f"multistation_{scope}_{cfg.bucket_minutes}m.parquet"

def build_multistation_bucket_parquet(
    cfg: MultiStationForecastConfig,
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

def load_or_build_station_buckets(cfg: MultiStationForecastConfig, force_rebuild: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

def _complete_one_station(g: pd.DataFrame, cfg: MultiStationForecastConfig) -> pd.DataFrame:
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

__all__ = [
    "_pass_source_path",
    "_default_station_buckets_path",
    "build_multistation_bucket_parquet",
    "load_or_build_station_buckets",
    "_complete_one_station",
]
