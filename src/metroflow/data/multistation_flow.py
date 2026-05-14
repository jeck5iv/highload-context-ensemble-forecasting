from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import duckdb
except ModuleNotFoundError:                    
    duckdb = None

from metroflow.data.access import build_full_parquet, _needs_rebuild


def _ensure_full_pass_parquet(
    *,
    data_dir: Path,
    cache_dir: Path,
    pass_filename: str,
    pass_sep: str = ';',
) -> Path:
    csv_path = data_dir / pass_filename
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    full_parquet = cache_dir / f'{csv_path.stem}_full.parquet'
    if _needs_rebuild(full_parquet, csv_path):
        build_full_parquet(csv_path, full_parquet, sep=pass_sep)
    return full_parquet


def _bucket_expr(bucket_minutes: int) -> str:
    return f"time_bucket(INTERVAL '{int(bucket_minutes)} minutes', CAST(p.TRAN_DATE AS TIMESTAMP))"


def build_multistation_bucket_parquet(
    *,
    data_dir: str | Path = './data',
    cache_dir: str | Path = './data/cache',
    pass_filename: str = 'PASS_ALL_202503242210.csv',
    pass_sep: str = ';',
    station_places_csv: str | Path = './data/station_place_map_sokolnicheskaya_places.csv',
    output_parquet: str | Path = './data/cache/multistation_sokolnicheskaya_15m.parquet',
    bucket_minutes: int = 15,
    transport_type_id: int | None = 1,
    validation_modes: list[int] | None = None,
    force: bool = False,
) -> Path:
    """Aggregate raw PASS_ALL events to station-level count buckets.

    Output schema includes station metadata columns, bucket timestamp, and count.
    The station mapping is one row per PLACE_ID, so several entrances are summed
    into the same station_key.
    """
    if duckdb is None:
        raise RuntimeError('duckdb is required for efficient multi-station aggregation')

    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)
    station_places_csv = Path(station_places_csv)
    output_parquet = Path(output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    if not station_places_csv.exists():
        raise FileNotFoundError(station_places_csv)

    full_parquet = _ensure_full_pass_parquet(
        data_dir=data_dir,
        cache_dir=cache_dir,
        pass_filename=pass_filename,
        pass_sep=pass_sep,
    )

    if output_parquet.exists() and not force and not _needs_rebuild(output_parquet, full_parquet, station_places_csv):
        return output_parquet

    where_parts = []
    if transport_type_id is not None:
        where_parts.append(f'p.TRANSPORT_TYPE_ID = {int(transport_type_id)}')
    if validation_modes:
        vm = ', '.join(str(int(x)) for x in validation_modes)
        where_parts.append(f'p.VALIDATION_MODE IN ({vm})')
    where_sql = ' AND '.join(where_parts) if where_parts else 'TRUE'
    bucket_col = f'bucket_{int(bucket_minutes)}m'

    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT
                m.station_key,
                CAST(m.ST_CODE AS BIGINT) AS ST_CODE,
                m.ST_NAME,
                m.ST_NAME_SHORT,
                CAST(m.LN_CODE AS BIGINT) AS LN_CODE,
                m.LN_NAME,
                m.LN_NAME_SHORT,
                m.station_label,
                {_bucket_expr(bucket_minutes)} AS {bucket_col},
                COUNT(*)::DOUBLE AS count
            FROM read_parquet('{str(full_parquet)}') AS p
            INNER JOIN read_csv_auto('{str(station_places_csv)}') AS m
                ON CAST(p.PLACE_ID AS BIGINT) = CAST(m.PLACE_ID AS BIGINT)
            WHERE {where_sql}
              AND p.TRAN_DATE IS NOT NULL
            GROUP BY
                m.station_key,
                m.ST_CODE,
                m.ST_NAME,
                m.ST_NAME_SHORT,
                m.LN_CODE,
                m.LN_NAME,
                m.LN_NAME_SHORT,
                m.station_label,
                {bucket_col}
            ORDER BY m.LN_CODE, m.ST_CODE, {bucket_col}
        ) TO '{str(output_parquet)}' (FORMAT PARQUET);
    """)
    con.close()
    return output_parquet


def load_multistation_bucket_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    bucket_cols = [c for c in df.columns if c.startswith('bucket_')]
    for c in bucket_cols:
        df[c] = pd.to_datetime(df[c])
    return df.sort_values(['station_key', bucket_cols[0] if bucket_cols else 'station_key']).reset_index(drop=True)
