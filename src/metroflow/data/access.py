
from __future__ import annotations

from pathlib import Path
import hashlib
try:
    import duckdb
except ModuleNotFoundError:
    duckdb = None
import pandas as pd

from metroflow.config import ExperimentConfig


KEEP_COLS = ['TRAN_DATE', 'TRANSPORT_TYPE_ID', 'PLACE_ID', 'VALIDATION_MODE', 'APP_ID']


def _needs_rebuild(target: Path, *sources: Path) -> bool:
    if (not target.exists()) or target.stat().st_size == 0:
        return True
    t = target.stat().st_mtime
    return any(s.exists() and s.stat().st_mtime > t for s in sources)


def build_full_parquet(csv_path: Path, full_parquet: Path, sep: str = ';') -> None:
    if duckdb is None:
        import pandas as pd
        df = pd.read_csv(csv_path, sep=sep)
        df.to_parquet(full_parquet, index=False)
        return
    con = duckdb.connect()
    con.execute(f"""
        COPY (
          SELECT * FROM read_csv_auto('{str(csv_path)}', delim='{sep}')
        ) TO '{str(full_parquet)}' (FORMAT PARQUET);
    """)
    con.close()


def build_slim_filtered_parquet(source_parquet: Path, target_parquet: Path, filters: list[str]) -> None:
    cols_sql = ', '.join(KEEP_COLS)
    where_sql = ' AND '.join(f'({f})' for f in filters) if filters else 'TRUE'
    if duckdb is None:
        import pandas as pd
        df = pd.read_parquet(source_parquet)[KEEP_COLS]
        for f in filters:
            if f.startswith('TRANSPORT_TYPE_ID = '):
                val = int(f.split('=')[1].strip())
                df = df[df['TRANSPORT_TYPE_ID'] == val]
            elif f.startswith('PLACE_ID IN ('):
                vals = [int(x.strip()) for x in f.split('(')[1].split(')')[0].split(',')]
                df = df[df['PLACE_ID'].isin(vals)]
            elif f.startswith('VALIDATION_MODE IN ('):
                vals = [int(x.strip()) for x in f.split('(')[1].split(')')[0].split(',')]
                df = df[df['VALIDATION_MODE'].isin(vals)]
        df.to_parquet(target_parquet, index=False)
        return
    con = duckdb.connect()
    con.execute(f"""
        COPY (
          SELECT {cols_sql}
          FROM read_parquet('{str(source_parquet)}')
          WHERE {where_sql}
        ) TO '{str(target_parquet)}' (FORMAT PARQUET);
    """)
    con.close()


def load_parquet_to_pandas(parquet_path: Path) -> pd.DataFrame:
    if duckdb is None:
        return pd.read_parquet(parquet_path)
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{str(parquet_path)}')").df()
    con.close()
    return df


def make_station_filters(cfg: ExperimentConfig) -> list[str]:
    filters: list[str] = []
    if cfg.data.transport_type_id is not None:
        filters.append(f'TRANSPORT_TYPE_ID = {cfg.data.transport_type_id}')
    if cfg.data.place_ids:
        place_sql = ', '.join(str(x) for x in cfg.data.place_ids)
        filters.append(f'PLACE_ID IN ({place_sql})')
    if cfg.data.validation_modes:
        vm_sql = ', '.join(str(x) for x in cfg.data.validation_modes)
        filters.append(f'VALIDATION_MODE IN ({vm_sql})')
    return filters



def _filter_signature(filters: list[str]) -> str:
    payload = ' AND '.join(filters) if filters else 'all'
    return hashlib.md5(payload.encode('utf-8')).hexdigest()[:10]


def load_pass_data(cfg: ExperimentConfig) -> pd.DataFrame:
    data_dir = Path(cfg.data.data_dir)
    cache_dir = Path(cfg.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / cfg.data.pass_filename
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    full_parquet = cache_dir / f'{csv_path.stem}_full.parquet'

    if _needs_rebuild(full_parquet, csv_path):
        build_full_parquet(csv_path, full_parquet, sep=cfg.data.pass_sep)

    filters = make_station_filters(cfg)
    filt_parquet = cache_dir / f'{csv_path.stem}_filtered_{_filter_signature(filters)}.parquet'
    if _needs_rebuild(filt_parquet, full_parquet):
        build_slim_filtered_parquet(full_parquet, filt_parquet, filters)

    df = load_parquet_to_pandas(filt_parquet)
    df['TRAN_DATE'] = pd.to_datetime(df['TRAN_DATE'], errors='coerce')
    df = df.dropna(subset=['TRAN_DATE']).sort_values('TRAN_DATE').reset_index(drop=True)
    return df


def load_station_metadata(cfg: ExperimentConfig) -> pd.DataFrame | None:
    if not cfg.data.station_metadata_path:
        return None
    path = Path(cfg.data.station_metadata_path)
    if not path.exists():
        return None
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    if path.suffix.lower() in {'.parquet', '.pq'}:
        return pd.read_parquet(path)
    raise ValueError(f'Unsupported station metadata format: {path}')
