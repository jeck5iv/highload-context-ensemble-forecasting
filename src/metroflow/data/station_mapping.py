from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


REF_PLACES_FILENAME = 'REF_PSG_PLACES_202503251822.csv'

PLACE_COLUMNS = [
    'PLACE_ID',
    'CODE',
    'NAME',
    'TYPE_ID',
    'IS_PARKING',
    'IS_TEST',
    'ST_CODE',
    'ST_NAME',
    'ST_NAME_SHORT',
    'LN_CODE',
    'LN_NAME',
    'LN_NAME_SHORT',
    'station_key',
    'station_label',
]

STATION_COLUMNS = [
    'station_key',
    'ST_CODE',
    'ST_NAME',
    'ST_NAME_SHORT',
    'LN_CODE',
    'LN_NAME',
    'LN_NAME_SHORT',
    'station_label',
    'n_place_ids',
    'place_ids',
]


def _clean_int_like(value) -> str:
    if pd.isna(value):
        return 'NA'
    try:
        return str(int(float(value)))
    except Exception:
        return re.sub(r'\W+', '_', str(value)).strip('_') or 'NA'


def load_ref_places(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep=';', encoding='utf-8')


def build_place_station_map(
    ref_df: pd.DataFrame,
    *,
    transport_type_id: int | None = 1,
    line_short: str | None = None,
    include_test: bool = False,
) -> pd.DataFrame:
    """Return one row per passenger place/entrance mapped to a station-line key.

    The key intentionally uses both LN_CODE and ST_CODE, not station name alone.
    This avoids mixing stations with equal names or transfer stations represented
    separately on different lines.
    """
    df = ref_df.copy()

    if transport_type_id is not None:
        df = df[df['TYPE_ID'] == transport_type_id]
    if not include_test and 'IS_TEST' in df.columns:
        df = df[df['IS_TEST'].fillna(0).astype(int) == 0]
    if line_short is not None:
        df = df[df['LN_NAME_SHORT'].fillna('').astype(str).eq(line_short)]

    df = df[df['PLACE_ID'].notna() & df['ST_CODE'].notna() & df['LN_CODE'].notna()].copy()

    ln_codes = df['LN_CODE'].map(_clean_int_like)
    st_codes = df['ST_CODE'].map(_clean_int_like)
    df['station_key'] = 'ln' + ln_codes.str.zfill(5) + '_st' + st_codes.str.zfill(5)
    df['station_label'] = (
        df['ST_NAME'].fillna(df['ST_NAME_SHORT']).astype(str).str.strip()
        + ' | '
        + df['LN_NAME_SHORT'].fillna(df['LN_NAME']).astype(str).str.strip()
    )

    for col in ['PLACE_ID', 'CODE', 'TYPE_ID', 'IS_PARKING', 'IS_TEST']:
        if col in df.columns:
            df[col] = df[col].astype('Int64')

    out = df[PLACE_COLUMNS].sort_values(['LN_CODE', 'ST_CODE', 'PLACE_ID']).reset_index(drop=True)
    return out


def build_station_summary(place_map: pd.DataFrame) -> pd.DataFrame:
    """Return one row per station-line key with all PLACE_ID values listed."""
    if place_map.empty:
        return pd.DataFrame(columns=STATION_COLUMNS)

    def _join_place_ids(s: pd.Series) -> str:
        vals = sorted(int(x) for x in s.dropna().astype(int).unique())
        return ','.join(map(str, vals))

    agg = (
        place_map
        .groupby('station_key', as_index=False)
        .agg(
            ST_CODE=('ST_CODE', 'first'),
            ST_NAME=('ST_NAME', 'first'),
            ST_NAME_SHORT=('ST_NAME_SHORT', 'first'),
            LN_CODE=('LN_CODE', 'first'),
            LN_NAME=('LN_NAME', 'first'),
            LN_NAME_SHORT=('LN_NAME_SHORT', 'first'),
            station_label=('station_label', 'first'),
            n_place_ids=('PLACE_ID', 'nunique'),
            place_ids=('PLACE_ID', _join_place_ids),
        )
        .sort_values(['LN_CODE', 'ST_CODE'])
        .reset_index(drop=True)
    )
    return agg[STATION_COLUMNS]


def build_and_save_station_maps(
    *,
    data_dir: str | Path = './data',
    ref_filename: str = REF_PLACES_FILENAME,
    line_short: str = 'Сокольническая',
    transport_type_id: int = 1,
    include_test: bool = False,
) -> dict[str, Path]:
    """Build reviewable station-place maps for all TYPE_ID=1 places and one line."""
    data_dir = Path(data_dir)
    ref_df = load_ref_places(data_dir / ref_filename)

    metro_places = build_place_station_map(
        ref_df,
        transport_type_id=transport_type_id,
        include_test=include_test,
    )
    metro_stations = build_station_summary(metro_places)

    slug = 'sokolnicheskaya' if line_short == 'Сокольническая' else re.sub(r'\W+', '_', line_short.lower()).strip('_')
    line_places = build_place_station_map(
        ref_df,
        transport_type_id=transport_type_id,
        line_short=line_short,
        include_test=include_test,
    )
    line_stations = build_station_summary(line_places)

    outputs = {
        'metro_places': data_dir / 'station_place_map_metro_places.csv',
        'metro_stations': data_dir / 'station_place_map_metro_stations.csv',
        f'{slug}_places': data_dir / f'station_place_map_{slug}_places.csv',
        f'{slug}_stations': data_dir / f'station_place_map_{slug}_stations.csv',
    }

    metro_places.to_csv(outputs['metro_places'], index=False)
    metro_stations.to_csv(outputs['metro_stations'], index=False)
    line_places.to_csv(outputs[f'{slug}_places'], index=False)
    line_stations.to_csv(outputs[f'{slug}_stations'], index=False)
    return outputs
