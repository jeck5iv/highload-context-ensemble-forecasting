from __future__ import annotations

from ..common import *
from ..config import (
    DEFAULT_CLASSIC_METRO_LINE_CODES,
    _scope_name,
    MultiStationForecastConfig,
)

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
    cfg: MultiStationForecastConfig,
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

__all__ = [
    "_find_ref_places",
    "_clean_col_name",
    "read_places_reference",
    "_station_key",
    "_add_station_structure_features",
    "build_station_place_maps",
]
