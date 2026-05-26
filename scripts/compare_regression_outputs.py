from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

NUMERIC_FILES = [
    "micro_metrics.csv",
    "macro_metrics.csv",
    "station_bootstrap_delta_ci.csv",
    "station_bootstrap_metric_ci.csv",
    "bootstrap_reference_models.csv",
]
EXACT_FILES = ["stations_used.csv"]
OPTIONAL_SCHEMA_FILES = ["stage_timings.csv"]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in df.columns if df[c].dtype == object or c.lower() in {"model", "baseline", "metric", "mode"}]
    if "Unnamed: 0" in df.columns:
        sort_cols = ["Unnamed: 0"] + [c for c in sort_cols if c != "Unnamed: 0"]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort")
    return df.reset_index(drop=True)


def _compare_numeric(old_dir: Path, new_dir: Path, filename: str, rtol: float, atol: float) -> None:
    old = _sort_frame(_read(old_dir / filename))
    new = _sort_frame(_read(new_dir / filename))
    pd.testing.assert_frame_equal(old, new, check_dtype=False, check_exact=False, rtol=rtol, atol=atol)


def _compare_exact(old_dir: Path, new_dir: Path, filename: str) -> None:
    old = _sort_frame(_read(old_dir / filename))
    new = _sort_frame(_read(new_dir / filename))
    pd.testing.assert_frame_equal(old, new, check_dtype=False, check_exact=True)


def compare_outputs(old_dir: Path, new_dir: Path, rtol: float = 1e-9, atol: float = 1e-9) -> None:
    for filename in NUMERIC_FILES:
        _compare_numeric(old_dir, new_dir, filename, rtol=rtol, atol=atol)
    for filename in EXACT_FILES:
        _compare_exact(old_dir, new_dir, filename)
    for filename in OPTIONAL_SCHEMA_FILES:
        old_path = old_dir / filename
        new_path = new_dir / filename
        if old_path.exists() and new_path.exists():
            old_cols = list(pd.read_csv(old_path, nrows=0).columns)
            new_cols = list(pd.read_csv(new_path, nrows=0).columns)
            if old_cols != new_cols:
                raise AssertionError(f"{filename}: schema mismatch: {old_cols} != {new_cols}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-dir", required=True, type=Path)
    parser.add_argument("--new-dir", required=True, type=Path)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-9)
    args = parser.parse_args()
    compare_outputs(args.old_dir, args.new_dir, rtol=args.rtol, atol=args.atol)
    print("regression outputs match")


if __name__ == "__main__":
    main()
