from __future__ import annotations

import argparse
from pathlib import Path

from metroflow.multistation.stage1 import MultiStationStage1Config, run_multistation_stage1, save_multistation_stage1_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-station Stage 1 global boosting/meta experiment.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--pass-filename", default="PASS_ALL_202503242210.csv")
    parser.add_argument("--cache-dir", default="./data/cache")
    parser.add_argument("--output-dir", default="./outputs/multistation_stage1_classic_metro_15m_60m_opday3")
    parser.add_argument("--station-buckets-parquet", default=None)
    parser.add_argument("--line-filter", default=None)
    parser.add_argument("--include-non-metro", action="store_true", help="Use all TYPE_ID=1 objects instead of the official Moscow Metro line whitelist.")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--force-rebuild-buckets", action="store_true")
    args = parser.parse_args()

    cfg = MultiStationStage1Config(
        data_dir=args.data_dir,
        pass_filename=args.pass_filename,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        station_buckets_parquet=args.station_buckets_parquet,
        line_filter=args.line_filter,
        classic_metro_only=not args.include_non_metro,
        n_bootstrap=args.n_bootstrap,
    )
    result = run_multistation_stage1(cfg, force_rebuild_buckets=args.force_rebuild_buckets)
    paths = save_multistation_stage1_result(result)
    print("Saved:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print("\nMicro metrics:")
    print(result["micro_table"])
    print("\nMacro metrics:")
    print(result["macro_table"])


if __name__ == "__main__":
    main()
