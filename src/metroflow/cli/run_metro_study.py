from __future__ import annotations

import argparse

from metroflow.studies.metro_short_history_forecasting import (
    MultiStationForecastConfig,
    run_multistation_forecasting_experiment,
    save_multistation_forecasting_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-station forecasting experiment.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--pass-filename", default="PASS_ALL_202503242210.csv")
    parser.add_argument("--cache-dir", default="./data/cache")
    parser.add_argument("--output-dir", default="./outputs/multistation_forecasting_classic_metro_15m_60m_opday3")
    parser.add_argument("--station-buckets-parquet", default=None)
    parser.add_argument("--line-filter", default=None)
    parser.add_argument("--include-non-metro", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--force-rebuild-buckets", action="store_true")
    args = parser.parse_args()

    cfg = MultiStationForecastConfig(
        data_dir=args.data_dir,
        pass_filename=args.pass_filename,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        station_buckets_parquet=args.station_buckets_parquet,
        line_filter=args.line_filter,
        classic_metro_only=not args.include_non_metro,
        n_bootstrap=args.n_bootstrap,
    )
    result = run_multistation_forecasting_experiment(cfg, force_rebuild_buckets=args.force_rebuild_buckets)
    paths = save_multistation_forecasting_result(result)
    print("Saved:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
