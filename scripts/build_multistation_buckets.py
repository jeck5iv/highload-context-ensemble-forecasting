from __future__ import annotations

import argparse
from pathlib import Path

from metroflow.data.multistation_flow import build_multistation_bucket_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description='Aggregate PASS_ALL events to station-level time buckets.')
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--cache-dir', default='./data/cache')
    parser.add_argument('--pass-filename', default='PASS_ALL_202503242210.csv')
    parser.add_argument('--pass-sep', default=';')
    parser.add_argument('--station-places-csv', default='./data/station_place_map_sokolnicheskaya_places.csv')
    parser.add_argument('--output-parquet', default='./data/cache/multistation_sokolnicheskaya_15m.parquet')
    parser.add_argument('--bucket-minutes', type=int, default=15)
    parser.add_argument('--transport-type-id', type=int, default=1)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    out = build_multistation_bucket_parquet(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        pass_filename=args.pass_filename,
        pass_sep=args.pass_sep,
        station_places_csv=args.station_places_csv,
        output_parquet=args.output_parquet,
        bucket_minutes=args.bucket_minutes,
        transport_type_id=args.transport_type_id,
        force=args.force,
    )
    print(out)


if __name__ == '__main__':
    main()
