from __future__ import annotations

import argparse
from pathlib import Path

from metroflow.data.station_mapping import build_and_save_station_maps, load_ref_places, build_place_station_map, build_station_summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Build reviewable station-place mapping CSV files from REF_PSG_PLACES.')
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--ref-filename', default='REF_PSG_PLACES_202503251822.csv')
    parser.add_argument('--line-short', default='Сокольническая')
    parser.add_argument('--transport-type-id', type=int, default=1)
    parser.add_argument('--include-test', action='store_true')
    args = parser.parse_args()

    outputs = build_and_save_station_maps(
        data_dir=args.data_dir,
        ref_filename=args.ref_filename,
        line_short=args.line_short,
        transport_type_id=args.transport_type_id,
        include_test=args.include_test,
    )

    ref = load_ref_places(Path(args.data_dir) / args.ref_filename)
    metro_places = build_place_station_map(ref, transport_type_id=args.transport_type_id, include_test=args.include_test)
    line_places = build_place_station_map(ref, transport_type_id=args.transport_type_id, line_short=args.line_short, include_test=args.include_test)
    print('Created files:')
    for key, path in outputs.items():
        print(f'  {key}: {path}')
    print(f'All TYPE_ID={args.transport_type_id}: {len(build_station_summary(metro_places))} station-line groups, {len(metro_places)} place rows')
    print(f'{args.line_short}: {len(build_station_summary(line_places))} station-line groups, {len(line_places)} place rows')


if __name__ == '__main__':
    main()
