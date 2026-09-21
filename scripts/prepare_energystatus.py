from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mden_battery.data import (
    prepare_log_age_dataset,
    prepare_result_dataset,
    recursively_extract,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare EnergyStatus battery-aging datasets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Dataset directory or a directory containing archives.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Prepared output directory."
    )
    parser.add_argument("--kind", choices=["result", "log_age"], required=True)
    parser.add_argument("--rated-capacity-ah", type=float, default=3.0)
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Recursively extract archives before preparation.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="Rows per chunk for LOG_AGE processing.",
    )
    parser.add_argument(
        "--only-30s",
        action="store_true",
        help="Skip 2-s driving-profile LOG_AGE files.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    source = args.input
    if args.extract:
        extract_dir = args.output / "extracted"
        recursively_extract(source, extract_dir)
        source = extract_dir

    if args.kind == "result":
        outputs = prepare_result_dataset(
            source,
            args.output / "prepared_result",
            rated_capacity_ah=args.rated_capacity_ah,
        )
    else:
        outputs = prepare_log_age_dataset(
            source,
            args.output / "prepared_log_age",
            rated_capacity_ah=args.rated_capacity_ah,
            include_profile_2s=not args.only_30s,
            chunksize=args.chunksize,
        )

    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
