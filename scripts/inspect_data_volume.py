"""Count unique physical cells and their selected CSV bytes without loading CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mden_battery.data.log_age import discover_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--target-gb', type=float, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not np.isfinite(args.target_gb) or args.target_gb <= 0:
        parser.error('target-gb must be positive')
    sources = discover_sources(args.raw_dir)
    rows = sources.to_dicts()
    np.random.default_rng(args.seed).shuffle(rows)
    sizes = np.array([r['size_bytes'] for r in rows], dtype=np.int64)
    n = min(len(rows), max(3, int(np.searchsorted(sizes.cumsum(), args.target_gb * 1e9)) + 1))
    print(json.dumps({
        'physical_cells': len(rows), 'selected_sources_gb': float(sizes.sum() / 1e9),
        'median_csv_mb': float(np.median(sizes) / 1e6),
        'target_gb': args.target_gb, 'estimated_cells': n,
        'estimated_gb': float(sizes[:n].sum() / 1e9),
        'target_available': bool(sizes.sum() >= args.target_gb * 1e9 and n >= 3),
        'note': 'Decimal GB; 30s preferred over 2s. Capacity/valid windows not checked yet.',
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
