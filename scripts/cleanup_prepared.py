"""Preview/delete recognized NPY caches. Does not touch raw data or runs."""
import argparse
import json
from pathlib import Path

from mden_battery.cache import cleanup_prepared_cache

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True, help='One prepared root, or project data directory with --all-roots')
parser.add_argument('--all-roots', action='store_true')
parser.add_argument('--include-current', action='store_true')
parser.add_argument('--apply', action='store_true', help='Without this flag, only print a preview')
args = parser.parse_args()
roots = sorted(args.root.glob('prepared_log_age_30s_*_v8')) if args.all_roots else [args.root]
for root in roots:
    print(json.dumps({'root': str(root), 'results': cleanup_prepared_cache(
        root, include_current=args.include_current, dry_run=not args.apply)}, indent=2))
