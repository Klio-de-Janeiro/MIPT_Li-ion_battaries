"""KIT LOG_AGE adapter. This is not the MIT label/split protocol of the paper."""
from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
from tqdm.auto import tqdm
from ..cache import GenerationLease

INPUT_COLUMNS = ['cycle', 'time_s', 'voltage_V', 'current_A', 'temperature_C']
ARRAY_COLUMNS = INPUT_COLUMNS + ['soc', 'soh']
RAW_COLUMNS = ['timestamp_s', 'EFC', 'v_raw_V', 'i_raw_A', 't_cell_degC', 'soc_est', 'cap_aged_est_Ah']
PATTERN = re.compile(r'cell_log_age_(2|30)s_P(\d+)_(\d+)_S(\d+)_C(\d+)\.csv$', re.I)
PREPROCESSING_VERSION = 8


@dataclass(frozen=True)
class PrepConfig:
    raw_dir: Path
    prepared_dir: Path
    cell_limit: int = 10
    selection_seed: int = 42
    split_seed: int = 42
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    rated_capacity_ah: float = 3.0
    interval_s: int = 30
    time_tolerance_s: float = 0.25
    input_length: int = 32
    horizon: int = 8
    stride: int = 8
    chunk_rows: int = 200_000
    soc_scale: float = 100.0
    max_soh: float = 1.2
    min_anchors: int = 2
    show_progress: bool = True
    target_raw_gb: float | None = None
    max_anchor_gap_s: float | None = None

    def __post_init__(self):
        if self.max_anchor_gap_s is not None and (not np.isfinite(self.max_anchor_gap_s) or self.max_anchor_gap_s <= 0):
            raise ValueError("max_anchor_gap_s must be positive or None")
        if self.target_raw_gb is not None and (
                not np.isfinite(self.target_raw_gb) or self.target_raw_gb <= 0):
            raise ValueError('target_raw_gb must be positive or None')
        numeric = (self.rated_capacity_ah, self.soc_scale, self.max_soh,
                   self.time_tolerance_s, self.val_fraction, self.test_fraction)
        if not np.isfinite(numeric).all() or self.max_soh <= 0 or self.min_anchors < 2:
            raise ValueError('Finite label/split settings and >=2 capacity anchors are required')
        if not 0 <= self.time_tolerance_s < 1:
            raise ValueError('Time tolerance must be in [0, 1) seconds')
        if self.cell_limit < 3 or self.rated_capacity_ah <= 0 or self.soc_scale <= 0:
            raise ValueError('Need >=3 cells and positive capacity/SOC scale')
        if min(self.input_length, self.horizon, self.stride, self.chunk_rows) <= 0:
            raise ValueError('Window/chunk dimensions must be positive')
        if self.interval_s != 30:
            raise ValueError('This adapter explicitly supports a 30-second target grid')
        if min(self.val_fraction, self.test_fraction) <= 0 or self.val_fraction + self.test_fraction >= 1:
            raise ValueError('Invalid split fractions')


def scan_columns(path: Path, columns=RAW_COLUMNS) -> pl.LazyFrame:
    return pl.scan_csv(path, separator=';', infer_schema=False).select(
        [pl.col(c).cast(pl.Float64, strict=False) for c in columns]
    )


def discover_sources(root: Path) -> pl.DataFrame:
    rows = []
    for path in sorted(Path(root).rglob('cell_log_age_*s_*.csv')):
        match = PATTERN.fullmatch(path.name)
        if match is None:
            continue
        interval, parameter, replicate, slave, channel = map(int, match.groups())
        st = path.stat()
        rows.append(dict(cell_id=f'P{parameter:03d}_{replicate}',
                         sampling_interval_s=interval, slave=slave, channel=channel,
                         path=str(path.resolve()), size_bytes=st.st_size, mtime_ns=st.st_mtime_ns))
    if not rows:
        raise FileNotFoundError(f'LOG_AGE files not found: {root}')
    selected = []
    for cell in sorted({r['cell_id'] for r in rows}):
        group = [r for r in rows if r['cell_id'] == cell]
        chosen = [r for r in group if r['sampling_interval_s'] == 30] or group
        if len(chosen) != 1:
            raise ValueError(f'Ambiguous files for physical cell {cell}: {[r["path"] for r in chosen]}')
        selected.append(chosen[0])
    return pl.DataFrame(selected)


def load_capacity_anchors(path: Path, source_interval: int, cfg: PrepConfig):
    """Sparse capacity results, ordered by measurement availability time.

    EFC is not a valid interpolation axis for calendar aging: EFC can remain
    constant while capacity changes. No extrapolation outside anchors below.
    """
    frame = (scan_columns(path, ['timestamp_s', 'cap_aged_est_Ah'])
             .filter(pl.all_horizontal(pl.all().is_finite()),
                     pl.col('cap_aged_est_Ah') > 0,
                     pl.col('cap_aged_est_Ah') <= cfg.max_soh * cfg.rated_capacity_ah)
             .group_by('timestamp_s').agg(pl.col('cap_aged_est_Ah').median())
             .sort('timestamp_s').collect(engine='streaming'))
    return (frame['timestamp_s'].to_numpy() + source_interval,
            frame['cap_aged_est_Ah'].to_numpy())


def split_cells(cell_ids: list[str], cfg: PrepConfig) -> dict[str, str]:
    cells = np.array(sorted(set(cell_ids)), dtype=object)
    if len(cells) != len(cell_ids):
        raise ValueError('Duplicate physical cell in split input')
    n_test = max(1, math.ceil(len(cells) * cfg.test_fraction))
    n_val = max(1, math.ceil(len(cells) * cfg.val_fraction))
    if n_test + n_val >= len(cells):
        raise ValueError('Not enough cells for three splits')
    np.random.default_rng(cfg.split_seed).shuffle(cells)
    return {str(c): ('test' if i < n_test else 'val' if i < n_test + n_val else 'train')
            for i, c in enumerate(cells)}


def resample_query(query: pl.LazyFrame, source_interval: int, cfg: PrepConfig) -> pl.LazyFrame:
    """Validate time-based bins BEFORE reducing. Never bridge a missing row.

    Published LOG_AGE timestamps label the left edge of averages. Convert
    them to availability at the right edge. SOC is averaged just as in the
    upstream 30-second files. It is a block estimate, not an instantaneous truth.
    """
    essential = RAW_COLUMNS[:-1]
    valid = pl.all_horizontal([pl.col(c).is_finite() for c in essential])
    valid &= pl.col('soc_est').is_between(-0.01 * cfg.soc_scale, 1.01 * cfg.soc_scale)
    timestamp = pl.col('timestamp_s')
    invalid_order = (timestamp.filter(timestamp.is_finite()).diff() <= 0).any()
    # Preserve this check through group_by/sort: sorting bins must not conceal
    # a reversed or duplicate source timestamp, even across bin boundaries.
    query = query.with_columns(valid.fill_null(False).alias('_valid'),
                               invalid_order.alias('_source_order_invalid'))
    if source_interval == cfg.interval_s:
        return query.with_columns((pl.col('timestamp_s') + source_interval).alias('timestamp_s'))
    if source_interval != 2:
        raise ValueError(f'Unsupported source resolution {source_interval}')
    dt = pl.col('timestamp_s').diff().drop_nulls()
    return (query.with_columns((pl.col('timestamp_s') / cfg.interval_s).floor().alias('_bin'))
            .group_by('_bin').agg(
                pl.len().alias('_rows'), pl.col('_valid').all().alias('_valid'),
                pl.col('_source_order_invalid').any(),
                ((dt - source_interval).abs() <= cfg.time_tolerance_s).all().alias('_regular'),
                *[pl.col(c).mean().alias(c) for c in ['EFC', 'v_raw_V', 'i_raw_A', 't_cell_degC', 'soc_est']],
                pl.col('cap_aged_est_Ah').last(),
            ).with_columns(
                ((pl.col('_bin') + 1) * cfg.interval_s).alias('timestamp_s'),
                (pl.col('_valid') & pl.col('_regular') & (pl.col('_rows') == cfg.interval_s // source_interval)).alias('_valid'),
            ).sort('_bin').drop('_bin', '_regular', '_rows'))


def segment_offsets(times: np.ndarray, valid: np.ndarray, cfg: PrepConfig) -> np.ndarray:
    indices = np.flatnonzero(valid)
    if not len(indices):
        return np.array([0], dtype=np.int64)
    boundary = ((np.diff(indices) != 1) |
                (np.abs(np.diff(times[indices]) - cfg.interval_s) > cfg.time_tolerance_s))
    return np.r_[0, np.flatnonzero(boundary) + 1, len(indices)].astype(np.int64)


def count_windows(offsets: np.ndarray, cfg: PrepConfig) -> np.ndarray:
    return np.maximum(0, 1 + (np.diff(offsets) - cfg.input_length - cfg.horizon) // cfg.stride)


def update_statistics(count, mean, m2, values):
    if not len(values):
        return count, mean, m2
    value = np.asarray(values, dtype=np.float64)
    n = len(value)
    mu = value.mean(0)
    delta = mu - mean
    total = count + n
    return total, mean + delta * n / total, m2 + ((value - mu)**2).sum(0) + delta**2 * count * n / total


def prepared_values(frame, anchors, cfg):
    t = frame['timestamp_s'].to_numpy()
    if (('_source_order_invalid' in frame.columns and frame['_source_order_invalid'].any())
            or np.any(np.diff(t[np.isfinite(t)]) <= 0)):
        raise ValueError('Non-increasing timestamps; inspect source rather than silently sorting')
    capacity = np.interp(t, *anchors, left=np.nan, right=np.nan)
    if cfg.max_anchor_gap_s is not None:
        times = anchors[0]
        right = np.searchsorted(times, t, side="left")
        lo = np.clip(right - 1, 0, len(times)-1)
        hi = np.clip(right, 0, len(times)-1)
        exact = np.isin(t, times)
        capacity[(times[hi] - times[lo] > cfg.max_anchor_gap_s) & ~exact] = np.nan
    soc = frame['soc_est'].to_numpy() / cfg.soc_scale
    data = np.column_stack([
        frame['EFC'].to_numpy(), t, frame['v_raw_V'].to_numpy(),
        frame['i_raw_A'].to_numpy(), frame['t_cell_degC'].to_numpy(),
        np.clip(soc, 0, 1), capacity / cfg.rated_capacity_ah,
    ])
    valid = frame['_valid'].to_numpy().astype(bool) & np.isfinite(data).all(1)
    valid &= (soc >= -0.01) & (soc <= 1.01) & (data[:, 6] > 0) & (data[:, 6] <= cfg.max_soh)
    offsets = segment_offsets(t, valid, cfg)
    # Remove short segments: their values would never be model inputs/targets.
    kept, lengths = [], []
    filtered = data[valid]
    for start, end in zip(offsets[:-1], offsets[1:]):
        if end - start >= cfg.input_length + cfg.horizon:
            kept.append(filtered[start:end]); lengths.append(end - start)
    if not kept:
        raise ValueError('No complete windows after filtering and anchor coverage restriction')
    return np.concatenate(kept), np.r_[0, np.cumsum(lengths)], int((~valid).sum())


def _signature(cfg, manifest):
    config = asdict(cfg)
    for key in ('raw_dir', 'prepared_dir', 'show_progress'):
        config.pop(key)
    if cfg.max_anchor_gap_s is None:
        config.pop("max_anchor_gap_s")  # preserve unchanged v8 caches
    if cfg.target_raw_gb is None:
        config.pop('target_raw_gb')  # keep existing count-based caches usable
    else:
        config.pop('cell_limit')  # byte budget determines the cell count
    return {'version': PREPROCESSING_VERSION, 'config': config,
            'sources': manifest.sort('cell_id').to_dicts(),
            'labels': 'SOC=mean(soc_est)/soc_scale; SOH=linear_time_capacity/rated; no extrapolation',
            'time': 'LOG_AGE left edge -> right edge availability; float64 before scaling',
            'split': 'disjoint physical P+replicate cells', 'columns': ARRAY_COLUMNS}


def _write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def prepare_dataset(cfg: PrepConfig, *, rebuild=False) -> tuple[Path, bool]:
    """Versioned transactional generation, immutable raw CSV, bounded per-cell RAM.

    Only the CURRENT pointer is replaced after full validation. A failed run
    cannot mark partially scaled arrays as a valid cache. Prior generations
    remain available. A cache hit does not parse capacity columns again.
    """
    root = Path(cfg.prepared_dir)
    root.mkdir(parents=True, exist_ok=True)
    available = discover_sources(cfg.raw_dir)
    pointer = root / 'CURRENT.json'
    if pointer.is_file() and not rebuild:
        old = json.loads(pointer.read_text())
        existing = root / old['generation']
        meta = json.loads((existing / 'preprocessing.json').read_text())
        selected_ids = [r['cell_id'] for r in meta['sources']]
        candidate = available.filter(pl.col('cell_id').is_in(selected_ids))
        if _signature(cfg, candidate) == meta:
            validate_prepared_dataset(existing, cfg, full=False)
            return existing, True
    rows = available.to_dicts()
    np.random.default_rng(cfg.selection_seed).shuffle(rows)
    selected, cache = [], {}
    selected_bytes = 0
    target_bytes = None if cfg.target_raw_gb is None else cfg.target_raw_gb * 1e9
    for row in tqdm(rows, desc='Проверка ёмкости', disable=not cfg.show_progress):
        anchors = load_capacity_anchors(Path(row['path']), row['sampling_interval_s'], cfg)
        if len(anchors[0]) < cfg.min_anchors:
            continue
        selected.append(row); cache[row['cell_id']] = anchors
        selected_bytes += row['size_bytes']
        if (target_bytes is None and len(selected) == cfg.cell_limit) or (
                target_bytes is not None and selected_bytes >= target_bytes and len(selected) >= 3):
            break
    if target_bytes is not None and (selected_bytes < target_bytes or len(selected) < 3):
        raise ValueError(f'Only {len(selected)} eligible cells / {selected_bytes / 1e9:.3f} GB; '
                         f'requested {cfg.target_raw_gb} GB and at least 3 cells')
    if target_bytes is None and len(selected) != cfg.cell_limit:
        raise ValueError(f'Only {len(selected)} cells with capacity anchors; requested {cfg.cell_limit}')
    if cfg.show_progress:
        print(f'Selected {len(selected)} physical cells, {selected_bytes / 1e9:.3f} GB CSV')
    manifest = pl.DataFrame(selected).sort('cell_id')
    splits = split_cells(manifest['cell_id'].to_list(), cfg)
    generation = root / ('generation_' + uuid.uuid4().hex[:12])
    generation.mkdir()
    lease = GenerationLease(generation)
    try:
        _write_json(generation / ".mden_generation.json", {"owner": "mden_prepared_cache"})
        (generation / 'arrays').mkdir()
        stats = (0, np.zeros(5), np.zeros(5))
        index = []
        for row in tqdm(manifest.to_dicts(), desc='Подготовка батарей', disable=not cfg.show_progress):
            frame = resample_query(scan_columns(Path(row['path'])), row['sampling_interval_s'], cfg).collect(engine='streaming')
            values, offsets, dropped = prepared_values(frame, cache[row['cell_id']], cfg)
            split = splits[row['cell_id']]
            raw_path = generation / 'arrays' / (row['cell_id'] + '.raw.npy')
            np.save(raw_path, values, allow_pickle=False)  # float64 until scaler is known
            if split == 'train':
                for start in range(0, len(values), cfg.chunk_rows):
                    stats = update_statistics(*stats, values[start:start+cfg.chunk_rows, :5])
            index.append(dict(cell_id=row['cell_id'], split=split,
                              path='arrays/' + row['cell_id'] + '.npy', rows=len(values),
                              segment_offsets=json.dumps(offsets.tolist()),
                              windows=int(count_windows(offsets, cfg).sum()), dropped_rows=dropped,
                          capacity_anchors=len(cache[row['cell_id']][0]),
                          max_anchor_gap_s=float(np.diff(cache[row['cell_id']][0]).max()),
                              first_time_s=float(values[0, 1]), last_time_s=float(values[-1, 1])))
            del values, frame
        count, mean, m2 = stats
        std = np.sqrt(m2 / count)
        std = np.where(std < 1e-12, 1.0, std)
        for row in index:
            dest = generation / row['path']
            raw = dest.with_suffix('.raw.npy')
            values = np.load(raw, mmap_mode='r')
            out = np.lib.format.open_memmap(dest, mode='w+', dtype=np.float32, shape=values.shape)
            for start in range(0, len(values), cfg.chunk_rows):
                chunk = np.array(values[start:start+cfg.chunk_rows], copy=True)
                chunk[:, :5] = (chunk[:, :5] - mean) / std
                out[start:start+len(chunk)] = chunk.astype(np.float32)
            out.flush(); del out, values
            raw.unlink()  # only our disposable intermediate, never user raw data
        manifest.write_csv(generation / 'manifest.csv')
        pl.DataFrame(index).write_csv(generation / 'prepared_index.csv')
        _write_json(generation / 'scaler.json', {'feature': INPUT_COLUMNS, 'mean': mean.tolist(), 'std': std.tolist(), 'count': count})
        _write_json(generation / 'preprocessing.json', _signature(cfg, manifest))
        validate_prepared_dataset(generation, cfg, full=True)
        _write_json(pointer, {'generation': generation.name})
        return generation, False
    finally:
        lease.close()


def validate_prepared_dataset(generation: Path, cfg: PrepConfig, *, full=True) -> pl.DataFrame:
    index = pl.read_csv(generation / 'prepared_index.csv')
    expected_count = cfg.cell_limit if cfg.target_raw_gb is None else index.height
    if index.height != expected_count or index['cell_id'].n_unique() != expected_count or index.height < 3:
        raise ValueError('Incorrect or duplicate physical cells')
    if cfg.target_raw_gb is not None:
        manifest = pl.read_csv(generation / 'manifest.csv')
        if (set(manifest['cell_id']) != set(index['cell_id'])
                or manifest['size_bytes'].sum() < cfg.target_raw_gb * 1e9):
            raise ValueError('Prepared sources do not satisfy the raw byte budget')
    expected = split_cells(index['cell_id'].to_list(), cfg)
    scaler = json.loads((generation / 'scaler.json').read_text())
    means, stds = np.asarray(scaler['mean']), np.asarray(scaler['std'])
    if (scaler['feature'] != INPUT_COLUMNS or means.shape != (5,) or stds.shape != (5,)
            or not np.isfinite(means).all() or not np.isfinite(stds).all()
            or not (stds > 0).all() or scaler['count'] <= 0):
        raise ValueError('Invalid train scaler')
    for row in index.iter_rows(named=True):
        if row['split'] != expected[row['cell_id']]:
            raise ValueError('Incorrect split')
        values = np.load(generation / row['path'], mmap_mode='r', allow_pickle=False)
        offsets = np.array(json.loads(row['segment_offsets']))
        if values.shape != (row['rows'], 7) or values.dtype != np.float32:
            raise ValueError('Incorrect NPY shape/dtype')
        if (offsets.ndim != 1 or len(offsets) < 2 or not np.issubdtype(offsets.dtype, np.integer)
                or offsets[0] != 0 or offsets[-1] != len(values)
                or np.any(np.diff(offsets) < cfg.input_length + cfg.horizon)):
            raise ValueError('Incorrect offsets')
        if count_windows(offsets, cfg).sum() != row['windows']:
            raise ValueError('Incorrect window count')
        starts = range(0, len(values), cfg.chunk_rows) if full else [0, max(0, len(values)-cfg.chunk_rows)]
        for start in starts:
            chunk = values[start:start+cfg.chunk_rows]
            if (not np.isfinite(chunk).all() or chunk[:, 5].min() < 0 or chunk[:, 5].max() > 1
                    or chunk[:, 6].min() <= 0 or chunk[:, 6].max() > cfg.max_soh + 1e-6):
                raise ValueError(f'Invalid values: {row["cell_id"]}')
    return index
