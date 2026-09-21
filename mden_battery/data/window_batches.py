from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import torch

from .article_preprocessing import WindowConfig
from ..cache import GenerationLease


class WindowBatchDataset:
    """Keep row arrays once; materialize only a batch of 40-row windows.

    Uniform shuffle over windows mixes cells/ages. Window starts are located
    by binary search in a small segment table. No windows cross a segment.
    CPU mode uses mmap; CUDA mode copies only row arrays to one flat cache.
    """

    def __init__(self, generation: Path, *, split: str, config: WindowConfig,
                 batch_size=512, shuffle=False, seed=42, cache_device='cpu', physics=False):
        self.generation, self.split, self.config = Path(generation), split, config
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0
        self.physics = physics
        self._lease = GenerationLease(self.generation)
        self._closed = False
        scaler = json.loads((self.generation / "scaler.json").read_text())
        self.current_mean, self.current_std = scaler["mean"][3], scaler["std"][3]
        self.cache_device = torch.device(cache_device)
        self.index = pl.read_csv(self.generation / 'prepared_index.csv').filter(pl.col('split') == split).sort('cell_id')
        if self.index.is_empty():
            raise ValueError(f'Empty split {split}')
        self.arrays, self.segments = [], []
        self.cell_ids = self.index['cell_id'].to_list()
        self.total_rows = int(self.index['rows'].sum())
        self.flat = (torch.empty((self.total_rows, 7), device=self.cache_device, dtype=torch.float32)
                     if self.cache_device.type == 'cuda' else None)
        base, cumulative = 0, 0
        for source, row in enumerate(self.index.iter_rows(named=True)):
            array = np.load(self.generation / row['path'], mmap_mode='c', allow_pickle=False)
            if self.flat is not None:
                self.flat[base:base+len(array)].copy_(torch.from_numpy(array))
                self.arrays.append(None)
            else:
                self.arrays.append(array)
            offsets = json.loads(row['segment_offsets'])
            for start, end in zip(offsets[:-1], offsets[1:]):
                n = max(0, 1 + (end-start-config.input_length-config.horizon)//config.stride)
                if n:
                    self.segments.append((source, start, base+start, cumulative, n))
                    cumulative += n
            base += len(array)
        self._segments = np.array(self.segments, dtype=np.int64)
        if not len(self.segments):
            raise ValueError('No complete windows')
        self._ends = self._segments[:, 3] + self._segments[:, 4]
        self.total_windows = cumulative
        self.set_batch_size(batch_size)

    def close(self):
        """Release NPY maps and the cleanup lease after all consumers finish."""
        if not self._closed:
            for array in self.arrays:
                if array is not None and getattr(array, "_mmap", None) is not None:
                    array._mmap.close()
            self.arrays.clear()
            self.flat = None
            self._lease.close()
            self._closed = True

    def set_batch_size(self, size: int):
        if int(size) <= 0:
            raise ValueError('batch size must be positive')
        self.batch_size = int(size)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return math.ceil(self.total_windows / self.batch_size)

    def locate(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= self.total_windows):
            raise IndexError('Window outside split')
        segment = self._segments[np.searchsorted(self._ends, ids, side='right')]
        displacement = (ids - segment[:, 3]) * self.config.stride
        return segment[:, 0], segment[:, 1] + displacement, segment[:, 2] + displacement

    def fetch(self, ids, *, metadata=False):
        if self._closed:
            raise RuntimeError("Dataset is closed; recreate it before fetching")
        ids = np.asarray(ids, dtype=np.int64)
        source, starts, global_starts = self.locate(ids)
        length = self.config.input_length + self.config.horizon
        if self.flat is not None:
            row_idx = torch.as_tensor(global_starts, device=self.cache_device)[:, None]
            row_idx = row_idx + torch.arange(length, device=self.cache_device)
            values = self.flat[row_idx]
        else:
            packed = np.empty((len(ids), length, 7), dtype=np.float32)
            steps = np.arange(length)
            for src in np.unique(source):
                positions = np.flatnonzero(source == src)
                rows = starts[positions, None] + steps
                packed[positions] = self.arrays[src][rows]
            values = torch.from_numpy(packed)
        cut = self.config.input_length
        result = {'x': values[:, :cut, :5].contiguous(),
                  'y_soc': values[:, cut:, 5].contiguous(),
                  'y_soh': values[:, cut:, 6].contiguous()}
        if self.physics:
            result["physics_current_a"] = (values[:, cut:, 3] * self.current_std + self.current_mean).contiguous()
        if metadata:
            return result, {'source': source, 'start': starts,
                            'history_soc': values[:, :cut, 5], 'history_soh': values[:, :cut, 6]}
        return result

    def sample_batch(self, size):
        if not 0 < size <= self.total_windows:
            raise ValueError('Probe batch exceeds available windows')
        ids = np.random.default_rng(self.seed).choice(self.total_windows, size, replace=False)
        return self.fetch(ids)

    def example_ids(self, count=2, seed=42):
        rng = np.random.default_rng(seed)
        sources = rng.choice(len(self.cell_ids), size=min(count, len(self.cell_ids)), replace=False)
        result = []
        for source in sources:
            segments = self._segments[self._segments[:, 0] == source]
            n = int(segments[:, 4].sum())
            draw = int(rng.integers(n))
            ends = np.cumsum(segments[:, 4])
            index = int(np.searchsorted(ends, draw, side='right'))
            before = 0 if index == 0 else ends[index-1]
            result.append(int(segments[index, 3] + draw - before))
        return result

    def __iter__(self):
        ids = np.arange(self.total_windows)
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(ids)
        for offset in range(0, self.total_windows, self.batch_size):
            yield self.fetch(ids[offset:offset+self.batch_size])


def choose_cache_device(generation, device, max_fraction=0.25):
    device = torch.device(device)
    if device.type != 'cuda':
        return torch.device('cpu')
    index = pl.read_csv(Path(generation) / 'prepared_index.csv')
    bytes_needed = int(index['rows'].sum()) * 7 * 4
    free, total = torch.cuda.mem_get_info(device)
    return device if bytes_needed < min(free, total) * max_fraction else torch.device('cpu')
