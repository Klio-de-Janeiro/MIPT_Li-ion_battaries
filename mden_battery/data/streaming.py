from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .article_preprocessing import WindowConfig
from .energystatus import read_prepared_frame

DEFAULT_INPUT_COLUMNS = [
    "cycle",
    "time_s",
    "voltage_V",
    "current_A",
    "temperature_C",
]


class PreparedWindowIterableDataset(IterableDataset):
    """Создаёт окна из подготовленных частей без пересечения батарей."""

    def __init__(
        self,
        index_csv: str | Path,
        *,
        split: str = "train",
        input_cols: Sequence[str] = DEFAULT_INPUT_COLUMNS,
        soc_col: str = "soc",
        soh_col: str = "soh",
        config: WindowConfig | None = None,
        scaler_csv: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.index_csv = Path(index_csv)
        self.split = split
        self.input_cols = list(input_cols)
        self.soc_col = soc_col
        self.soh_col = soh_col
        self.config = config or WindowConfig()
        self.scaler_csv = (
            Path(scaler_csv) if scaler_csv is not None else None
        )
        self._source_cache: dict[str, np.ndarray] = {}
        self._source_segments = {}

    def _load_index(self) -> pd.DataFrame:
        """Загружает индекс и оставляет строки указанной выборки."""
        index = pd.read_csv(self.index_csv)
        index = index[index["split"] == self.split].copy()
        return index.sort_values(
            ["source", "chunk"]
        ).reset_index(drop=True)

    def _load_scaler(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Загружает параметры стандартизации входных признаков."""
        if self.scaler_csv is None:
            return None, None
        scaler = pd.read_csv(self.scaler_csv).set_index("feature")
        missing = [
            column
            for column in self.input_cols
            if column not in scaler.index
        ]
        if missing:
            raise KeyError(f"Scaler misses features: {missing}")
        mean = scaler.loc[
            self.input_cols,
            "mean",
        ].to_numpy(np.float64)
        std = scaler.loc[
            self.input_cols,
            "std",
        ].to_numpy(np.float64)
        return mean, np.where(std == 0, 1.0, std)

    @staticmethod
    def _assign_sources(index: pd.DataFrame) -> list[str]:
        """Балансирует батареи между процессами по числу строк."""
        source_sizes = (
            index.groupby("source")["rows"]
            .sum()
            .sort_values(ascending=False)
        )
        worker = get_worker_info()
        if worker is None:
            return source_sizes.index.tolist()

        assignments = [[] for _ in range(worker.num_workers)]
        loads = [0] * worker.num_workers
        for source, rows in source_sizes.items():
            worker_id = int(np.argmin(loads))
            assignments[worker_id].append(source)
            loads[worker_id] += int(rows)
        return assignments[worker.id]

    def _load_source(
        self,
        index: pd.DataFrame,
        source: str,
        mean: np.ndarray | None,
        std: np.ndarray | None,
    ) -> np.ndarray:
        """Загружает и сохраняет одну батарею в RAM процесса."""
        if source in self._source_cache:
            return self._source_cache[source]

        columns = self.input_cols + [self.soc_col, self.soh_col]
        source_rows = index[index["source"] == source]
        frames = []
        for path in source_rows.sort_values("chunk")["path"]:
            frame = read_prepared_frame(path)
            values = frame[columns].dropna().to_numpy(
                dtype=np.float64,
                copy=True,
            )
            if values.size:
                frames.append(values)
        if not frames:
            values = np.empty((0, len(columns)), dtype=np.float32)
        else:
            values = np.concatenate(frames, axis=0)
        match = re.search(r"log_age_(\d+)s", str(source))
        interval = float(match.group(1)) if match else None
        times = values[:, self.input_cols.index("time_s")] if len(values) else np.array([])
        delta = np.diff(times)
        breaks = delta <= 0
        if interval is not None:
            breaks |= np.abs(delta - interval) > min(1.0, interval * 0.1)
        offsets = np.r_[0, np.flatnonzero(breaks)+1, len(values)]
        self._source_segments[source] = list(zip(offsets[:-1], offsets[1:]))
        if mean is not None and std is not None:
            feature_count = len(self.input_cols)
            values[:, :feature_count] -= mean
            values[:, :feature_count] /= std
        values = values.astype(np.float32)
        self._source_cache[source] = values
        return values

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Возвращает отдельные окна из RAM-кеша."""
        index = self._load_index()
        if index.empty:
            return
        sources = self._assign_sources(index)
        mean, std = self._load_scaler()
        input_length = self.config.input_length
        horizon = self.config.horizon
        total_length = input_length + horizon
        feature_count = len(self.input_cols)

        for source in sources:
            values = self._load_source(index, source, mean, std)
            for left, right in self._source_segments[source]:
                for start in range(left, right-total_length+1, self.config.stride):
                    cut = start + input_length
                    yield {"x": torch.from_numpy(values[start:cut, :feature_count].copy()),
                           "y_soc": torch.from_numpy(values[cut:cut+horizon, feature_count].copy()),
                           "y_soh": torch.from_numpy(values[cut:cut+horizon, feature_count+1].copy())}


class PreparedBatchIterableDataset(PreparedWindowIterableDataset):
    """Кеширует батареи в RAM и возвращает готовые векторные батчи."""

    def __init__(
        self,
        index_csv: str | Path,
        *,
        batch_size: int,
        shuffle_batches: bool = False,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(index_csv, **kwargs)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches
        self.seed = seed
        self._iteration = 0

    def _build_batch_specs(
        self,
        sources: list[str],
        source_values: dict[str, np.ndarray],
    ) -> list[tuple[str, int, int]]:
        """Создаёт описания батчей без материализации окон."""
        total_length = self.config.input_length + self.config.horizon
        specs = []
        for source in sources:
            for left, right in self._source_segments[source]:
                window_count = max(0, 1 + (right-left-total_length)//self.config.stride)
                for first in range(0, window_count, self.batch_size):
                    size = min(self.batch_size, window_count-first)
                    specs.append((source, left + first*self.config.stride, size))
        return specs

    def _make_batch(
        self,
        values: np.ndarray,
        first_window: int,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Векторно создаёт один батч входов и целевых значений."""
        input_length = self.config.input_length
        horizon = self.config.horizon
        feature_count = len(self.input_cols)
        starts = first_window + np.arange(batch_size, dtype=np.int64) * self.config.stride
        input_indices = starts[:, None] + np.arange(input_length)
        target_indices = (
            starts[:, None] + input_length + np.arange(horizon)
        )

        x = np.ascontiguousarray(
            values[input_indices, :feature_count]
        )
        y_soc = np.ascontiguousarray(
            values[target_indices, feature_count]
        )
        y_soh = np.ascontiguousarray(
            values[target_indices, feature_count + 1]
        )
        return {
            "x": torch.from_numpy(x),
            "y_soc": torch.from_numpy(y_soc),
            "y_soh": torch.from_numpy(y_soh),
        }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Возвращает готовые батчи и сохраняет данные между эпохами."""
        index = self._load_index()
        if index.empty:
            return
        sources = self._assign_sources(index)
        mean, std = self._load_scaler()
        source_values = {
            source: self._load_source(index, source, mean, std)
            for source in sources
        }
        specs = self._build_batch_specs(sources, source_values)

        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        rng = np.random.default_rng(
            self.seed + self._iteration + worker_id * 100_000
        )
        self._iteration += 1
        if self.shuffle_batches:
            rng.shuffle(specs)

        for source, first_window, batch_size in specs:
            yield self._make_batch(
                source_values[source],
                first_window,
                batch_size,
            )
