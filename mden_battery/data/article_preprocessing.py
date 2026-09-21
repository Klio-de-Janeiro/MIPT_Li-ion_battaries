from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ArticleOutlierConfig:
    min_points_per_cycle: int = 10
    min_soc_range: float = 0.2
    min_duration_s: float = 10 * 60
    max_duration_s: float = 100 * 60
    soc_tolerance: float = 0.01
    max_delta_soc_std: float = 0.05
    max_soh_cycle_jump: float = 0.005


@dataclass(frozen=True)
class WindowConfig:
    input_length: int = 32
    horizon: int = 8
    stride: int = 8

    def __post_init__(self):
        if min(self.input_length, self.horizon, self.stride) <= 0:
            raise ValueError("Window lengths and stride must be positive")


def compute_article_labels(
    df: pd.DataFrame,
    *,
    cycle_col: str = "cycle",
    time_col: str = "time_s",
    current_col: str = "current_A",
    rated_capacity_ah: float,
) -> pd.DataFrame:
    """Compute SOC/SOH labels following Sec. 3.1.1, Eq. (1)-(3).

    Current convention follows the paper exactly: charging current is positive
    and discharging current is negative. Capacity integration is reset at the
    initial point of every cycle. No offset/shift is introduced: Eq. (1) is
    implemented as the cumulative discrete integral from the cycle start.
    """
    required = {cycle_col, time_col, current_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    if rated_capacity_ah <= 0:
        raise ValueError("rated_capacity_ah must be positive")

    pieces = []
    group_keys = ["cell_id", cycle_col] if "cell_id" in df.columns else cycle_col
    for cycle, g in df.groupby(group_keys, sort=False):
        g = g.sort_values(time_col).copy()
        t = g[time_col].to_numpy(dtype=float)
        i = g[current_col].to_numpy(dtype=float)
        if len(g) == 0:
            continue
        dt = np.diff(t, prepend=t[0])
        dt[0] = 0.0
        q_remaining = np.cumsum(i * dt) / 3600.0
        q_max = (np.max(q_remaining) if np.isfinite(t).all() and np.isfinite(i).all()
                 and (len(t) < 2 or np.all(np.diff(t) > 0)) else np.nan)
        if not np.isfinite(q_max) or q_max <= 0:
            g["soc"] = np.nan
            g["soh"] = np.nan
            g["q_remaining_ah"] = q_remaining
            g["q_cycle_max_ah"] = q_max
        else:
            g["q_remaining_ah"] = q_remaining
            g["q_cycle_max_ah"] = q_max
            g["soc"] = q_remaining / q_max
            g["soh"] = q_max / rated_capacity_ah
        pieces.append(g)
    if not pieces:
        return df.copy().assign(soc=np.nan, soh=np.nan)
    return pd.concat(pieces, ignore_index=True)


def filter_article_outliers(
    df: pd.DataFrame,
    *,
    cycle_col: str = "cycle",
    time_col: str = "time_s",
    soc_col: str = "soc",
    soh_col: str = "soh",
    config: ArticleOutlierConfig | None = None,
) -> pd.DataFrame:
    """Apply the explicit outlier rules listed in Sec. 3.1.2."""
    cfg = config or ArticleOutlierConfig()
    if "cell_id" in df.columns:
        pieces = []
        for cell, group in df.groupby("cell_id", sort=False):
            filtered = filter_article_outliers(group.drop(columns="cell_id"), cycle_col=cycle_col, time_col=time_col, soc_col=soc_col, soh_col=soh_col, config=cfg)
            filtered["cell_id"] = cell
            pieces.append(filtered)
        return pd.concat(pieces, ignore_index=True) if pieces else df.iloc[:0].copy()
    keep_cycles = []
    cycle_soh = []

    for cycle, g in df.groupby(cycle_col, sort=True):
        g = g.sort_values(time_col)
        soc = g[soc_col]
        t = g[time_col]
        valid = True
        valid &= len(g) >= cfg.min_points_per_cycle
        valid &= np.isfinite(g[[soc_col, soh_col, time_col]].to_numpy(dtype=float)).all()
        valid &= bool((g[soh_col] > 0).all())
        if valid:
            valid &= float(soc.max() - soc.min()) >= cfg.min_soc_range
            duration = float(t.iloc[-1] - t.iloc[0])
            valid &= cfg.min_duration_s <= duration <= cfg.max_duration_s
            valid &= float(soc.min()) >= -cfg.soc_tolerance
            valid &= float(soc.max()) <= 1.0 + cfg.soc_tolerance
            if len(soc) > 1:
                valid &= (
                    float(np.std(np.diff(soc.to_numpy(dtype=float))))
                    <= cfg.max_delta_soc_std
                )
        if valid:
            keep_cycles.append(cycle)
            cycle_soh.append((cycle, float(g[soh_col].median())))

    # SOH jump rule is applied between consecutive retained cycles.
    rejected_soh = set()
    for (prev_c, prev_h), (cur_c, cur_h) in pairwise(cycle_soh):
        if (
            np.isfinite(prev_h)
            and np.isfinite(cur_h)
            and abs(cur_h - prev_h) > cfg.max_soh_cycle_jump
        ):
            rejected_soh.add(cur_c)

    keep = [c for c in keep_cycles if c not in rejected_soh]
    return df[df[cycle_col].isin(keep)].copy().reset_index(drop=True)


class SlidingWindowDataset(Dataset):
    """Paper-style 32 -> 8 windows with stride 8.

    By default windows are generated *within a single cycle*, matching the
    statement in Sec. 3.2.1 that every input window covers a local segment of
    one cycle. Pass ``group_col=None`` only for datasets that do not expose
    discrete cycle boundaries (for example the separate EnergyStatus adapter).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        input_cols: Sequence[str],
        *,
        soc_col: str = "soc",
        soh_col: str = "soh",
        group_col: str | None = "cycle",
        config: WindowConfig | None = None,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> None:
        self.cfg = config or WindowConfig()
        self.input_cols = list(input_cols)
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        if group_col is None:
            groups = df.groupby("cell_id", sort=False) if "cell_id" in df.columns else [(None, df)]
        else:
            if group_col not in df.columns:
                raise KeyError(f"Grouping column {group_col!r} is missing.")
            keys = ["cell_id", group_col] if "cell_id" in df.columns and group_col != "cell_id" else group_col
            groups = df.groupby(keys, sort=False)
        for _, g in groups:
            g = (
                g[self.input_cols + [soc_col, soh_col]]
                .reset_index(drop=True)
            )
            x = g[self.input_cols].to_numpy(np.float32, copy=True)
            if mean is not None and std is not None:
                x = (x - mean.astype(np.float32)) / np.where(
                    std == 0, 1.0, std
                ).astype(np.float32)
            y_soc = g[soc_col].to_numpy(np.float32, copy=True)
            y_soh = g[soh_col].to_numpy(np.float32, copy=True)
            total = self.cfg.input_length + self.cfg.horizon
            finite = np.isfinite(x).all(1) & np.isfinite(y_soc) & np.isfinite(y_soh)
            for start in range(0, len(g) - total + 1, self.cfg.stride):
                if not finite[start:start+total].all():
                    continue
                cut = start + self.cfg.input_length
                self.samples.append(
                    (
                        x[start:cut],
                        y_soc[cut : cut + self.cfg.horizon],
                        y_soh[cut : cut + self.cfg.horizon],
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x, y_soc, y_soh = self.samples[idx]
        return {
            "x": torch.from_numpy(x),
            "y_soc": torch.from_numpy(y_soc),
            "y_soh": torch.from_numpy(y_soh),
        }
