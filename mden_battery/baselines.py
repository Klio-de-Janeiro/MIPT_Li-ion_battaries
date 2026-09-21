from __future__ import annotations

import numpy as np


class CoulombCountingSOC:
    """Minimal SOC baseline for sanity checks."""

    def __init__(self, capacity_ah: float, initial_soc: float = 1.0) -> None:
        self.capacity_ah = float(capacity_ah)
        self.initial_soc = float(initial_soc)

    def predict(self, current_a: np.ndarray, time_s: np.ndarray) -> np.ndarray:
        i = np.asarray(current_a, dtype=float)
        t = np.asarray(time_s, dtype=float)
        dt = np.diff(t, prepend=t[0])
        dt[0] = 0.0
        soc = self.initial_soc + np.cumsum(i * dt) / (
            3600.0 * self.capacity_ah
        )
        return np.clip(soc, 0.0, 1.0)
