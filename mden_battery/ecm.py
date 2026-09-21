from __future__ import annotations

import numpy as np


def simulate_second_order_ecm(
    current_a: np.ndarray,
    dt_s: float,
    ocv_v: float,
    r0: float,
    r1: float,
    c1: float,
    r2: float,
    c2: float,
) -> np.ndarray:
    """Simple 2-RC ECM baseline; not part of the MDEN paper."""
    i = np.asarray(current_a, dtype=float)
    v1 = 0.0
    v2 = 0.0
    out = np.empty_like(i)
    a1 = np.exp(-dt_s / max(r1 * c1, 1e-12))
    a2 = np.exp(-dt_s / max(r2 * c2, 1e-12))
    for k, cur in enumerate(i):
        v1 = a1 * v1 + r1 * (1 - a1) * cur
        v2 = a2 * v2 + r2 * (1 - a2) * cur
        out[k] = ocv_v + r0 * cur + v1 + v2
    return out
