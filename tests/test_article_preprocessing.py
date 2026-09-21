import numpy as np
import pandas as pd

from mden_battery.data.article_preprocessing import (
    SlidingWindowDataset,
    compute_article_labels,
    filter_article_outliers,
)


def _cycle(cycle: int, current_a: float = 1.1, n: int = 11) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cycle": cycle,
            "time_s": np.linspace(0, 600, n),
            "current_A": np.full(n, current_a),
            "voltage_V": np.linspace(3.0, 4.1, n),
            "temperature_C": np.linspace(25, 28, n),
        }
    )


def test_article_soc_soh_labels_follow_discrete_current_integral():
    raw = _cycle(1)
    labeled = compute_article_labels(raw, rated_capacity_ah=1.1)
    assert labeled.loc[0, "q_remaining_ah"] == 0.0
    assert np.isclose(labeled["soc"].iloc[-1], 1.0)
    expected_qmax = 1.1 * 600 / 3600
    assert np.isclose(labeled["q_cycle_max_ah"].iloc[0], expected_qmax)
    assert np.isclose(labeled["soh"].iloc[0], expected_qmax / 1.1)
    assert labeled["soh"].nunique() == 1


def test_article_outlier_filter_keeps_valid_cycle_and_removes_short_one():
    valid = compute_article_labels(_cycle(1), rated_capacity_ah=1.1)
    invalid = compute_article_labels(_cycle(2, n=5), rated_capacity_ah=1.1)
    filtered = filter_article_outliers(
        pd.concat([valid, invalid], ignore_index=True)
    )
    assert set(filtered["cycle"].unique()) == {1}


def test_sliding_windows_are_32_to_8_with_stride_8():
    n = 80
    frame = pd.DataFrame(
        {
            # Sec. 3.2.1 states that one window is a local segment within
            # one cycle, so all rows belong to the same cycle here.
            "cycle": np.ones(n, dtype=float),
            "time_s": np.arange(n, dtype=float),
            "voltage_V": np.linspace(3.0, 4.2, n),
            "current_A": np.ones(n),
            "temperature_C": np.linspace(25, 30, n),
            "soc": np.linspace(0, 1, n),
            "soh": np.full(n, 0.95),
            "cell_id": "cell_1",
        }
    )
    ds = SlidingWindowDataset(
        frame,
        ["cycle", "time_s", "voltage_V", "current_A", "temperature_C"],
        # Default group_col="cycle" is the paper-faithful behavior.
    )
    assert len(ds) == 6
    sample = ds[0]
    assert sample["x"].shape == (32, 5)
    assert sample["y_soc"].shape == (8,)
    assert sample["y_soh"].shape == (8,)
