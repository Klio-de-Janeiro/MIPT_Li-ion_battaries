import numpy as np

from mden_battery.baselines import CoulombCountingSOC
from mden_battery.ecm import simulate_second_order_ecm


def test_coulomb_counting_soc_changes_with_current():
    baseline = CoulombCountingSOC(capacity_ah=1.0, initial_soc=0.5)
    time_s = np.array([0.0, 1800.0, 3600.0])
    current_a = np.array([0.0, 0.5, 0.5])
    soc = baseline.predict(current_a, time_s)
    assert np.all(np.diff(soc) >= 0)
    assert np.all((soc >= 0) & (soc <= 1))


def test_second_order_ecm_returns_finite_voltage_trace():
    current = np.array([0.0, 1.0, 1.0, 0.0, -0.5])
    voltage = simulate_second_order_ecm(
        current,
        dt_s=1.0,
        ocv_v=3.7,
        r0=0.01,
        r1=0.02,
        c1=1000.0,
        r2=0.03,
        c2=2000.0,
    )
    assert voltage.shape == current.shape
    assert np.isfinite(voltage).all()
