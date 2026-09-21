"""Discrete charge-balance regularization for interval-mean KIT labels.

This is a weak, supervised physics constraint, not a PDE-solving PINN.
Future measured current is loss-only supervision. The network never sees it.
"""
from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PhysicsConfig:
    charge_weight: float = 0.01
    bounds_weight: float = 0.01
    soh_smooth_weight: float = 0.0
    interval_s: float = 30.0
    rated_capacity_ah: float = 3.0
    charge_sign: float = 1.0  # KIT: positive current charges the cell
    residual_scale: float = 0.01  # 1 percentage point of SOC per transition
    residual_tolerance: float = 0.002  # 0.2 pp, experimental, not calibrated
    soh_upper: float = 1.2  # dataset operating range, NOT a universal law

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Physics settings must be finite')
        if min(self.charge_weight, self.bounds_weight, self.soh_smooth_weight,
               self.residual_tolerance) < 0:
            raise ValueError('Physics weights/tolerance must be nonnegative')
        if min(self.interval_s, self.rated_capacity_ah, self.residual_scale, self.soh_upper) <= 0:
            raise ValueError('Physics scales must be positive')
        if self.charge_sign not in (-1, 1):
            raise ValueError('charge_sign must be +1 or -1')


def charge_increment(current, reference_soh, config):
    """Delta of adjacent interval-mean SOC under piecewise-constant current.

For bin width dt, mean(z)[k+1]-mean(z)[k] = dt*(I[k]+I[k+1])/(2*3600*Q).
This identity is exact for constant Q, eta=1 and constant current in each
bin. Mean current alone is insufficient for exact sub-bin pulse integration.
Q comes from existing target labels and is detached: predicted SOH cannot
grow to evade this penalty. No gradient/physical supervision of SOH via Q.
"""
    if current.ndim != 2 or current.shape != reference_soh.shape or current.shape[1] < 2:
        raise ValueError('Charge loss requires matching [B,H>=2] current/SOH')
    current = current.detach().float()
    reference_soh = reference_soh.detach().float()
    if not torch.isfinite(current).all() or not torch.isfinite(reference_soh).all() or (reference_soh <= 0).any():
        raise ValueError('Finite physical current and positive reference SOH required')
    capacity = config.rated_capacity_ah * (reference_soh[:, :-1] + reference_soh[:, 1:]) / 2
    return config.charge_sign * config.interval_s * (current[:, :-1] + current[:, 1:]) / (2 * 3600 * capacity)


def physics_terms(pred, reference_soh, current, config):
    soc, soh = pred['soc'].float(), pred['soh'].float()
    zero = soc.new_zeros(())
    charge, residual_rmse = zero, zero
    if config.charge_weight:
        if current is None:
            raise ValueError('Charge loss enabled but batch has no physics_current_a')
        expected = charge_increment(current, reference_soh, config)
        residual = soc.diff(dim=1) - expected
        excess = (residual.abs() - config.residual_tolerance).clamp_min(0) / config.residual_scale
        charge = F.smooth_l1_loss(excess, torch.zeros_like(excess))
        residual_rmse = residual.square().mean().sqrt()
    bounds = (F.relu(-soc).square() + F.relu(soc - 1).square()).mean()
    bounds = bounds + (F.relu(-soh).square() + F.relu(soh - config.soh_upper).square()).mean()
    smooth = soh.diff(dim=1).square().mean() if soh.shape[1] > 1 else zero
    total = config.charge_weight * charge + config.bounds_weight * bounds + config.soh_smooth_weight * smooth
    return dict(physics_loss=total, charge_loss=charge, bounds_loss=bounds,
                soh_smooth_loss=smooth, charge_residual_mse=residual_rmse.square())


@torch.no_grad()
def diagnose_physics_targets(batch, config):
    """Use TRAIN only to assess label/physics mismatch before choosing weights."""
    expected = charge_increment(batch['physics_current_a'], batch['y_soh'], config)
    observed = batch['y_soc'].float().diff(dim=1)
    residual = (observed - expected).flatten()
    return dict(transitions=residual.numel(),
                target_charge_rmse_pp=float(residual.square().mean().sqrt().cpu()) * 100,
                target_charge_p95_abs_pp=float(torch.quantile(residual.abs(), 0.95).cpu()) * 100,
                fraction_above_tolerance=float((residual.abs() > config.residual_tolerance).float().mean().cpu()),
                expected_change_rms_pp=float(expected.square().mean().sqrt().cpu()) * 100)
