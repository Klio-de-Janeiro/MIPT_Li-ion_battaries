from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import asdict
from .physics import PhysicsConfig, physics_terms


class UncertaintyPredictor(nn.Module):
    """Eq. 20: Pool -> Flatten -> Linear -> Softplus, independently per sample."""

    def __init__(self, feature_dim: int, sigma_floor: float = 1e-4) -> None:
        super().__init__()
        if sigma_floor <= 0:
            raise ValueError("sigma_floor must be positive")
        self.sigma_floor = sigma_floor
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(feature_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, float(torch.log(torch.expm1(torch.tensor(1.0)))))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=features.device.type, enabled=False):
            x = self.pool(features.float().transpose(1, 2)).flatten(1)
            return F.softplus(self.linear(x)).squeeze(-1).clamp_min(self.sigma_floor)


class MDENJointLoss(nn.Module):
    """Eq. 19-21, averaged over samples after horizon MSE.

    The per-sample reduction is explicit: the paper does not specify how a
    vector of feature-derived sigmas is reduced against batch-level MSE.
    Sensitive arithmetic stays FP32 under AMP. A numerical sigma floor is
    required: log(1+sigma) alone does NOT prevent sigma -> 0 when MSE -> 0.
    """

    def __init__(self, feature_dim: int, sigma_floor: float = 1e-4,
                 physics: dict | PhysicsConfig | None = None) -> None:
        super().__init__()
        self.physics = (PhysicsConfig(**physics) if isinstance(physics, dict) else physics)
        self.sigma_floor = sigma_floor
        self.soc_uncertainty = UncertaintyPredictor(feature_dim, sigma_floor)
        self.soh_uncertainty = UncertaintyPredictor(feature_dim, sigma_floor)

    def loss_config(self):
        return {'sigma_floor': self.sigma_floor,
                'physics': asdict(self.physics) if self.physics is not None else None}

    def from_batch(self, pred, batch):
        return self(pred, batch['y_soc'], batch['y_soh'], batch.get('physics_current_a'))

    def forward(self, pred: dict[str, torch.Tensor], y_soc: torch.Tensor,
                y_soh: torch.Tensor, physics_current_a: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if pred['soc'].shape != y_soc.shape or pred['soh'].shape != y_soh.shape:
            raise ValueError("Prediction/target mismatch")
        if y_soc.ndim != 2 or y_soc.shape != y_soh.shape:
            raise ValueError("Targets must have matching [B,H] shapes")
        with torch.autocast(device_type=y_soc.device.type, enabled=False):
            mse_soc = (pred['soc'].float() - y_soc.float()).square().mean(1)
            mse_soh = (pred['soh'].float() - y_soh.float()).square().mean(1)
            soc = self.soc_uncertainty(pred['soc_fused'])
            soh = self.soh_uncertainty(pred['soh_fused'])
            weighted_soc = mse_soc / (2 * soc.square())
            weighted_soh = mse_soh / (2 * soh.square())
            regularizer = soc.log1p() + soh.log1p()
            total = weighted_soc + weighted_soh + regularizer
            result = {
                'loss': total.mean(), 'data_loss': total.mean(), 'mse_soc': mse_soc.mean(), 'mse_soh': mse_soh.mean(),
                'sigma_soc': soc.mean(), 'sigma_soh': soh.mean(),
                'weighted_soc': weighted_soc.mean(), 'weighted_soh': weighted_soh.mean(),
                'regularizer': regularizer.mean(),
            }
            if self.physics is not None:
                terms = physics_terms(pred, y_soh, physics_current_a, self.physics)
                result.update(terms)
                result['loss'] = result['loss'] + terms['physics_loss']
        return result


PINNJointLoss = MDENJointLoss
