from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class OverfitResult:
    """Metrics collected while fitting one fixed mini-batch."""

    loss: list[float]
    soc_rmse: list[float]
    soh_rmse: list[float]
    steps: int
    converged: bool


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _accumulate_metrics(
    sums: dict[str, float],
    losses: dict[str, torch.Tensor],
    batch_size: int,
) -> None:
    for key in sums:
        sums[key] += float(losses[key].detach()) * batch_size


def _validate_batch_limit(max_batches: int | None) -> None:
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive or None")


def train_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    gradient_clip_norm: float | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run one training epoch for synchronous SOC/SOH estimation."""
    _validate_batch_limit(max_batches)
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive or None")

    model.train()
    criterion.train()
    sums = {"loss": 0.0, "mse_soc": 0.0, "mse_soh": 0.0}
    samples = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        x = batch["x"].to(device)
        y_soc = batch["y_soc"].to(device)
        y_soh = batch["y_soh"].to(device)

        optimizer.zero_grad(set_to_none=True)
        losses = criterion(model(x), y_soc, y_soh)
        losses["loss"].backward()
        if gradient_clip_norm is not None:
            parameters = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=gradient_clip_norm,
            )
        optimizer.step()

        batch_size = x.shape[0]
        samples += batch_size
        _accumulate_metrics(sums, losses, batch_size)

    if samples == 0:
        raise ValueError("Training loader produced no batches")
    return {key: value / samples for key, value in sums.items()}


@torch.no_grad()
def eval_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate one epoch without updating trainable parameters."""
    _validate_batch_limit(max_batches)
    model.eval()
    criterion.eval()
    sums = {"loss": 0.0, "mse_soc": 0.0, "mse_soh": 0.0}
    samples = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        x = batch["x"].to(device)
        y_soc = batch["y_soc"].to(device)
        y_soh = batch["y_soh"].to(device)
        losses = criterion(model(x), y_soc, y_soh)

        batch_size = x.shape[0]
        samples += batch_size
        _accumulate_metrics(sums, losses, batch_size)

    if samples == 0:
        raise ValueError("Evaluation loader produced no batches")
    return {key: value / samples for key, value in sums.items()}


def _prediction_metrics(
    prediction: dict[str, torch.Tensor],
    y_soc: torch.Tensor,
    y_soh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    soc_mse = F.mse_loss(prediction["soc"], y_soc)
    soh_mse = F.mse_loss(prediction["soh"], y_soh)
    return soc_mse + soh_mse, soc_mse.sqrt(), soh_mse.sqrt()


def overfit_one_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    steps: int = 1_000,
    learning_rate: float = 3e-3,
    target_rmse: float = 0.02,
) -> OverfitResult:
    """Fit one fixed batch without dropout, weight decay or clipping."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if target_rmse <= 0:
        raise ValueError("target_rmse must be positive")

    required = {"x", "y_soc", "y_soh"}
    missing = required - set(batch)
    if missing:
        raise KeyError(f"Batch misses keys: {sorted(missing)}")

    x = batch["x"].detach().to(device)
    y_soc = batch["y_soc"].detach().to(device)
    y_soh = batch["y_soh"].detach().to(device)
    tensors = {"x": x, "y_soc": y_soc, "y_soh": y_soh}
    non_finite = [
        name
        for name, tensor in tensors.items()
        if not torch.isfinite(tensor).all()
    ]
    if non_finite:
        raise ValueError(f"Non-finite tensors: {non_finite}")

    model.to(device)
    previous_mode = model.training
    dropout_modules = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.Dropout)
    ]
    dropout_rates = [module.p for module in dropout_modules]
    for module in dropout_modules:
        module.p = 0.0
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    loss_history: list[float] = []
    soc_history: list[float] = []
    soh_history: list[float] = []
    completed_steps = 0
    converged = False

    try:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            total, soc_rmse, soh_rmse = _prediction_metrics(
                model(x),
                y_soc,
                y_soh,
            )
            loss_history.append(float(total.detach()))
            soc_history.append(float(soc_rmse.detach()))
            soh_history.append(float(soh_rmse.detach()))

            if max(soc_history[-1], soh_history[-1]) <= target_rmse:
                converged = True
                break

            total.backward()
            optimizer.step()
            completed_steps += 1
        else:
            with torch.no_grad():
                total, soc_rmse, soh_rmse = _prediction_metrics(
                    model(x),
                    y_soc,
                    y_soh,
                )
            loss_history.append(float(total))
            soc_history.append(float(soc_rmse))
            soh_history.append(float(soh_rmse))
            converged = max(soc_history[-1], soh_history[-1]) <= target_rmse
    finally:
        for module, rate in zip(dropout_modules, dropout_rates):
            module.p = rate
        model.train(previous_mode)

    return OverfitResult(
        loss=loss_history,
        soc_rmse=soc_history,
        soh_rmse=soh_history,
        steps=completed_steps,
        converged=converged,
    )
