"""Training and evaluation shared by the audited notebook and regression tests."""
from __future__ import annotations

import contextlib
import gc
import inspect
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Precision:
    device: torch.device
    enabled: bool
    dtype: torch.dtype

    def context(self):
        return torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.enabled)

    def scaler(self):
        return torch.amp.GradScaler('cuda', enabled=self.enabled and self.dtype == torch.float16)


def configure_precision(device, use_amp=True, *, amp_dtype='auto', allow_tf32=False):
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    enabled = use_amp and device.type == 'cuda'
    native_bf16 = enabled and torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported()
    # A hardware-safe selection, not a diagnosis of any past slow run.
    if amp_dtype not in {'auto', 'float16', 'bfloat16', 'float32'}:
        raise ValueError('Unknown AMP dtype')
    if amp_dtype == 'bfloat16' and enabled and not native_bf16:
        raise ValueError('Native BF16 is unavailable on this device; use float16')
    dtype = torch.bfloat16 if native_bf16 else torch.float16
    if amp_dtype in {'float16', 'bfloat16'}:
        dtype = getattr(torch, amp_dtype)
    if amp_dtype == 'float32':
        enabled = False
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.benchmark = False
    return Precision(device, enabled, dtype)


def move_batch_to_device(batch, device):
    return {name: tensor.contiguous().to(device, non_blocking=tensor.is_pinned())
            for name, tensor in batch.items()}


def prefetch_batches(loader, device, enabled=True):
    """Bounded CPU prefetch + pinned copies on a separate CUDA stream.

    Only CPU batches run in the worker. Already-cached CUDA batches pass
    through unchanged. Keep two prepared batches at most, not whole epochs.
    """
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    if not enabled or device.type != 'cuda':
        yield from loader
        return
    iterator = iter(loader)
    first = next(iterator, None)
    if first is None:
        return
    if all(v.device == device for v in first.values()):
        yield first
        yield from iterator
        return

    def pin(batch):
        if batch is None:
            return None
        if any(v.device.type != 'cpu' for v in batch.values()):
            raise ValueError('Prefetch expects consistently CPU batches')
        return {k: v.contiguous().pin_memory() for k, v in batch.items()}

    def next_cpu():
        return pin(next(iterator, None))

    stream = torch.cuda.Stream(device=device)
    with ThreadPoolExecutor(max_workers=1) as worker:
        future = worker.submit(next_cpu)
        with torch.cuda.stream(stream):
            ready = move_batch_to_device(pin(first), device)
        while ready is not None:
            current = torch.cuda.current_stream(device)
            current.wait_stream(stream)
            batch = ready
            for tensor in batch.values():
                tensor.record_stream(current)
            cpu_batch = future.result()
            if cpu_batch is None:
                ready = None
            else:
                future = worker.submit(next_cpu)
                with torch.cuda.stream(stream):
                    ready = move_batch_to_device(cpu_batch, device)
            yield batch


def build_optimizer(model, criterion, *, learning_rate=1e-3, weight_decay=1e-4):
    parameters = list(model.parameters()) + list(criterion.parameters())
    decay, no_decay = [], []
    for p in parameters:
        if p.requires_grad:
            (no_decay if getattr(p, '_no_weight_decay', False) else decay).append(p)
    groups = [{'params': decay, 'weight_decay': weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]
    kwargs = {'lr': learning_rate}
    if parameters[0].device.type == 'cuda' and 'fused' in inspect.signature(torch.optim.AdamW).parameters:
        kwargs['fused'] = True
    return torch.optim.AdamW(groups, **kwargs)


def compute_batch_loss(criterion, outputs, batch):
    if hasattr(criterion, "from_batch"):
        return criterion.from_batch(outputs, batch)
    return criterion(outputs, batch["y_soc"], batch["y_soh"])


METRIC_NAMES = ("loss", "mse_soc", "mse_soh", "sigma_soc", "sigma_soh",
                "weighted_soc", "weighted_soh", "regularizer", "data_loss",
                "physics_loss", "charge_loss", "bounds_loss", "soh_smooth_loss", "charge_residual_mse")


class MetricAccumulator:
    """Reduce GPU tensors per batch, copy compact statistics once per epoch."""

    def __init__(self, horizon, device):
        self.n = 0
        self.sums = torch.zeros(len(METRIC_NAMES), device=device, dtype=torch.float64)
        self.errors = torch.zeros(4, horizon, device=device, dtype=torch.float64)
        self.target_mean = torch.zeros(2, device=device, dtype=torch.float64)
        self.target_m2 = torch.zeros(2, device=device, dtype=torch.float64)
        self.target_abs = torch.zeros(2, device=device, dtype=torch.float64)
        self.target_n = 0

    @torch.no_grad()
    def add(self, outputs, losses, batch):
        size = batch['x'].shape[0]
        self.n += size
        names = METRIC_NAMES
        zero = losses['loss'].detach().new_zeros(())
        self.sums += torch.stack([losses.get(k, zero).detach() for k in names]).double()*size
        e_soc = outputs['soc'].detach().float() - batch['y_soc'].float()
        e_soh = outputs['soh'].detach().float() - batch['y_soh'].float()
        self.errors += torch.stack([e_soc.square().sum(0), e_soh.square().sum(0), e_soc.abs().sum(0), e_soh.abs().sum(0)]).double()
        target = torch.stack([batch['y_soc'], batch['y_soh']]).reshape(2, -1).float()
        n = target.shape[1]
        variance, mu = torch.var_mean(target, dim=1, correction=0)
        mu = mu.double()
        delta = mu - self.target_mean
        total = self.target_n + n
        self.target_m2 += variance.double()*n + delta.square()*self.target_n*n/total
        self.target_mean += delta*n/total
        self.target_abs += target.abs().sum(1)
        self.target_n = total

    def result(self):
        if not self.n:
            raise ValueError('Loader produced no samples')
        if not torch.isfinite(self.sums).all() or not torch.isfinite(self.errors).all():
            raise FloatingPointError('Non-finite epoch statistics')
        sums = (self.sums/self.n).cpu().tolist()
        errors = self.errors.cpu().numpy()
        m2, denom = self.target_m2.cpu().numpy(), self.target_abs.cpu().numpy()
        names = METRIC_NAMES
        metrics = dict(zip(names, sums))
        metrics['charge_residual_rmse'] = math.sqrt(max(0., metrics['charge_residual_mse']))
        for i, task in enumerate(('soc', 'soh')):
            rmse_h = np.sqrt(errors[i]/self.n)
            metrics['rmse_' + task + '_h'] = rmse_h.tolist()
            metrics['mae_' + task + '_h'] = (errors[i+2]/self.n).tolist()
            metrics['rmse_' + task] = float(np.sqrt(np.mean(errors[i]/self.n)))
            metrics['mae_' + task] = float(np.mean(errors[i+2]/self.n))
            metrics['r2_' + task] = float(1-errors[i].sum()/m2[i]) if m2[i] > 1e-12 else None
            metrics['wmape_' + task] = float(errors[i+2].sum()/denom[i]) if denom[i] > 1e-12 else None
        metrics['balanced_rmse'] = math.sqrt((metrics['rmse_soc']**2 + metrics['rmse_soh']**2)/2)
        metrics['samples'] = self.n
        return metrics


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _finish_step(optimizer, scaler, parameters, factor, clip_norm):
    scaler.unscale_(optimizer)
    if factor != 1.0:
        grads = [p.grad for p in parameters if p.grad is not None]
        torch._foreach_mul_(grads, factor)
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_norm, error_if_nonfinite=not scaler.is_enabled())
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return (~torch.isfinite(norm)).to(torch.int32)


def train_epoch_amp(model, criterion, loader, optimizer, scaler, precision,
                    *, accumulation_steps=1, gradient_clip_norm=1.0, prefetch=True):
    if accumulation_steps < 1 or gradient_clip_norm <= 0:
        raise ValueError('Invalid accumulation or clipping')
    model.train(); criterion.train()
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    optimizer.zero_grad(set_to_none=True)
    acc = MetricAccumulator(model.config.horizon, precision.device)
    pending, pending_samples, reference = 0, 0, None
    skipped = torch.zeros((), device=precision.device, dtype=torch.int32)
    if precision.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(precision.device)
    _sync(precision.device); started = time.perf_counter()
    for batch in prefetch_batches(loader, precision.device, prefetch):
        batch = move_batch_to_device(batch, precision.device)
        n = batch['x'].shape[0]
        reference = reference or n*accumulation_steps
        with precision.context():
            outputs = model(batch['x'])
        losses = compute_batch_loss(criterion, outputs, batch)
        if not torch.isfinite(losses['loss']):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError('Non-finite loss before optimizer update')
        scaler.scale(losses['loss'] * (n/reference)).backward()
        pending += 1; pending_samples += n
        acc.add(outputs, losses, batch)
        if pending == accumulation_steps:
            skipped += _finish_step(optimizer, scaler, parameters, reference/pending_samples, gradient_clip_norm)
            pending, pending_samples = 0, 0
    if pending:
        skipped += _finish_step(optimizer, scaler, parameters, reference/pending_samples, gradient_clip_norm)
    _sync(precision.device)
    metrics = acc.result()
    metrics['samples_per_second'] = acc.n / max(time.perf_counter()-started, 1e-9)
    metrics['skipped_steps'] = int(skipped.cpu())
    metrics['peak_vram_fraction'] = (torch.cuda.max_memory_allocated(precision.device) / torch.cuda.get_device_properties(precision.device).total_memory
                                    if precision.device.type == 'cuda' else 0.0)
    metrics['peak_reserved_fraction'] = (torch.cuda.max_memory_reserved(precision.device) / torch.cuda.get_device_properties(precision.device).total_memory
                                        if precision.device.type == 'cuda' else 0.0)
    return metrics


@torch.inference_mode()
def eval_epoch_amp(model, criterion, loader, precision, *, prefetch=True):
    model.eval(); criterion.eval()
    acc = MetricAccumulator(model.config.horizon, precision.device)
    _sync(precision.device); started = time.perf_counter()
    for batch in prefetch_batches(loader, precision.device, prefetch):
        batch = move_batch_to_device(batch, precision.device)
        with precision.context():
            outputs = model(batch['x'])
        losses = compute_batch_loss(criterion, outputs, batch)
        acc.add(outputs, losses, batch)
    _sync(precision.device)
    metrics = acc.result()
    metrics['samples_per_second'] = acc.n / max(time.perf_counter()-started, 1e-9)
    return metrics


def choose_micro_batch_size(model_config, dataset, precision, *, fixed=None,
                            candidates=(64, 128, 256, 512, 1024, 2048, 4096),
                            learning_rate=1e-3, weight_decay=1e-4,
                            max_vram_fraction=0.88, warmup_steps=3, timed_steps=10,
                            fem_execution='dynamic', compile_scan=False, loss_config=None):
    """Actual full training-step probe, finite loss/gradients, representative windows.

    Select the smallest batch within 5% of best throughput. This is only a
    local timing probe; it cannot promise best generalization or all-shape OOM safety.
    Probe models never touch the real training model or the dataset epoch.
    """
    from .model import MDEN
    from .loss import MDENJointLoss
    from .performance import configure_execution, verify_execution
    if fixed is not None and fixed <= 0:
        raise ValueError('Fixed batch must be positive')
    if not 0 < max_vram_fraction < 1 or min(warmup_steps, timed_steps) < 1:
        raise ValueError('Use positive probe steps and a VRAM fraction in (0,1)')
    if precision.device.type == 'cpu':
        size = min(fixed or 16, dataset.total_windows)
        return size, {'batch_size': size, 'probes': [], 'samples_per_second': None, 'reason': 'CPU: probe skipped'}
    sizes = [fixed] if fixed is not None else [b for b in candidates if b <= dataset.total_windows]
    if not sizes:
        sizes = [min(16, dataset.total_windows)]
    records = []
    for size in sizes:
        if size > dataset.total_windows:
            raise ValueError('Fixed probe batch exceeds dataset size')
        gc.collect(); torch.cuda.empty_cache()
        model = criterion = optimizer = batch = scaler = None
        try:
            with torch.random.fork_rng(devices=[precision.device.index or 0]):
                torch.manual_seed(42)
                model = MDEN(model_config).to(precision.device)
                configure_execution(model, fem_execution, compile_scan)
                criterion = MDENJointLoss(model_config.input_dim, **(loss_config or {})).to(precision.device)
                optimizer = build_optimizer(model, criterion, learning_rate=learning_rate, weight_decay=weight_decay)
                scaler = precision.scaler()
                verification = verify_execution(
                    model, dataset.sample_batch(min(4, dataset.total_windows)),
                    precision, fem_execution, compile_scan)
                resolved_execution = configure_execution(model, fem_execution, compile_scan)
                # Include batch materialization and transfer in the timing.
                rng = np.random.default_rng(42)
                ids = [rng.choice(dataset.total_windows, int(size), replace=False)
                       for _ in range(3)]
                def batches(steps):
                    for j in range(steps):
                        yield dataset.fetch(ids[j % len(ids)])
                train_epoch_amp(model, criterion, batches(warmup_steps), optimizer, scaler, precision)
                metrics = train_epoch_amp(model, criterion, batches(timed_steps), optimizer, scaler, precision)
                if metrics['skipped_steps']:
                    raise FloatingPointError('Probe skipped an AMP update; lower scale or disable AMP to diagnose')
                peak = metrics['peak_vram_fraction']
                records.append(dict(batch_size=int(size), fits=peak <= max_vram_fraction,
                                    peak_fraction=peak, samples_per_second=metrics['samples_per_second'],
                                    peak_reserved_fraction=metrics['peak_reserved_fraction'],
                                    compile_scan=resolved_execution['compile_scan'],
                                    verification=verification))
        except torch.cuda.OutOfMemoryError:
            records.append(dict(batch_size=int(size), fits=False, peak_fraction=None, samples_per_second=0.0))
        finally:
            model = criterion = optimizer = batch = scaler = None
            gc.collect(); torch.cuda.empty_cache()
        print(records[-1])
    valid = [r for r in records if r['fits']]
    if not valid:
        raise RuntimeError('No probe fits memory budget. Reduce fixed batch/candidates or choose CPU data cache.')
    best_speed = max(r['samples_per_second'] for r in valid)
    chosen = min((r for r in valid if r['samples_per_second'] >= 0.95*best_speed), key=lambda r: r['batch_size'])
    return chosen['batch_size'], dict(chosen, probes=records)
