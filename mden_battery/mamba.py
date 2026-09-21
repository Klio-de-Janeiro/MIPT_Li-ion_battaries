from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """RMS normalization with stable reductions under mixed precision."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            value = x.to(dtype)
            value = value * (value.square().mean(-1, keepdim=True) + self.eps).rsqrt()
            return (value * self.weight.to(dtype)).to(x.dtype)


def selective_scan_native(
    u: torch.Tensor, delta: torch.Tensor, a: torch.Tensor,
    b: torch.Tensor, c: torch.Tensor, d: torch.Tensor,
    mode: str = "parallel",
) -> torch.Tensor:
    """Mamba-1 recurrence; inclusive associative scan or serial reference.

    u, delta: [B,T,D]; A: [D,N]; B,C: [B,T,N]; D: [D].
    Both implementations accumulate in FP32 (FP64 for numerical tests).
    Parallel scan uses log2(T) stages, without unstable division by cumprod.
    """
    dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=u.device.type, enabled=False):
        u, delta, a, b, c, d = [v.to(dtype) for v in (u, delta, a, b, c, d)]
        decay = torch.exp(delta.unsqueeze(-1) * a)
        drive = delta.unsqueeze(-1) * b.unsqueeze(2) * u.unsqueeze(-1)
        if mode == "serial":
            state = torch.zeros_like(drive[:, 0])
            states = []
            for t in range(u.shape[1]):
                state = decay[:, t] * state + drive[:, t]
                states.append(state)
            state_all = torch.stack(states, dim=1)
        elif mode == "parallel":
            offset = 1
            while offset < u.shape[1]:
                drive = torch.cat([
                    drive[:, :offset],
                    drive[:, offset:] + decay[:, offset:] * drive[:, :-offset],
                ], dim=1)
                decay = torch.cat([
                    decay[:, :offset],
                    decay[:, offset:] * decay[:, :-offset],
                ], dim=1)
                offset *= 2
            state_all = drive
        else:
            raise ValueError("scan must be serial or parallel")
        return (state_all * c.unsqueeze(2)).sum(-1) + d * u


class NativeMambaBlock(nn.Module):
    """Figure 4(a): two input RMSNorms, mixer, Linear, output RMSNorm.

    figure4 is a literal reconstruction of the visible diagram: no external
    residual. Independent input norms are an explicit interpretation of the
    two drawn RMSNorm blocks. standard retains a conventional residual block
    for ablations; it is not labeled a faithful reproduction of Figure 4.
    SSM parameters/initialization follow official Mamba-1, not unreported
    hyperparameters of the MDEN authors.
    """

    def __init__(
        self, d_model: int, d_state: int = 16, d_conv: int = 4,
        expand: int = 2, dt_rank: int | None = None, dropout: float = 0.0,
        scan: str = "parallel", layout: str = "figure4",
        backend: str = "native",
    ) -> None:
        super().__init__()
        if min(d_model, d_state, d_conv, expand) <= 0:
            raise ValueError("Mamba dimensions must be positive")
        if layout not in {"figure4", "standard"}:
            raise ValueError("layout must be figure4 or standard")
        self.d_model, self.d_state = d_model, d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        self.scan, self.layout, self.backend = scan, layout, backend
        self.norm = RMSNorm(d_model)
        self.gate_norm = RMSNorm(d_model) if layout == "figure4" else nn.Identity()
        self.output_norm = RMSNorm(d_model) if layout == "figure4" else nn.Identity()
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv, padding=d_conv - 1,
            groups=self.d_inner,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank**-0.5, self.dt_rank**-0.5)
        dt = torch.exp(torch.rand(self.d_inner) * math.log(0.1 / 0.001) + math.log(0.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.dt_proj.bias._no_reinit = True
        self.A_log = nn.Parameter(torch.arange(1, d_state + 1).float().log().repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _selective_scan(self, u: torch.Tensor) -> torch.Tensor:
        params = self.x_proj(u)
        raw_dt, b, c = params.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=u.device.type, enabled=False):
            delta = F.softplus(F.linear(raw_dt.to(dtype), self.dt_proj.weight.to(dtype), self.dt_proj.bias.to(dtype)))
            a = -self.A_log.to(dtype).exp()
            if self.backend == "official":
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                if u.device.type != "cuda":
                    raise RuntimeError("Official selective scan requires CUDA; select native for CPU")
                result = selective_scan_fn(
                    u.float().transpose(1, 2).contiguous(), delta.float().transpose(1, 2).contiguous(),
                    a.float(), b.float().transpose(1, 2).contiguous(), c.float().transpose(1, 2).contiguous(),
                    self.D.float(), delta_softplus=False,
                ).transpose(1, 2)
            else:
                scan_fn = getattr(self, "scan_function", selective_scan_native)
                result = scan_fn(u, delta, a, b, c, self.D, self.scan)
        return result.to(u.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layout == "figure4":
            w_u, w_gate = self.in_proj.weight.chunk(2, dim=0)
            u = F.linear(self.norm(x), w_u)
            gate = F.linear(self.gate_norm(x), w_gate)
        else:
            u, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        u = F.silu(self.conv1d(u.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2))
        y = self._selective_scan(u) * F.silu(gate)
        y = self.dropout(self.out_proj(y))
        return self.output_norm(y) if self.layout == "figure4" else x + y


class MambaSequence(nn.Module):
    """Same learned parameters for native and official selective-scan kernels."""

    def __init__(
        self, d_model: int, d_state: int = 16, d_conv: int = 4,
        expand: int = 2, dropout: float = 0.0, backend: str = "native",
        scan: str = "parallel", layout: str = "figure4",
    ) -> None:
        super().__init__()
        if backend not in {"native", "official", "auto"}:
            raise ValueError("backend must be native, official or auto")
        resolved = "native"
        if backend in {"auto", "official"}:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA unavailable")
                resolved = "official"
            except (ImportError, OSError, RuntimeError) as error:
                if backend == "official":
                    raise RuntimeError("Compatible mamba_ssm/CUDA required") from error
                warnings.warn(f"Mamba auto resolved to native: {error}", stacklevel=2)
        self.backend = resolved
        self.impl = NativeMambaBlock(
            d_model, d_state, d_conv, expand, dropout=dropout,
            backend=resolved, scan=scan, layout=layout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.impl(x)
