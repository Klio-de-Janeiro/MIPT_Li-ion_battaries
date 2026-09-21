from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class FFTPeriodSelection:
    """FFT-derived periods and adaptive per-sample weights."""

    periods: torch.Tensor  # [B,k] per_sample or [k] batch_shared
    frequency_indices: torch.Tensor  # same shape as periods; positive FFT-bin indices
    scores: torch.Tensor  # [B, k], amplitudes used for softmax aggregation


def fft_select_periods(
    x: torch.Tensor, top_k: int, mode: str = "per_sample",
) -> FFTPeriodSelection:
    """Eq. 7-9: non-DC top-k, ceil(T/f); FFT always in FP32 or FP64.

    Avg/Softmax axes are not fully specified in the article. The default
    averages channels within each sample; scores are FFT amplitudes. This
    explicit interpretation makes inference independent of other samples.
    batch_shared retains the old TimesNet-style approximation for ablations.
    Ties use stable frequency order, avoiding batch-dependent topk ties.
    """
    if x.ndim != 3 or x.shape[1] < 3 or top_k <= 0:
        raise ValueError("Expected [B,T>=3,C] and top_k > 0")
    if mode not in {"per_sample", "batch_shared"}:
        raise ValueError(f"Unknown FFT selection: {mode}")
    t = x.shape[1]
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type, enabled=False):
        amp = torch.fft.rfft(x.to(dtype), dim=1).abs().mean(-1)
        positive = amp[:, 1:]
        k = min(int(top_k), t // 2)
        ranking = positive if mode == "per_sample" else positive.mean(0)
        idx = torch.argsort(ranking, dim=-1, descending=True, stable=True)[..., :k] + 1
        scores = amp.gather(1, idx) if mode == "per_sample" else amp[:, idx]
        periods = torch.div(t + idx - 1, idx, rounding_mode="floor")
    return FFTPeriodSelection(periods, idx.detach(), scores)


class ConvRound(nn.Module):
    """One Conv(.) round in Fig. 3 / Eq. (11).

    1x1 channel expansion -> 3x3 grouped convolution -> 1x1 reduction.
    The paper specifies grouped convolution but not the number of groups;
    ``groups`` is therefore an explicit reproducibility hyperparameter.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 32,
        groups: int = 4,
        dropout: float = 0.0,
        activation: str = "identity",
    ) -> None:
        super().__init__()
        if hidden_channels % groups != 0:
            raise ValueError(
                f"hidden_channels={hidden_channels} must be divisible "
                f"by groups={groups}."
            )

        self.expand = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.group_conv = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
        )
        self.reduce = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        if activation not in {"identity", "gelu"}:
            raise ValueError("activation must be identity or gelu")
        self.activation = nn.Identity() if activation == "identity" else nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fuse_linear = activation == "identity" and dropout == 0.0

    def fused_parameters(self):
        """Collapse linear 1x1 -> grouped 3x3 -> 1x1, keeping bias at edges.

        A constant-one input channel represents expansion bias *inside* the
        image. Padding that channel with zero exactly preserves edge behavior.
        Parameters stay differentiable; nothing is detached or cached across steps.
        """
        e, g, r = self.expand, self.group_conv, self.reduce
        groups, width = g.groups, g.in_channels // g.groups
        with torch.autocast(device_type=e.weight.device.type, enabled=False):
            w = torch.cat([e.weight[:, :, 0, 0], e.bias[:, None]], dim=1)
            w = w.reshape(groups, width, -1)
            middle = g.weight.reshape(groups, width, width, 3, 3)
            hidden = torch.einsum("gohxy,ghc->gocxy", middle, w)
            kernel = torch.einsum("ago,gocxy->acxy", r.weight[:, :, 0, 0].reshape(-1, groups, width), hidden)
            bias = r.weight[:, :, 0, 0] @ g.bias + r.bias
        return kernel, bias

    @staticmethod
    def apply_fused(x, parameters):
        ones = torch.ones_like(x[:, :1])
        return F.conv2d(torch.cat([x, ones], dim=1), *parameters, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fuse_linear:
            return self.apply_fused(x, self.fused_parameters())
        x = self.dropout(self.activation(self.expand(x)))
        x = self.dropout(self.activation(self.group_conv(x)))
        return self.reduce(x)


class FeatureExtractionModule(nn.Module):
    """Frequency-aware Feature Extraction Module (FEM) from Sec. 3.2.1.

    Pipeline:
      X_1D -> FFT/top-k -> k period-dependent 2D tensors -> Round_4(Conv)
      -> reshape/remove padding -> adaptive softmax aggregation (no extra residual in Eq. 12).

    Important distinction:
    * one FEM contains four Conv(.) rounds (Eq. 11, Round_4);
    * the SOC/shared/SOH expert networks stack 2/4/6 FEMs respectively
      (Sec. 3.2.1 and the FEM-count ablation in Sec. 4.3.1).
    """

    def __init__(
        self,
        channels: int,
        top_k: int = 4,
        hidden_channels: int = 32,
        groups: int = 4,
        dropout: float = 0.0,
        conv_rounds: int = 4,
        residual: bool = False,
        fft_mode: str = "per_sample",
        activation: str = "identity",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.top_k = top_k
        self.residual = residual
        self.fft_mode = fft_mode
        # Runtime choice only: parameter names and model mathematics stay the same.
        self.execution = "dynamic"
        self.rounds = nn.Sequential(
            *[
                ConvRound(
                    channels=channels,
                    hidden_channels=hidden_channels,
                    groups=groups,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(conv_rounds)
            ]
        )

    @staticmethod
    def _to_2d(x: torch.Tensor, period: int) -> tuple[torch.Tensor, int]:
        """Pad along time and reshape [B,T,C] -> [B,C,H,period]."""
        b, t, c = x.shape
        length = ((t + period - 1) // period) * period
        pad_t = length - t
        if pad_t:
            x = F.pad(x, (0, 0, 0, pad_t), mode="constant", value=0.0)
        x2d = (
            x.reshape(b, length // period, period, c)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return x2d, length

    @staticmethod
    def _to_1d(x2d: torch.Tensor, original_t: int) -> torch.Tensor:
        """[B,C,H,W] -> [B,T,C] and remove zero padding (Rem in Eq. 12)."""
        b, c, h, w = x2d.shape
        x = x2d.permute(0, 2, 3, 1).contiguous().reshape(b, h * w, c)
        return x[:, :original_t, :]

    def _kernels(self):
        """Fuse all four linear rounds together, reducing tiny kernel launches."""
        if not all(r.fuse_linear for r in self.rounds):
            return [r.fused_parameters() if r.fuse_linear else None for r in self.rounds]
        first = self.rounds[0]
        groups = first.group_conv.groups
        width = first.group_conv.in_channels // groups
        count = len(self.rounds)
        with torch.autocast(device_type=first.expand.weight.device.type, enabled=False):
            expand = torch.stack([r.expand.weight[:, :, 0, 0] for r in self.rounds])
            expand_bias = torch.stack([r.expand.bias for r in self.rounds])
            middle = torch.stack([r.group_conv.weight for r in self.rounds])
            middle_bias = torch.stack([r.group_conv.bias for r in self.rounds])
            reduce = torch.stack([r.reduce.weight[:, :, 0, 0] for r in self.rounds])
            reduce_bias = torch.stack([r.reduce.bias for r in self.rounds])
            w = torch.cat([expand, expand_bias[:, :, None]], dim=-1)
            w = w.reshape(count, groups, width, -1)
            middle = middle.reshape(count, groups, width, width, 3, 3)
            hidden = torch.einsum('rgohxy,rghc->rgocxy', middle, w)
            kernel = torch.einsum('rago,rgocxy->racxy',
                                  reduce.reshape(count, -1, groups, width), hidden)
            bias = (reduce @ middle_bias[:, :, None]).squeeze(-1) + reduce_bias
        return list(zip(kernel.unbind(), bias.unbind()))

    def forward(self, x: torch.Tensor, selection=None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,C], got {tuple(x.shape)}")
        if x.shape[-1] != self.channels:
            raise ValueError(
                f"Expected C={self.channels}, got C={x.shape[-1]} "
                f"for shape={tuple(x.shape)}"
            )

        _, t, _ = x.shape
        if selection is None:
            selection = fft_select_periods(x, self.top_k, self.fft_mode)
        weights = selection.scores.softmax(-1)
        periods = selection.periods
        if periods.ndim == 1:
            periods = periods.unsqueeze(0).expand(x.shape[0], -1)
        if self.execution != "dynamic":
            out = self._forward_static(x, periods, weights)
            return out + x if self.residual else out
        # One schedule transfer per FEM instead of .item() for every branch.
        schedule = periods.detach().cpu().numpy()
        kernels = self._kernels()
        out = torch.zeros_like(x)
        for period in np.unique(schedule):
            rows = np.flatnonzero(np.any(schedule == period, axis=1))
            indices = torch.as_tensor(rows, device=x.device)
            branch = self._to_2d(x.index_select(0, indices), int(period))[0]
            for round_, kernel in zip(self.rounds, kernels):
                branch = round_.apply_fused(branch, kernel) if kernel is not None else round_(branch)
            restored = self._to_1d(branch, t)
            mask = periods.index_select(0, indices) == int(period)
            # Equal periods are the same deterministic Conv path. Sum their
            # weights before convolution; gradients through scores remain.
            w = (weights.index_select(0, indices) * mask).sum(-1)
            contribution = restored.to(x.dtype) * w[:, None, None].to(x.dtype)
            out = out.index_add(0, indices, contribution)
        return out + x if self.residual else out

    def _forward_static(self, x, periods, weights):
        """GPU-only routing over the finite set ceil(T/f), with no host reads.

        static: evaluate every period on the batch, then weight on device.
        matrix: build each affine Conv chain on basis vectors, then apply
        it to the full batch by GEMM. Only valid with identity and no dropout.
        Padding cells participate in every round, exactly as in dynamic mode.
        """
        if self.execution not in {"static", "matrix"}:
            raise ValueError("FEM execution must be dynamic, static or matrix")
        t, c = x.shape[1:]
        possible = sorted({(t + f - 1) // f for f in range(1, t // 2 + 1)})
        kernels = self._kernels()
        if self.execution == "matrix":
            if not all(r.fuse_linear for r in self.rounds):
                raise ValueError("matrix FEM requires identity and dropout=0")
            if torch.is_autocast_enabled(x.device.type) or x.dtype not in {torch.float32, torch.float64}:
                raise ValueError("matrix FEM requires FP32/FP64 with autocast disabled")
            return self._forward_matrix(x, possible, periods, weights, kernels)
        out = torch.zeros_like(x)
        for period in possible:
            branch = self._to_2d(x, period)[0]
            for round_, kernel in zip(self.rounds, kernels):
                branch = (round_.apply_fused(branch, kernel)
                          if kernel is not None else round_(branch))
            w = (weights * (periods == period)).sum(-1)
            out = out + self._to_1d(branch, t).to(x.dtype) * w[:, None, None].to(x.dtype)
        return out

    def _forward_matrix(self, x, possible, periods, weights, kernels):
        """Construct coefficients directly, avoiding subtracting large biases.

        First T*C basis rows carry only the linear response. The final row
        carries all affine offsets. Parameters remain connected to autograd;
        no weight-dependent matrix is cached across optimizer steps.
        """
        t, c = x.shape[1:]
        n = t * c
        if n > 512:
            raise ValueError("matrix FEM is for short windows (T*C <= 512)")
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        operators, offsets = [], []
        with torch.autocast(device_type=x.device.type, enabled=False):
            basis = F.pad(torch.eye(n, device=x.device, dtype=dtype), (0, 0, 0, 1))
            basis = basis.reshape(n + 1, t, c)
            marker = torch.zeros(n + 1, 1, 1, 1, device=x.device, dtype=dtype)
            marker[-1] = 1
            for period in possible:
                branch = self._to_2d(basis, period)[0]
                # Expansion bias exists inside the padded image, not beyond it.
                ones = marker.expand(-1, 1, branch.shape[2], branch.shape[3])
                for kernel, bias in kernels:
                    branch = F.conv2d(torch.cat([branch, ones], 1), kernel,
                                      padding=1) + marker * bias[None, :, None, None]
                affine = self._to_1d(branch, t).reshape(n + 1, n)
                operators.append(affine[:n])
                offsets.append(affine[n])
        # One large matmul replaces batch-dependent convolution activations.
        y = x.reshape(x.shape[0], n) @ torch.cat(operators, dim=1)
        y = y.reshape(x.shape[0], len(possible), n)
        y = y.to(x.dtype) + torch.stack(offsets).to(x.dtype)
        choices = torch.tensor(possible, device=x.device)
        w = (weights[:, :, None] * (periods[:, :, None] == choices)).sum(1)
        return (y * w[:, :, None].to(x.dtype)).sum(1).reshape_as(x)


class FeatureExtractionNetwork(nn.Module):
    """Stack of FEMs forming one expert network."""

    def __init__(
        self,
        channels: int,
        depth: int,
        top_k: int = 4,
        hidden_channels: int = 32,
        groups: int = 4,
        dropout: float = 0.0,
        conv_rounds: int = 4,
        fft_mode: str = "per_sample",
        activation: str = "identity",
        residual: bool = False,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.layers = nn.ModuleList(
            [
                FeatureExtractionModule(
                    channels=channels,
                    top_k=top_k,
                    hidden_channels=hidden_channels,
                    groups=groups,
                    dropout=dropout,
                    conv_rounds=conv_rounds,
                    fft_mode=fft_mode,
                    activation=activation,
                    residual=residual,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, first_selection=None) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, selection=first_selection if i == 0 else None)
        return x


class MDENFeatureExtractor(nn.Module):
    """Multi-depth expert feature extractor: SOC=2, shared=4, SOH=6 FEMs."""

    def __init__(
        self,
        input_dim: int = 5,
        top_k: int = 4,
        hidden_channels: int = 32,
        groups: int = 4,
        dropout: float = 0.0,
        soc_depth: int = 2,
        shared_depth: int = 4,
        soh_depth: int = 6,
        conv_rounds: int = 4,
        fft_mode: str = "per_sample",
        activation: str = "identity",
        residual: bool = False,
    ) -> None:
        super().__init__()
        kwargs = {
            "channels": input_dim,
            "top_k": top_k,
            "hidden_channels": hidden_channels,
            "groups": groups,
            "dropout": dropout,
            "conv_rounds": conv_rounds,
            "fft_mode": fft_mode,
            "activation": activation,
            "residual": residual,
        }
        self.soc_net = FeatureExtractionNetwork(depth=soc_depth, **kwargs)
        self.shared_net = FeatureExtractionNetwork(
            depth=shared_depth, **kwargs
        )
        self.soh_net = FeatureExtractionNetwork(depth=soh_depth, **kwargs)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        first = self.soc_net.layers[0]
        selection = fft_select_periods(x, first.top_k, first.fft_mode)
        return {
            "soc_specific": self.soc_net(x, selection),
            "shared": self.shared_net(x, selection),
            "soh_specific": self.soh_net(x, selection),
        }
