from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .feature_extraction import MDENFeatureExtractor
from .mamba import MambaSequence


@dataclass
class MDENConfig:
    """Configuration of the MDEN architecture."""

    input_dim: int = 5
    horizon: int = 8
    top_k: int = 4
    fem_hidden_channels: int = 32
    fem_groups: int = 4
    fem_dropout: float = 0.0
    fem_activation: str = "identity"
    fem_residual: bool = False
    fft_mode: str = "per_sample"
    soc_depth: int = 2
    shared_depth: int = 4
    soh_depth: int = 6
    conv_rounds: int = 4
    fusion_hidden: int = 32
    sequence_dim: int = 5
    lstm_layers: int = 1
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_backend: Literal["auto", "official", "native"] = "native"
    mamba_scan: str = "parallel"
    mamba_layout: str = "figure4"
    prediction_hidden: int = 32
    dropout: float = 0.0


class TaskAssessment(nn.Module):
    """E_sox(X_input) -> two softmax weights (Eq. 14).

    The article states that weights are obtained by analyzing the input but
    does not publish the exact MLP dimensions.  This compact pool+MLP module
    preserves that documented behavior.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        return self.net(x_input.transpose(1, 2))


class TaskFeatureFusion(nn.Module):
    """Eq. (13): F_w = w1 * shared + w2 * task-specific."""

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.assessment = TaskAssessment(input_dim, hidden_dim)

    def forward(
        self,
        x_input: torch.Tensor,
        shared: torch.Tensor,
        task_specific: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.assessment(x_input)
        w_shared = weights[:, 0].view(-1, 1, 1)
        w_task = weights[:, 1].view(-1, 1, 1)
        return w_shared * shared + w_task * task_specific, weights


class JointPredictionHead(nn.Module):
    """Eq. (17-18): concatenate task features, then 1x1-ReLU-1x1 predictor."""

    def __init__(
        self,
        feature_dim: int,
        horizon: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(feature_dim, 2 * horizon)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Conv1d(2 * feature_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 2 * horizon, kernel_size=1),
        )

    def forward(
        self,
        soc_features: torch.Tensor,
        soh_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1x1 convolutions act independently at every time position.
        # Slicing before the head is equivalent and avoids computing 31 unused outputs.
        fused = torch.cat([soc_features[:, -1:], soh_features[:, -1:]], dim=-1).transpose(1, 2)
        out = self.net(fused)[:, :, -1]
        soc, soh = torch.split(out, self.horizon, dim=1)
        return soc, soh


class MDEN(nn.Module):
    """Multi-Depth Expert Network for synchronous SOC/SOH estimation."""

    def __init__(self, config: MDENConfig | None = None) -> None:
        super().__init__()
        self.config = config or MDENConfig()
        c = self.config

        self.extractor = MDENFeatureExtractor(
            input_dim=c.input_dim,
            top_k=c.top_k,
            hidden_channels=c.fem_hidden_channels,
            groups=c.fem_groups,
            dropout=c.fem_dropout,
            soc_depth=c.soc_depth,
            shared_depth=c.shared_depth,
            soh_depth=c.soh_depth,
            conv_rounds=c.conv_rounds,
            fft_mode=c.fft_mode,
            activation=c.fem_activation,
            residual=c.fem_residual,
        )

        self.soc_fusion = TaskFeatureFusion(c.input_dim, c.fusion_hidden)
        self.soh_fusion = TaskFeatureFusion(c.input_dim, c.fusion_hidden)

        self.soc_in = (
            nn.Identity()
            if c.sequence_dim == c.input_dim
            else nn.Linear(c.input_dim, c.sequence_dim)
        )
        self.soh_in = (
            nn.Identity()
            if c.sequence_dim == c.input_dim
            else nn.Linear(c.input_dim, c.sequence_dim)
        )
        self.soc_lstm = nn.LSTM(
            input_size=c.sequence_dim,
            hidden_size=c.sequence_dim,
            num_layers=c.lstm_layers,
            batch_first=True,
            dropout=c.dropout if c.lstm_layers > 1 else 0.0,
        )
        self.soh_mamba = MambaSequence(
            d_model=c.sequence_dim,
            d_state=c.mamba_d_state,
            d_conv=c.mamba_d_conv,
            expand=c.mamba_expand,
            dropout=c.dropout,
            backend=c.mamba_backend,
            scan=c.mamba_scan,
            layout=c.mamba_layout,
        )
        self.dropout = nn.Dropout(c.dropout)
        self.predictor = JointPredictionHead(
            feature_dim=c.sequence_dim,
            horizon=c.horizon,
            hidden_dim=c.prediction_hidden,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.config.input_dim:
            raise ValueError("MDEN expects [B,T,input_dim]")
        features = self.extractor(x)

        soc_fw, soc_weights = self.soc_fusion(
            x, features["shared"], features["soc_specific"]
        )
        soh_fw, soh_weights = self.soh_fusion(
            x, features["shared"], features["soh_specific"]
        )

        soc_seq, _ = self.soc_lstm(self.soc_in(soc_fw))
        soh_seq = self.soh_mamba(self.soh_in(soh_fw))
        soc_seq = self.dropout(soc_seq)
        soh_seq = self.dropout(soh_seq)

        soc_pred, soh_pred = self.predictor(soc_seq, soh_seq)
        return {
            "soc": soc_pred,
            "soh": soh_pred,
            "soc_fused": soc_fw,
            "soh_fused": soh_fw,
            "soc_deep": soc_seq,
            "soh_deep": soh_seq,
            "soc_weights": soc_weights,
            "soh_weights": soh_weights,
        }


# Compatibility alias for the previous repository API.
EndToEndSOCSOHModelSharedCNN = MDEN
