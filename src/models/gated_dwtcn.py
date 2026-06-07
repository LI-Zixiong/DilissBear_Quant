"""
Gated Depthwise-TCN — lightweight temporal model with controlled feature interaction.

Architecture:
    Depthwise causal conv per feature → each factor learns its own temporal pattern
    Low-rank gate (A @ B @ x)          → light cross-factor interaction, r << n_features
    Pointwise projection               → scalar output per stock

Complexity: O(T × F × k) temporal + O(T × F × r) interaction. Linear in both F and T.

Suitable as a third/fourth model alongside LGBM (tree nonlinear), DLinear (static
trend-seasonal), and XGBoost (colsample-diversified tree).

@JulongQuant
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GatedDWTcnConfig:
    seq_len: int = 20
    n_features: int = 30
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4)
    hidden_dim: int = 32
    gate_rank: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.seq_len < 2:
            raise ValueError(f"seq_len must be >= 2, got {self.seq_len}")
        if self.n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {self.n_features}")
        if self.kernel_size < 2:
            raise ValueError(f"kernel_size must be >= 2, got {self.kernel_size}")
        if not self.dilations:
            raise ValueError("dilations must be non-empty")
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {self.hidden_dim}")
        if self.gate_rank < 1:
            raise ValueError(f"gate_rank must be >= 1, got {self.gate_rank}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


class _CausalDepthwiseConv(nn.Module):
    """Per-feature causal 1D conv. groups=n_features = each feature independent."""

    def __init__(self, n_features: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=n_features,
            padding=0,  # manual causal padding
            bias=False,
        )
        self.pad = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F] → [B, F, L]
        x = x.permute(0, 2, 1)
        x = nn.functional.pad(x, (self.pad, 0))  # causal: pad left only
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # [B, L, F]
        return x


class _LowRankGate(nn.Module):
    """Low-rank factor interaction gate: sigmoid(A @ B @ x_pooled)."""

    def __init__(self, n_features: int, rank: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.A = nn.Linear(n_features, rank, bias=False)
        self.B = nn.Linear(rank, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(n_features, hidden_dim)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.A.weight)
        nn.init.xavier_uniform_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F] — mean-pool over time
        pooled = x.mean(dim=1)  # [B, F]
        cross = self.B(self.A(pooled))  # [B, hidden_dim]
        gate = torch.sigmoid(self.gate_proj(pooled))  # [B, hidden_dim]
        return self.drop(cross * gate)


class GatedDWTcnRegressor(nn.Module):
    """Gated Depthwise-TCN for panel factor regression.

    Input:  [B, seq_len, n_features]
    Output: [B, 1]
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
        hidden_dim: int = 32,
        gate_rank: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features

        # Depthwise temporal stack
        self.convs = nn.ModuleList([
            _CausalDepthwiseConv(n_features, kernel_size, d)
            for d in dilations
        ])
        self.temporal_proj = nn.Linear(n_features, hidden_dim)

        # Low-rank cross-factor interaction
        self.gate = _LowRankGate(n_features, gate_rank, hidden_dim, dropout)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Temporal: stacked depthwise causal convs
        out = x
        for conv in self.convs:
            residual = out
            out = conv(out)
            out = out + residual  # per-block residual
            out = nn.functional.relu(out)

        # Mean-pool over time + project
        out = out.mean(dim=1)  # [B, F]
        out = self.temporal_proj(out)  # [B, hidden_dim]

        # Cross-factor gate
        gate_out = self.gate(x)  # [B, hidden_dim]
        out = out + gate_out

        return self.head(out)


def build_gated_dwtcn(config: GatedDWTcnConfig) -> GatedDWTcnRegressor:
    return GatedDWTcnRegressor(
        seq_len=config.seq_len,
        n_features=config.n_features,
        kernel_size=config.kernel_size,
        dilations=config.dilations,
        hidden_dim=config.hidden_dim,
        gate_rank=config.gate_rank,
        dropout=config.dropout,
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    config = GatedDWTcnConfig(seq_len=20, n_features=30)
    model = build_gated_dwtcn(config)

    x = torch.randn(4, 20, 30)
    y = model(x)
    expected = (4, 1)
    if y.shape != expected:
        raise RuntimeError(f"Smoke test shape mismatch: expected {expected}, got {tuple(y.shape)}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Smoke test passed {tuple(y.shape)}")
    print(f"[OK] Total parameters: {total_params:,}")
    print(f"[OK] Expected ~{30*3 + 30*4*2 + 30*32 + 32*16 + 16*1:,}")
