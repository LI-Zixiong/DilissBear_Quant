from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DLinearConfig:
    """
    DLinear model config for panel factor regression.

    seq_len:
        Length of historical window. In your previous notebook, this was 60.

    n_features:
        Number of factor columns. If you remove raw price and keep 12 style factors,
        this should be 12.

    moving_avg_kernel:
        Kernel size used to extract trend component.

    dropout:
        Dropout rate for regularization.

    @JulongQuant
    """

    seq_len: int = 60
    n_features: int = 12
    moving_avg_kernel: int = 25
    dropout: float = 0.1
    n_industries: int = 0
    ind_rank: int = 0

    def __post_init__(self) -> None:
        if self.moving_avg_kernel < 1:
            raise ValueError(
                f"moving_avg_kernel must be >= 1, but got {self.moving_avg_kernel}"
            )
        if self.seq_len < 1:
            raise ValueError(
                f"seq_len must be >= 1, but got {self.seq_len}"
            )
        if self.n_features < 1:
            raise ValueError(
                f"n_features must be >= 1, but got {self.n_features}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0.0, 1.0), but got {self.dropout}"
            )


class MovingAverage(nn.Module):
    """
    Moving average block used by DLinear.

    Input shape:
        x: [batch_size, seq_len, n_features]

    Output shape:
        trend: [batch_size, seq_len, n_features]
    """

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError(
                f"kernel_size must be >= 1, but got {kernel_size}"
            )
        self.kernel_size = kernel_size

        if kernel_size > 1:
            self.avg = nn.AvgPool1d(
                kernel_size=kernel_size,
                stride=1,
                padding=0,
            )
        else:
            self.avg = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size <= 1:
            return x

        # x: [B, L, C]
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left

        # Replicate edge values to keep output length unchanged.
        front = x[:, :1, :].repeat(1, pad_left, 1)
        end = x[:, -1:, :].repeat(1, pad_right, 1)

        x_padded = torch.cat([front, x, end], dim=1)

        # AvgPool1d expects [B, C, L]
        trend = self.avg(x_padded.permute(0, 2, 1))
        trend = trend.permute(0, 2, 1)

        return trend


class DLinearPanelRegressor(nn.Module):
    """
    DLinear-style model for stock panel factor regression.

    This model does NOT use historical target returns as input.
    It only uses observable factor columns.

    Input:
        x: [batch_size, seq_len, n_features]

    Output:
        pred: [batch_size, 1]
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        moving_avg_kernel: int = 25,
        dropout: float = 0.1,
        n_industries: int = 0,
        ind_rank: int = 0,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.n_features = n_features

        self.decomposition = MovingAverage(moving_avg_kernel)

        # DLinear projects each feature's time series from seq_len -> 1.
        self.seasonal_linear = nn.Linear(seq_len, 1)
        self.trend_linear = nn.Linear(seq_len, 1)

        self.dropout = nn.Dropout(dropout)

        # Low-rank industry-factor interaction.  industry_id is static per stock
        # (not in the time series).  rank-4 embedding → per-factor modulation.
        self.uses_industry_id = n_industries > 0 and ind_rank > 0
        self.ind_embedding: nn.Embedding | None = None
        self.ind_to_factor: nn.Linear | None = None
        if self.uses_industry_id:
            self.ind_embedding = nn.Embedding(n_industries, ind_rank)
            self.ind_to_factor = nn.Linear(ind_rank, n_features, bias=False)
            nn.init.zeros_(self.ind_to_factor.weight)  # start with no modulation

        # Then combine all factor-level predictions into one return prediction.
        self.feature_head = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, n_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_features, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Uniform prior (equal-weight averaging across time steps) with small
        # noise to break symmetry. Safer than Xavier for (1, seq_len) shapes
        # where input-output variance assumptions break down.
        nn.init.constant_(self.seasonal_linear.weight, 1.0 / self.seq_len)
        nn.init.constant_(self.trend_linear.weight, 1.0 / self.seq_len)
        with torch.no_grad():
            self.seasonal_linear.weight.add_(torch.randn_like(self.seasonal_linear.weight) * 0.01)
            self.trend_linear.weight.add_(torch.randn_like(self.trend_linear.weight) * 0.01)

        nn.init.zeros_(self.seasonal_linear.bias)
        nn.init.zeros_(self.trend_linear.bias)

        for layer in self.feature_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, industry_id: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Tensor with shape [B, L, C]
        industry_id:
            Optional long tensor [B] with per-sample industry index (0..n_ind-1).
            When provided and ind_embedding is configured, applies low-rank
            industry-factor modulation to per_feature_pred before the head.

        Returns
        -------
        Tensor with shape [B, 1]
        """

        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape [B, L, C], but got {tuple(x.shape)}"
            )

        if x.shape[1] != self.seq_len:
            raise ValueError(
                f"Expected seq_len={self.seq_len}, but got {x.shape[1]}"
            )

        if x.shape[2] != self.n_features:
            raise ValueError(
                f"Expected n_features={self.n_features}, but got {x.shape[2]}"
            )

        # Decompose into trend and seasonal components.
        trend = self.decomposition(x)
        seasonal = x - trend

        # Convert [B, L, C] -> [B, C, L]
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)

        # Apply linear projection along the time dimension.
        seasonal_out = self.seasonal_linear(seasonal)
        trend_out = self.trend_linear(trend)

        # [B, C, 1] -> [B, C]
        per_feature_pred = seasonal_out + trend_out
        per_feature_pred = per_feature_pred.squeeze(-1)

        per_feature_pred = self.dropout(per_feature_pred)

        # Low-rank industry-factor interaction: each industry learns a rank-4
        # embedding that modulates per-factor predictions before the head.
        if self.ind_embedding is not None and industry_id is not None:
            ind_emb = self.ind_embedding(industry_id)        # [B, rank]
            factor_mod = self.ind_to_factor(ind_emb)          # [B, C]
            per_feature_pred = per_feature_pred * (1.0 + factor_mod)

        # [B, C] -> [B, 1]
        pred = self.feature_head(per_feature_pred)

        return pred


def build_dlinear_model(config: DLinearConfig) -> DLinearPanelRegressor:
    """
    Helper function for building the model from config.
    """

    return DLinearPanelRegressor(
        seq_len=config.seq_len,
        n_features=config.n_features,
        moving_avg_kernel=config.moving_avg_kernel,
        dropout=config.dropout,
        n_industries=config.n_industries,
        ind_rank=config.ind_rank,
    )


if __name__ == "__main__":
    # Quick smoke test.
    config = DLinearConfig(
        seq_len=60,
        n_features=12,
        moving_avg_kernel=25,
        dropout=0.1,
    )

    model = build_dlinear_model(config)

    x = torch.randn(32, 60, 12)
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)