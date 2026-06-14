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
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.n_features = n_features

        self.decomposition = MovingAverage(moving_avg_kernel)

        # DLinear projects each feature's time series from seq_len -> 1.
        self.seasonal_linear = nn.Linear(seq_len, 1)
        self.trend_linear = nn.Linear(seq_len, 1)

        self.dropout = nn.Dropout(dropout)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Tensor with shape [B, L, C]

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