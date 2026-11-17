"""
Baseline models for Multi-Horizon Electricity Load Forecasting:
1. SeasonalNaiveBaseline (Domain benchmark: Weekly seasonal persistence)
2. MLPBaseline (Parametric baseline: Multi-Layer Perceptron on flattened past + future)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class SeasonalNaiveBaseline:
    """
    Non-parametric Seasonal Naive baseline.
    
    Given hourly electricity demand, load exhibits a strong 168-hour (7 days) periodicity.
    The weekly seasonal naive forecast takes the observed values from the same time 
    one week prior (t - 168 to t - 145), corresponding to indices [0:24] of x_past.
    """
    def __init__(self, mode: str = "weekly"):
        self.mode = mode

    def predict(self, x_past: torch.Tensor, x_future_cal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x_past: (B, 168, 9) historical features where feature 0 is target load.
            x_future_cal: (B, 24, 8) future calendar features (unused in naive baseline).
        Returns:
            preds: (B, 24, 3) where columns correspond to [q10, q50, q90].
        """
        if self.mode == "weekly":
            # 168 hours ago is the beginning of the 168-hour window
            pred_median = x_past[:, :24, 0]  # (B, 24)
        elif self.mode == "daily":
            # 24 hours ago is the end of the 168-hour window
            pred_median = x_past[:, -24:, 0]  # (B, 24)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Broadcast across quantiles [q10, q50, q90]
        # In seasonal naive, point forecast is replicated across quantiles
        preds = pred_median.unsqueeze(-1).repeat(1, 1, 3)
        return preds

    def __call__(self, x_past: torch.Tensor, x_future_cal: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.predict(x_past, x_future_cal)

class MLPBaseline(nn.Module):
    """
    Parametric MLP Baseline:
    Flattens past window (168 * 9) and future calendar (24 * 8), passes through
    fully connected layers with dropout, and outputs (B, 24, 3) quantiles with
    non-crossing guarantee.
    """
    def __init__(
        self,
        lookback: int = 168,
        past_dim: int = 9,
        horizon: int = 24,
        future_cal_dim: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        in_dim = (lookback * past_dim) + (horizon * future_cal_dim)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Non-crossing quantile heads: median, delta_low (q10), delta_high (q90)
        self.fc_q50 = nn.Linear(hidden_dim // 2, horizon)
        self.fc_delta_low = nn.Linear(hidden_dim // 2, horizon)
        self.fc_delta_high = nn.Linear(hidden_dim // 2, horizon)

    def forward(self, x_past: torch.Tensor, x_future_cal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_past: (B, 168, 9)
            x_future_cal: (B, 24, 8)
        Returns:
            (B, 24, 3)
        """
        B = x_past.size(0)
        past_flat = x_past.reshape(B, -1)
        future_flat = x_future_cal.reshape(B, -1)
        x = torch.cat([past_flat, future_flat], dim=-1)

        feat = self.net(x)
        q50 = self.fc_q50(feat)                                   # (B, 24)
        q10 = q50 - F.softplus(self.fc_delta_low(feat))           # q10 <= q50
        q90 = q50 + F.softplus(self.fc_delta_high(feat))          # q90 >= q50

        # Stack into (B, 24, 3)
        return torch.stack([q10, q50, q90], dim=-1)
