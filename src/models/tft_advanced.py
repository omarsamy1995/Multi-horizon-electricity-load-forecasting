"""
Advanced Architecture: Temporal Fusion Multi-Horizon Forecasting Model.

Written from scratch using custom components:
- GatedResidualNetwork (GRN)
- GatedLinearUnit (GLU)
- InterpretableMultiHeadAttention
- VariableSelectionNetwork (VSN)

Unified Interface:
    Input:
        x_past: (B, 168, 9)
        x_future_cal: (B, 24, 8)
    Output:
        (B, 24, 3) for quantiles [0.1, 0.5, 0.9]
"""

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.custom_layers import (
    GatedResidualNetwork,
    GatedLinearUnit,
    InterpretableMultiHeadAttention
)

class TemporalFusionModel(nn.Module):
    """
    Temporal Fusion Multi-Horizon Forecasting Architecture.
    
    Combines:
    1. Feature Transformations via Gated Residual Networks (GRN)
    2. Recurrent Locality Processing (LSTM / GRU) for short-term temporal dependencies
    3. Interpretable Multi-Head Temporal Self-Attention for long-term historical dependencies (e.g. 7-day periodicity)
    4. Gated Skip Connections with Layer Normalization
    5. Non-crossing Quantile Output Heads [0.1, 0.5, 0.9]
    """
    def __init__(
        self,
        past_dim: int = 9,
        future_cal_dim: int = 8,
        hidden_dim: int = 128,
        n_heads: int = 4,
        num_lstm_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.past_dim = past_dim
        self.future_cal_dim = future_cal_dim
        self.hidden_dim = hidden_dim

        # 1. Input Feature Encoders via GRN
        self.past_grn = GatedResidualNetwork(past_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.future_grn = GatedResidualNetwork(future_cal_dim, hidden_dim, hidden_dim, dropout=dropout)

        # 2. Locality Processing via Recurrent Layer
        self.past_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )
        self.future_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )

        # Gated Skip connections after LSTM
        self.past_post_lstm_glu = GatedLinearUnit(hidden_dim, hidden_dim)
        self.past_post_lstm_norm = nn.LayerNorm(hidden_dim)

        self.future_post_lstm_glu = GatedLinearUnit(hidden_dim, hidden_dim)
        self.future_post_lstm_norm = nn.LayerNorm(hidden_dim)

        # 3. Interpretable Temporal Multi-Head Attention
        # Future steps query past steps + future steps
        self.temporal_attn = InterpretableMultiHeadAttention(
            d_model=hidden_dim,
            n_heads=n_heads,
            dropout=dropout
        )
        self.post_attn_glu = GatedLinearUnit(hidden_dim, hidden_dim)
        self.post_attn_norm = nn.LayerNorm(hidden_dim)

        # 4. Position-wise Feedforward GRN
        self.pos_wise_grn = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout=dropout
        )

        # 5. Non-crossing Quantile Output Heads
        self.fc_q50 = nn.Linear(hidden_dim, 1)
        self.fc_delta_low = nn.Linear(hidden_dim, 1)
        self.fc_delta_high = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x_past: torch.Tensor,
        x_future_cal: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x_past: (B, 168, 9) historical observations
            x_future_cal: (B, 24, 8) future calendar features
            return_attention: If True, returns (quantiles, attention_weights)
        Returns:
            quantiles: (B, 24, 3) representing [q10, q50, q90]
        """
        B, L_past, _ = x_past.size()
        _, L_future, _ = x_future_cal.size()

        # Step 1: Feature transformations via GRN
        past_feat = self.past_grn(x_past)          # (B, 168, hidden_dim)
        future_feat = self.future_grn(x_future_cal) # (B, 24, hidden_dim)

        # Step 2: Locality Processing
        past_lstm_out, (h_n, c_n) = self.past_lstm(past_feat)
        past_local = self.past_post_lstm_norm(past_feat + self.past_post_lstm_glu(past_lstm_out))

        future_lstm_out, _ = self.future_lstm(future_feat, (h_n, c_n))
        future_local = self.future_post_lstm_norm(future_feat + self.future_post_lstm_glu(future_lstm_out))

        # Full temporal sequence: past (168) + future (24) = 192 steps
        full_seq = torch.cat([past_local, future_local], dim=1) # (B, 192, hidden_dim)

        # Step 3: Interpretable Multi-Head Temporal Attention
        # Future queries (24) attend to full sequence (192) or past sequence (168)
        attn_out, attn_weights = self.temporal_attn(
            query=future_local,
            key=full_seq,
            value=full_seq
        )  # attn_out: (B, 24, hidden_dim), attn_weights: (B, 24, 192)

        # Gated residual connection around attention
        gated_attn = self.post_attn_norm(future_local + self.post_attn_glu(attn_out))

        # Step 4: Position-wise GRN
        final_rep = self.pos_wise_grn(gated_attn) # (B, 24, hidden_dim)

        # Step 5: Non-crossing quantile projection heads
        q50 = self.fc_q50(final_rep)                                    # (B, 24, 1)
        delta_low = F.softplus(self.fc_delta_low(final_rep))           # (B, 24, 1)
        delta_high = F.softplus(self.fc_delta_high(final_rep))         # (B, 24, 1)

        q10 = q50 - delta_low
        q90 = q50 + delta_high

        quantiles = torch.cat([q10, q50, q90], dim=-1)                 # (B, 24, 3)

        if return_attention:
            return quantiles, attn_weights

        return quantiles
