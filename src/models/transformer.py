"""
Modern Model: Time-Series Transformer Encoder-Decoder for Multi-Horizon Forecasting.

Architecture:
- Positional Encoding: Sinusoidal or learnable temporal position embeddings
- Encoder: Multi-layer Transformer Encoder processing past observations (168 timesteps)
- Decoder: Multi-layer Transformer Decoder with causal self-attention & cross-attention
- Quantile Projection Head: Produces non-crossing [q10, q50, q90]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    """
    Encoder-Decoder Transformer for multi-horizon forecasting with future calendar decoder.
    
    Unified Interface:
        x_past: (B, 168, 9)
        x_future_cal: (B, 24, 8)
    Returns:
        (B, 24, 3) quantiles [0.1, 0.5, 0.9]
    """
    def __init__(
        self,
        past_dim: int = 9,
        future_cal_dim: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.enc_embedding = nn.Linear(past_dim, d_model)
        self.dec_embedding = nn.Linear(future_cal_dim, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=300, dropout=dropout)

        # Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Non-crossing quantile projection heads
        self.fc_q50 = nn.Linear(d_model, 1)
        self.fc_delta_low = nn.Linear(d_model, 1)
        self.fc_delta_high = nn.Linear(d_model, 1)

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)
        return mask

    def forward(self, x_past: torch.Tensor, x_future_cal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_past: (B, 168, 9)
            x_future_cal: (B, 24, 8)
        Returns:
            (B, 24, 3)
        """
        B, L_past, _ = x_past.size()
        _, L_future, _ = x_future_cal.size()

        # 1. Encode past
        enc_in = self.enc_embedding(x_past) * math.sqrt(self.d_model)
        enc_in = self.pos_encoder(enc_in)
        memory = self.transformer_encoder(enc_in)  # (B, 168, d_model)

        # 2. Decode future
        dec_in = self.dec_embedding(x_future_cal) * math.sqrt(self.d_model)
        dec_in = self.pos_encoder(dec_in)

        # Causal mask so decoder future steps only attend to prior decoder steps
        tgt_mask = self._generate_causal_mask(L_future, x_past.device)
        dec_out = self.transformer_decoder(
            tgt=dec_in,
            memory=memory,
            tgt_mask=tgt_mask
        )  # (B, 24, d_model)

        # 3. Quantile heads
        q50 = self.fc_q50(dec_out)                                      # (B, 24, 1)
        delta_low = F.softplus(self.fc_delta_low(dec_out))             # (B, 24, 1)
        delta_high = F.softplus(self.fc_delta_high(dec_out))           # (B, 24, 1)

        q10 = q50 - delta_low
        q90 = q50 + delta_high

        return torch.cat([q10, q50, q90], dim=-1)  # (B, 24, 3)
