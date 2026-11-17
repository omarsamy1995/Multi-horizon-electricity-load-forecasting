"""
Classic Deep Learning Model: Sequence-to-Sequence GRU with Future Calendar Decoder.

Architecture:
- Encoder: Multi-layer GRU processing historical observations (168 timesteps, 9 features)
- Decoder: GRU taking future calendar features (24 timesteps, 8 features) + Bahdanau Temporal Attention
- Quantile Projection Head: Guarantees non-crossing quantiles [q10, q50, q90]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class BahdanauAttention(nn.Module):
    """
    Additive Attention mechanism (Bahdanau et al.)
    Computes attention alignment between decoder state and encoder memory.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w_enc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_dec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Current decoder state (B, 1, hidden_dim)
            memory: All encoder hidden states (B, L_enc, hidden_dim)
        Returns:
            context: (B, 1, hidden_dim)
            weights: (B, 1, L_enc)
        """
        score = self.v(torch.tanh(self.w_enc(memory) + self.w_dec(query))) # (B, L_enc, 1)
        score = score.transpose(1, 2)                                      # (B, 1, L_enc)
        weights = F.softmax(score, dim=-1)                                 # (B, 1, L_enc)
        context = torch.bmm(weights, memory)                               # (B, 1, hidden_dim)
        return context, weights

class Seq2SeqGRU(nn.Module):
    """
    Seq2Seq GRU with Future Calendar Conditioning and Temporal Attention.
    
    Accepts:
        x_past: Tensor of shape (B, 168, 9)
        x_future_cal: Tensor of shape (B, 24, 8)
    Returns:
        Tensor of shape (B, 24, 3) for quantiles [0.1, 0.5, 0.9]
    """
    def __init__(
        self,
        past_dim: int = 9,
        future_cal_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention

        # Encoder
        self.enc_in_proj = nn.Linear(past_dim, hidden_dim)
        self.encoder_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Decoder
        self.dec_in_proj = nn.Linear(future_cal_dim, hidden_dim)
        decoder_input_dim = hidden_dim * 2 if use_attention else hidden_dim
        self.decoder_gru = nn.GRU(
            input_size=decoder_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        if use_attention:
            self.attention = BahdanauAttention(hidden_dim)

        # Output projection for non-crossing quantiles
        head_dim = hidden_dim * 2 if use_attention else hidden_dim
        self.fc_q50 = nn.Linear(head_dim, 1)
        self.fc_delta_low = nn.Linear(head_dim, 1)
        self.fc_delta_high = nn.Linear(head_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_past: torch.Tensor,
        x_future_cal: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x_past: (B, 168, 9)
            x_future_cal: (B, 24, 8)
            return_attention: If True, also return attention weights
        Returns:
            quantiles: (B, 24, 3)
        """
        B, T_future, _ = x_future_cal.size()

        # 1. Encode past
        enc_in = self.dropout(F.relu(self.enc_in_proj(x_past))) # (B, 168, hidden_dim)
        enc_out, h_state = self.encoder_gru(enc_in)             # enc_out: (B, 168, hidden_dim), h_state: (layers, B, hidden_dim)

        # 2. Decode future step-by-step
        dec_inputs = self.dropout(F.relu(self.dec_in_proj(x_future_cal))) # (B, 24, hidden_dim)

        step_q50 = []
        step_delta_low = []
        step_delta_high = []
        attn_list = []

        cur_h = h_state
        last_context = torch.zeros(B, 1, self.hidden_dim, device=x_past.device)

        for t in range(T_future):
            cal_step = dec_inputs[:, t : t + 1, :]  # (B, 1, hidden_dim)

            if self.use_attention:
                # Query with top layer hidden state
                query = cur_h[-1:].transpose(0, 1)  # (B, 1, hidden_dim)
                context, attn_w = self.attention(query, enc_out)
                attn_list.append(attn_w)
                dec_in_step = torch.cat([cal_step, context], dim=-1) # (B, 1, 2*hidden_dim)
            else:
                dec_in_step = cal_step

            dec_out, cur_h = self.decoder_gru(dec_in_step, cur_h)

            if self.use_attention:
                head_feat = torch.cat([dec_out, context], dim=-1)
            else:
                head_feat = dec_out

            q50 = self.fc_q50(head_feat)
            d_low = F.softplus(self.fc_delta_low(head_feat))
            d_high = F.softplus(self.fc_delta_high(head_feat))

            step_q50.append(q50)
            step_delta_low.append(d_low)
            step_delta_high.append(d_high)

        q50_all = torch.cat(step_q50, dim=1)                 # (B, 24, 1)
        q10_all = q50_all - torch.cat(step_delta_low, dim=1)  # (B, 24, 1)
        q90_all = q50_all + torch.cat(step_delta_high, dim=1) # (B, 24, 1)

        quantiles = torch.cat([q10_all, q50_all, q90_all], dim=-1) # (B, 24, 3)

        if return_attention and self.use_attention:
            attn_all = torch.cat(attn_list, dim=1) # (B, 24, 168)
            return quantiles, attn_all

        return quantiles
