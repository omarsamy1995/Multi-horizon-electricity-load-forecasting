"""
Custom PyTorch components written from scratch for Multi-Horizon Forecasting:
1. MultiHorizonQuantileLoss (Pinball Loss for arbitrary quantiles [0.1, 0.5, 0.9])
2. GatedLinearUnit (GLU)
3. GatedResidualNetwork (GRN)
4. InterpretableMultiHeadAttention (Interpretable temporal self/cross-attention)
5. VariableSelectionNetwork (VSN)
"""

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHorizonQuantileLoss(nn.Module):
    """
    Multi-Horizon Quantile Loss (Pinball Loss).
    
    L_q(y, y_hat) = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    
    Parameters:
        quantiles: List of float quantiles to optimize (default: [0.1, 0.5, 0.9])
    """
    def __init__(self, quantiles: Optional[List[float]] = None):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Predictions tensor of shape (B, Horizon, num_quantiles)
            y_true: Ground truth target of shape (B, Horizon) or (B, Horizon, 1)
        Returns:
            Scalar aggregated pinball loss.
        """
        if y_true.dim() == 2:
            y_true = y_true.unsqueeze(-1)  # (B, Horizon, 1)

        # Broadcast error: (B, Horizon, num_quantiles)
        error = y_true - y_pred
        q = self.quantiles.view(1, 1, -1)  # (1, 1, num_quantiles)

        # Pinball loss per element
        loss = torch.max(q * error, (q - 1.0) * error)
        return torch.mean(loss)

    def per_quantile_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Computes loss separated by quantile (num_quantiles,)."""
        if y_true.dim() == 2:
            y_true = y_true.unsqueeze(-1)
        error = y_true - y_pred
        q = self.quantiles.view(1, 1, -1)
        loss = torch.max(q * error, (q - 1.0) * error)
        return torch.mean(loss, dim=(0, 1))

class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU):
    GLU(x) = Linear_1(x) * sigmoid(Linear_2(x))
    """
    def __init__(self, input_dim: int, output_dim: Optional[int] = None):
        super().__init__()
        output_dim = output_dim or input_dim
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        val, gate = torch.chunk(out, chunks=2, dim=-1)
        return val * torch.sigmoid(gate)

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) as introduced in Temporal Fusion Transformers.
    
    Provides non-linear processing with adaptive gating to dynamically skip
    layers when complex transformations are unnecessary.
    
    GRN(a, c) = LayerNorm(Skip(a) + GLU(W_2 * ELU(W_1 * a + W_c * c + b_1) + b_2))
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.context_dim = context_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if context_dim is not None and context_dim > 0:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        else:
            self.context_fc = None

        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, self.output_dim * 2)
        self.dropout = nn.Dropout(dropout)

        if self.input_dim != self.output_dim:
            self.skip_proj = nn.Linear(input_dim, self.output_dim)
        else:
            self.skip_proj = nn.Identity()

        self.layer_norm = nn.LayerNorm(self.output_dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = self.fc1(x)
        if self.context_fc is not None and context is not None:
            if context.dim() == 2 and x.dim() == 3:
                hidden = hidden + self.context_fc(context).unsqueeze(1)
            else:
                hidden = hidden + self.context_fc(context)

        hidden = self.elu(hidden)
        hidden = self.dropout(hidden)
        
        gate_out = self.fc2(hidden)
        val, gate = torch.chunk(gate_out, chunks=2, dim=-1)
        gated = val * torch.sigmoid(gate)

        skip = self.skip_proj(x)
        return self.layer_norm(skip + gated)

class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention layer.
    
    Instead of projecting concatenated heads, TFT computes shared attention
    weights across heads or projects head values directly, allowing exact
    extraction of temporal attention weights across horizons.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, L_q, d_model)
            key:   (B, L_k, d_model)
            value: (B, L_k, d_model)
            mask:  Optional attention mask (B, 1, L_q, L_k) or (1, 1, L_q, L_k)
        Returns:
            out: (B, L_q, d_model)
            attn_weights: (B, L_q, L_k) interpretable attention map averaged over heads
        """
        B, L_q, _ = query.size()
        L_k = key.size(1)

        # (B, n_heads, L, d_k)
        Q = self.q_proj(query).view(B, L_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(B, L_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(B, L_k, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product scores: (B, n_heads, L_q, L_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights_heads = F.softmax(scores, dim=-1)
        attn_weights_heads = self.dropout(attn_weights_heads)

        # Compute head outputs: (B, n_heads, L_q, d_k)
        context = torch.matmul(attn_weights_heads, V)
        # Transpose and combine heads: (B, L_q, d_model)
        context = context.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        out = self.out_proj(context)

        # Mean attention weights across heads for interpretability: (B, L_q, L_k)
        interpretable_attn = attn_weights_heads.mean(dim=1)

        return out, interpretable_attn

class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) for feature importance and non-linear feature transformation.
    """
    def __init__(
        self,
        num_features: int,
        feature_dim: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Individual GRN per variable
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(feature_dim, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_features)
        ])

        # Selection weight network
        self.flattened_grn = GatedResidualNetwork(
            num_features * feature_dim,
            hidden_dim,
            num_features,
            dropout=dropout
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feature_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feature_list: List of length num_features, each of shape (B, L, feature_dim)
        Returns:
            combined: (B, L, hidden_dim)
            weights:  (B, L, num_features)
        """
        # Flattened inputs for selection weights: (B, L, num_features * feature_dim)
        flattened = torch.cat(feature_list, dim=-1)
        weights = self.softmax(self.flattened_grn(flattened))  # (B, L, num_features)

        # Process each variable through its GRN
        transformed = torch.stack(
            [grn(f) for grn, f in zip(self.var_grns, feature_list)],
            dim=-2
        )  # (B, L, num_features, hidden_dim)

        # Weighted combination across features
        combined = torch.sum(transformed * weights.unsqueeze(-1), dim=-2)
        return combined, weights
