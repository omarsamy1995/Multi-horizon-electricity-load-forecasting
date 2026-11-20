"""
Advanced Architecture: Temporal Fusion Multi-Horizon Forecasting Model.

Written from scratch in tf.keras using custom components:
- GatedResidualNetwork (GRN)
- GatedLinearUnit (GLU)
- InterpretableMultiHeadAttention
- NonCrossingQuantileHead

Unified Interface:
    Input:
        x_past: (B, 168, 9)
        x_future_cal: (B, 24, 8)
    Output:
        (B, 24, 3) for quantiles [0.1, 0.5, 0.9]
"""

from typing import Optional, Tuple, Union, Dict, Any
import tensorflow as tf

from src.models.custom_layers import (
    GatedResidualNetwork,
    GatedLinearUnit,
    InterpretableMultiHeadAttention,
    NonCrossingQuantileHead
)


@tf.keras.utils.register_keras_serializable(package="models")
class TemporalFusionModel(tf.keras.Model):
    """
    Temporal Fusion Multi-Horizon Forecasting Architecture in tf.keras.
    
    Combines:
    1. Feature Transformations via Gated Residual Networks (GRN)
    2. Recurrent Locality Processing (LSTM) for short-term temporal dependencies
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
        dropout: float = 0.1,
        name: str = "tft_advanced",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.past_dim = past_dim
        self.future_cal_dim = future_cal_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_lstm_layers = num_lstm_layers
        self.dropout_rate = dropout

        # 1. Input Feature Encoders via GRN
        self.past_grn = GatedResidualNetwork(
            past_dim, hidden_dim, hidden_dim, dropout=dropout, name="past_grn"
        )
        self.future_grn = GatedResidualNetwork(
            future_cal_dim, hidden_dim, hidden_dim, dropout=dropout, name="future_grn"
        )

        # 2. Locality Processing via Recurrent Layer (LSTM)
        self.past_lstm = tf.keras.layers.LSTM(
            hidden_dim,
            return_sequences=True,
            return_state=True,
            dropout=dropout,
            name="past_lstm"
        )
        self.future_lstm = tf.keras.layers.LSTM(
            hidden_dim,
            return_sequences=True,
            dropout=dropout,
            name="future_lstm"
        )

        # Gated Skip connections after LSTM
        self.past_post_lstm_glu = GatedLinearUnit(hidden_dim, hidden_dim, name="past_post_lstm_glu")
        self.past_post_lstm_norm = tf.keras.layers.LayerNormalization(axis=-1, name="past_post_lstm_norm")

        self.future_post_lstm_glu = GatedLinearUnit(hidden_dim, hidden_dim, name="future_post_lstm_glu")
        self.future_post_lstm_norm = tf.keras.layers.LayerNormalization(axis=-1, name="future_post_lstm_norm")

        # 3. Interpretable Temporal Multi-Head Attention
        self.temporal_attn = InterpretableMultiHeadAttention(
            d_model=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            name="temporal_attn"
        )
        self.post_attn_glu = GatedLinearUnit(hidden_dim, hidden_dim, name="post_attn_glu")
        self.post_attn_norm = tf.keras.layers.LayerNormalization(axis=-1, name="post_attn_norm")

        # 4. Position-wise Feedforward GRN
        self.pos_wise_grn = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout=dropout, name="pos_wise_grn"
        )

        # 5. Non-crossing Quantile Output Heads
        self.quantile_head = NonCrossingQuantileHead(output_horizon=None, name="quantile_head")

    def call(
        self,
        inputs: Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]],
        x_future_cal: Optional[tf.Tensor] = None,
        return_attention: bool = False,
        training: Optional[bool] = None
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Args:
            inputs: x_past (B, 168, 9) historical observations or tuple (x_past, x_future_cal)
            x_future_cal: (B, 24, 8) future calendar features
            return_attention: If True, returns (quantiles, attention_weights)
        Returns:
            quantiles: (B, 24, 3) representing [q10, q50, q90]
        """
        if isinstance(inputs, (tuple, list)):
            x_past, x_future_cal = inputs[0], inputs[1]
        elif isinstance(inputs, dict):
            x_past = inputs.get("x_past", inputs.get("past"))
            x_future_cal = inputs.get("x_future_cal", inputs.get("future_cal"))
        else:
            x_past = inputs

        # Step 1: Feature transformations via GRN
        past_feat = self.past_grn(x_past, training=training)          # (B, 168, hidden_dim)
        future_feat = self.future_grn(x_future_cal, training=training) # (B, 24, hidden_dim)

        # Step 2: Locality Processing
        past_lstm_out, h_state, c_state = self.past_lstm(past_feat, training=training)
        past_local = self.past_post_lstm_norm(past_feat + self.past_post_lstm_glu(past_lstm_out))

        future_lstm_out = self.future_lstm(future_feat, initial_state=[h_state, c_state], training=training)
        future_local = self.future_post_lstm_norm(future_feat + self.future_post_lstm_glu(future_lstm_out))

        # Full temporal sequence: past (168) + future (24) = 192 steps
        full_seq = tf.concat([past_local, future_local], axis=1)  # (B, 192, hidden_dim)

        # Step 3: Interpretable Multi-Head Temporal Attention
        attn_out, attn_weights = self.temporal_attn(
            query=future_local,
            key=full_seq,
            value=full_seq,
            training=training
        )  # attn_out: (B, 24, hidden_dim), attn_weights: (B, 24, 192)

        # Gated residual connection around attention
        gated_attn = self.post_attn_norm(future_local + self.post_attn_glu(attn_out))

        # Step 4: Position-wise GRN
        final_rep = self.pos_wise_grn(gated_attn, training=training)  # (B, 24, hidden_dim)

        # Step 5: Non-crossing quantile projection heads
        quantiles = self.quantile_head(final_rep)  # (B, 24, 3)

        if return_attention:
            return quantiles, attn_weights

        return quantiles

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "past_dim": self.past_dim,
            "future_cal_dim": self.future_cal_dim,
            "hidden_dim": self.hidden_dim,
            "n_heads": self.n_heads,
            "num_lstm_layers": self.num_lstm_layers,
            "dropout": self.dropout_rate
        })
        return config
