"""
Modern Model: Time-Series Transformer Encoder-Decoder for Multi-Horizon Forecasting.

Architecture in tf.keras:
- Positional Encoding: Sinusoidal temporal position embeddings
- Encoder: Multi-layer Transformer Encoder processing past observations (168 timesteps)
- Decoder: Multi-layer Transformer Decoder with causal self-attention & cross-attention
- Quantile Projection Head: Produces non-crossing [q10, q50, q90] via NonCrossingQuantileHead
"""

import math
from typing import Optional, Union, Tuple, Dict, Any
import tensorflow as tf

from src.models.custom_layers import NonCrossingQuantileHead


@tf.keras.utils.register_keras_serializable(package="models")
class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
    """Sinusoidal positional encoding for temporal sequences."""
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        self.dropout_rate = dropout
        self.dropout = tf.keras.layers.Dropout(dropout)

        # Precompute sinusoidal positional table
        pe = np_pe = tf.zeros((max_len, d_model)).numpy()
        position = tf.reshape(tf.range(0, max_len, dtype=tf.float32), (-1, 1))
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * (-math.log(10000.0) / d_model))

        sin_vals = tf.sin(position * div_term)
        cos_vals = tf.cos(position * div_term)

        pe = tf.Variable(
            initial_value=tf.concat([
                tf.reshape(tf.stack([sin_vals, cos_vals], axis=-1), (max_len, d_model))
            ], axis=-1),
            trainable=False,
            name="pe_table"
        )
        self.pe = pe

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        pe_slice = tf.expand_dims(self.pe[:seq_len, :], axis=0)
        x = x + pe_slice
        return self.dropout(x, training=training)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "max_len": self.max_len,
            "dropout": self.dropout_rate
        })
        return config


@tf.keras.utils.register_keras_serializable(package="models")
class TransformerEncoderBlock(tf.keras.layers.Layer):
    """Single Transformer Encoder block: MultiHeadAttention + GELU FeedForward + LayerNorms."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 256, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout

        self.self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=nhead,
            key_dim=d_model // nhead,
            dropout=dropout,
            name="self_attn"
        )
        self.ln1 = tf.keras.layers.LayerNormalization(axis=-1, name="ln1")
        self.drop1 = tf.keras.layers.Dropout(dropout, name="drop1")

        self.ffn_dense1 = tf.keras.layers.Dense(dim_feedforward, activation="gelu", name="ffn_dense1")
        self.ffn_dense2 = tf.keras.layers.Dense(d_model, name="ffn_dense2")
        self.ln2 = tf.keras.layers.LayerNormalization(axis=-1, name="ln2")
        self.drop2 = tf.keras.layers.Dropout(dropout, name="drop2")

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        attn_out = self.self_attn(query=x, value=x, key=x, training=training)
        x = self.ln1(x + self.drop1(attn_out, training=training))

        ffn_out = self.ffn_dense2(self.ffn_dense1(x))
        x = self.ln2(x + self.drop2(ffn_out, training=training))
        return x

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "nhead": self.nhead,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout_rate
        })
        return config


@tf.keras.utils.register_keras_serializable(package="models")
class TransformerDecoderBlock(tf.keras.layers.Layer):
    """Single Transformer Decoder block: Causal Self-Attn + Cross-Attn + GELU FeedForward + LayerNorms."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 256, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout

        self.self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=nhead,
            key_dim=d_model // nhead,
            dropout=dropout,
            name="causal_self_attn"
        )
        self.ln1 = tf.keras.layers.LayerNormalization(axis=-1, name="ln1")
        self.drop1 = tf.keras.layers.Dropout(dropout, name="drop1")

        self.cross_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=nhead,
            key_dim=d_model // nhead,
            dropout=dropout,
            name="cross_attn"
        )
        self.ln2 = tf.keras.layers.LayerNormalization(axis=-1, name="ln2")
        self.drop2 = tf.keras.layers.Dropout(dropout, name="drop2")

        self.ffn_dense1 = tf.keras.layers.Dense(dim_feedforward, activation="gelu", name="ffn_dense1")
        self.ffn_dense2 = tf.keras.layers.Dense(d_model, name="ffn_dense2")
        self.ln3 = tf.keras.layers.LayerNormalization(axis=-1, name="ln3")
        self.drop3 = tf.keras.layers.Dropout(dropout, name="drop3")

    def call(self, tgt: tf.Tensor, memory: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        # Causal self-attention
        self_attn_out = self.self_attn(
            query=tgt,
            value=tgt,
            key=tgt,
            use_causal_mask=True,
            training=training
        )
        tgt = self.ln1(tgt + self.drop1(self_attn_out, training=training))

        # Cross-attention to encoder memory
        cross_attn_out = self.cross_attn(
            query=tgt,
            value=memory,
            key=memory,
            training=training
        )
        tgt = self.ln2(tgt + self.drop2(cross_attn_out, training=training))

        # Position-wise Feedforward
        ffn_out = self.ffn_dense2(self.ffn_dense1(tgt))
        tgt = self.ln3(tgt + self.drop3(ffn_out, training=training))
        return tgt

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "nhead": self.nhead,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout_rate
        })
        return config


@tf.keras.utils.register_keras_serializable(package="models")
class TimeSeriesTransformer(tf.keras.Model):
    """
    Encoder-Decoder Transformer for multi-horizon forecasting with future calendar decoder in tf.keras.
    
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
        dropout: float = 0.1,
        name: str = "transformer",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.past_dim = past_dim
        self.future_cal_dim = future_cal_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout

        # Embeddings
        self.enc_embedding = tf.keras.layers.Dense(d_model, name="enc_embedding")
        self.dec_embedding = tf.keras.layers.Dense(d_model, name="dec_embedding")
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=300, dropout=dropout, name="pos_enc")

        # Transformer Backbone
        self.encoder_blocks = [
            TransformerEncoderBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                name=f"enc_block_{i}"
            )
            for i in range(num_encoder_layers)
        ]

        self.decoder_blocks = [
            TransformerDecoderBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                name=f"dec_block_{i}"
            )
            for i in range(num_decoder_layers)
        ]

        # Non-crossing quantile projection head
        self.quantile_head = NonCrossingQuantileHead(output_horizon=None, name="quantile_head")

    def call(
        self,
        inputs: Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]],
        x_future_cal: Optional[tf.Tensor] = None,
        training: Optional[bool] = None
    ) -> tf.Tensor:
        """
        Args:
            inputs: x_past (B, 168, 9) or tuple (x_past, x_future_cal)
            x_future_cal: (B, 24, 8)
        Returns:
            (B, 24, 3)
        """
        if isinstance(inputs, (tuple, list)):
            x_past, x_future_cal = inputs[0], inputs[1]
        elif isinstance(inputs, dict):
            x_past = inputs.get("x_past", inputs.get("past"))
            x_future_cal = inputs.get("x_future_cal", inputs.get("future_cal"))
        else:
            x_past = inputs

        scale = math.sqrt(self.d_model)

        # 1. Encode past sequence
        enc_in = self.enc_embedding(x_past) * scale
        enc_in = self.pos_encoder(enc_in, training=training)

        memory = enc_in
        for block in self.encoder_blocks:
            memory = block(memory, training=training)

        # 2. Decode future calendar sequence
        dec_in = self.dec_embedding(x_future_cal) * scale
        dec_in = self.pos_encoder(dec_in, training=training)

        dec_out = dec_in
        for block in self.decoder_blocks:
            dec_out = block(dec_out, memory=memory, training=training)

        # 3. Non-crossing quantile head
        return self.quantile_head(dec_out)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "past_dim": self.past_dim,
            "future_cal_dim": self.future_cal_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout_rate
        })
        return config
