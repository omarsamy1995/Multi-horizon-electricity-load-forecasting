"""
Modern Attention Deep Learning: Time-Series Transformer Encoder-Decoder in tf.keras.
"""

import math
from typing import Optional, Union, Tuple, Dict, Any
import tensorflow as tf
from src.models.custom_layers import NonCrossingQuantileHead


@tf.keras.utils.register_keras_serializable(package="models")
class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
    """Sinusoidal positional encoding table added to sequence embeddings."""
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        self.dropout_rate = dropout
        self.dropout = tf.keras.layers.Dropout(dropout)

        position = tf.reshape(tf.range(0, max_len, dtype=tf.float32), (-1, 1))
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * (-math.log(10000.0) / d_model))
        sin_vals = tf.sin(position * div_term)
        cos_vals = tf.cos(position * div_term)
        pe_table = tf.reshape(tf.stack([sin_vals, cos_vals], axis=-1), (max_len, d_model))

        self.pe = tf.Variable(initial_value=pe_table, trainable=False, name="pe_table")

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        x = x + tf.expand_dims(self.pe[:seq_len, :], axis=0)
        return self.dropout(x, training=training)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"d_model": self.d_model, "max_len": self.max_len, "dropout": self.dropout_rate})
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

        self.self_attn = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model // nhead, dropout=dropout, name="self_attn")
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
        return self.ln2(x + self.drop2(ffn_out, training=training))

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
    """Decoder block: Causal Self-Attn + Encoder-Decoder Cross-Attn + GELU FeedForward + LayerNorms."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 256, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout

        self.self_attn = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model // nhead, dropout=dropout, name="causal_self_attn")
        self.ln1 = tf.keras.layers.LayerNormalization(axis=-1, name="ln1")
        self.drop1 = tf.keras.layers.Dropout(dropout, name="drop1")

        self.cross_attn = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model // nhead, dropout=dropout, name="cross_attn")
        self.ln2 = tf.keras.layers.LayerNormalization(axis=-1, name="ln2")
        self.drop2 = tf.keras.layers.Dropout(dropout, name="drop2")

        self.ffn_dense1 = tf.keras.layers.Dense(dim_feedforward, activation="gelu", name="ffn_dense1")
        self.ffn_dense2 = tf.keras.layers.Dense(d_model, name="ffn_dense2")
        self.ln3 = tf.keras.layers.LayerNormalization(axis=-1, name="ln3")
        self.drop3 = tf.keras.layers.Dropout(dropout, name="drop3")

    def call(self, tgt: tf.Tensor, memory: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        self_attn_out = self.self_attn(query=tgt, value=tgt, key=tgt, use_causal_mask=True, training=training)
        tgt = self.ln1(tgt + self.drop1(self_attn_out, training=training))

        cross_attn_out = self.cross_attn(query=tgt, value=memory, key=memory, training=training)
        tgt = self.ln2(tgt + self.drop2(cross_attn_out, training=training))

        ffn_out = self.ffn_dense2(self.ffn_dense1(tgt))
        return self.ln3(tgt + self.drop3(ffn_out, training=training))

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
    """Transformer Encoder-Decoder mapping x_past (B, 168, 9) and x_future_cal (B, 24, 8) to (B, 24, 3)."""
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

        self.enc_embedding = tf.keras.layers.Dense(d_model, name="enc_embedding")
        self.dec_embedding = tf.keras.layers.Dense(d_model, name="dec_embedding")
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=300, dropout=dropout, name="pos_enc")

        self.encoder_blocks = [
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout, name=f"enc_block_{i}")
            for i in range(num_encoder_layers)
        ]
        self.decoder_blocks = [
            TransformerDecoderBlock(d_model, nhead, dim_feedforward, dropout, name=f"dec_block_{i}")
            for i in range(num_decoder_layers)
        ]
        self.quantile_head = NonCrossingQuantileHead(output_horizon=None, name="quantile_head")

    def call(
        self,
        inputs: Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]],
        x_future_cal: Optional[tf.Tensor] = None,
        training: Optional[bool] = None
    ) -> tf.Tensor:
        if isinstance(inputs, (tuple, list)):
            x_past, x_future_cal = inputs[0], inputs[1]
        elif isinstance(inputs, dict):
            x_past = inputs.get("x_past", inputs.get("past"))
            x_future_cal = inputs.get("x_future_cal", inputs.get("future_cal"))
        else:
            x_past = inputs

        scale = math.sqrt(self.d_model)
        # 1. Encode past sequence: (B, 168, 9) -> (B, 168, d_model)
        memory = self.pos_encoder(self.enc_embedding(x_past) * scale, training=training)
        for block in self.encoder_blocks:
            memory = block(memory, training=training)

        # 2. Decode future calendar sequence: (B, 24, 8) -> (B, 24, d_model)
        dec_out = self.pos_encoder(self.dec_embedding(x_future_cal) * scale, training=training)
        for block in self.decoder_blocks:
            dec_out = block(dec_out, memory=memory, training=training)

        return self.quantile_head(dec_out)  # (B, 24, 3)

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
