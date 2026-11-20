"""
Custom TensorFlow 2.x / Keras primitives for multi-horizon probabilistic forecasting.
Includes Pinball Loss, Non-Crossing Quantile Head, GLU, GRN, Interpretable Attention, and VSN.
"""

import math
from typing import List, Optional, Tuple, Union
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="custom_layers")
class MultiHorizonQuantileLoss(tf.keras.losses.Loss):
    """Pinball loss for multi-quantile regression: L_q = max(q*e, (q-1)*e)."""
    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        reduction: str = "sum_over_batch_size",
        name: str = "multi_horizon_quantile_loss",
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.quantiles = [float(q) for q in (quantiles or [0.1, 0.5, 0.9])]

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # y_true: (B, H) or (B, H, 1) | y_pred: (B, H, 3)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        if len(y_true.shape) == 2 or tf.rank(y_true) == 2:
            y_true = tf.expand_dims(y_true, axis=-1)  # (B, H, 1)

        error = y_true - y_pred  # (B, H, 3)
        q = tf.reshape(tf.constant(self.quantiles, dtype=tf.float32), (1, 1, -1))
        loss = tf.maximum(q * error, (q - 1.0) * error)  # Pinball loss per element
        return tf.reduce_mean(loss, axis=[-2, -1])  # Mean over horizon and quantiles -> (B,)

    def per_quantile_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Computes mean pinball loss separated by quantile -> (3,)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        if len(y_true.shape) == 2 or tf.rank(y_true) == 2:
            y_true = tf.expand_dims(y_true, axis=-1)
        error = y_true - y_pred
        q = tf.reshape(tf.constant(self.quantiles, dtype=tf.float32), (1, 1, -1))
        loss = tf.maximum(q * error, (q - 1.0) * error)
        return tf.reduce_mean(loss, axis=[0, 1])

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"quantiles": self.quantiles})
        return config


@tf.keras.utils.register_keras_serializable(package="custom_layers")
class NonCrossingQuantileHead(tf.keras.layers.Layer):
    """Guarantees q10 <= q50 <= q90 via Softplus positive offsets delta_low and delta_high."""
    def __init__(self, output_horizon: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.output_horizon = output_horizon
        out_units = output_horizon if output_horizon is not None else 1
        self.fc_q50 = tf.keras.layers.Dense(out_units, name="fc_q50")
        self.fc_delta_low = tf.keras.layers.Dense(out_units, name="fc_delta_low")
        self.fc_delta_high = tf.keras.layers.Dense(out_units, name="fc_delta_high")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # inputs: (B, D) if output_horizon=H else (B, H, D)
        q50 = self.fc_q50(inputs)
        delta_low = tf.nn.softplus(self.fc_delta_low(inputs))
        delta_high = tf.nn.softplus(self.fc_delta_high(inputs))
        q10 = q50 - delta_low   # strictly <= q50
        q90 = q50 + delta_high  # strictly >= q50

        # Output shape: (B, H, 3) where columns are [q10, q50, q90]
        if self.output_horizon is not None:
            return tf.stack([q10, q50, q90], axis=-1)
        return tf.concat([q10, q50, q90], axis=-1)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"output_horizon": self.output_horizon})
        return config


@tf.keras.utils.register_keras_serializable(package="custom_layers")
class GatedLinearUnit(tf.keras.layers.Layer):
    """GLU(x) = Linear_1(x) * sigmoid(Linear_2(x))."""
    def __init__(self, input_dim: Optional[int] = None, output_dim: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        units = (self.output_dim * 2) if self.output_dim is not None else None
        self.fc = tf.keras.layers.Dense(units, name="fc") if units is not None else None

    def build(self, input_shape):
        if self.output_dim is None:
            self.output_dim = int(input_shape[-1])
        if self.fc is None:
            self.fc = tf.keras.layers.Dense(self.output_dim * 2, name="fc")
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        out = self.fc(x)
        val, gate = tf.split(out, num_or_size_splits=2, axis=-1)
        return val * tf.sigmoid(gate)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"input_dim": self.input_dim, "output_dim": self.output_dim})
        return config


@tf.keras.utils.register_keras_serializable(package="custom_layers")
class GatedResidualNetwork(tf.keras.layers.Layer):
    """GRN: Non-linear processing with gated skip connection and optional context injection."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.context_dim = context_dim
        self.dropout_rate = dropout

        self.fc1 = tf.keras.layers.Dense(hidden_dim, name="fc1")
        self.context_fc = tf.keras.layers.Dense(hidden_dim, use_bias=False, name="context_fc") if (context_dim and context_dim > 0) else None
        self.elu = tf.keras.layers.ELU()
        self.fc2 = tf.keras.layers.Dense(self.output_dim * 2, name="fc2")
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.skip_proj = tf.keras.layers.Dense(self.output_dim, name="skip_proj") if self.input_dim != self.output_dim else None
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1, name="layer_norm")

    def call(self, x: tf.Tensor, context: Optional[tf.Tensor] = None, training: Optional[bool] = None) -> tf.Tensor:
        hidden = self.fc1(x)
        if self.context_fc is not None and context is not None:
            c = self.context_fc(context)
            if len(c.shape) == 2 and len(x.shape) == 3:
                c = tf.expand_dims(c, axis=1)
            hidden = hidden + c

        hidden = self.dropout(self.elu(hidden), training=training)
        val, gate = tf.split(self.fc2(hidden), num_or_size_splits=2, axis=-1)
        gated = val * tf.sigmoid(gate)
        skip = self.skip_proj(x) if self.skip_proj is not None else x
        return self.layer_norm(skip + gated)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "context_dim": self.context_dim,
            "dropout": self.dropout_rate
        })
        return config


@tf.keras.utils.register_keras_serializable(package="custom_layers")
class InterpretableMultiHeadAttention(tf.keras.layers.Layer):
    """Multi-Head Attention returning both context and head-averaged attention map."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout_rate = dropout

        self.q_proj = tf.keras.layers.Dense(d_model, name="q_proj")
        self.k_proj = tf.keras.layers.Dense(d_model, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(d_model, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(d_model, name="out_proj")
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: Optional[bool] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # query: (B, L_q, D) | key, value: (B, L_k, D)
        B = tf.shape(query)[0]
        L_q, L_k = tf.shape(query)[1], tf.shape(key)[1]

        # Project and split heads: (B, n_heads, L, d_k)
        Q = tf.transpose(tf.reshape(self.q_proj(query), (B, L_q, self.n_heads, self.d_k)), [0, 2, 1, 3])
        K = tf.transpose(tf.reshape(self.k_proj(key), (B, L_k, self.n_heads, self.d_k)), [0, 2, 1, 3])
        V = tf.transpose(tf.reshape(self.v_proj(value), (B, L_k, self.n_heads, self.d_k)), [0, 2, 1, 3])

        scores = tf.matmul(Q, K, transpose_b=True) * self.scale
        if mask is not None:
            scores = scores + (1.0 - tf.cast(mask, scores.dtype)) * -1e9

        attn_weights = self.dropout(tf.nn.softmax(scores, axis=-1), training=training)
        context = tf.matmul(attn_weights, V)  # (B, n_heads, L_q, d_k)
        context = tf.reshape(tf.transpose(context, [0, 2, 1, 3]), (B, L_q, self.d_model))
        out = self.out_proj(context)  # (B, L_q, d_model)

        # Average attention weights across heads for interpretability -> (B, L_q, L_k)
        interpretable_attn = tf.reduce_mean(attn_weights, axis=1)
        return out, interpretable_attn

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"d_model": self.d_model, "n_heads": self.n_heads, "dropout": self.dropout_rate})
        return config


@tf.keras.utils.register_keras_serializable(package="custom_layers")
class VariableSelectionNetwork(tf.keras.layers.Layer):
    """VSN: Softmax-weighted feature combination using individual GRNs."""
    def __init__(self, num_features: int, feature_dim: int, hidden_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        self.var_grns = [
            GatedResidualNetwork(feature_dim, hidden_dim, hidden_dim, dropout=dropout, name=f"var_grn_{i}")
            for i in range(num_features)
        ]
        self.flattened_grn = GatedResidualNetwork(
            num_features * feature_dim, hidden_dim, num_features, dropout=dropout, name="flattened_grn"
        )
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, feature_list: List[tf.Tensor], training: Optional[bool] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        # feature_list: list of num_features tensors of shape (B, L, feature_dim)
        flattened = tf.concat(feature_list, axis=-1)
        weights = self.softmax(self.flattened_grn(flattened, training=training))  # (B, L, num_features)

        transformed = tf.stack([grn(f, training=training) for grn, f in zip(self.var_grns, feature_list)], axis=-2)
        combined = tf.reduce_sum(transformed * tf.expand_dims(weights, axis=-1), axis=-2)  # (B, L, hidden_dim)
        return combined, weights

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "num_features": self.num_features,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_rate
        })
        return config
