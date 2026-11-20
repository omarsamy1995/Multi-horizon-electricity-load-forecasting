"""
Custom TensorFlow 2.x / Keras components written from scratch for Multi-Horizon Forecasting:
1. MultiHorizonQuantileLoss (subclassed tf.keras.losses.Loss for pinball loss [0.10, 0.50, 0.90])
2. NonCrossingQuantileHead (subclassed tf.keras.layers.Layer with softplus guarantee q10 <= q50 <= q90)
3. GatedLinearUnit (GLU)
4. GatedResidualNetwork (GRN)
5. InterpretableMultiHeadAttention (Interpretable temporal self/cross-attention)
6. VariableSelectionNetwork (VSN)
"""

import math
from typing import List, Optional, Tuple, Union
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="custom_layers")
class MultiHorizonQuantileLoss(tf.keras.losses.Loss):
    """
    Multi-Horizon Quantile Loss (Pinball Loss) for arbitrary quantiles.
    
    L_q(y, y_hat) = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    
    Parameters:
        quantiles: List of float quantiles to optimize (default: [0.1, 0.5, 0.9])
    """
    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        reduction: str = "sum_over_batch_size",
        name: str = "multi_horizon_quantile_loss",
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = [float(q) for q in quantiles]

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Args:
            y_true: Ground truth target of shape (B, Horizon) or (B, Horizon, 1)
            y_pred: Predictions tensor of shape (B, Horizon, num_quantiles)
        Returns:
            Scalar aggregated pinball loss (or per-sample loss vector).
        """
        # Handle transposed arguments if passed as (y_pred, y_true)
        if (y_true.shape[-1] is not None and y_true.shape[-1] == len(self.quantiles)
                and (y_pred.shape[-1] is None or y_pred.shape[-1] != len(self.quantiles))):
            y_true, y_pred = y_pred, y_true

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if len(y_true.shape) == 2 or tf.rank(y_true) == 2:
            y_true = tf.expand_dims(y_true, axis=-1)  # (B, Horizon, 1)

        # Broadcast error: (B, Horizon, num_quantiles)
        error = y_true - y_pred
        q = tf.reshape(tf.constant(self.quantiles, dtype=tf.float32), (1, 1, -1))

        # Pinball loss per element
        loss = tf.maximum(q * error, (q - 1.0) * error)
        # Reduce over horizon and quantiles
        return tf.reduce_mean(loss, axis=[-2, -1])

    def per_quantile_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Computes loss separated by quantile (num_quantiles,)."""
        if (y_true.shape[-1] is not None and y_true.shape[-1] == len(self.quantiles)
                and (y_pred.shape[-1] is None or y_pred.shape[-1] != len(self.quantiles))):
            y_true, y_pred = y_pred, y_true

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
        config.update({
            "quantiles": self.quantiles
        })
        return config


@tf.keras.utils.register_keras_serializable(package="custom_layers")
class NonCrossingQuantileHead(tf.keras.layers.Layer):
    """
    Non-Crossing Quantile Projection Head.
    Guarantees q10 <= q50 <= q90 with 0.00% crossing violations by predicting
    median q50 and strictly positive offsets delta_low and delta_high via softplus:
      q10 = q50 - softplus(delta_low)
      q90 = q50 + softplus(delta_high)
      quantiles = [q10, q50, q90]
    
    Supports:
      - 3D features (B, Horizon, D): output_horizon=None -> dense(1) per head -> (B, Horizon, 3)
      - 2D features (B, D): output_horizon=H -> dense(H) per head -> (B, H, 3)
    """
    def __init__(self, output_horizon: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.output_horizon = output_horizon
        out_units = output_horizon if output_horizon is not None else 1
        self.fc_q50 = tf.keras.layers.Dense(out_units, name="fc_q50")
        self.fc_delta_low = tf.keras.layers.Dense(out_units, name="fc_delta_low")
        self.fc_delta_high = tf.keras.layers.Dense(out_units, name="fc_delta_high")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        q50 = self.fc_q50(inputs)
        delta_low = tf.nn.softplus(self.fc_delta_low(inputs))
        delta_high = tf.nn.softplus(self.fc_delta_high(inputs))

        q10 = q50 - delta_low
        q90 = q50 + delta_high

        if self.output_horizon is not None:
            # inputs is (B, D) -> q50 is (B, H) -> stack to (B, H, 3)
            return tf.stack([q10, q50, q90], axis=-1)
        else:
            # inputs is (B, H, D) -> q50 is (B, H, 1) -> concat to (B, H, 3)
            return tf.concat([q10, q50, q90], axis=-1)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "output_horizon": self.output_horizon
        })
        return config


@tf.keras.utils.register_keras_serializable(package="custom_layers")
class GatedLinearUnit(tf.keras.layers.Layer):
    """
    Gated Linear Unit (GLU):
    GLU(x) = Linear_1(x) * sigmoid(Linear_2(x))
    """
    def __init__(self, input_dim: Optional[int] = None, output_dim: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        units = (self.output_dim * 2) if self.output_dim is not None else None
        self.fc = tf.keras.layers.Dense(units) if units is not None else None

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
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim
        })
        return config


@tf.keras.utils.register_keras_serializable(package="custom_layers")
class GatedResidualNetwork(tf.keras.layers.Layer):
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
        if context_dim is not None and context_dim > 0:
            self.context_fc = tf.keras.layers.Dense(hidden_dim, use_bias=False, name="context_fc")
        else:
            self.context_fc = None

        self.elu = tf.keras.layers.ELU()
        self.fc2 = tf.keras.layers.Dense(self.output_dim * 2, name="fc2")
        self.dropout = tf.keras.layers.Dropout(dropout)

        if self.input_dim != self.output_dim:
            self.skip_proj = tf.keras.layers.Dense(self.output_dim, name="skip_proj")
        else:
            self.skip_proj = None

        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1, name="layer_norm")

    def call(
        self,
        x: tf.Tensor,
        context: Optional[tf.Tensor] = None,
        training: Optional[bool] = None
    ) -> tf.Tensor:
        hidden = self.fc1(x)
        if self.context_fc is not None and context is not None:
            c = self.context_fc(context)
            if len(c.shape) == 2 and len(x.shape) == 3:
                c = tf.expand_dims(c, axis=1)
            hidden = hidden + c

        hidden = self.elu(hidden)
        hidden = self.dropout(hidden, training=training)

        gate_out = self.fc2(hidden)
        val, gate = tf.split(gate_out, num_or_size_splits=2, axis=-1)
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
    """
    Interpretable Multi-Head Attention layer.
    
    Instead of projecting concatenated heads, TFT computes shared attention
    weights across heads or projects head values directly, allowing exact
    extraction of temporal attention weights across horizons.
    """
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
        """
        Args:
            query: (B, L_q, d_model)
            key:   (B, L_k, d_model)
            value: (B, L_k, d_model)
            mask:  Optional attention mask
        Returns:
            out: (B, L_q, d_model)
            interpretable_attn: (B, L_q, L_k) interpretable attention map averaged over heads
        """
        B = tf.shape(query)[0]
        L_q = tf.shape(query)[1]
        L_k = tf.shape(key)[1]

        # Reshape to (B, L, n_heads, d_k) -> transpose to (B, n_heads, L, d_k)
        Q = tf.transpose(tf.reshape(self.q_proj(query), (B, L_q, self.n_heads, self.d_k)), [0, 2, 1, 3])
        K = tf.transpose(tf.reshape(self.k_proj(key), (B, L_k, self.n_heads, self.d_k)), [0, 2, 1, 3])
        V = tf.transpose(tf.reshape(self.v_proj(value), (B, L_k, self.n_heads, self.d_k)), [0, 2, 1, 3])

        # Scaled dot-product: (B, n_heads, L_q, L_k)
        scores = tf.matmul(Q, K, transpose_b=True) * self.scale
        if mask is not None:
            mask = tf.cast(mask, scores.dtype)
            scores = scores + (1.0 - mask) * -1e9

        attn_weights_heads = tf.nn.softmax(scores, axis=-1)
        attn_weights_heads = self.dropout(attn_weights_heads, training=training)

        # Head outputs: (B, n_heads, L_q, d_k)
        context = tf.matmul(attn_weights_heads, V)
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, (B, L_q, self.d_model))
        out = self.out_proj(context)

        # Mean attention weights across heads for interpretability: (B, L_q, L_k)
        interpretable_attn = tf.reduce_mean(attn_weights_heads, axis=1)

        return out, interpretable_attn

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "dropout": self.dropout_rate
        })
        return config


@tf.keras.utils.register_keras_serializable(package="custom_layers")
class VariableSelectionNetwork(tf.keras.layers.Layer):
    """
    Variable Selection Network (VSN) for feature importance and non-linear feature transformation.
    """
    def __init__(
        self,
        num_features: int,
        feature_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        **kwargs
    ):
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
            num_features * feature_dim,
            hidden_dim,
            num_features,
            dropout=dropout,
            name="flattened_grn"
        )
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, feature_list: List[tf.Tensor], training: Optional[bool] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            feature_list: List of length num_features, each of shape (B, L, feature_dim)
        Returns:
            combined: (B, L, hidden_dim)
            weights:  (B, L, num_features)
        """
        flattened = tf.concat(feature_list, axis=-1)
        weights = self.softmax(self.flattened_grn(flattened, training=training))

        transformed_list = [grn(f, training=training) for grn, f in zip(self.var_grns, feature_list)]
        transformed = tf.stack(transformed_list, axis=-2)  # (B, L, num_features, hidden_dim)

        combined = tf.reduce_sum(transformed * tf.expand_dims(weights, axis=-1), axis=-2)
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
