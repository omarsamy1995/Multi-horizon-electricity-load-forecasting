"""
Baseline models for Multi-Horizon Electricity Load Forecasting:
1. SeasonalNaiveBaseline (Domain benchmark: Weekly seasonal persistence)
2. MLPBaseline (Parametric baseline: Multi-Layer Perceptron on flattened past + future in tf.keras)
"""

from typing import Optional, Union, Tuple, Dict, Any
import numpy as np
import tensorflow as tf

from src.models.custom_layers import NonCrossingQuantileHead


class SeasonalNaiveBaseline:
    """
    Non-parametric Seasonal Naive baseline.
    
    Given hourly electricity demand, load exhibits a strong 168-hour (7 days) periodicity.
    The weekly seasonal naive forecast takes the observed values from the same time 
    one week prior (t - 168 to t - 145), corresponding to indices [0:24] of x_past.
    """
    def __init__(self, mode: str = "weekly"):
        self.mode = mode

    def predict(
        self,
        x_past: Union[tf.Tensor, np.ndarray],
        x_future_cal: Optional[Union[tf.Tensor, np.ndarray]] = None
    ) -> Union[tf.Tensor, np.ndarray]:
        """
        Args:
            x_past: (B, 168, 9) historical features where feature 0 is target load.
            x_future_cal: (B, 24, 8) future calendar features (unused in naive baseline).
        Returns:
            preds: (B, 24, 3) where columns correspond to [q10, q50, q90].
        """
        is_tf = tf.is_tensor(x_past)
        if is_tf:
            if self.mode == "weekly":
                pred_median = x_past[:, :24, 0]  # (B, 24)
            elif self.mode == "daily":
                pred_median = x_past[:, -24:, 0]  # (B, 24)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            preds = tf.repeat(tf.expand_dims(pred_median, axis=-1), repeats=3, axis=-1)
            return preds
        else:
            x_past_np = np.asarray(x_past)
            if self.mode == "weekly":
                pred_median = x_past_np[:, :24, 0]
            elif self.mode == "daily":
                pred_median = x_past_np[:, -24:, 0]
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            preds = np.repeat(np.expand_dims(pred_median, axis=-1), repeats=3, axis=-1)
            return preds

    def __call__(
        self,
        x_past: Union[tf.Tensor, np.ndarray],
        x_future_cal: Optional[Union[tf.Tensor, np.ndarray]] = None
    ) -> Union[tf.Tensor, np.ndarray]:
        return self.predict(x_past, x_future_cal)


@tf.keras.utils.register_keras_serializable(package="models")
class MLPBaseline(tf.keras.Model):
    """
    Parametric MLP Baseline:
    Flattens past window (168 * 9) and future calendar (24 * 8), passes through
    fully connected layers with dropout, and outputs (B, 24, 3) quantiles with
    non-crossing guarantee via NonCrossingQuantileHead.
    """
    def __init__(
        self,
        lookback: int = 168,
        past_dim: int = 9,
        horizon: int = 24,
        future_cal_dim: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        name: str = "mlp_baseline",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.lookback = lookback
        self.past_dim = past_dim
        self.horizon = horizon
        self.future_cal_dim = future_cal_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        self.flatten_past = tf.keras.layers.Flatten(name="flatten_past")
        self.flatten_future = tf.keras.layers.Flatten(name="flatten_future")

        self.fc1 = tf.keras.layers.Dense(hidden_dim, name="fc1")
        self.ln1 = tf.keras.layers.LayerNormalization(axis=-1, name="ln1")
        self.relu1 = tf.keras.layers.ReLU(name="relu1")
        self.drop1 = tf.keras.layers.Dropout(dropout, name="drop1")

        self.fc2 = tf.keras.layers.Dense(hidden_dim // 2, name="fc2")
        self.ln2 = tf.keras.layers.LayerNormalization(axis=-1, name="ln2")
        self.relu2 = tf.keras.layers.ReLU(name="relu2")
        self.drop2 = tf.keras.layers.Dropout(dropout, name="drop2")

        self.head = NonCrossingQuantileHead(output_horizon=horizon, name="quantile_head")

    def call(
        self,
        inputs: Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]],
        x_future_cal: Optional[tf.Tensor] = None,
        training: Optional[bool] = None
    ) -> tf.Tensor:
        """
        Accepts:
            inputs: x_past (B, 168, 9) or tuple (x_past, x_future_cal)
            x_future_cal: (B, 24, 8) if inputs is solely x_past
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

        past_flat = self.flatten_past(x_past)
        future_flat = self.flatten_future(x_future_cal)
        x = tf.concat([past_flat, future_flat], axis=-1)

        feat = self.fc1(x)
        feat = self.ln1(feat)
        feat = self.relu1(feat)
        feat = self.drop1(feat, training=training)

        feat = self.fc2(feat)
        feat = self.ln2(feat)
        feat = self.relu2(feat)
        feat = self.drop2(feat, training=training)

        return self.head(feat)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "lookback": self.lookback,
            "past_dim": self.past_dim,
            "horizon": self.horizon,
            "future_cal_dim": self.future_cal_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_rate
        })
        return config
