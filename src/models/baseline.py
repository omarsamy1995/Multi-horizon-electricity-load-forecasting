"""
Baseline models: SeasonalNaiveBaseline (168h weekly persistence) and MLPBaseline in tf.keras.
"""

from typing import Optional, Union, Tuple, Dict, Any
import numpy as np
import tensorflow as tf
from src.models.custom_layers import NonCrossingQuantileHead


class SeasonalNaiveBaseline:
    """Non-parametric baseline copying past weekly values (t-168 to t-145) for all 3 quantiles."""
    def __init__(self, mode: str = "weekly"):
        self.mode = mode

    def predict(
        self,
        x_past: Union[tf.Tensor, np.ndarray],
        x_future_cal: Optional[Union[tf.Tensor, np.ndarray]] = None
    ) -> Union[tf.Tensor, np.ndarray]:
        # x_past: (B, 168, 9) where feature 0 is load -> preds: (B, 24, 3)
        if tf.is_tensor(x_past):
            slice_idx = slice(0, 24) if self.mode == "weekly" else slice(-24, None)
            pred_median = x_past[:, slice_idx, 0]  # (B, 24)
            return tf.repeat(tf.expand_dims(pred_median, axis=-1), repeats=3, axis=-1)

        x_np = np.asarray(x_past)
        slice_idx = slice(0, 24) if self.mode == "weekly" else slice(-24, None)
        pred_median = x_np[:, slice_idx, 0]
        return np.repeat(np.expand_dims(pred_median, axis=-1), repeats=3, axis=-1)

    def __call__(self, x_past, x_future_cal=None):
        return self.predict(x_past, x_future_cal)


@tf.keras.utils.register_keras_serializable(package="models")
class MLPBaseline(tf.keras.Model):
    """Flattens past (168*9) and future (24*8), passes through Dense+LN blocks to NonCrossingQuantileHead."""
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
        # Handles single inputs or tuple (x_past, x_future_cal)
        if isinstance(inputs, (tuple, list)):
            x_past, x_future_cal = inputs[0], inputs[1]
        elif isinstance(inputs, dict):
            x_past = inputs.get("x_past", inputs.get("past"))
            x_future_cal = inputs.get("x_future_cal", inputs.get("future_cal"))
        else:
            x_past = inputs

        # Flatten past (B, 1512) and future (B, 192) -> concat to (B, 1704)
        x = tf.concat([self.flatten_past(x_past), self.flatten_future(x_future_cal)], axis=-1)
        feat = self.drop1(self.relu1(self.ln1(self.fc1(x))), training=training)
        feat = self.drop2(self.relu2(self.ln2(self.fc2(feat))), training=training)
        return self.head(feat)  # (B, 24, 3)

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
