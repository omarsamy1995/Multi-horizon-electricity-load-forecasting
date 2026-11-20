"""
TensorFlow 2.x Dataset and DataLoader pipelines for multi-horizon electricity load forecasting.
Constructs sliding windows: x_past (B, 168, 9), x_future_cal (B, 24, 8), y_target (B, 24).
"""

import os
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf


class MultiHorizonLoadDataset:
    """Sliding-window dataset generating past context and known future calendar features."""
    def __init__(self, df: pd.DataFrame, lookback: int = 168, horizon: int = 24):
        self.data = np.asarray(df.values, dtype=np.float32)
        self.lookback = lookback
        self.horizon = horizon
        self.length = max(0, len(self.data) - lookback - horizon + 1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # x_past: (168, 9) [load + 8 calendar] | x_future_cal: (24, 8) [calendar only] | y: (24,)
        x_past = self.data[idx : idx + self.lookback]
        x_future_cal = self.data[idx + self.lookback : idx + self.lookback + self.horizon, 1:]
        y_target = self.data[idx + self.lookback : idx + self.lookback + self.horizon, 0]
        return x_past, x_future_cal, y_target

    def to_tf_dataset(
        self,
        batch_size: int = 64,
        shuffle: bool = False,
        drop_remainder: bool = False,
        seed: Optional[int] = None
    ) -> tf.data.Dataset:
        total_window = self.lookback + self.horizon
        windows = np.lib.stride_tricks.sliding_window_view(
            self.data, (total_window, self.data.shape[1])
        )[:, 0, :, :]

        past = np.ascontiguousarray(windows[:, :self.lookback, :], dtype=np.float32)
        future_cal = np.ascontiguousarray(windows[:, self.lookback:, 1:], dtype=np.float32)
        y = np.ascontiguousarray(windows[:, self.lookback:, 0], dtype=np.float32)

        ds = tf.data.Dataset.from_tensor_slices((past, future_cal, y))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(past), 10000), seed=seed, reshuffle_each_iteration=True)

        return ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(tf.data.AUTOTUNE)


def get_dataloaders(
    processed_dir: str = "data/processed",
    lookback: int = 168,
    horizon: int = 24,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: Optional[int] = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Loads processed parquets and returns (train_loader, val_loader, test_loader)."""
    train_df = pd.read_parquet(os.path.join(processed_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(processed_dir, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(processed_dir, "test.parquet"))

    train_ds = MultiHorizonLoadDataset(train_df, lookback, horizon).to_tf_dataset(batch_size, shuffle=True, drop_remainder=True, seed=seed)
    val_ds = MultiHorizonLoadDataset(val_df, lookback, horizon).to_tf_dataset(batch_size, shuffle=False)
    test_ds = MultiHorizonLoadDataset(test_df, lookback, horizon).to_tf_dataset(batch_size, shuffle=False)
    return train_ds, val_ds, test_ds
