"""
TensorFlow 2.x Dataset and DataLoader definitions for multi-horizon electricity load forecasting.
Outputs tf.data.Dataset pipelines while preserving chronological leakage-free splits and scaling.
"""

import os
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf


class MultiHorizonLoadDataset:
    """
    Sliding window dataset for multi-horizon forecasting.
    
    Yields:
        x_past: Historical window of shape (lookback, num_features) = (168, 9)
                Features: [target, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, is_weekend, is_holiday]
        x_future_cal: Future calendar window of shape (horizon, num_cal_features) = (24, 8)
                      Features: [hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, is_weekend, is_holiday]
        y_target: Ground truth future target values of shape (horizon,) = (24,)
    """
    def __init__(self, df: pd.DataFrame, lookback: int = 168, horizon: int = 24):
        self.data = np.asarray(df.values, dtype=np.float32)
        self.lookback = lookback
        self.horizon = horizon
        self.length = max(0, len(self.data) - self.lookback - self.horizon + 1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_past = self.data[idx : idx + self.lookback]                                           # (168, 9)
        x_future_cal = self.data[idx + self.lookback : idx + self.lookback + self.horizon, 1:]    # (24, 8)
        y_target = self.data[idx + self.lookback : idx + self.lookback + self.horizon, 0]        # (24,)
        return x_past, x_future_cal, y_target

    def to_tf_dataset(
        self,
        batch_size: int = 64,
        shuffle: bool = False,
        drop_remainder: bool = False,
        seed: Optional[int] = None
    ) -> tf.data.Dataset:
        total_window = self.lookback + self.horizon
        if self.length <= 0:
            raise ValueError(f"Data length ({len(self.data)}) is too small for lookback={self.lookback} and horizon={self.horizon}")

        windows = np.lib.stride_tricks.sliding_window_view(
            self.data, (total_window, self.data.shape[1])
        )[:, 0, :, :]

        past = np.ascontiguousarray(windows[:, :self.lookback, :], dtype=np.float32)
        future_cal = np.ascontiguousarray(windows[:, self.lookback:, 1:], dtype=np.float32)
        y = np.ascontiguousarray(windows[:, self.lookback:, 0], dtype=np.float32)

        ds = tf.data.Dataset.from_tensor_slices((past, future_cal, y))

        if shuffle:
            buffer_size = min(len(past), 10000)
            ds = ds.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True)

        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds


def get_dataloaders(
    processed_dir: str = "data/processed",
    lookback: int = 168,
    horizon: int = 24,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: Optional[int] = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Load preprocessed train/val/test parquets and wrap them in optimized tf.data.Dataset pipelines.
    """
    train_path = os.path.join(processed_dir, "train.parquet")
    val_path = os.path.join(processed_dir, "val.parquet")
    test_path = os.path.join(processed_dir, "test.parquet")

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing preprocessed dataset file: '{p}'. "
                f"Please run 'python data/preprocess.py' first."
            )

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    train_dataset = MultiHorizonLoadDataset(train_df, lookback=lookback, horizon=horizon)
    val_dataset = MultiHorizonLoadDataset(val_df, lookback=lookback, horizon=horizon)
    test_dataset = MultiHorizonLoadDataset(test_df, lookback=lookback, horizon=horizon)

    train_loader = train_dataset.to_tf_dataset(
        batch_size=batch_size,
        shuffle=True,
        drop_remainder=True,
        seed=seed
    )
    val_loader = val_dataset.to_tf_dataset(
        batch_size=batch_size,
        shuffle=False,
        drop_remainder=False
    )
    test_loader = test_dataset.to_tf_dataset(
        batch_size=batch_size,
        shuffle=False,
        drop_remainder=False
    )

    return train_loader, val_loader, test_loader
