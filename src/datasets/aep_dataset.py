"""
PyTorch Dataset and DataLoader definitions for multi-horizon electricity load forecasting.
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

class MultiHorizonLoadDataset(Dataset):
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
        self.data = torch.tensor(df.values, dtype=torch.float32)
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.data) - self.lookback - self.horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_past = self.data[idx : idx + self.lookback]                                           # (168, 9)
        x_future_cal = self.data[idx + self.lookback : idx + self.lookback + self.horizon, 1:]    # (24, 8)
        y_target = self.data[idx + self.lookback : idx + self.lookback + self.horizon, 0]        # (24,)
        return x_past, x_future_cal, y_target

def get_dataloaders(
    processed_dir: str = "data/processed",
    lookback: int = 168,
    horizon: int = 24,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load preprocessed train/val/test parquets and wrap them in PyTorch DataLoaders.
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
