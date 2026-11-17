"""
Preprocess AEP Hourly Electricity Load Dataset for Multi-Horizon Deep Learning.
- Fills missing timestamps with linear interpolation
- Extracts cyclical temporal features (sin/cos of hour, dow, month)
- Adds US Holiday and weekend indicators
- Performs chronological split (70% Train, 15% Val, 15% Test)
- Normalizes target using StandardScaler fitted strictly on Train partition
"""

import os
import sys
import argparse
import joblib
import holidays
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def clean_time_series(df: pd.DataFrame, target_col: str = "AEP_MW") -> pd.DataFrame:
    """Sort time series, eliminate duplicate timestamps, and interpolate missing hours."""
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").drop_duplicates(subset=["Datetime"]).set_index("Datetime")
    
    # Enforce strictly hourly frequency
    df = df.asfreq("h")
    
    missing_before = df[target_col].isna().sum()
    df[target_col] = df[target_col].interpolate(method="linear")
    print(f"[INFO] Missing hourly timestamps filled via linear interpolation: {missing_before}")
    
    return df

def extract_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar features and apply continuous cyclical encoding (sin/cos)."""
    df = df.copy()
    us_holidays = holidays.US()
    
    hour = df.index.hour
    dow = df.index.day_of_week
    month = df.index.month
    
    # Cyclical sin/cos encodings
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    
    # Binary calendar flags
    df["is_weekend"] = (dow >= 5).astype(np.float32)
    df["is_holiday"] = pd.Series(df.index.date, index=df.index).apply(
        lambda d: 1.0 if d in us_holidays else 0.0
    ).astype(np.float32)
    
    return df

def run_preprocessing(raw_path: str, out_dir: str, scaler_dir: str, target_col: str = "AEP_MW"):
    print(f"[INFO] Loading raw dataset from: {raw_path}")
    df = pd.read_csv(raw_path)
    
    df = clean_time_series(df, target_col=target_col)
    df = extract_calendar_features(df)
    
    # Feature ordering: Target column first, followed by the 8 calendar features
    feature_cols = [
        target_col, "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "is_weekend", "is_holiday"
    ]
    df = df[feature_cols].astype(np.float32)
    
    # Chronological partition (70% Train, 15% Val, 15% Test)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Leakage-free normalization: fit exclusively on Train
    scaler = StandardScaler()
    train_df[target_col] = scaler.fit_transform(train_df[[target_col]])
    val_df[target_col] = scaler.transform(val_df[[target_col]])
    test_df[target_col] = scaler.transform(test_df[[target_col]])
    
    # Save processed parquets and scaler
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)
    
    train_df.to_parquet(os.path.join(out_dir, "train.parquet"))
    val_df.to_parquet(os.path.join(out_dir, "val.parquet"))
    test_df.to_parquet(os.path.join(out_dir, "test.parquet"))
    scaler_out_path = os.path.join(scaler_dir, "aep_scaler.pkl")
    joblib.dump(scaler, scaler_out_path)
    
    print("-" * 50)
    print(f"[SUCCESS] Preprocessing complete!")
    print(f"Train shape : {train_df.shape} | Range: {train_df.index.min()} -> {train_df.index.max()}")
    print(f"Val shape   : {val_df.shape}   | Range: {val_df.index.min()} -> {val_df.index.max()}")
    print(f"Test shape  : {test_df.shape}  | Range: {test_df.index.min()} -> {test_df.index.max()}")
    print(f"Scaler mean : {scaler.mean_[0]:.2f} MW | std: {scaler.scale_[0]:.2f} MW")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess AEP Hourly Load Dataset")
    parser.add_argument("--raw_path", type=str, default="data/raw/AEP_hourly.csv")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--scaler_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    
    run_preprocessing(args.raw_path, args.out_dir, args.scaler_dir)
