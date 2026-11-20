import os, sys
sys.path.insert(0, os.path.abspath("."))
import torch, joblib, numpy as np, pandas as pd
from evaluate import load_model_from_checkpoint
from src.datasets.aep_dataset import get_dataloaders

scaler = joblib.load('checkpoints/aep_scaler.pkl')
test_df = pd.read_parquet('data/processed/test.parquet')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, _ = load_model_from_checkpoint('checkpoints/seq2seq_gru_seed123_best.pt', device)
model.eval()

# Let's inspect sliding windows over test_df
lookback = 168
horizon = 24
N = len(test_df) - lookback - horizon + 1

timestamps = test_df.index
data_vals = test_df.values

all_maes = []
all_peak_errors = []
worst_windows = []

# Subsample or run in batches
batch_size = 256
_, _, test_loader = get_dataloaders(processed_dir='data/processed', lookback=168, horizon=24, batch_size=batch_size)

all_preds = []
all_targets = []
with torch.no_grad():
    for x_past, x_future_cal, y_target in test_loader:
        x_past = x_past.to(device)
        x_future_cal = x_future_cal.to(device)
        y_pred = model(x_past, x_future_cal)
        all_preds.append(y_pred.cpu().numpy())
        all_targets.append(y_target.numpy())

preds_arr = np.concatenate(all_preds, axis=0) # (N, 24, 3)
targets_arr = np.concatenate(all_targets, axis=0) # (N, 24)

# Invert scale
y_true_mw = scaler.inverse_transform(targets_arr.reshape(-1, 1)).reshape(N, 24)
y_pred_mw = scaler.inverse_transform(preds_arr[:, :, 1].reshape(-1, 1)).reshape(N, 24)
abs_err = np.abs(y_true_mw - y_pred_mw)
window_maes = np.mean(abs_err, axis=1) # (N,)

# Find error categories
# 1. Extreme Heatwave / Cold Snap (> 2500 MW window MAE)
extreme_temp_mask = window_maes > 2000
count_extreme = int(np.sum(extreme_temp_mask))

# 2. Holiday Drops (dates where is_holiday == 1 in future)
future_holidays = [np.any(data_vals[i+lookback : i+lookback+horizon, 8] > 0.5) for i in range(N)]
future_holidays = np.array(future_holidays)
holiday_errs = window_maes[future_holidays]
holiday_high_err = np.sum(holiday_errs > 1000)

# 3. Fast Ramp (gradient in target > 2000 MW within 3 hours)
ramps = [np.max(np.abs(np.diff(y_true_mw[i]))) > 1500 for i in range(N)]
ramps = np.array(ramps)
ramp_high_err = np.sum((ramps) & (window_maes > 1000))

# 4. Long-horizon drift: Error at 24h > 1.5 * Error at 1h
drift_mask = (abs_err[:, 23] > 1.5 * abs_err[:, 0]) & (abs_err[:, 23] > 1000)
count_drift = int(np.sum(drift_mask))

# Find top 5 worst timestamps
worst_indices = np.argsort(window_maes)[::-1][:10]

print("=== ERROR DIAGNOSTICS FOR TABLE 6 ===")
print(f"Total test windows evaluated: {N}")
print(f"Mean window MAE: {np.mean(window_maes):.2f} MW")
print(f"Windows with MAE > 1500 MW: {np.sum(window_maes > 1500)} ({np.sum(window_maes > 1500)/N*100:.2f}%)")
print(f"Windows with MAE > 2000 MW: {count_extreme} ({count_extreme/N*100:.2f}%)")
print(f"Holiday windows count: {np.sum(future_holidays)}, with MAE > 1000 MW: {holiday_high_err}")
print(f"Steep ramp windows count: {np.sum(ramps)}, with MAE > 1000 MW: {ramp_high_err}")
print(f"Long-horizon drift windows count: {count_drift} ({count_drift/N*100:.2f}%)")

print("\nTop 5 Worst Windows:")
for rank, idx in enumerate(worst_indices[:5]):
    dt_start = timestamps[idx + lookback]
    dt_end = timestamps[idx + lookback + horizon - 1]
    peak_true = np.max(y_true_mw[idx])
    peak_pred = np.max(y_pred_mw[idx])
    print(f"Rank {rank+1}: Window Index {idx} | Range: {dt_start} to {dt_end} | Window MAE: {window_maes[idx]:.2f} MW | Max Error: {np.max(abs_err[idx]):.2f} MW | True Peak: {peak_true:.1f} MW, Pred Peak: {peak_pred:.1f} MW")
