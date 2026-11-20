import os
import sys
sys.path.insert(0, os.path.abspath("."))
import glob
import torch
import joblib
import numpy as np
import pandas as pd
from evaluate import load_model_from_checkpoint, evaluate_model, evaluate_seasonal_naive
from src.datasets.aep_dataset import get_dataloaders

scaler = joblib.load('checkpoints/aep_scaler.pkl')
_, _, test_loader = get_dataloaders(processed_dir='data/processed', lookback=168, horizon=24, batch_size=128)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models = ['mlp_baseline', 'seq2seq_gru', 'transformer', 'tft_advanced']
results = {}

for m in models:
    ckpts = sorted(glob.glob(f'checkpoints/{m}_seed*_best.pt'))
    results[m] = []
    for c in ckpts:
        model, info = load_model_from_checkpoint(c, device)
        met, _ = evaluate_model(model, test_loader, scaler, device, model_name=m)
        results[m].append(met)
        print(f"{c}: MAE={met['MAE (MW)']}, RMSE={met['RMSE (MW)']}, MAPE={met['MAPE (%)']}, Cov={met.get('Coverage_80_Interval (%)')}")

print("\n=== SUMMARY MEAN +- SD ACROSS 3 SEEDS ===")
for m in models:
    maes = [r['MAE (MW)'] for r in results[m]]
    rmses = [r['RMSE (MW)'] for r in results[m]]
    mapes = [r['MAPE (%)'] for r in results[m]]
    covs = [r.get('Coverage_80_Interval (%)', 0) for r in results[m]]
    mae_1h = [r['MAE_1h'] for r in results[m]]
    mae_6h = [r['MAE_6h'] for r in results[m]]
    mae_12h = [r['MAE_12h'] for r in results[m]]
    mae_24h = [r['MAE_24h'] for r in results[m]]
    lats = [r['Latency (ms/sample)'] for r in results[m]]
    print(f"Model: {m}")
    print(f"  MAE (MW): {np.mean(maes):.2f} +/- {np.std(maes):.2f}")
    print(f"  RMSE (MW): {np.mean(rmses):.2f} +/- {np.std(rmses):.2f}")
    print(f"  MAPE (%): {np.mean(mapes):.2f} +/- {np.std(mapes):.2f}")
    print(f"  Coverage (%): {np.mean(covs):.2f} +/- {np.std(covs):.2f}")
    print(f"  1h MAE: {np.mean(mae_1h):.2f} +/- {np.std(mae_1h):.2f}")
    print(f"  6h MAE: {np.mean(mae_6h):.2f} +/- {np.std(mae_6h):.2f}")
    print(f"  12h MAE: {np.mean(mae_12h):.2f} +/- {np.std(mae_12h):.2f}")
    print(f"  24h MAE: {np.mean(mae_24h):.2f} +/- {np.std(mae_24h):.2f}")
    print(f"  Latency: {np.mean(lats):.4f} ms/sample")
