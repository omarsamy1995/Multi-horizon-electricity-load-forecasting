import os, json, glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Horizon MAE comparison
models_info = [
    ("Seasonal Naive", "seasonal_naive", [1438.00, 1437.89, 1437.81, 1437.75, 1437.76, 1437.77, 1437.76, 1437.73, 1437.70, 1437.66, 1437.62, 1437.59, 1437.55, 1437.50, 1437.45, 1437.41, 1437.37, 1437.35, 1437.33, 1437.31, 1437.30, 1437.30, 1437.30, 1437.30], "#7f7f7f", "--"),
    ("MLP Baseline", "mlp_baseline", None, "#1f77b4", "-"),
    ("Seq2Seq GRU", "seq2seq_gru", None, "#2ca02c", "-"),
    ("Transformer", "transformer", None, "#d62728", "-"),
    ("TFT Advanced", "tft_advanced", None, "#9467bd", "-")
]

plt.figure(figsize=(11, 6))
horizons = np.arange(1, 25)

for label, m_key, default_curve, color, ls in models_info:
    json_path = f"experiments/{m_key}_metrics.json"
    if os.path.exists(json_path):
        with open(json_path) as fp:
            d = json.load(fp)
        curve = d.get("all_horizon_mae", default_curve)
    else:
        curve = default_curve
    if curve is not None:
        plt.plot(horizons, curve, label=label, color=color, linestyle=ls, linewidth=2.2, marker="o" if ls == "-" else None, markersize=4)

plt.title("Multi-Horizon Error Progression across 24 Forecast Hours (Test Set)", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Forecast Horizon Ahead (Hours)", fontsize=12)
plt.ylabel("Mean Absolute Error (MW)", fontsize=12)
plt.xticks(horizons)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11, loc="center right")
plt.tight_layout()
plt.savefig("experiments/all_models_horizon_comparison.png", dpi=300)
plt.close()
print("[SAVED] experiments/all_models_horizon_comparison.png")

# 2. Training Convergence Curves
plt.figure(figsize=(10, 5.5))
histories = [
    ("MLP Baseline (Seed 42)", "experiments/mlp_baseline_seed42_history.csv", "#1f77b4"),
    ("Seq2Seq GRU (Seed 123)", "experiments/seq2seq_gru_seed123_history.csv", "#2ca02c"),
    ("Transformer (Seed 42)", "experiments/transformer_seed42_history.csv", "#d62728"),
    ("TFT Advanced (Seed 999)", "experiments/tft_advanced_seed999_history.csv", "#9467bd")
]

for label, path, color in histories:
    if os.path.exists(path):
        df = pd.read_csv(path)
        plt.plot(df["epoch"], df["val_loss"], label=label, color=color, linewidth=2.0, marker="s", markersize=4)

plt.title("Validation Quantile Loss Convergence across Epochs", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Validation Pinball Loss", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("experiments/training_convergence_comparison.png", dpi=300)
plt.close()
print("[SAVED] experiments/training_convergence_comparison.png")
