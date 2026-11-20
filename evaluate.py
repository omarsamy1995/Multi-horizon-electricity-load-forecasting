"""
Unified Evaluation and Benchmarking Script for Multi-Horizon Electricity Load Forecasting in TensorFlow 2.x.
Computes physical MW metrics (MAE, RMSE, MAPE, 80% coverage), latency profiling, and ablation testing.
"""

import os
import sys
import time
import json
import argparse
from typing import Tuple, Optional, Dict, Any
import joblib
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.datasets.aep_dataset import get_dataloaders
from src.models import (
    SeasonalNaiveBaseline,
    MLPBaseline,
    Seq2SeqGRU,
    TimeSeriesTransformer,
    TemporalFusionModel
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_median: np.ndarray,
    y_q10: np.ndarray = None,
    y_q90: np.ndarray = None
) -> dict:
    """Computes unscaled physical MW metrics across all 24 horizons."""
    diff = y_true - y_pred_median
    abs_diff = np.abs(diff)

    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mape = float(np.mean(abs_diff / (np.abs(y_true) + 1e-8)) * 100.0)

    # Per-horizon MAE for h=1..24
    horizon_mae = [round(float(np.mean(abs_diff[:, h])), 2) for h in range(y_true.shape[1])]

    metrics = {
        "MAE (MW)": round(mae, 2),
        "RMSE (MW)": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "MAE_1h": horizon_mae[0],
        "MAE_6h": horizon_mae[5] if len(horizon_mae) >= 6 else None,
        "MAE_12h": horizon_mae[11] if len(horizon_mae) >= 12 else None,
        "MAE_24h": horizon_mae[23] if len(horizon_mae) >= 24 else None,
        "all_horizon_mae": horizon_mae
    }

    # Empirical Coverage of [q10, q90] Interval (Nominal: 80%)
    if y_q10 is not None and y_q90 is not None:
        in_band = (y_true >= y_q10) & (y_true <= y_q90)
        metrics["Coverage_80_Interval (%)"] = round(float(np.mean(in_band) * 100.0), 2)

    return metrics


def evaluate_seasonal_naive(test_loader, scaler) -> Tuple[dict, tuple]:
    """Evaluates non-parametric 168h weekly persistence baseline."""
    baseline = SeasonalNaiveBaseline(mode="weekly")
    all_preds, all_targets = [], []

    t0 = time.time()
    for x_past, _, y_target in test_loader:
        preds = baseline.predict(x_past)
        all_preds.append(preds.numpy() if hasattr(preds, "numpy") else preds)
        all_targets.append(y_target.numpy() if hasattr(y_target, "numpy") else y_target)
    elapsed = time.time() - t0

    preds_arr = np.concatenate(all_preds, axis=0)     # (N, 24, 3)
    targets_arr = np.concatenate(all_targets, axis=0) # (N, 24)
    N, H, _ = preds_arr.shape

    # Inverse transform to MW
    y_true_mw = scaler.inverse_transform(targets_arr.reshape(-1, 1)).reshape(N, H)
    y_pred_mw = scaler.inverse_transform(preds_arr[:, :, 1].reshape(-1, 1)).reshape(N, H)
    y_q10_mw = scaler.inverse_transform(preds_arr[:, :, 0].reshape(-1, 1)).reshape(N, H)
    y_q90_mw = scaler.inverse_transform(preds_arr[:, :, 2].reshape(-1, 1)).reshape(N, H)

    metrics = compute_metrics(y_true_mw, y_pred_mw, y_q10_mw, y_q90_mw)
    metrics["Latency (ms/sample)"] = round((elapsed / N) * 1000.0, 4)
    metrics["Model"] = "Seasonal Naive (Weekly)"
    return metrics, (y_true_mw, y_pred_mw, y_q10_mw, y_q90_mw)


def instantiate_model_from_type(mtype: str, cfg: dict) -> tf.keras.Model:
    """Instantiates a model architecture from its type name and configuration dictionary."""
    mtype = mtype.lower()
    p_dim, f_dim = cfg.get("past_dim", 9), cfg.get("future_cal_dim", 8)
    if mtype in ["mlp", "mlp_baseline", "baseline"]:
        return MLPBaseline(
            lookback=cfg.get("lookback", 168), past_dim=p_dim, horizon=cfg.get("horizon", 24),
            future_cal_dim=f_dim, hidden_dim=cfg.get("hidden_dim", 256), dropout=cfg.get("dropout", 0.0)
        )
    elif mtype in ["seq2seq", "seq2seq_gru", "gru"]:
        return Seq2SeqGRU(
            past_dim=p_dim, future_cal_dim=f_dim, hidden_dim=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 2), dropout=cfg.get("dropout", 0.0), use_attention=cfg.get("use_attention", True)
        )
    elif mtype in ["transformer", "transformer_enc_dec"]:
        return TimeSeriesTransformer(
            past_dim=p_dim, future_cal_dim=f_dim, d_model=cfg.get("d_model", 128),
            nhead=cfg.get("nhead", 4), num_encoder_layers=cfg.get("num_encoder_layers", 2),
            num_decoder_layers=cfg.get("num_decoder_layers", 2), dim_feedforward=cfg.get("dim_feedforward", 256),
            dropout=cfg.get("dropout", 0.0)
        )
    elif mtype in ["tft", "tft_advanced", "temporal_fusion"]:
        return TemporalFusionModel(
            past_dim=p_dim, future_cal_dim=f_dim, hidden_dim=cfg.get("hidden_dim", 128),
            n_heads=cfg.get("n_heads", 4), num_lstm_layers=cfg.get("num_lstm_layers", 1), dropout=cfg.get("dropout", 0.0)
        )
    raise ValueError(f"Unknown model type: {mtype}")


def load_model_from_checkpoint(checkpoint_path: str):
    """Instantiates architecture and loads saved weights from .weights.h5 or checkpoint path."""
    base_no_ext = os.path.splitext(checkpoint_path)[0]
    if base_no_ext.endswith(".weights"):
        base_no_ext = os.path.splitext(base_no_ext)[0]
    meta_path = base_no_ext + ".json"

    ckpt_info = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            ckpt_info = json.load(f)

    cfg = ckpt_info.get("config", {}).get("model", {})
    model_name = ckpt_info.get("model_name", "")

    # Infer model from filename if not in metadata
    if not model_name or not cfg:
        fname = os.path.basename(checkpoint_path).lower()
        mapping = [
            ("mlp", "mlp_baseline", "configs/baseline.yaml"),
            ("seq2seq", "seq2seq_gru", "configs/seq2seq_gru.yaml"),
            ("gru", "seq2seq_gru", "configs/seq2seq_gru.yaml"),
            ("tft", "tft_advanced", "configs/tft_advanced.yaml"),
            ("transformer", "transformer", "configs/transformer.yaml")
        ]
        for key, name, cfg_file in mapping:
            if key in fname:
                model_name = name
                if os.path.exists(cfg_file):
                    with open(cfg_file, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f)["model"]
                break

    mtype = cfg.get("type", model_name)
    model = instantiate_model_from_type(mtype, cfg)

    # Initialize weights via dummy forward pass
    dummy_past = tf.zeros((1, 168, cfg.get("past_dim", 9)), dtype=tf.float32)
    dummy_cal = tf.zeros((1, 24, cfg.get("future_cal_dim", 8)), dtype=tf.float32)
    _ = model(dummy_past, dummy_cal, training=False)

    # Load weights
    weights_h5 = base_no_ext + ".weights.h5" if not checkpoint_path.endswith(".weights.h5") else checkpoint_path
    if os.path.exists(weights_h5):
        model.load_weights(weights_h5)
    elif os.path.exists(checkpoint_path) and not checkpoint_path.endswith(".pt"):
        model.load_weights(checkpoint_path)
    else:
        alt_weights = os.path.join("checkpoints", f"{model_name}_best.weights.h5")
        if os.path.exists(alt_weights):
            model.load_weights(alt_weights)

    return model, {"model_name": model_name, "config": {"model": cfg}}


def evaluate_model(
    model: tf.keras.Model,
    test_loader,
    scaler,
    device: str = None,
    ablation_no_calendar: bool = False,
    model_name: str = "Model"
) -> Tuple[dict, tuple]:
    """Runs test inference, per-sample latency measurement, and unscaling."""
    all_preds, all_targets = [], []

    # Warmup for latency profiling
    dummy_p = tf.zeros((1, 168, 9), dtype=tf.float32)
    dummy_c = tf.zeros((1, 24, 8), dtype=tf.float32)
    for _ in range(5):
        _ = model(dummy_p, dummy_c, training=False)

    total_time, total_samples = 0.0, 0
    for x_past, x_future_cal, y_target in test_loader:
        if ablation_no_calendar:
            x_future_cal = tf.zeros_like(x_future_cal)  # Ablation: zero out calendar

        t_start = time.time()
        y_pred = model(x_past, x_future_cal, training=False)
        total_time += (time.time() - t_start)
        total_samples += int(tf.shape(x_past)[0])

        all_preds.append(y_pred.numpy() if hasattr(y_pred, "numpy") else y_pred)
        all_targets.append(y_target.numpy() if hasattr(y_target, "numpy") else y_target)

    preds_arr = np.concatenate(all_preds, axis=0)     # (N, 24, 3)
    targets_arr = np.concatenate(all_targets, axis=0) # (N, 24)
    N, H, _ = preds_arr.shape

    # Inverse transform to MW
    y_true_mw = scaler.inverse_transform(targets_arr.reshape(-1, 1)).reshape(N, H)
    y_pred_mw = scaler.inverse_transform(preds_arr[:, :, 1].reshape(-1, 1)).reshape(N, H)
    y_q10_mw = scaler.inverse_transform(preds_arr[:, :, 0].reshape(-1, 1)).reshape(N, H)
    y_q90_mw = scaler.inverse_transform(preds_arr[:, :, 2].reshape(-1, 1)).reshape(N, H)

    metrics = compute_metrics(y_true_mw, y_pred_mw, y_q10_mw, y_q90_mw)
    metrics["Latency (ms/sample)"] = round((total_time / max(1, total_samples)) * 1000.0, 4)
    metrics["Model"] = model_name
    metrics["Ablation_No_Calendar"] = ablation_no_calendar
    return metrics, (y_true_mw, y_pred_mw, y_q10_mw, y_q90_mw)


def save_evaluation_plots(metrics: dict, plot_data: tuple, model_name: str, out_dir: str = "experiments"):
    """Generates and exports publication-ready evaluation plots."""
    os.makedirs(out_dir, exist_ok=True)
    y_true_mw, y_pred_mw, y_q10_mw, y_q90_mw = plot_data

    # 1. Horizon MAE Progression Curve
    plt.figure(figsize=(9, 5))
    horizons = np.arange(1, len(metrics["all_horizon_mae"]) + 1)
    plt.plot(horizons, metrics["all_horizon_mae"], marker="o", color="#1f77b4", linewidth=2.5, label=f"{model_name} MAE")
    plt.title(f"Forecast Error per Horizon (1h - 24h) - {model_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Forecast Horizon Ahead (Hours)", fontsize=11)
    plt.ylabel("MAE (MW)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(horizons)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_horizon_mae.png"), dpi=200)
    plt.close()

    # 2. Sample 24h Forecast with 80% Uncertainty Ribbon
    sample_idx = 100
    plt.figure(figsize=(10, 5))
    h_steps = np.arange(1, 25)
    plt.plot(h_steps, y_true_mw[sample_idx], label="Ground Truth (Actual MW)", color="black", linewidth=2.5)
    plt.plot(h_steps, y_pred_mw[sample_idx], label="Predicted (Median q50)", color="#2ca02c", linestyle="--", linewidth=2)
    plt.fill_between(h_steps, y_q10_mw[sample_idx], y_q90_mw[sample_idx], color="#2ca02c", alpha=0.25, label="80% Uncertainty Band [q10, q90]")
    plt.title(f"24-Hour Electricity Load Forecast - {model_name} (Sample #{sample_idx})", fontsize=13, fontweight="bold")
    plt.xlabel("Forecast Horizon Ahead (Hours)", fontsize=11)
    plt.ylabel("Electricity Demand (MW)", fontsize=11)
    plt.xticks(h_steps)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_sample_forecast.png"), dpi=200)
    plt.close()

    # 3. Absolute Residual Error Distribution
    plt.figure(figsize=(8, 4.5))
    errors = np.abs(y_true_mw - y_pred_mw).flatten()
    sns.histplot(errors, bins=50, kde=True, color="#d62728")
    plt.axvline(np.mean(errors), color="black", linestyle="--", label=f"Mean Error: {np.mean(errors):.1f} MW")
    plt.title(f"Absolute Error Distribution - {model_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Absolute Error (MW)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_error_distribution.png"), dpi=200)
    plt.close()
    print(f"[PLOTS] Saved evaluation figures to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-horizon forecasting models on test set")
    parser.add_argument("--baseline", action="store_true", help="Evaluate Seasonal Naive baseline")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.weights.h5/.keras/.pt)")
    parser.add_argument("--scaler_path", type=str, default="checkpoints/aep_scaler.pkl", help="Path to scaler pickle")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to processed parquets")
    parser.add_argument("--ablation_no_calendar", action="store_true", help="Ablation: Zero out future calendar")
    parser.add_argument("--save_plots", action="store_true", help="Generate and save evaluation plots")
    parser.add_argument("--save_metrics", action="store_true", help="Save metrics to experiments JSON/CSV")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu)")
    args = parser.parse_args()

    scaler = joblib.load(args.scaler_path)
    _, _, test_loader = get_dataloaders(processed_dir=args.data_dir, lookback=168, horizon=24, batch_size=128)

    if args.baseline or not args.checkpoint:
        print("\n" + "=" * 60 + "\nEVALUATING: Seasonal Naive Baseline (Weekly 168h)\n" + "=" * 60)
        metrics, plot_data = evaluate_seasonal_naive(test_loader, scaler)
        model_name = "seasonal_naive"
    else:
        model, ckpt_info = load_model_from_checkpoint(args.checkpoint)
        model_name = ckpt_info.get("model_name", "model")
        if args.ablation_no_calendar:
            model_name += "_ablation_no_cal"
        print(f"\n" + "=" * 60 + f"\nEVALUATING: {model_name.upper()} (Checkpoint: {args.checkpoint})\n" + "=" * 60)
        metrics, plot_data = evaluate_model(
            model=model, test_loader=test_loader, scaler=scaler,
            device=args.device, ablation_no_calendar=args.ablation_no_calendar, model_name=model_name
        )

    # Print summary metrics
    print("\n" + "-" * 45 + f"\nEVALUATION METRICS [{model_name}]:\n" + "-" * 45)
    for k, v in metrics.items():
        if k != "all_horizon_mae":
            print(f"{k:26s}: {v}")
    print("-" * 45 + "\n")

    if args.save_plots:
        save_evaluation_plots(metrics, plot_data, model_name)

    if args.save_metrics:
        os.makedirs("experiments", exist_ok=True)
        with open(os.path.join("experiments", f"{model_name}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        flat = {k: v for k, v in metrics.items() if k != "all_horizon_mae"}
        pd.DataFrame([flat]).to_csv(os.path.join("experiments", f"{model_name}_metrics.csv"), index=False)
        print(f"[METRICS] Saved metrics to experiments/{model_name}_metrics.json and .csv")


if __name__ == "__main__":
    main()
