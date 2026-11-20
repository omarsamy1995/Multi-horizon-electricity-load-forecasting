"""
Unified Evaluation and Benchmarking Script for Multi-Horizon Electricity Load Forecasting in TensorFlow 2.x.

Supports:
- Seasonal Naive baseline evaluation (--baseline)
- Deep Learning checkpoint evaluation (--checkpoint <path>)
- Calendar ablation study (--ablation_no_calendar)
- Per-horizon error metrics (1h, 6h, 12h, 24h)
- Exact latency profiling (ms per sample)
- Publication-quality error & forecast plot generation (--save_plots)
- Metrics export to CSV/JSON (--save_metrics)
"""

import os
import sys
import time
import json
import argparse
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
    TemporalFusionModel,
    MultiHorizonQuantileLoss,
    NonCrossingQuantileHead,
    GatedResidualNetwork,
    GatedLinearUnit,
    InterpretableMultiHeadAttention
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_median: np.ndarray,
    y_q10: np.ndarray = None,
    y_q90: np.ndarray = None
) -> dict:
    """
    Compute domain evaluation metrics on unscaled physical MW scale.
    
    Args:
        y_true: (N, 24) Ground truth electricity demand in MW
        y_pred_median: (N, 24) Point / median forecast in MW
        y_q10: (N, 24) Optional 10th percentile forecast
        y_q90: (N, 24) Optional 90th percentile forecast
    """
    diff = y_true - y_pred_median
    abs_diff = np.abs(diff)

    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mape = float(np.mean(abs_diff / (np.abs(y_true) + 1e-8)) * 100.0)

    # Per-horizon MAE
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
        coverage_pct = float(np.mean(in_band) * 100.0)
        metrics["Coverage_80_Interval (%)"] = round(coverage_pct, 2)

    return metrics


def evaluate_seasonal_naive(test_loader, scaler) -> dict:
    """Evaluate non-parametric weekly seasonal naive baseline."""
    baseline = SeasonalNaiveBaseline(mode="weekly")
    all_preds = []
    all_targets = []

    t0 = time.time()
    for x_past, _, y_target in test_loader:
        preds = baseline.predict(x_past)  # (B, 24, 3)
        all_preds.append(preds.numpy() if hasattr(preds, "numpy") else preds)
        all_targets.append(y_target.numpy() if hasattr(y_target, "numpy") else y_target)
    elapsed = time.time() - t0

    preds_arr = np.concatenate(all_preds, axis=0)     # (N, 24, 3)
    targets_arr = np.concatenate(all_targets, axis=0) # (N, 24)

    # Unscale using scaler
    N, H, _ = preds_arr.shape
    y_true_mw = scaler.inverse_transform(targets_arr.reshape(-1, 1)).reshape(N, H)
    y_pred_mw = scaler.inverse_transform(preds_arr[:, :, 1].reshape(-1, 1)).reshape(N, H)
    y_q10_mw = scaler.inverse_transform(preds_arr[:, :, 0].reshape(-1, 1)).reshape(N, H)
    y_q90_mw = scaler.inverse_transform(preds_arr[:, :, 2].reshape(-1, 1)).reshape(N, H)

    metrics = compute_metrics(y_true_mw, y_pred_mw, y_q10_mw, y_q90_mw)
    latency_ms_per_sample = (elapsed / N) * 1000.0
    metrics["Latency (ms/sample)"] = round(latency_ms_per_sample, 4)
    metrics["Model"] = "Seasonal Naive (Weekly)"

    return metrics, (y_true_mw, y_pred_mw, y_q10_mw, y_q90_mw)


def instantiate_model_from_type(mtype: str, cfg: dict) -> tf.keras.Model:
    mtype = mtype.lower()
    if mtype in ["mlp", "mlp_baseline", "baseline"]:
        return MLPBaseline(
            lookback=cfg.get("lookback", 168),
            past_dim=cfg.get("past_dim", 9),
            horizon=cfg.get("horizon", 24),
            future_cal_dim=cfg.get("future_cal_dim", 8),
            hidden_dim=cfg.get("hidden_dim", 256),
            dropout=cfg.get("dropout", 0.0)
        )
    elif mtype in ["seq2seq", "seq2seq_gru", "gru"]:
        return Seq2SeqGRU(
            past_dim=cfg.get("past_dim", 9),
            future_cal_dim=cfg.get("future_cal_dim", 8),
            hidden_dim=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.0),
            use_attention=cfg.get("use_attention", True)
        )
    elif mtype in ["transformer", "transformer_enc_dec"]:
        return TimeSeriesTransformer(
            past_dim=cfg.get("past_dim", 9),
            future_cal_dim=cfg.get("future_cal_dim", 8),
            d_model=cfg.get("d_model", 128),
            nhead=cfg.get("nhead", 4),
            num_encoder_layers=cfg.get("num_encoder_layers", 2),
            num_decoder_layers=cfg.get("num_decoder_layers", 2),
            dim_feedforward=cfg.get("dim_feedforward", 256),
            dropout=cfg.get("dropout", 0.0)
        )
    elif mtype in ["tft", "tft_advanced", "temporal_fusion"]:
        return TemporalFusionModel(
            past_dim=cfg.get("past_dim", 9),
            future_cal_dim=cfg.get("future_cal_dim", 8),
            hidden_dim=cfg.get("hidden_dim", 128),
            n_heads=cfg.get("n_heads", 4),
            num_lstm_layers=cfg.get("num_lstm_layers", 1),
            dropout=cfg.get("dropout", 0.0)
        )
    else:
        raise ValueError(f"Unknown model type: {mtype}")


def load_model_from_checkpoint(checkpoint_path: str):
    """Load architecture and weights from saved TensorFlow checkpoint (.keras or .weights.h5 or json)."""
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

    if not model_name or not cfg:
        fname = os.path.basename(checkpoint_path).lower()
        if "mlp" in fname:
            model_name = "mlp_baseline"
            config_file = "configs/baseline.yaml"
        elif "seq2seq" in fname or "gru" in fname:
            model_name = "seq2seq_gru"
            config_file = "configs/seq2seq_gru.yaml"
        elif "tft" in fname:
            model_name = "tft_advanced"
            config_file = "configs/tft_advanced.yaml"
        elif "transformer" in fname:
            model_name = "transformer"
            config_file = "configs/transformer.yaml"
        else:
            raise ValueError(f"Cannot deduce model type for {checkpoint_path}")

        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)["model"]

    mtype = cfg.get("type", model_name)
    model = instantiate_model_from_type(mtype, cfg)
    dummy_past = tf.zeros((1, 168, cfg.get("past_dim", 9)), dtype=tf.float32)
    dummy_cal = tf.zeros((1, 24, cfg.get("future_cal_dim", 8)), dtype=tf.float32)
    _ = model(dummy_past, dummy_cal, training=False)

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
) -> dict:
    """Evaluate a TensorFlow model with optional future calendar ablation."""
    all_preds = []
    all_targets = []

    # Warmup for latency profiling
    dummy_x = tf.zeros((1, 168, 9), dtype=tf.float32)
    dummy_cal = tf.zeros((1, 24, 8), dtype=tf.float32)
    for _ in range(5):
        _ = model(dummy_x, dummy_cal, training=False)

    total_infer_time = 0.0
    total_samples = 0

    for x_past, x_future_cal, y_target in test_loader:
        if ablation_no_calendar:
            x_future_cal = tf.zeros_like(x_future_cal)

        t_start = time.time()
        y_pred = model(x_past, x_future_cal, training=False)
        total_infer_time += (time.time() - t_start)
        total_samples += int(tf.shape(x_past)[0])

        all_preds.append(y_pred.numpy() if hasattr(y_pred, "numpy") else y_pred)
        all_targets.append(y_target.numpy() if hasattr(y_target, "numpy") else y_target)

    preds_arr = np.concatenate(all_preds, axis=0)     # (N, 24, 3)
    targets_arr = np.concatenate(all_targets, axis=0) # (N, 24)

    # Inverse transform
    N, H, _ = preds_arr.shape
    y_true_mw = scaler.inverse_transform(targets_arr.reshape(-1, 1)).reshape(N, H)
    y_pred_mw = scaler.inverse_transform(preds_arr[:, :, 1].reshape(-1, 1)).reshape(N, H)
    y_q10_mw = scaler.inverse_transform(preds_arr[:, :, 0].reshape(-1, 1)).reshape(N, H)
    y_q90_mw = scaler.inverse_transform(preds_arr[:, :, 2].reshape(-1, 1)).reshape(N, H)

    metrics = compute_metrics(y_true_mw, y_pred_mw, y_q10_mw, y_q90_mw)
    latency_ms = (total_infer_time / max(1, total_samples)) * 1000.0
    metrics["Latency (ms/sample)"] = round(latency_ms, 4)
    metrics["Model"] = model_name
    metrics["Ablation_No_Calendar"] = ablation_no_calendar

    return metrics, (y_true_mw, y_pred_mw, y_q10_mw, y_q90_mw)


def save_evaluation_plots(metrics: dict, plot_data: tuple, model_name: str, out_dir: str = "experiments"):
    """Generate and save publication-quality evaluation figures."""
    os.makedirs(out_dir, exist_ok=True)
    y_true_mw, y_pred_mw, y_q10_mw, y_q90_mw = plot_data

    # 1. Horizon MAE Curve
    plt.figure(figsize=(9, 5))
    horizons = np.arange(1, len(metrics["all_horizon_mae"]) + 1)
    plt.plot(horizons, metrics["all_horizon_mae"], marker="o", color="#1f77b4", linewidth=2.5, label=f"{model_name} MAE")
    plt.title(f"Forecast Error per Horizon (1h - 24h) - {model_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Forecast Horizon (Hours Ahead)", fontsize=11)
    plt.ylabel("MAE (MW)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(horizons)
    plt.legend()
    plt.tight_layout()
    plot_path_horizon = os.path.join(out_dir, f"{model_name}_horizon_mae.png")
    plt.savefig(plot_path_horizon, dpi=200)
    plt.close()
    print(f"[PLOT] Saved horizon curve to: {plot_path_horizon}")

    # 2. Sample 24-hour Multi-Horizon Forecast with Uncertainty Ribbon
    sample_idx = 100
    plt.figure(figsize=(10, 5))
    h_steps = np.arange(1, 25)
    plt.plot(h_steps, y_true_mw[sample_idx], label="Ground Truth (Actual MW)", color="black", linewidth=2.5)
    plt.plot(h_steps, y_pred_mw[sample_idx], label="Predicted (Median q50)", color="#2ca02c", linestyle="--", linewidth=2)
    plt.fill_between(
        h_steps,
        y_q10_mw[sample_idx],
        y_q90_mw[sample_idx],
        color="#2ca02c",
        alpha=0.25,
        label="80% Uncertainty Band [q10, q90]"
    )
    plt.title(f"24-Hour Electricity Load Forecast - {model_name} (Sample #{sample_idx})", fontsize=13, fontweight="bold")
    plt.xlabel("Forecast Horizon (Hours Ahead)", fontsize=11)
    plt.ylabel("Electricity Demand (MW)", fontsize=11)
    plt.xticks(h_steps)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plot_path_sample = os.path.join(out_dir, f"{model_name}_sample_forecast.png")
    plt.savefig(plot_path_sample, dpi=200)
    plt.close()
    print(f"[PLOT] Saved sample forecast to: {plot_path_sample}")

    # 3. Absolute Error Distribution
    plt.figure(figsize=(8, 4.5))
    errors = np.abs(y_true_mw - y_pred_mw).flatten()
    sns.histplot(errors, bins=50, kde=True, color="#d62728")
    plt.axvline(np.mean(errors), color="black", linestyle="--", label=f"Mean Error: {np.mean(errors):.1f} MW")
    plt.title(f"Absolute Error Distribution - {model_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Absolute Error (MW)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plot_path_hist = os.path.join(out_dir, f"{model_name}_error_distribution.png")
    plt.savefig(plot_path_hist, dpi=200)
    plt.close()
    print(f"[PLOT] Saved error distribution to: {plot_path_hist}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-horizon forecasting models on test set")
    parser.add_argument("--baseline", action="store_true", help="Evaluate Seasonal Naive baseline")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.keras/.weights.h5)")
    parser.add_argument("--scaler_path", type=str, default="checkpoints/aep_scaler.pkl", help="Path to scaler pickle")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to processed parquets")
    parser.add_argument("--ablation_no_calendar", action="store_true", help="Ablation: Zero-out future calendar inputs")
    parser.add_argument("--save_plots", action="store_true", help="Generate and save evaluation plots")
    parser.add_argument("--save_metrics", action="store_true", help="Save metrics to experiments JSON/CSV")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu)")
    args = parser.parse_args()

    if not os.path.exists(args.scaler_path):
        raise FileNotFoundError(f"Scaler not found at {args.scaler_path}. Run 'python data/preprocess.py' first.")

    scaler = joblib.load(args.scaler_path)

    _, _, test_loader = get_dataloaders(processed_dir=args.data_dir, lookback=168, horizon=24, batch_size=128)

    if args.baseline:
        print("\n" + "=" * 60)
        print("EVALUATING: Seasonal Naive Baseline (Weekly 168h)")
        print("=" * 60)
        metrics, plot_data = evaluate_seasonal_naive(test_loader, scaler)
        model_name = "seasonal_naive"
    elif args.checkpoint:
        model, ckpt_info = load_model_from_checkpoint(args.checkpoint)
        model_name = ckpt_info.get("model_name", "model")
        print("\n" + "=" * 60)
        print(f"EVALUATING: {model_name.upper()} (Checkpoint: {args.checkpoint})")
        if args.ablation_no_calendar:
            print("[ABLATION MODE] Future calendar features are zeroed out!")
            model_name += "_ablation_no_cal"
        print("=" * 60)
        metrics, plot_data = evaluate_model(
            model=model,
            test_loader=test_loader,
            scaler=scaler,
            device=args.device,
            ablation_no_calendar=args.ablation_no_calendar,
            model_name=model_name
        )
    else:
        print("No evaluation mode specified. Defaulting to Seasonal Naive baseline (--baseline).")
        metrics, plot_data = evaluate_seasonal_naive(test_loader, scaler)
        model_name = "seasonal_naive"

    # Print Formatted Results
    print("\n" + "-" * 45)
    print(f"EVALUATION METRICS [{model_name}]:")
    print("-" * 45)
    for k, v in metrics.items():
        if k != "all_horizon_mae":
            print(f"{k:26s}: {v}")
    print("-" * 45 + "\n")

    if args.save_plots:
        save_evaluation_plots(metrics, plot_data, model_name)

    if args.save_metrics:
        os.makedirs("experiments", exist_ok=True)
        # JSON
        json_path = os.path.join("experiments", f"{model_name}_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        # CSV
        flat_metrics = {k: v for k, v in metrics.items() if k != "all_horizon_mae"}
        csv_path = os.path.join("experiments", f"{model_name}_metrics.csv")
        pd.DataFrame([flat_metrics]).to_csv(csv_path, index=False)
        print(f"[METRICS] Saved metrics to {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
