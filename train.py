"""
Unified Training Script for Multi-Horizon Electricity Load Forecasting in TensorFlow 2.x / Keras.
Supports modular YAML configs, multi-seed training (--seed), and computational cost logging for Table 7.
"""

import os
import sys
import time
import random
import argparse
import platform
import json
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.datasets.aep_dataset import get_dataloaders
from src.models import (
    MultiHorizonQuantileLoss,
    MLPBaseline,
    Seq2SeqGRU,
    TimeSeriesTransformer,
    TemporalFusionModel
)


def set_seed(seed: int):
    """Sets random seeds across all libraries for deterministic, reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    print(f"[INFO] Deterministic seed set to: {seed}")


def get_hardware_info() -> str:
    """Detects and returns active GPU or CPU hardware identifier."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return f"GPU: {', '.join([gpu.name for gpu in gpus])}"
    return f"CPU: {platform.processor() or platform.machine()} ({platform.system()})"


def calculate_model_size(model: tf.keras.Model) -> Dict[str, Any]:
    """Calculates total trainable parameters and model size in MB (4 bytes / float32)."""
    total_params = int(model.count_params())
    trainable_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    size_mb = (total_params * 4.0) / (1024.0 * 1024.0)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_mb": round(size_mb, 4)
    }


def build_model(model_cfg: Dict[str, Any]) -> tf.keras.Model:
    """Instantiates model architecture and builds weights via a dummy forward pass."""
    mtype = model_cfg.get("type", "").lower()
    p_dim = model_cfg.get("past_dim", 9)
    f_dim = model_cfg.get("future_cal_dim", 8)
    h_dim = model_cfg.get("hidden_dim", 128)
    drop = model_cfg.get("dropout", 0.1)

    if mtype in ["mlp", "mlp_baseline", "baseline"]:
        model = MLPBaseline(
            lookback=model_cfg.get("lookback", 168),
            past_dim=p_dim,
            horizon=model_cfg.get("horizon", 24),
            future_cal_dim=f_dim,
            hidden_dim=model_cfg.get("hidden_dim", 256),
            dropout=drop
        )
    elif mtype in ["seq2seq", "seq2seq_gru", "gru"]:
        model = Seq2SeqGRU(
            past_dim=p_dim,
            future_cal_dim=f_dim,
            hidden_dim=h_dim,
            num_layers=model_cfg.get("num_layers", 2),
            dropout=drop,
            use_attention=model_cfg.get("use_attention", True)
        )
    elif mtype in ["transformer", "transformer_enc_dec"]:
        model = TimeSeriesTransformer(
            past_dim=p_dim,
            future_cal_dim=f_dim,
            d_model=model_cfg.get("d_model", 128),
            nhead=model_cfg.get("nhead", 4),
            num_encoder_layers=model_cfg.get("num_encoder_layers", 2),
            num_decoder_layers=model_cfg.get("num_decoder_layers", 2),
            dim_feedforward=model_cfg.get("dim_feedforward", 256),
            dropout=drop
        )
    elif mtype in ["tft", "tft_advanced", "temporal_fusion"]:
        model = TemporalFusionModel(
            past_dim=p_dim,
            future_cal_dim=f_dim,
            hidden_dim=h_dim,
            n_heads=model_cfg.get("n_heads", 4),
            num_lstm_layers=model_cfg.get("num_lstm_layers", 1),
            dropout=drop
        )
    else:
        raise ValueError(f"Unsupported model type in config: {mtype}")

    # Build weights via dummy forward pass: x_past (1, 168, 9), x_future_cal (1, 24, 8)
    dummy_past = tf.zeros((1, 168, p_dim), dtype=tf.float32)
    dummy_cal = tf.zeros((1, 24, f_dim), dtype=tf.float32)
    _ = model(dummy_past, dummy_cal, training=False)
    return model


def train_model(
    config_path: str,
    seed: int = 42,
    epochs_override: int = None,
    device_override: str = None
):
    """Executes full training loop with AdamW, Pinball Loss, LR scheduling, and checkpointing."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    model_type = cfg["model"]["type"]
    batch_size = cfg["training"].get("batch_size", 64)
    epochs = epochs_override or cfg["training"].get("epochs", 15)
    lr = float(cfg["training"].get("learning_rate", 1e-3))
    weight_decay = float(cfg["training"].get("weight_decay", 1e-4))
    patience = cfg["training"].get("patience", 5)
    grad_clip = float(cfg["training"].get("gradient_clip_val", 1.0))
    quantiles = cfg["training"].get("quantiles", [0.1, 0.5, 0.9])
    data_dir = cfg["data"].get("processed_dir", "data/processed")
    lookback = cfg["data"].get("lookback", 168)
    horizon = cfg["data"].get("horizon", 24)

    set_seed(seed)
    if device_override and "cpu" in device_override.lower():
        tf.config.set_visible_devices([], "GPU")
    hardware_name = get_hardware_info()

    # Load data pipelines
    train_loader, val_loader, _ = get_dataloaders(
        processed_dir=data_dir, lookback=lookback, horizon=horizon, batch_size=batch_size, seed=seed
    )

    # Initialize model, loss, and AdamW optimizer with gradient clipping
    model = build_model(cfg["model"])
    stats = calculate_model_size(model)
    criterion = MultiHorizonQuantileLoss(quantiles=quantiles)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay, clipnorm=grad_clip)

    sched_cfg = cfg["training"].get("scheduler", {})
    sched_factor = float(sched_cfg.get("factor", 0.5))
    sched_patience = int(sched_cfg.get("patience", 2))
    min_lr = float(sched_cfg.get("min_lr", 1e-5))

    print("=" * 70)
    print(f"TRAINING PIPELINE: {model_name.upper()} (Seed: {seed}) [TensorFlow 2.x]")
    print(f"Hardware: {hardware_name} | Total Params: {stats['total_params']:,} ({stats['size_mb']} MB)")
    print("=" * 70)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    @tf.function
    def train_step(x_past, x_cal, y_tgt):
        with tf.GradientTape() as tape:
            y_pred = model(x_past, x_cal, training=True)
            loss = tf.reduce_mean(criterion(y_tgt, y_pred))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_step(x_past, x_cal, y_tgt):
        y_pred = model(x_past, x_cal, training=False)
        return tf.reduce_mean(criterion(y_tgt, y_pred))

    best_val_loss = float("inf")
    patience_counter = 0
    lr_patience_counter = 0
    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = float(np.mean([float(train_step(xp, xc, yt)) for xp, xc, yt in train_loader]))
        val_loss = float(np.mean([float(val_step(xp, xc, yt)) for xp, xc, yt in val_loader]))
        elapsed = time.time() - t0
        current_lr = float(optimizer.learning_rate.numpy())

        # Checkpoint if validation loss improves
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            lr_patience_counter = 0
            improved_flag = " [*BEST]"

            # Save checkpoint weights and metadata
            weights_path = os.path.join("checkpoints", f"{model_name}_seed{seed}_best.weights.h5")
            meta_path = os.path.join("checkpoints", f"{model_name}_seed{seed}_best.json")
            model.save_weights(weights_path)
            with open(meta_path, "w", encoding="utf-8") as fp:
                json.dump({
                    "epoch": epoch,
                    "model_name": model_name,
                    "model_type": model_type,
                    "val_loss": val_loss,
                    "config": cfg,
                    "seed": seed,
                    "quantiles": quantiles,
                    "stats": stats,
                    "hardware": hardware_name
                }, fp, indent=2)
        else:
            patience_counter += 1
            lr_patience_counter += 1
            improved_flag = ""
            if lr_patience_counter >= sched_patience:
                optimizer.learning_rate.assign(max(current_lr * sched_factor, min_lr))
                lr_patience_counter = 0

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "learning_rate": current_lr, "epoch_time_sec": round(elapsed, 2)
        })
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train: {train_loss:.5f} | Val: {val_loss:.5f} | LR: {current_lr:.6f} | Time: {elapsed:.1f}s{improved_flag}")

        if patience_counter >= patience:
            print(f"[EARLY STOPPING] Validation loss stalled for {patience} consecutive epochs.")
            break

    total_time = time.time() - start_time
    history_csv = os.path.join("experiments", f"{model_name}_seed{seed}_history.csv")
    pd.DataFrame(history).to_csv(history_csv, index=False)

    print("=" * 70)
    print(f"[FINISHED] Best Val Loss: {best_val_loss:.5f} | Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"[SAVED] History: {history_csv} | Weights: checkpoints/{model_name}_seed{seed}_best.weights.h5")
    print(f"\n--- TABLE 7 SUMMARY ---")
    print(f"Model: {model_name} | Params: {stats['total_params']:,} | Size: {stats['size_mb']} MB | Time: {total_time:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-horizon electricity load forecasting models in TensorFlow 2.x")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu)")
    args = parser.parse_args()

    train_model(config_path=args.config, seed=args.seed, epochs_override=args.epochs, device_override=args.device)
