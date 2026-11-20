"""
Unified Training Script for Multi-Horizon Electricity Load Forecasting in TensorFlow 2.x / Keras.
Supports multi-seed experiments (--seed) and modular YAML configurations (--config).
Logs computational costs (parameters, size in MB, training time, hardware used) for project Table 7.
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
    """Set random seed across all libraries for deterministic, reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    print(f"[INFO] Deterministic random seed set to: {seed}")


def get_hardware_info() -> str:
    """Detect and format available computing hardware."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        details = [gpu.name for gpu in gpus]
        return f"GPU: {', '.join(details)}"
    return f"CPU: {platform.processor() or platform.machine()} ({platform.system()})"


def calculate_model_size(model: tf.keras.Model) -> Dict[str, Any]:
    """Calculate parameter counts and approximate memory footprint in MB."""
    total_params = int(model.count_params())
    trainable_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    # 4 bytes per float32 parameter
    size_mb = (total_params * 4.0) / (1024.0 * 1024.0)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_mb": round(size_mb, 4)
    }


def build_model(model_cfg: Dict[str, Any]) -> tf.keras.Model:
    """Instantiate and build model according to config specification."""
    mtype = model_cfg.get("type", "").lower()

    if mtype in ["mlp", "mlp_baseline", "baseline"]:
        model = MLPBaseline(
            lookback=model_cfg.get("lookback", 168),
            past_dim=model_cfg.get("past_dim", 9),
            horizon=model_cfg.get("horizon", 24),
            future_cal_dim=model_cfg.get("future_cal_dim", 8),
            hidden_dim=model_cfg.get("hidden_dim", 256),
            dropout=model_cfg.get("dropout", 0.1)
        )
    elif mtype in ["seq2seq", "seq2seq_gru", "gru"]:
        model = Seq2SeqGRU(
            past_dim=model_cfg.get("past_dim", 9),
            future_cal_dim=model_cfg.get("future_cal_dim", 8),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.1),
            use_attention=model_cfg.get("use_attention", True)
        )
    elif mtype in ["transformer", "transformer_enc_dec"]:
        model = TimeSeriesTransformer(
            past_dim=model_cfg.get("past_dim", 9),
            future_cal_dim=model_cfg.get("future_cal_dim", 8),
            d_model=model_cfg.get("d_model", 128),
            nhead=model_cfg.get("nhead", 4),
            num_encoder_layers=model_cfg.get("num_encoder_layers", 2),
            num_decoder_layers=model_cfg.get("num_decoder_layers", 2),
            dim_feedforward=model_cfg.get("dim_feedforward", 256),
            dropout=model_cfg.get("dropout", 0.1)
        )
    elif mtype in ["tft", "tft_advanced", "temporal_fusion"]:
        model = TemporalFusionModel(
            past_dim=model_cfg.get("past_dim", 9),
            future_cal_dim=model_cfg.get("future_cal_dim", 8),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            n_heads=model_cfg.get("n_heads", 4),
            num_lstm_layers=model_cfg.get("num_lstm_layers", 1),
            dropout=model_cfg.get("dropout", 0.1)
        )
    else:
        raise ValueError(f"Unsupported model type in config: {mtype}")

    # Initialize model weights by calling with dummy batch
    dummy_past = tf.zeros((1, 168, model_cfg.get("past_dim", 9)), dtype=tf.float32)
    dummy_cal = tf.zeros((1, 24, model_cfg.get("future_cal_dim", 8)), dtype=tf.float32)
    _ = model(dummy_past, dummy_cal, training=False)
    return model


def train_model(
    config_path: str,
    seed: int = 42,
    epochs_override: int = None,
    device_override: str = None
):
    # 1. Load Configuration
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

    # 2. Setup Device & Seed
    set_seed(seed)
    if device_override:
        if "cpu" in device_override.lower():
            tf.config.set_visible_devices([], "GPU")
    hardware_name = get_hardware_info()

    # 3. Load Datasets
    train_loader, val_loader, _ = get_dataloaders(
        processed_dir=data_dir,
        lookback=lookback,
        horizon=horizon,
        batch_size=batch_size,
        seed=seed
    )

    # 4. Build Model & Criterion
    model = build_model(cfg["model"])
    stats = calculate_model_size(model)
    criterion = MultiHorizonQuantileLoss(quantiles=quantiles)

    # AdamW with gradient clipping (clipnorm=1.0)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr,
        weight_decay=weight_decay,
        clipnorm=grad_clip
    )

    sched_cfg = cfg["training"].get("scheduler", {})
    sched_factor = float(sched_cfg.get("factor", 0.5))
    sched_patience = int(sched_cfg.get("patience", 2))
    min_lr = float(sched_cfg.get("min_lr", 1e-5))

    print("=" * 70)
    print(f"TRAINING PIPELINE: {model_name.upper()} (Seed: {seed}) [TensorFlow 2.x]")
    print(f"Hardware          : {hardware_name}")
    print(f"Total Parameters  : {stats['total_params']:,}")
    print(f"Trainable Params  : {stats['trainable_params']:,}")
    print(f"Estimated Size    : {stats['size_mb']} MB")
    print(f"Quantiles         : {quantiles}")
    print("=" * 70)

    # Output directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    @tf.function
    def train_step(x_past, x_cal, y_tgt):
        with tf.GradientTape() as tape:
            y_pred = model(x_past, x_cal, training=True)
            loss_val = criterion(y_tgt, y_pred)
            loss = tf.reduce_mean(loss_val)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_step(x_past, x_cal, y_tgt):
        y_pred = model(x_past, x_cal, training=False)
        loss_val = criterion(y_tgt, y_pred)
        return tf.reduce_mean(loss_val)

    best_val_loss = float("inf")
    patience_counter = 0
    lr_patience_counter = 0
    history = []
    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Training Loop
        train_loss_sum = 0.0
        train_batches = 0
        for x_past, x_cal, y_tgt in train_loader:
            b_loss = train_step(x_past, x_cal, y_tgt)
            train_loss_sum += float(b_loss)
            train_batches += 1
        train_loss = train_loss_sum / max(1, train_batches)

        # Validation Loop
        val_loss_sum = 0.0
        val_batches = 0
        for x_past, x_cal, y_tgt in val_loader:
            b_loss = val_step(x_past, x_cal, y_tgt)
            val_loss_sum += float(b_loss)
            val_batches += 1
        val_loss = val_loss_sum / max(1, val_batches)

        elapsed = time.time() - t0
        current_lr = float(optimizer.learning_rate.numpy())

        # ReduceLROnPlateau Scheduler logic
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            lr_patience_counter = 0
            improved_flag = " [*BEST]"

            # Save best checkpoints (.keras and .weights.h5)
            keras_path = os.path.join("checkpoints", f"{model_name}_seed{seed}_best.keras")
            weights_path = os.path.join("checkpoints", f"{model_name}_seed{seed}_best.weights.h5")
            meta_path = os.path.join("checkpoints", f"{model_name}_seed{seed}_best.json")

            try:
                model.save(keras_path)
            except Exception:
                pass
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
                new_lr = max(current_lr * sched_factor, min_lr)
                optimizer.learning_rate.assign(new_lr)
                lr_patience_counter = 0

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
            "epoch_time_sec": round(elapsed, 2)
        })

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {elapsed:.1f}s{improved_flag}"
        )

        if patience_counter >= patience:
            print(f"[EARLY STOPPING] Validation loss did not improve for {patience} consecutive epochs.")
            break

    total_time = time.time() - start_total_time
    print("=" * 70)
    print(f"[FINISHED] Best Val Loss: {best_val_loss:.5f} | Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")

    # Save training history
    history_df = pd.DataFrame(history)
    history_csv = os.path.join("experiments", f"{model_name}_seed{seed}_history.csv")
    history_df.to_csv(history_csv, index=False)
    print(f"[SAVED] Training curves logged to: {history_csv}")
    print(f"[SAVED] Best checkpoints saved in checkpoints/{model_name}_seed{seed}_best.*")

    # Table 7 Computational Cost Summary
    print("\n--- TABLE 7 COMPUTATIONAL COST SUMMARY ---")
    print(f"Architecture : {model_name}")
    print(f"Seed         : {seed}")
    print(f"Parameters   : {stats['total_params']:,}")
    print(f"Model Size   : {stats['size_mb']} MB")
    print(f"Device       : {hardware_name}")
    print(f"Epochs Run   : {len(history)}")
    print(f"Total Time   : {total_time:.2f} s")
    print(f"Best Loss    : {best_val_loss:.5f}")
    print("-" * 42 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-horizon electricity load forecasting models in TensorFlow 2.x")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to model config YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu)")
    args = parser.parse_args()

    train_model(
        config_path=args.config,
        seed=args.seed,
        epochs_override=args.epochs,
        device_override=args.device
    )
