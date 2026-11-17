"""
Unified Training Script for Multi-Horizon Electricity Load Forecasting.
Supports multi-seed experiments (--seed) and modular YAML configurations (--config).
Logs computational costs (parameters, size in MB, training time, hardware used) for project Table 7.
"""

import os
import sys
import time
import random
import argparse
import platform
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[INFO] Deterministic random seed set to: {seed}")

def get_hardware_info(device: torch.device) -> str:
    """Detect and format available computing hardware."""
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return f"GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)"
    return f"CPU: {platform.processor() or platform.machine()} ({platform.system()})"

def calculate_model_size(model: nn.Module) -> Dict[str, Any]:
    """Calculate parameter counts and approximate memory footprint in MB."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_bytes + buffer_bytes) / (1024.0 * 1024.0)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_mb": round(size_mb, 4)
    }

def build_model(model_cfg: Dict[str, Any]) -> nn.Module:
    """Instantiate model according to config specification."""
    mtype = model_cfg.get("type", "").lower()
    
    if mtype in ["mlp", "mlp_baseline", "baseline"]:
        return MLPBaseline(
            lookback=model_cfg.get("lookback", 168),
            past_dim=model_cfg.get("past_dim", 9),
            horizon=model_cfg.get("horizon", 24),
            future_cal_dim=model_cfg.get("future_cal_dim", 8),
            hidden_dim=model_cfg.get("hidden_dim", 256),
            dropout=model_cfg.get("dropout", 0.1)
        )
    elif mtype in ["seq2seq", "seq2seq_gru", "gru"]:
        return Seq2SeqGRU(
            past_dim=model_cfg.get("past_dim", 9),
            future_cal_dim=model_cfg.get("future_cal_dim", 8),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.1),
            use_attention=model_cfg.get("use_attention", True)
        )
    elif mtype in ["transformer", "transformer_enc_dec"]:
        return TimeSeriesTransformer(
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
        return TemporalFusionModel(
            past_dim=model_cfg.get("past_dim", 9),
            future_cal_dim=model_cfg.get("future_cal_dim", 8),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            n_heads=model_cfg.get("n_heads", 4),
            num_lstm_layers=model_cfg.get("num_lstm_layers", 1),
            dropout=model_cfg.get("dropout", 0.1)
        )
    else:
        raise ValueError(f"Unsupported model type in config: {mtype}")

def train_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0
) -> float:
    model.train()
    total_loss = 0.0

    for x_past, x_future_cal, y_target in dataloader:
        x_past = x_past.to(device)
        x_future_cal = x_future_cal.to(device)
        y_target = y_target.to(device)

        optimizer.zero_grad()
        y_pred = model(x_past, x_future_cal)
        loss = criterion(y_pred, y_target)

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * x_past.size(0)

    return total_loss / len(dataloader.dataset)

@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    model.eval()
    total_loss = 0.0

    for x_past, x_future_cal, y_target in dataloader:
        x_past = x_past.to(device)
        x_future_cal = x_future_cal.to(device)
        y_target = y_target.to(device)

        y_pred = model(x_past, x_future_cal)
        loss = criterion(y_pred, y_target)
        total_loss += loss.item() * x_past.size(0)

    return total_loss / len(dataloader.dataset)

def train_model(config_path: str, seed: int = 42, epochs_override: int = None, device_override: str = None):
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
    num_workers = cfg["data"].get("num_workers", 0)

    # 2. Setup Device & Seed
    set_seed(seed)
    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hardware_name = get_hardware_info(device)

    # 3. Load DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        processed_dir=data_dir,
        lookback=lookback,
        horizon=horizon,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # 4. Build Model & Criterion
    model = build_model(cfg["model"]).to(device)
    stats = calculate_model_size(model)
    criterion = MultiHorizonQuantileLoss(quantiles=quantiles).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    sched_cfg = cfg["training"].get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=sched_cfg.get("factor", 0.5),
        patience=sched_cfg.get("patience", 2),
        min_lr=float(sched_cfg.get("min_lr", 1e-5))
    )

    print("=" * 70)
    print(f"TRAINING PIPELINE: {model_name.upper()} (Seed: {seed})")
    print(f"Hardware          : {hardware_name}")
    print(f"Total Parameters  : {stats['total_params']:,}")
    print(f"Trainable Params  : {stats['trainable_params']:,}")
    print(f"Estimated Size    : {stats['size_mb']} MB")
    print(f"Quantiles         : {quantiles}")
    print(f"Training Batches  : {len(train_loader)} | Val Batches: {len(val_loader)}")
    print("=" * 70)

    # Output directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []
    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, grad_clip)
        val_loss = evaluate_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
            "epoch_time_sec": round(elapsed, 2)
        })

        improved_flag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            improved_flag = " [*BEST]"
            
            # Save best checkpoint
            ckpt_filename = f"{model_name}_seed{seed}_best.pt"
            ckpt_path = os.path.join("checkpoints", ckpt_filename)
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_type": model_type,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": cfg,
                "seed": seed,
                "quantiles": quantiles,
                "stats": stats,
                "hardware": hardware_name
            }, ckpt_path)
        else:
            patience_counter += 1

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
    print(f"[SAVED] Best checkpoint saved to: checkpoints/{model_name}_seed{seed}_best.pt")

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
    parser = argparse.ArgumentParser(description="Train multi-horizon electricity load forecasting models")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to model config YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--device", type=str, default=None, help="Device override (cuda/cpu)")
    args = parser.parse_args()

    train_model(
        config_path=args.config,
        seed=args.seed,
        epochs_override=args.epochs,
        device_override=args.device
    )
