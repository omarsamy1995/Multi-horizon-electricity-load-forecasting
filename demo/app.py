"""
Streamlit Interactive Demo for Multi-Horizon Electricity Load Forecasting.
Enables real-time 24-hour load forecasting with 10%-90% uncertainty bands.
"""

import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import torch


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models import (
    SeasonalNaiveBaseline,
    MLPBaseline,
    Seq2SeqGRU,
    TimeSeriesTransformer,
    TemporalFusionModel
)

st.set_page_config(
    page_title="Multi-Horizon Load Forecasting | PJM AEP",
    page_icon="⚡",
    layout="wide"
)

SCALER_PATH = "checkpoints/aep_scaler.pkl"
TEST_DATA_PATH = "data/processed/test.parquet"

@st.cache_resource
def load_scaler():
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    return None

@st.cache_data
def load_test_dataframe():
    if os.path.exists(TEST_DATA_PATH):
        return pd.read_parquet(TEST_DATA_PATH)
    return None

def load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]["model"]
    mtype = cfg.get("type", "").lower()

    if mtype in ["mlp", "mlp_baseline", "baseline"]:
        model = MLPBaseline(
            lookback=cfg.get("lookback", 168),
            past_dim=cfg.get("past_dim", 9),
            horizon=cfg.get("horizon", 24),
            future_cal_dim=cfg.get("future_cal_dim", 8),
            hidden_dim=cfg.get("hidden_dim", 256),
            dropout=0.0
        )
    elif mtype in ["seq2seq", "seq2seq_gru", "gru"]:
        model = Seq2SeqGRU(
            past_dim=cfg.get("past_dim", 9),
            future_cal_dim=cfg.get("future_cal_dim", 8),
            hidden_dim=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 2),
            dropout=0.0,
            use_attention=cfg.get("use_attention", True)
        )
    elif mtype in ["transformer", "transformer_enc_dec"]:
        model = TimeSeriesTransformer(
            past_dim=cfg.get("past_dim", 9),
            future_cal_dim=cfg.get("future_cal_dim", 8),
            d_model=cfg.get("d_model", 128),
            nhead=cfg.get("nhead", 4),
            num_encoder_layers=cfg.get("num_encoder_layers", 2),
            num_decoder_layers=cfg.get("num_decoder_layers", 2),
            dim_feedforward=cfg.get("dim_feedforward", 256),
            dropout=0.0
        )
    elif mtype in ["tft", "tft_advanced", "temporal_fusion"]:
        model = TemporalFusionModel(
            past_dim=cfg.get("past_dim", 9),
            future_cal_dim=cfg.get("future_cal_dim", 8),
            hidden_dim=cfg.get("hidden_dim", 128),
            n_heads=cfg.get("n_heads", 4),
            num_lstm_layers=cfg.get("num_lstm_layers", 1),
            dropout=0.0
        )
    else:
        raise ValueError(f"Unknown model type: {mtype}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt

def main():
    st.title("⚡ Multi-Horizon Electricity Load Forecasting (AEP)")
    st.markdown(
        "**Deep Learning Final Project (#13)** | Forecasting 24-hour electric grid demand in MW with probabilistic confidence intervals."
    )

    scaler = load_scaler()
    test_df = load_test_dataframe()

    if scaler is None or test_df is None:
        st.error(
            "Required dataset or scaler not found! "
            "Please run `python data/preprocess.py` to prepare the dataset and scaler."
        )
        return

    # Sidebar
    st.sidebar.header("🕹️ Controls & Model Selection")

    # Discover available checkpoints
    available_ckpts = glob.glob("checkpoints/*_best.pt")
    model_choices = ["Seasonal Naive (Weekly 168h Baseline)"] + available_ckpts

    selected_model_option = st.sidebar.selectbox("Choose Forecasting Model", model_choices)
    ablation_cal = st.sidebar.checkbox("Ablation: Zero-out Future Calendar Features", value=False)

    total_samples = len(test_df) - 168 - 24 + 1
    st.sidebar.markdown(f"**Total Available Test Windows:** {total_samples:,}")

    # Random sample button or numeric index
    if "sample_idx" not in st.session_state:
        st.session_state.sample_idx = 100

    if st.sidebar.button("🎲 Pick Random Test Window"):
        st.session_state.sample_idx = int(np.random.randint(0, total_samples))

    sample_idx = st.sidebar.number_input(
        "Window Start Index",
        min_value=0,
        max_value=total_samples - 1,
        value=st.session_state.sample_idx,
        step=24
    )

    # Slice historical context and future horizon
    idx = int(sample_idx)
    raw_vals = test_df.values
    timestamps = test_df.index

    past_slice = raw_vals[idx : idx + 168]
    future_slice = raw_vals[idx + 168 : idx + 168 + 24]
    past_dates = timestamps[idx : idx + 168]
    future_dates = timestamps[idx + 168 : idx + 168 + 24]

    x_past = torch.tensor(past_slice, dtype=torch.float32).unsqueeze(0)             # (1, 168, 9)
    x_future_cal = torch.tensor(future_slice[:, 1:], dtype=torch.float32).unsqueeze(0) # (1, 24, 8)
    y_target_scaled = future_slice[:, 0]                                             # (24,)

    if ablation_cal:
        x_future_cal = torch.zeros_like(x_future_cal)

    # Run Prediction
    device = torch.device("cpu")
    if selected_model_option == "Seasonal Naive (Weekly 168h Baseline)":
        baseline = SeasonalNaiveBaseline(mode="weekly")
        preds = baseline.predict(x_past, x_future_cal) # (1, 24, 3)
        model_display_name = "Seasonal Naive Baseline"
    else:
        model, ckpt = load_checkpoint(selected_model_option, device)
        with torch.no_grad():
            preds = model(x_past, x_future_cal)
        model_display_name = ckpt.get("model_name", "Deep Model")

    preds_np = preds.squeeze(0).numpy() # (24, 3)

    # Inverse transform to MW scale
    y_true_mw = scaler.inverse_transform(y_target_scaled.reshape(-1, 1)).flatten()
    past_load_mw = scaler.inverse_transform(past_slice[:, 0].reshape(-1, 1)).flatten()

    pred_q10_mw = scaler.inverse_transform(preds_np[:, 0].reshape(-1, 1)).flatten()
    pred_q50_mw = scaler.inverse_transform(preds_np[:, 1].reshape(-1, 1)).flatten()
    pred_q90_mw = scaler.inverse_transform(preds_np[:, 2].reshape(-1, 1)).flatten()

    # Compute errors
    abs_errors = np.abs(y_true_mw - pred_q50_mw)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean((y_true_mw - pred_q50_mw) ** 2)))
    mape = float(np.mean(abs_errors / (y_true_mw + 1e-8)) * 100.0)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("24h MAE", f"{mae:.2f} MW")
    col2.metric("24h RMSE", f"{rmse:.2f} MW")
    col3.metric("24h MAPE", f"{mape:.2f} %")
    col4.metric("Window Start", str(future_dates[0])[:13] + ":00")

    # Plotly Forecast Visualization
    fig = go.Figure()

    # Past 7 days history (168 hours)
    fig.add_trace(go.Scatter(
        x=past_dates,
        y=past_load_mw,
        mode="lines",
        name="Historical Context (Past 7 Days)",
        line=dict(color="#4a90e2", width=1.5)
    ))

    # Ground Truth Future 24 hours
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=y_true_mw,
        mode="lines+markers",
        name="Actual Load (Ground Truth)",
        line=dict(color="#111111", width=2.5)
    ))

    # Predicted Median (q50)
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=pred_q50_mw,
        mode="lines+markers",
        name=f"{model_display_name} (Median q50)",
        line=dict(color="#2ca02c", width=2.5, dash="dash")
    ))

    # 10%-90% Uncertainty Band
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=pred_q90_mw,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="q90 Upper Bound"
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=pred_q10_mw,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(44, 160, 44, 0.22)",
        name="80% Confidence Interval [q10, q90]"
    ))

    fig.update_layout(
        title=f"Multi-Horizon Forecast: {future_dates[0]} to {future_dates[-1]} ({model_display_name})",
        xaxis_title="Timestamp",
        yaxis_title="Electricity Load (MW)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=550
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed Horizon Breakdown
    st.subheader("📊 Forecast Error Breakdown by Step (1h to 24h)")
    h_df = pd.DataFrame({
        "Horizon (Hour Ahead)": [f"t+{h+1}" for h in range(24)],
        "Timestamp": [str(d) for d in future_dates],
        "Actual (MW)": np.round(y_true_mw, 1),
        "Pred q10 (MW)": np.round(pred_q10_mw, 1),
        "Pred Median (MW)": np.round(pred_q50_mw, 1),
        "Pred q90 (MW)": np.round(pred_q90_mw, 1),
        "Absolute Error (MW)": np.round(abs_errors, 1),
        "Relative Error (%)": np.round((abs_errors / y_true_mw) * 100.0, 2)
    })

    st.dataframe(h_df, use_container_width=True)

if __name__ == "__main__":
    main()
