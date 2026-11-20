"""
Streamlit Live Interactive Dashboard for Multi-Horizon Electricity Load Forecasting.
Enables real-time 24h point forecasting (q50) and 80% uncertainty ribbons [q10, q90].
"""

import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import tensorflow as tf

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from evaluate import load_model_from_checkpoint
from src.models import SeasonalNaiveBaseline

st.set_page_config(page_title="Multi-Horizon Load Forecasting | PJM AEP", page_icon="⚡", layout="wide")

SCALER_PATH = "checkpoints/aep_scaler.pkl"
TEST_DATA_PATH = "data/processed/test.parquet"


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None


@st.cache_data
def load_test_dataframe():
    return pd.read_parquet(TEST_DATA_PATH) if os.path.exists(TEST_DATA_PATH) else None


def main():
    st.title("⚡ Multi-Horizon Electricity Load Forecasting (AEP)")
    st.markdown("**Topic #13** | 24-Hour probabilistic electric grid demand forecasting with non-crossing quantile bounds.")

    scaler, test_df = load_scaler(), load_test_dataframe()
    if scaler is None or test_df is None:
        st.error("Missing dataset or scaler! Please run `python data/preprocess.py` first.")
        return

    # Sidebar Controls
    st.sidebar.header("🕹️ Controls & Model Selection")
    ckpts = sorted(glob.glob("checkpoints/*_best.keras") + glob.glob("checkpoints/*_best.weights.h5") + glob.glob("checkpoints/*_best.pt"))
    unique_ckpts, seen = [], set()
    for c in ckpts:
        base = c.replace(".weights.h5", "").replace(".keras", "").replace(".pt", "")
        if base not in seen:
            seen.add(base)
            unique_ckpts.append(c)

    model_choices = ["Seasonal Naive (Weekly 168h Baseline)"] + unique_ckpts
    selected_model = st.sidebar.selectbox("Choose Forecasting Model", model_choices)
    ablation_cal = st.sidebar.checkbox("Ablation: Zero-out Future Calendar Features", value=False)

    total_samples = len(test_df) - 168 - 24 + 1
    st.sidebar.markdown(f"**Available Test Windows:** {total_samples:,}")

    if "sample_idx" not in st.session_state:
        st.session_state.sample_idx = 100
    if st.sidebar.button("🎲 Pick Random Test Window"):
        st.session_state.sample_idx = int(np.random.randint(0, total_samples))

    sample_idx = st.sidebar.number_input(
        "Window Start Index", min_value=0, max_value=total_samples - 1, value=st.session_state.sample_idx, step=24
    )

    # Slice historical context (168h) and future horizon (24h)
    idx = int(sample_idx)
    raw_vals, timestamps = test_df.values, test_df.index
    past_slice = raw_vals[idx : idx + 168]
    future_slice = raw_vals[idx + 168 : idx + 168 + 24]
    past_dates = timestamps[idx : idx + 168]
    future_dates = timestamps[idx + 168 : idx + 168 + 24]

    x_past = tf.constant(past_slice[np.newaxis, :, :], dtype=tf.float32)            # (1, 168, 9)
    x_future_cal = tf.constant(future_slice[np.newaxis, :, 1:], dtype=tf.float32)    # (1, 24, 8)
    if ablation_cal:
        x_future_cal = tf.zeros_like(x_future_cal)

    # Inference
    if selected_model == "Seasonal Naive (Weekly 168h Baseline)":
        preds = SeasonalNaiveBaseline(mode="weekly").predict(x_past, x_future_cal)
        model_display = "Seasonal Naive Baseline"
    else:
        model, ckpt_info = load_model_from_checkpoint(selected_model)
        preds = model(x_past, x_future_cal, training=False)
        model_display = ckpt_info.get("model_name", "Deep Model")

    preds_np = np.squeeze(preds.numpy() if hasattr(preds, "numpy") else preds, axis=0)  # (24, 3)

    # Inverse transform to MW scale
    y_true_mw = scaler.inverse_transform(future_slice[:, 0].reshape(-1, 1)).flatten()
    past_load_mw = scaler.inverse_transform(past_slice[:, 0].reshape(-1, 1)).flatten()
    q10_mw = scaler.inverse_transform(preds_np[:, 0].reshape(-1, 1)).flatten()
    q50_mw = scaler.inverse_transform(preds_np[:, 1].reshape(-1, 1)).flatten()
    q90_mw = scaler.inverse_transform(preds_np[:, 2].reshape(-1, 1)).flatten()

    # Metrics
    abs_errors = np.abs(y_true_mw - q50_mw)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean((y_true_mw - q50_mw) ** 2)))
    mape = float(np.mean(abs_errors / (y_true_mw + 1e-8)) * 100.0)

    # Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("24h MAE", f"{mae:.2f} MW")
    col2.metric("24h RMSE", f"{rmse:.2f} MW")
    col3.metric("24h MAPE", f"{mape:.2f} %")
    col4.metric("Window Start", str(future_dates[0])[:13] + ":00")

    # Plotly Forecast Visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=past_dates, y=past_load_mw, mode="lines", name="Historical Context (Past 7 Days)", line=dict(color="#4a90e2", width=1.5)))
    fig.add_trace(go.Scatter(x=future_dates, y=y_true_mw, mode="lines+markers", name="Actual Load (Ground Truth)", line=dict(color="#111111", width=2.5)))
    fig.add_trace(go.Scatter(x=future_dates, y=q50_mw, mode="lines+markers", name=f"{model_display} (Median q50)", line=dict(color="#2ca02c", width=2.5, dash="dash")))
    fig.add_trace(go.Scatter(x=future_dates, y=q90_mw, mode="lines", line=dict(width=0), showlegend=False, name="q90 Upper"))
    fig.add_trace(go.Scatter(x=future_dates, y=q10_mw, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(44, 160, 44, 0.22)", name="80% Prediction Interval [q10, q90]"))

    fig.update_layout(
        title=f"Multi-Horizon Forecast: {future_dates[0]} to {future_dates[-1]} ({model_display})",
        xaxis_title="Timestamp", yaxis_title="Electricity Demand (MW)", hovermode="x unified",
        template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=550
    )
    st.plotly_chart(fig, width="stretch")

    # Detailed Step-by-Step Breakdown
    st.subheader("📊 Forecast Error Breakdown by Step (1h to 24h)")
    h_df = pd.DataFrame({
        "Horizon": [f"t+{h+1}" for h in range(24)],
        "Timestamp": [str(d) for d in future_dates],
        "Actual (MW)": np.round(y_true_mw, 1),
        "Pred q10 (MW)": np.round(q10_mw, 1),
        "Pred Median (MW)": np.round(q50_mw, 1),
        "Pred q90 (MW)": np.round(q90_mw, 1),
        "MAE (MW)": np.round(abs_errors, 1),
        "Relative Error (%)": np.round((abs_errors / y_true_mw) * 100.0, 2)
    })
    st.dataframe(h_df, width="stretch")


if __name__ == "__main__":
    main()
