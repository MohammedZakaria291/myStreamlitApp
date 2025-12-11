# app.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ========================= Page Config =========================
st.set_page_config(
    page_title="AI Predictive Maintenance Dashboard",
    page_icon="robot",
    layout="wide"
)

st.title("AI-Powered Predictive Maintenance System")
st.markdown("### Real-time Machine Health Monitoring using Deep Learning")

# ========================= Load Model & Data =========================
@st.cache_resource
def load_model_and_scaler():
    model_path = "/content/drive/MyDrive/BEST_HEALTH_MODEL.pth"
    scaler_path = "/content/drive/MyDrive/scaler_health.pkl"
    data_path = "/content/drive/MyDrive/preprocessed_smart_data.csv"

    # Model Architecture (must match training)
    class LSTMHealth(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(9, 128, num_layers=2, batch_first=True, dropout=0.3)
            self.fc = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.fc(h[-1]) * 100

    model = LSTMHealth()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    return model, scaler, df

model, scaler, df = load_model_and_scaler()

# ========================= Feature Engineering Function =========================
def prepare_features(group):
    window = 5
    group = group.sort_values('timestamp').copy()
    group['temp_ma'] = group['temperature'].rolling(window, min_periods=1).mean()
    group['vib_ma']  = group['vibration'].rolling(window, min_periods=1).mean()
    group['temp_roc'] = group['temperature'].diff().fillna(0)
    group['vib_roc']  = group['vibration'].diff().fillna(0)
    return group

features_cols = [
    'temperature', 'vibration', 'humidity', 'pressure', 'energy_consumption',
    'temp_ma', 'vib_ma', 'temp_roc', 'vib_roc'
]

# ========================= Sidebar & Machine Selection =========================
st.sidebar.header("Machine Selection")
machine_ids = sorted(df['machine_id'].unique())
selected_machine = st.sidebar.selectbox("Select Machine ID", machine_ids, index=0)

# ========================= Main Dashboard =========================
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"### Machine {selected_machine}")
    st.markdown("---")

    # Filter data
    machine_data = df[df['machine_id'] == selected_machine].copy()
    machine_data = prepare_features(machine_data)
    machine_data = machine_data.dropna().reset_index(drop=True)

    if len(machine_data) < 20:
        st.error("Not enough data points for this machine (need ≥20 readings)")
        st.stop()

    # Prepare latest sequence
    latest = machine_data[features_cols].tail(20).values
    latest_scaled = scaler.transform(latest)
    seq = torch.tensor(latest_scaled, dtype=torch.float32).unsqueeze(0)  # (1, 20, 9)

    # Predict current health
    with torch.no_grad():
        current_health = model(seq).item()

    # Estimate degradation rate and RUL
    recent = machine_data.tail(30)
    health_drop = recent['health_score'].iloc[0] - recent['health_score'].iloc[-1]
    days_passed = max((recent['timestamp'].iloc[-1] - recent['timestamp'].iloc[0]).days, 1)
    degradation_rate = health_drop / days_passed

    # Risk classification
    if current_health < 70:
        risk = "CRITICAL"
        color = "red"
        est_rul = int((current_health - 20) / max(degradation_rate, 0.1))
    elif current_health < 85:
        risk = "WARNING"
        color = "orange"
        est_rul = int((current_health - 20) / max(degradation_rate, 0.1))
    else:
        risk = "HEALTHY"
        color = "green"
        est_rul = None

    # Metrics
    st.metric("Current Health Score", f"{current_health:.1f}/100", delta=f"{current_health-85:+.1f}")
    st.metric("Risk Level", risk)
    if est_rul and est_rul < 365:
        st.metric("Estimated Days Until Failure", f"{est_rul}", delta="Act Now!" if est_rul < 30 else None)
    else:
        st.metric("Estimated Days Until Failure", "Stable (>1 year)")

with col2:
    st.markdown("### Health Trend Over Time")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=machine_data['timestamp'],
        y=machine_data['health_score'],
        mode='lines+markers',
        name='Health Score',
        line=dict(color='#1f77b4', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=[machine_data['timestamp'].iloc[-1]],
        y=[current_health],
        mode='markers',
        marker=dict(color='red', size=14, symbol='star', line=dict(width=2, color='white')),
        name='Latest Prediction'
    ))

    fig.add_hline(y=85, line_dash="dash", line_color="orange", annotation_text=" Warning Threshold")
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text=" Critical Threshold")

    fig.update_layout(
        title=f"Health Degradation Trend – Machine {selected_machine}",
        xaxis_title="Date",
        yaxis_title="Health Score (100 = Excellent)",
        template="plotly_white",
        height=550,
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

# ========================= Maintenance Recommendation =========================
st.markdown("### Maintenance Recommendation")
if current_health < 70:
    st.error("Immediate maintenance required! High risk of failure.")
elif current_health < 85:
    st.warning("Schedule preventive maintenance soon.")
else:
    st.success("Machine is in excellent condition. Continue regular monitoring.")

st.info("Model: LSTM trained on sensor anomaly detection | Health Score: 0–100 | R² = 0.85+ on test data")