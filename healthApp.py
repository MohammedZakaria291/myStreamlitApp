# app.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import plotly.graph_objects as go
import os

# ========================= Page Config =========================
st.set_page_config(page_title="AI Predictive Maintenance", layout="wide", page_icon="robot")
st.title("AI-Powered Predictive Maintenance System")
st.markdown("### Real-time Machine Health Monitoring using Deep Learning")

# ========================= Load Model & Data =========================
@st.cache_resource
def load_model_and_scaler():
    # الملفات موجودة في نفس مجلد الـ app
    model_path = "BEST_HEALTH_MODEL.pth"
    scaler_path = "scaler_health.pkl"

    # تحميل النموذج
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
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # تحميل البيانات (ضروري لاختيار الآلات و الـ health_score)
    df = pd.read_csv("https://raw.githubusercontent.com/yourusername/predictive-maintenance/main/preprocessed_smart_data.csv")
    # لو مش عايز تحمل البيانات أونلاين، احملها في الريبو كمان واستخدم:
    # df = pd.read_csv("preprocessed_smart_data.csv")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return model, scaler, df

# تحميل كل حاجة
try:
    model, scaler, df = load_model_and_scaler()
except Exception as e:
    st.error("خطأ في تحميل النموذج أو البيانات. تأكد من رفع الملفات صح.")
    st.code(str(e))
    st.stop()

# ========================= Feature Engineering =========================
def prepare_features(group):
    window = 5
    group = group.sort_values('timestamp').copy()
    group['temp_ma'] = group['temperature'].rolling(window, min_periods=1).mean()
    group['vib_ma']  = group['vibration'].rolling(window, min_periods=1).mean()
    group['temp_roc'] = group['temperature'].diff().fillna(0)
    group['vib_roc']  = group['vibration'].diff().fillna(0)
    return group

features_cols = ['temperature','vibration','humidity','pressure','energy_consumption',
                 'temp_ma','vib_ma','temp_roc','vib_roc']

# ========================= UI =========================
st.sidebar.header("Machine Selection")
machine_ids = sorted(df['machine_id'].unique())
selected_machine = st.sidebar.selectbox("Select Machine ID", machine_ids)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"### Machine {selected_machine}")

    machine_data = df[df['machine_id'] == selected_machine].copy()
    machine_data = prepare_features(machine_data).dropna().reset_index(drop=True)

    if len(machine_data) < 20:
        st.error("Not enough data")
        st.stop()

    seq = torch.tensor(scaler.transform(machine_data[features_cols].tail(20)), 
                       dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        current_health = model(seq).item()

    recent = machine_data.tail(30)
    health_drop = recent['health_score'].iloc[0] - recent['health_score'].iloc[-1]
    days = max((recent['timestamp'].iloc[-1] - recent['timestamp'].iloc[0]).days, 1)
    rate = health_drop / days

    if current_health < 70:
        risk, color = "CRITICAL", "red"
        rul = int((current_health - 20) / max(rate, 0.1))
    elif current_health < 85:
        risk, color = "WARNING", "orange"
        rul = int((current_health - 20) / max(rate, 0.1))
    else:
        risk, color = "HEALTHY", "green"
        rul = None

    st.metric("Health Score", f"{current_health:.1f}/100", delta=f"{current_health-85:+.1f}")
    st.metric("Risk Level", risk)
    st.metric("Est. Days to Failure", f"{rul if rul and rul<365 else '>365'}" if rul else "Stable")

with col2:
    st.markdown("### Health Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=machine_data['timestamp'], y=machine_data['health_score'],
                             mode='lines+markers', name='Health Score', line=dict(width=3)))
    fig.add_trace(go.Scatter(x=[machine_data['timestamp'].iloc[-1]], y=[current_health],
                             mode='markers', marker=dict(color='red', size=14, symbol='star')))
    fig.add_hline(y=85, line_dash="dash", line_color="orange")
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.update_layout(height=550, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

st.success("Machine is healthy") if current_health >= 85 else st.warning("Plan maintenance") if current_health >= 70 else st.error("Critical – Act now!")
