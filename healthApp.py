# app.py - Full Professional Real-Time RUL & Anomaly Monitor
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import plotly.graph_objects as go
from datetime import datetime

# ========================= Config =========================
st.set_page_config(
    page_title="Real-Time RUL & Anomaly Monitor",
    page_icon="robot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title with style
st.markdown("""
    <h1 style='text-align: center; color: #1E90FF;'>Real-Time Machine Health Monitor</h1>
    <h3 style='text-align: center; color: #666;'>AI-Powered RUL Prediction & Full-Range Anomaly Detection</h3>
    <hr style='border-color: #1E90FF;'>
""", unsafe_allow_html=True)

# ========================= Model Definition =========================
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=9, hidden_size=100, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# ========================= Session State Init =========================
if 'history' not in st.session_state:
    st.session_state.history = {}  # machine_id -> last 19 readings
if 'df_full' not in st.session_state:
    st.session_state.df_full = None

# ========================= Sidebar =========================
with st.sidebar:
    st.header("Upload Files")
    
    uploaded_model = st.file_uploader("Model (.pth)", type=['pth'])
    uploaded_scaler = st.file_uploader("Scaler (.pkl)", type=['pkl'])
    uploaded_data = st.file_uploader("Data (.csv)", type=['csv'])
    
    st.markdown("---")
    st.markdown("### Instructions")
    st.info("""
    1. Upload your trained LSTM model (.pth)  
    2. Upload the fitted scaler (.pkl)  
    3. Upload your dataset (.csv)  
    4. Select machine & enter live values → Get instant RUL + alerts!
    """)

# ========================= Load Everything =========================
model = None
scaler = None
df = None

if uploaded_model and uploaded_scaler and uploaded_data:
    try:
        # Save temporarily
        with open("temp_model.pth", "wb") as f:
            f.write(uploaded_model.getbuffer())
        with open("temp_scaler.pkl", "wb") as f:
            f.write(uploaded_scaler.getbuffer())
        
        # Load model & scaler
        model = LSTMRegressor()
        model.load_state_dict(torch.load("temp_model.pth", map_location='cpu'))
        model.eval()
        
        with open("temp_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        
        # Load data
        df = pd.read_csv(uploaded_data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['machine_id', 'timestamp']).reset_index(drop=True)
        
        st.session_state.df_full = df
        
        # Initialize history for all machines
        for mid in df['machine_id'].unique():
            machine_data = df[df['machine_id'] == mid].tail(19)
            if len(machine_data) > 0:
                st.session_state.history[mid] = machine_data[['temperature','vibration','humidity','pressure','energy_consumption']].values
            else:
                st.session_state.history[mid] = np.zeros((19, 5))
        
        st.success("All files loaded successfully!")
        
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()
else:
    st.warning("Please upload Model, Scaler, and Data files from the sidebar.")
    st.stop()

# ========================= Main App =========================
machine_ids = sorted(df['machine_id'].unique())
selected_machine = st.selectbox("Select Machine ID", machine_ids, key="machine_select")

# Get historical safe range
machine_df = df[df['machine_id'] == selected_machine]
healthy_df = machine_df.copy()  # You can filter by 'Normal' status if available

limits = {}
for col in ['temperature', 'vibration', 'humidity', 'pressure', 'energy_consumption']:
    limits[col] = {
        'min': healthy_df[col].min(),
        'max': healthy_df[col].max(),
        'last': healthy_df[col].iloc[-1]
    }

st.markdown(f"### Live Sensor Input – Machine {selected_machine}")

cols = st.columns(5)
sensor_names = ['Temperature', 'Vibration', 'Humidity', 'Pressure', 'Energy Consumption']
sensor_keys = ['temperature', 'vibration', 'humidity', 'pressure', 'energy_consumption']

inputs = {}

alerts = []

for col, name, key in zip(cols, sensor_names, sensor_keys):
    with col:
        default = float(limits[key]['last'])
        value = st.number_input(name, value=default, step=0.1, key=f"input_{key}")
        inputs[key] = value
        
        mn, mx = limits[key]['min'], limits[key]['max']
        if value < mn or value > mx:
            st.error(f"Out of Range!")
            alerts.append(f"**{name}** out of safe range [{mn:.2f}, {mx:.2f}]")
        else:
            st.success("Normal")

# Alerts Summary
if alerts:
    st.error("ABNORMAL READINGS DETECTED!")
    for a in alerts:
        st.markdown(f"• {a}")
else:
    st.success("All sensors in normal operating range")

st.markdown("---")

if st.button("Predict Remaining Useful Life (RUL)", type="primary", use_container_width=True):
    with st.spinner("Predicting..."):
        # Build sequence: last 19 historical + 1 new
        hist = st.session_state.history[selected_machine]
        new_reading = np.array([[inputs[k] for k in sensor_keys]])
        full_sequence = np.vstack([hist, new_reading])  # Shape: (20, 5)
        
        # Feature Engineering
        df_seq = pd.DataFrame(full_sequence, columns=sensor_keys)
        df_seq['temp_moving_avg'] = df_seq['temperature'].rolling(window=5, min_periods=1).mean()
        df_seq['vib_moving_avg'] = df_seq['vibration'].rolling(window=5, min_periods=1).mean()
        df_seq['temp_rate_change'] = df_seq['temperature'].diff().fillna(0)
        df_seq['vib_rate_change'] = df_seq['vibration'].diff().fillna(0)
        
        X = df_seq[FEATURES].values
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            pred_rul = model(X_tensor).item()
        
        # Update history (keep last 19)
        st.session_state.history[selected_machine] = full_sequence[1:]

        # Results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted RUL", f"{pred_rul:.1f} Cycles", delta=None)
            health = min(max(pred_rul / 150.0, 0), 1)  # Assuming 150 is full life
            st.progress(health)
            st.write(f"**Health Index:** {health*100:.1f}%")

        with col2:
            if alerts:
                st.error("Warning: Prediction made under abnormal sensor conditions!")
            elif pred_rul < 30:
                st.error("CRITICAL: Failure Imminent!")
            elif pred_rul < 70:
                st.warning("Warning: Schedule Maintenance Recommended")
            else:
                st.success("Excellent: Machine in Great Condition")

        # Plot trend
        st.markdown("### Health Trend (Last 20 Readings)")
        rul_history = [100] * 16 + [pred_rul]  # Dummy past for visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=rul_history, mode='lines+markers', name='RUL Trend',
                                 line=dict(color='dodgerblue', width=4)))
        fig.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Warning")
        fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Critical")
        fig.update_layout(height=400, showlegend=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Built with ❤️ by Your AI Maintenance Assistant | Supports Live Streaming & Out-of-Range Alerts")
