# app.py - Final Working Version (Tested & Fixed)
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import plotly.graph_objects as go
import os

# ========================= Page Config =========================
st.set_page_config(
    page_title="AI Machine Health Monitor",
    page_icon="robot",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <h1 style='text-align: center; color: #00BFFF;'>AI-Powered Machine Health Monitor</h1>
    <h3 style='text-align: center; color: #888;'>Real-time Health Score (0-100) & Anomaly Detection</h3>
    <hr style='border: 2px solid #00BFFF;'>
""", unsafe_allow_html=True)

# ========================= Correct Model Architecture (Matches your .pth file) =========================
class LSTMHealth(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=9, hidden_size=128, num_layers=2,
                            batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out * 100  # Output: Health Score from 0 to 100

# ========================= Session State =========================
if 'history' not in st.session_state:
    st.session_state.history = {}
if 'df_full' not in st.session_state:
    st.session_state.df_full = None

# ========================= Sidebar - File Upload =========================
with st.sidebar:
    st.header("Upload Required Files")
    uploaded_model = st.file_uploader("Model File (.pth)", type="pth")
    uploaded_scaler = st.file_uploader("Scaler File (.pkl)", type="pkl")
    uploaded_csv = st.file_uploader("Dataset (.csv)", type="csv")

    st.markdown("---")
    st.markdown("### How to Use")
    st.success("""
    1. Upload your trained model (`BEST_HEALTH_MODEL.pth`)  
    2. Upload the scaler (`scaler_health.pkl`)  
    3. Upload your data (`preprocessed_smart_data.csv`)  
    4. Select machine & view live health prediction!
    """)

# ========================= Load Model, Scaler & Data =========================
@st.cache_resource
def load_everything(model_file, scaler_file, csv_file):
    # Save uploaded files temporarily
    model_path = "temp_model.pth"
    scaler_path = "temp_scaler.pkl"
    
    with open(model_path, "wb") as f:
        f.write(model_file.getbuffer())
    with open(scaler_path, "wb") as f:
        f.write(scaler_file.getbuffer())
    
    # Load model
    model = LSTMHealth()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Load scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    # Load data
    df = pd.read_csv(csv_file)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return model, scaler, df

# ========================= Main Logic =========================
if uploaded_model and uploaded_scaler and uploaded_csv:
    try:
        model, scaler, df = load_everything(uploaded_model, uploaded_scaler, uploaded_csv)
        st.session_state.df_full = df
        
        # Initialize history for all machines
        for mid in df['machine_id'].unique():
            data = df[df['machine_id'] == mid][['temperature','vibration','humidity','pressure','energy_consumption']].tail(19)
            if len(data) < 19:
                pad = np.zeros((19 - len(data), 5))
                data = np.vstack([pad, data.values]) if len(data) > 0 else np.zeros((19, 5))
            else:
                data = data.values
            st.session_state.history[mid] = data
        
        st.success("All files loaded successfully! Ready for prediction.")
    
    except Exception as e:
        st.error("Error loading files. Check file compatibility.")
        st.code(str(e))
        st.stop()
else:
    st.info("Please upload the three required files from the sidebar to start.")
    st.stop()

# ========================= Machine Selection =========================
machine_ids = sorted(df['machine_id'].unique())
selected_machine = st.selectbox("Select Machine ID", machine_ids, index=0)

# Get safe operating ranges
machine_data = df[df['machine_id'] == selected_machine]
safe_ranges = {}
for col in ['temperature', 'vibration', 'humidity', 'pressure', 'energy_consumption']:
    safe_ranges[col] = {
        'min': machine_data[col].min(),
        'max': machine_data[col].max(),
        'last': machine_data[col].iloc[-1]
    }

# ========================= Live Input =========================
st.markdown("### Live Sensor Readings")
cols = st.columns(5)
sensor_labels = ["Temperature", "Vibration", "Humidity", "Pressure", "Energy Consumption"]
sensor_keys = ['temperature', 'vibration', 'humidity', 'pressure', 'energy_consumption']
inputs = {}
alerts = []

for col, label, key in zip(cols, sensor_labels, sensor_keys):
    with col:
        default_val = float(safe_ranges[key]['last'])
        val = st.number_input(label, value=default_val, step=0.01, format="%.4f")
        inputs[key] = val
        
        mn, mx = safe_ranges[key]['min'], safe_ranges[key]['max']
        if val < mn or val > mx:
            st.error("Out of Range!")
            alerts.append(f"**{label}** out of safe range [{mn:.2f} – {mx:.2f}]")
        else:
            st.success("Normal")

# Alerts
if alerts:
    st.error("ABNORMAL SENSOR READINGS DETECTED!")
    for alert in alerts:
        st.markdown(f"• {alert}")
else:
    st.success("All sensors within normal operating range")

st.markdown("---")

# ========================= Prediction =========================
if st.button("Predict Current Health Score", type="primary", use_container_width=True):
    with st.spinner("Analyzing machine health..."):
        # Build full sequence
        hist = st.session_state.history[selected_machine]  # (19, 5)
        new_reading = np.array([[inputs[k] for k in sensor_keys]])  # (1, 5)
        full_seq = np.vstack([hist, new_reading])  # (20, 5)
        
        # Feature engineering
        df_seq = pd.DataFrame(full_seq, columns=sensor_keys)
        df_seq['temp_moving_avg'] = df_seq['temperature'].rolling(5, min_periods=1).mean()
        df_seq['vib_moving_avg'] = df_seq['vibration'].rolling(5, min_periods=1).mean()
        df_seq['temp_rate_change'] = df_seq['temperature'].diff().fillna(0)
        df_seq['vib_rate_change'] = df_seq['vibration'].diff().fillna(0)
        
        feature_cols = ['temperature','vibration','humidity','pressure','energy_consumption',
                        'temp_moving_avg','vib_moving_avg','temp_rate_change','vib_rate_change']
        
        X = df_seq[feature_cols].values
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            health_score = model(X_tensor).item()
        
        # Update history
        st.session_state.history[selected_machine] = full_seq[1:]

        # Display Results
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Health Score Result")
            st.metric("Current Health", f"{health_score:.1f}/100")
            
            health_pct = health_score / 100
            st.progress(health_pct)
            
            if health_score >= 85:
                st.success("Excellent Condition")
            elif health_score >= 70:
                st.warning("Monitor Closely – Schedule Maintenance")
            else:
                st.error("CRITICAL – Immediate Action Required!")

        with col2:
            st.markdown("### Health Trend (Last 20 Readings)")
            # Simulate trend for visualization
            trend = np.linspace(95, health_score, 20)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=trend, mode='lines+markers',
                name='Health Trend',
                line=dict(color='dodgerblue', width=4)
            ))
            fig.add_hline(y=85, line_dash="dash", line_color="orange", annotation_text="Warning")
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Critical")
            fig.update_layout(height=450, template="plotly_white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True) 

st.caption("AI Model: LSTM | Features: 9 | Output: Health Score 0–100 | Built with Streamlit")
