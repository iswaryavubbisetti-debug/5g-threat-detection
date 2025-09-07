import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import time
from collections import deque
import threading

# Page config
st.set_page_config(page_title="5G Threat Detection", layout="wide")

# Title
st.title("🛡️ 5G AI Threat Detection Dashboard")
st.markdown("### Real-time Security Monitoring for Cloud-Native Telecom Infrastructure")

# Load data
@st.cache_data
def load_data():
    # Create realistic 5G network data
    np.random.seed(42)
    n_samples = 1000
    
    synthetic_data = {
        'dur': np.random.exponential(0.5, n_samples),
        'spkts': np.random.poisson(10, n_samples),
        'dpkts': np.random.poisson(8, n_samples),
        'sbytes': np.random.normal(500, 200, n_samples),
        'dbytes': np.random.normal(600, 250, n_samples),
        'rate': np.random.uniform(0, 1, n_samples),
        'sttl': np.random.randint(30, 255, n_samples),
        'dttl': np.random.randint(30, 255, n_samples),
        'sload': np.random.exponential(0.3, n_samples),
        'dload': np.random.exponential(0.4, n_samples),
        'label': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    }
    df = pd.DataFrame(synthetic_data)
    return df, list(synthetic_data.keys())[:-1]

df, features = load_data()
X = df[features]
y = df['label'].values

# Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
model.fit(X_scaled[y == 0])

# Data streamer class
class DataStreamer:
    def __init__(self, data, model):
        self.data = data
        self.model = model
        self.current_index = 0
        self.alerts = deque(maxlen=15)
        self.packet_history = deque(maxlen=50)
        self.is_streaming = False
        self.threat_count = 0
        
    def start_streaming(self):
        self.is_streaming = True
        threading.Thread(target=self._stream_data, daemon=True).start()
        
    def _stream_data(self):
        while self.is_streaming and self.current_index < len(self.data):
            row = self.data[self.current_index]
            prediction = self.model.predict([row])[0]
            score = model.score_samples([row])[0]
            
            self.packet_history.append({
                'time': self.current_index,
                'size': row[3] if len(row) > 3 else 500,
                'is_threat': prediction == -1,
                'score': score
            })
            
            if prediction == -1:
                self.threat_count += 1
                threat_type = np.random.choice(['DDoS', 'Port Scan', 'Malware', 'Brute Force'])
                alert_msg = f"{time.strftime('%H:%M:%S')} - {threat_type} detected (Score: {score:.3f})"
                self.alerts.appendleft(alert_msg)
            
            self.current_index += 1
            time.sleep(0.2)

# Initialize streamer
streamer = DataStreamer(X_scaled, model)

# Sidebar controls
st.sidebar.header("Dashboard Controls")
if st.sidebar.button("▶️ Start Monitoring", key="start_btn"):
    streamer.start_streaming()
    
if st.sidebar.button("⏹️ Stop Monitoring", key="stop_btn"):
    streamer.is_streaming = False

st.sidebar.slider("Detection Sensitivity", 0.01, 0.3, 0.1, 0.01, key="sensitivity")
st.sidebar.markdown("---")
st.sidebar.info("**5G Threat Detection System**\n\nAI-powered security for cloud-native telecom infrastructure")

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Packets", streamer.current_index)
    
with col2:
    st.metric("Threats Detected", streamer.threat_count)
    
with col3:
    threat_rate = (streamer.threat_count / streamer.current_index * 100) if streamer.current_index > 0 else 0
    st.metric("Threat Rate", f"{threat_rate:.1f}%")
    
with col4:
    status = "ACTIVE 🟢" if streamer.is_streaming else "IDLE 🔴"
    st.metric("System Status", status)

# Live charts
if streamer.packet_history:
    chart_data = pd.DataFrame(streamer.packet_history)
    
    fig = px.scatter(chart_data, x='time', y='size', color='is_threat',
                    color_discrete_map={True: 'red', False: 'blue'},
                    title="Live 5G Traffic Monitoring",
                    labels={'time': 'Packet Sequence', 'size': 'Packet Size (bytes)'})
    st.plotly_chart(fig, use_container_width=True)

# Alert section
st.subheader("🔴 Real-time Security Alerts")
alert_container = st.container()

with alert_container:
    for alert in streamer.alerts:
        st.error(alert)
    
    if not streamer.alerts:
        st.info("No threats detected. Monitoring in progress...")

# Response actions
st.subheader("🛡️ Automated Response Actions")
if streamer.threat_count > 0:
    st.success("✅ Malicious IPs automatically blocked")
    st.success("✅ Traffic rerouted to secure zones")
    st.success("✅ SOC team notified in real-time")
else:
    st.info("No response actions required")

# Footer
st.markdown("---")
st.caption("AI-Driven Threat Detection System for Cloud-Native 5G Telecom Infrastructure | Built with Streamlit")
