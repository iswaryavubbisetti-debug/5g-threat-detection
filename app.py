import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import IsolationForest
import time

# ------------------------------
# Streamlit Dashboard UI
# ------------------------------
st.set_page_config(page_title="5G AI Threat Detection Dashboard", layout="wide")

# Initialize session state
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False
if "traffic_data" not in st.session_state:
    st.session_state.traffic_data = []

st.sidebar.title("Dashboard Controls")
if st.sidebar.button("Start Monitoring"):
    st.session_state.monitoring = True
if st.sidebar.button("Stop Monitoring"):
    st.session_state.monitoring = False
if st.sidebar.button("Retry Download"):
    st.session_state.traffic_data = []

detection_sensitivity = st.sidebar.slider("Detection Sensitivity", 0.01, 0.50, 0.10)

st.sidebar.markdown("---")
st.sidebar.subheader("5G Threat Detection System")
st.sidebar.write("AI-powered security for cloud-native telecom infrastructure")

st.title("🛡️ 5G AI Threat Detection Dashboard")
st.subheader("Real-time Security Monitoring for Cloud-Native Telecom Infrastructure")

# ------------------------------
# Load Dataset
# ------------------------------
def generate_synthetic_data():
    np.random.seed(42)
    data = np.random.rand(100, 10)
    return pd.DataFrame(data, columns=[f"feature_{i}" for i in range(10)])

file_path = os.path.join("data", "UNSW-NB15_small.csv")
uploaded_file = st.file_uploader("Upload UNSW-NB15 dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Loaded dataset from uploaded file.")
elif os.path.exists(file_path):
    df = pd.read_csv(file_path)
    st.success("✅ Loaded dataset from local path.")
else:
    st.warning("⚠️ Dataset not found, using synthetic data instead.")
    df = generate_synthetic_data()

# ------------------------------
# AI Model (Isolation Forest)
# ------------------------------
model = IsolationForest(contamination=detection_sensitivity, random_state=42)
model.fit(df.select_dtypes(include=[np.number]))
predictions = model.predict(df.select_dtypes(include=[np.number]))
df["threat"] = np.where(predictions == -1, 1, 0)

# ------------------------------
# Dashboard Metrics
# ------------------------------
total_packets = len(df)
threats_detected = df["threat"].sum()
threat_rate = (threats_detected / total_packets) * 100 if total_packets > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Packets", total_packets)
col2.metric("Threats Detected", threats_detected)
col3.metric("Threat Rate", f"{threat_rate:.2f}%")
col4.metric("System Status", "ACTIVE" if st.session_state.monitoring else "IDLE")

# ------------------------------
# Real-time Alerts
# ------------------------------
st.subheader("🔴 Real-time Security Alerts")
if threats_detected > 0 and st.session_state.monitoring:
    st.error(f"⚠️ {threats_detected} threats detected in live data stream!")
elif st.session_state.monitoring:
    st.info("✅ No threats detected at the moment.")
else:
    st.info("⏸ Monitoring stopped.")

# ------------------------------
# Traffic Graph (Packet Flow)
# ------------------------------
st.subheader("📊 Network Traffic Overview")

if st.session_state.monitoring:
    # Simulate live packet traffic
    new_point = {
        "time": len(st.session_state.traffic_data) + 1,
        "packets": total_packets,
        "threats": threats_detected,
    }
    st.session_state.traffic_data.append(new_point)

traffic_df = pd.DataFrame(st.session_state.traffic_data)

if not traffic_df.empty:
    st.line_chart(traffic_df.set_index("time")[["packets", "threats"]])
else:
    st.info("No traffic data available. Start monitoring to see live traffic.")
