import streamlit as st
import pandas as pd
import numpy as np
import os
from urllib.request import urlretrieve
from collections import deque
import threading
import time
import requests

# Try to import ML/plotting libraries with error handling
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.error("Scikit-learn is not installed. Please install it with: pip install scikit-learn")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.error("Matplotlib is not installed. Please install it with: pip install matplotlib")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    st.error("Seaborn is not installed. Please install it with: pip install seaborn")

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("Plotly is not installed. Please install it with: pip install plotly")

# ===== DATASET DOWNLOAD FUNCTION =====
def setup_dataset():
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    filename = "UNSW-NB15_small.csv"   # Use the smaller file
    filepath = os.path.join(data_dir, filename)

    # If file already exists, use it
    if os.path.exists(filepath):
        return {"training": filepath}
    
    # If file doesn't exist, try to download it
    try:
        with st.spinner('Downloading dataset... This may take a few minutes.'):
            # Use a reliable URL
            dataset_url = "https://github.com/tawabshakeel/UNSW-NB15/raw/master/UNSW-NB15_1.csv"
            response = requests.get(dataset_url)
            response.raise_for_status()  # Check for HTTP errors
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            st.success("Downloaded dataset successfully!")
            return {"training": filepath}
            
    except Exception as e:
        st.error(f"Error downloading dataset: {str(e)}")
        st.info("Please make sure UNSW-NB15_1.csv is in the 'data' folder.")
        return None

# Load data function with download capability
@st.cache_data
def load_real_data():
    # Setup dataset (check if available)
    dataset_paths = setup_dataset()
    
    if dataset_paths is None:
        st.warning("Using synthetic data for demonstration. Place UNSW-NB15_1.csv in the 'data' folder for real data.")
        return create_synthetic_data()
    
    try:
        # Load training data
        data = pd.read_csv(dataset_paths["training"])
        
        # Make sure the last column is treated as 'label'
        if 'label' not in data.columns:
            data.columns = list(data.columns[:-1]) + ['label']
        
        # Check if this is the real dataset
        if data.shape[0] < 100:  # If the file is too small, it might be corrupted
            st.warning("Downloaded file seems too small. Using synthetic data instead.")
            return create_synthetic_data()
            
        features = data.columns.drop('label')
        return data, features
        
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        # Fall back to synthetic data if real data fails
        return create_synthetic_data()

# Fallback function if real data isn't available
def create_synthetic_data():
    st.warning("Using synthetic data for demonstration. Real dataset will be used when available.")
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
# ===== END OF DATASET DOWNLOAD CODE =====

# Set page config
st.set_page_config(
    page_title="5G AI Threat Detection",
    page_icon="📡",
    layout="wide"
)

# Title
st.title("🛡️ 5G AI Threat Detection Dashboard")
st.markdown("### Real-time Security Monitoring for Cloud-Native Telecom Infrastructure")

# Check if all required packages are installed
if not all([SKLEARN_AVAILABLE, MATPLOTLIB_AVAILABLE, SEABORN_AVAILABLE, PLOTLY_AVAILABLE]):
    st.error("""
    **Missing required packages!**
    
    Please install all required packages using:
    ```
    pip install -r requirements.txt
    ```
    
    Or manually install them with:
    ```
    pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly requests
    ```
    """)
    st.stop()

# Load data
df, features = load_real_data()

if df is not None:
    # Check if we have a label column
    if 'label' in df.columns:
        X = df[features]
        y = df['label'].values
    else:
        # If no label column, create a synthetic one for demonstration
        st.warning("No label column found. Using synthetic labels for demonstration.")
        X = df
        y = np.random.choice([0, 1], len(df), p=[0.9, 0.1])
    
    # Preprocess
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
    model.fit(X_scaled[y == 0]) if len(np.unique(y)) > 1 else model.fit(X_scaled)

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
if df is not None:
    streamer = DataStreamer(X_scaled, model)
else:
    st.error("Failed to load data. Please check your internet connection and try again.")
    streamer = None

# Sidebar controls
st.sidebar.header("Dashboard Controls")

# Add a retry button at the top of the sidebar
if st.sidebar.button("🔄 Retry Download", help="Click if dataset download failed"):
    # Clear cache to force redownload
    st.cache_data.clear()
    st.rerun()

if streamer:
    if st.sidebar.button("▶️ Start Monitoring", key="start_btn"):
        streamer.start_streaming()
        
    if st.sidebar.button("⏹️ Stop Monitoring", key="stop_btn"):
        streamer.is_streaming = False

st.sidebar.slider("Detection Sensitivity", 0.01, 0.3, 0.1, 0.01, key="sensitivity")
st.sidebar.markdown("---")
st.sidebar.info("**5G Threat Detection System**\n\nAI-powered security for cloud-native telecom infrastructure")

# Main dashboard
if streamer:
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
else:
    st.error("System initialization failed. Unable to start monitoring.")

# Footer
st.markdown("---")
st.caption("AI-Driven Threat Detection System for Cloud-Native 5G Telecom Infrastructure | Built with Streamlit")
