import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

@st.cache_resource
def load_data_and_model():
    with open('df.pkl', 'rb') as f_df:
        df = pickle.load(f_df)
    with open('pipeline.pkl', 'rb') as f_pipe:
        pipeline = pickle.load(f_pipe)
    return df, pipeline

try:
    df, pipeline = load_data_and_model()
except FileNotFoundError:
    st.error("Missing model artifacts! Please run your training script first to generate 'df.pkl' and 'pipeline.pkl'.")
    st.stop()

st.title("💻 Laptop Price Predictor Dashboard")
st.markdown("Estimate retail market prices using your trained Scikit-Learn pipeline based on core hardware configurations.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Brand & Form Factor")
    company = st.selectbox("Manufacturer (Company):", sorted(df['Company'].unique()))
    type_name = st.selectbox("Laptop Category (TypeName):", sorted(df['TypeName'].unique()))
    os = st.selectbox("Operating System (os):", sorted(df['os'].unique()))
    weight = st.number_input("Weight (kg):", min_value=0.5, max_value=5.0, value=1.8, step=0.05)

with col2:
    st.subheader("2. Processing & Storage")
    cpu_brand = st.selectbox("CPU Family (Cpu Brand):", sorted(df['Cpu Brand'].unique()))
    gpu_brand = st.selectbox("GPU Manufacturer (Gpu Brand):", sorted(df['Gpu Brand'].unique()))
    ram = st.selectbox("System Memory / RAM (GB):", [2, 4, 6, 8, 12, 16, 24, 32, 64], index=3)
    
    sub_col_a, sub_col_b = st.columns(2)
    with sub_col_a:
        ssd = st.selectbox("SSD Storage (GB):", [0, 8, 128, 256, 512, 1024], index=3)
    with sub_col_b:
        hdd = st.selectbox("HDD Storage (GB):", [0, 128, 256, 512, 1024, 2048], index=0)

with col3:
    st.subheader("3. Display & Resolution")
    screen_size = st.slider("Screen Size (Inches):", min_value=10.0, max_value=18.5, value=15.6, step=0.1)
    resolution = st.selectbox(
        "Screen Resolution:",
        [
            "1920x1080",
            "1366x768",
            "1600x900",
            "3840x2160",
            "3200x1800",
            "2880x1800",
            "2560x1600",
            "2560x1440",
            "2304x1440"
        ]
    )
    touchscreen = st.radio("Touchscreen Support:", ["No", "Yes"], horizontal=True)
    ips = st.radio("IPS Panel Display:", ["No", "Yes"], horizontal=True)

st.divider()

if st.button("🚀 Calculate Estimated Market Price", use_container_width=True):
    # Parse screen dimensions and compute PPI
    x_res, y_res = map(int, resolution.split('x'))
    ppi = (((x_res ** 2) + (y_res ** 2)) ** 0.5) / screen_size

    # Convert binary choices to 0/1 integers
    touchscreen_val = 1 if touchscreen == "Yes" else 0
    ips_val = 1 if ips == "Yes" else 0

    # Build input DataFrame adhering to the exact column schema of X
    input_data = pd.DataFrame([{
        'Company': company,
        'TypeName': type_name,
        'Ram': int(ram),
        'Weight': float(weight),
        'TouchScreen': touchscreen_val,
        'Ips': ips_val,
        'ppi': np.float32(ppi),
        'Cpu Brand': cpu_brand,
        'HDD': int(hdd),
        'SSD': int(ssd),
        'Gpu Brand': gpu_brand,
        'os': os
    }])

    # Predict log price and reverse with exp
    pred_log_price = pipeline.predict(input_data)[0]
    predicted_price = np.exp(pred_log_price)

    # Display results
    res_col1, res_col2 = st.columns([1, 1])
    with res_col1:
        st.success(f"### Predicted Price: ₹ {predicted_price:,.2f}")
        st.caption("Valuation derived from trained regression pipeline.")
        
    with res_col2:
        st.info(f"**Computed Display Density:** `{ppi:.2f} PPI` | **Total Storage:** `{ssd + hdd} GB`")