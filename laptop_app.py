import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIG ---
st.set_page_config(page_title="Laptop Price Predictor", layout="centered")

# --- STEP 1: LOAD & TRAIN MODEL (AUTOMATIC) ---
@st.cache_data # Isse bar-bar training nahi hogi, fast chalega
def train_model():
    try:
        df = pd.read_csv('laptopPrice.csv')
        
        # Cleaning
        def clean_to_int(text):
            match = re.search(r'\d+', str(text))
            return int(match.group()) if match else 0

        df['ram_gb'] = df['ram_gb'].apply(clean_to_int)
        df['ssd'] = df['ssd'].apply(clean_to_int)
        df['hdd'] = df['hdd'].apply(clean_to_int)
        df['processor_gnrtn'] = df['processor_gnrtn'].apply(clean_to_int)
        df['Touchscreen'] = df['Touchscreen'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
        
        features = ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn', 'ram_gb', 'ram_type', 'ssd', 'hdd']
        X = df[features].copy()
        y = df['Price']

        # Encoding
        le_dict = {}
        cat_cols = ['brand', 'processor_brand', 'processor_name', 'ram_type']
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model, le_dict, df, features
    except FileNotFoundError:
        st.error("Error: 'laptopPrice.csv' file nahi mili! Please check karein.")
        return None, None, None, None

model, le_dict, df, features = train_model()

# --- STEP 2: USER INTERFACE (UI) ---
st.title("💻 Laptop Price Predictor")
st.write("Apne laptop ki specs select karein aur price jaanein.")

if df is not None:
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", df['brand'].unique())
        p_brand = st.selectbox("Processor Brand", df['processor_brand'].unique())
        p_name = st.selectbox("Processor Name", df['processor_name'].unique())
        p_gen = st.selectbox("Generation", sorted(df['processor_gnrtn'].unique()))

    with col2:
        ram = st.selectbox("RAM (GB)", sorted(df['ram_gb'].unique()))
        ram_type = st.selectbox("RAM Type", df['ram_type'].unique())
        ssd = st.selectbox("SSD (GB)", sorted(df['ssd'].unique()))
        hdd = st.selectbox("HDD (GB)", sorted(df['hdd'].unique()))

    # --- PREDICTION LOGIC ---
    if st.button("Predict Price", use_container_width=True):
        # Prepare Input
        input_data = pd.DataFrame([[brand, p_brand, p_name, p_gen, ram, ram_type, ssd, hdd]], 
                                 columns=features)
        
        # Apply Encoding
        for col in le_dict:
            input_data[col] = le_dict[col].transform(input_data[col])

        # Predict
        res = model.predict(input_data)[0]
        
        st.success(f"### Estimated Price: ₹{round(res, 2)}")