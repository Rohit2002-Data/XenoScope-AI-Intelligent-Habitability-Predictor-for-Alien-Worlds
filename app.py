# app.py
import streamlit as st
import pandas as pd
import joblib
import os
from fetch_nasa import get_exoplanet_data

st.title("🪐 Exoplanet Habitability Predictor")
st.markdown("Predict whether an exoplanet could be habitable based on its features.")

# ✅ Safely load the trained model (.pkl)
model_path = os.path.join(os.path.dirname(__file__), "habitability_model.pkl")
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("❌ Model file not found. Please run `model.py` to create `habitability_model.pkl`.")
    st.stop()  # Prevent app from running if model is missing

# 🔢 Features to be used
features = ['pl_orbper', 'pl_rade', 'pl_bmasse', 'st_teff', 'pl_eqt', 'st_rad', 'st_lum']

# 🔘 User Input Method
option = st.radio("🔍 Choose input method:", ["Manual Entry", "Select from NASA live data"])

if option == "Manual Entry":
    inputs = []
    for feat in features:
        value = st.number_input(f"Enter {feat}:", min_value=0.0)
        inputs.append(value)

    if st.button("🔮 Predict"):
        pred = model.predict([inputs])[0]
        st.success("✅ Ha
