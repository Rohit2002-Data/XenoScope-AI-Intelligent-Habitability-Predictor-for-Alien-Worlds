# app.py
import streamlit as st
import pandas as pd
import joblib
from fetch_nasa import get_exoplanet_data

# Load model
model = joblib.load("habitability_model.pkl")

st.title("🪐 Exoplanet Habitability Predictor")
st.markdown("Predict whether an exoplanet could be habitable based on its features.")

option = st.radio("🔍 Choose input method:", ["Manual Entry", "Select from NASA live data"])

features = ['pl_orbper', 'pl_rade', 'pl_bmasse', 'st_teff', 'pl_eqt', 'st_rad', 'st_lum']

if option == "Manual Entry":
    inputs = []
    for feat in features:
        value = st.number_input(f"Enter {feat}:", min_value=0.0)
        inputs.append(value)

    if st.button("Predict"):
        pred = model.predict([inputs])[0]
        st.success("✅ Habitable" if pred else "❌ Not Habitable")

else:
    df = get_exoplanet_data()
    selected = st.selectbox("Choose an exoplanet", df['pl_name'])
    row = df[df['pl_name'] == selected][features]
    st.write("🔭 Selected planet data:")
    st.dataframe(row)

    if st.button("Predict Habitability"):
        pred = model.predict(row)[0]
        st.success("✅ Habitable" if pred else "❌ Not Habitable")
