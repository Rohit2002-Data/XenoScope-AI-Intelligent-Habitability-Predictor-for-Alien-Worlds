# app.py
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from fetch_nasa import get_exoplanet_data

st.set_page_config(page_title="XenoScope AI", page_icon="🪐")
st.title("🪐 XenoScope AI — Exoplanet Habitability Predictor")
st.markdown("Predict whether an exoplanet could be habitable based on NASA data and test the model on real unseen exoplanets.")

# 🔭 Load NASA dataset
raw_df = get_exoplanet_data()
features = ['pl_orbper', 'pl_rade', 'pl_bmasse', 'st_teff', 'pl_eqt', 'st_rad', 'st_lum']
df = raw_df[["pl_name"] + features].dropna()

# 🧠 Labeling rule
df['habitable'] = (
    (df['pl_eqt'] >= 180) & (df['pl_eqt'] <= 310) &
    (df['pl_rade'] >= 0.5) & (df['pl_rade'] <= 2.0)
).astype(int)

# 🔀 Train/test split
X = df[features]
y = df['habitable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ✅ Train model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 📊 Evaluation
with st.expander("📈 Test Model on Testing Dataset"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"✅ **Accuracy:** `{acc*100:.2f}%`")

    st.write("📊 **Confusion Matrix**")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Habitable", "Habitable"], yticklabels=["Not Habitable", "Habitable"])
    st.pyplot(fig)

    if st.checkbox("🔍 Show test predictions"):
        test_results = X_test.copy()
        test_results["Actual"] = y_test.values
        test_results["Predicted"] = y_pred
        st.dataframe(test_results)

# 🔘 User input
option = st.radio("🔍 Choose input method:", ["Manual Entry", "Select from NASA live data"])

if option == "Manual Entry":
    inputs = []
    st.subheader("🧪 Enter Planet Features")

    if st.button("🧬 Simulate Earth-like Planet"):
        inputs = [365.25, 1.0, 1.0, 5778, 288, 1.0, 1.0]
        st.success("✅ Earth-like values prefilled!")
    else:
        for feat in features:
            value = st.number_input(f"Enter {feat}:", min_value=0.0)
            inputs.append(value)

    if st.button("🔮 Predict"):
        if len(inputs) == len(features):
            pred = model.predict([inputs])[0]
            st.success("✅ Habitable 🌍" if pred else "❌ Not Habitable 🪐")
        else:
            st.error("❌ Please fill all input values.")

else:
    st.subheader("🔭 Select from NASA Live Exoplanets")
    selected = st.selectbox("Choose an exoplanet", df['pl_name'])
    row = df[df['pl_name'] == selected][features]
    st.write("🔍 Planet Features:")
    st.dataframe(row)

    if st.button("🔮 Predict Habitability"):
        pred = model.predict(row)[0]
        st.success("✅ Habitable 🌍" if pred else "❌ Not Habitable 🪐")
