# model.py
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from fetch_nasa import get_exoplanet_data

def train_and_save_model():
    df = get_exoplanet_data()
    features = ['pl_orbper', 'pl_rade', 'pl_bmasse', 'st_teff', 'pl_eqt', 'st_rad', 'st_lum']
    X = df[features]
    y = df['habitable']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "habitability_model.pkl")
    print("✅ Model trained and saved as 'habitability_model.pkl'.")

    # ✅ TESTING
    print("\n🧪 Testing Results:")
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_and_save_model()

