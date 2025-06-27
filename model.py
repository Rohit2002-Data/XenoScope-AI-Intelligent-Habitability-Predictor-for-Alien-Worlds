# model.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from fetch_nasa import get_exoplanet_data

def train_and_save_model():
    df = get_exoplanet_data()
    features = ['pl_orbper', 'pl_rade', 'pl_bmasse', 'st_teff', 'pl_eqt', 'st_rad', 'st_lum']
    X = df[features]
    y = df['habitable']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "habitability_model.pkl")
    print("✅ Model saved!")

if __name__ == "__main__":
    train_and_save_model()
