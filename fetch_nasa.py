# fetch_nasa.py
import pandas as pd

def get_exoplanet_data():
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,pl_orbper,pl_rade,pl_bmasse,st_teff,pl_eqt,st_rad,st_lum,sy_dist+from+ps&format=csv"
    df = pd.read_csv(url)

    # Drop rows with missing values
    df = df.dropna()

    # Label habitability (custom rule)
    df['habitable'] = (
        (df['pl_eqt'] >= 180) & (df['pl_eqt'] <= 310) &
        (df['pl_rade'] >= 0.5) & (df['pl_rade'] <= 2.0)
    ).astype(int)

    return df
