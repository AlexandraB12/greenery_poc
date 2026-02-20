import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# load_data() → lit le CSV robustement, peu importe où l'on exécute le script.
def load_data(filename="metrics_output.csv"):
    """Charge le CSV depuis la racine du projet"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, filename)
    return pd.read_csv(path)

# get_features(df) → retourne uniquement les colonnes que l'on veut dans la régression.
def get_features(df):
    """
    Retourne les colonnes à utiliser pour la régression.
    Ici, on fusionne brightness et sky_ratio pour réduire le VIF.
    """
    # Fusion pour réduire la multicolinéarité
    df = df.copy()  # pour ne pas modifier le df original
    df["luminance"] = 0.5 * df["brightness"] + 0.5 * df["sky_ratio"]

    feature_cols = [
        "greenery_ratio",
        "luminance",
        "visual_complexity",
        "building_regularity"
    ]
    return df[feature_cols]

# standardize_features(X) → standardise les métriques pour que les β soient comparables.
# Renvoie X_scaled et le scaler
def standardize_features(X):
    """
    Standardise X pour que chaque variable ait moyenne=0 et écart-type=1
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled_df, scaler


