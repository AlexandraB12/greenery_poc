# linear.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from utils import load_data, get_features, standardize_features

from utils import load_data, get_features

def compute_vif(X):
    """Calcule le VIF pour chaque variable."""
    vif = pd.DataFrame()
    vif["variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

def standardize_X(X):
    """Standardise les variables pour comparer les coefficients."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)

# 1️⃣ Charger les données
df = load_data()

# 2️⃣ Extraire les features
X = get_features(df)

# 3️⃣ Standardiser
X_scaled, scaler = standardize_features(X)

# 4️⃣ Sauvegarder le DataFrame standardisé
X_scaled.to_csv("metrics_standardized.csv", index=False)
print("✅ Standardized metrics saved in metrics_standardized.csv")


def create_target_score(df):
    """Crée automatiquement un target_score synthétique si aucune cible n'est présente."""
    if "target_score" not in df.columns:
        print("⚡ target_score non trouvé. Création d'un score synthétique.")
        df["target_score"] = (
            0.4 * df["greenery_ratio"] +
            0.2 * df["brightness"] +
            0.1 * df["sky_ratio"] +
            0.2 * df["visual_complexity"] -
            0.1 * df["building_regularity"]
        )
    return df

def run_linear_regression(target_col=None):
    # 1️⃣ Charger les données
    df = load_data()

    # 2️⃣ Créer target_score si nécessaire
    df = create_target_score(df)

    # 3️⃣ Définir X et y
    X = get_features(df)
    if target_col is None:
        target_col = "target_score"
    y = df[target_col]

    # 4️⃣ Vérifier la multicolinéarité
    print("\n🔹 Variance Inflation Factor (VIF) :")
    vif = compute_vif(X)
    print(vif)

    # 5️⃣ Standardiser X
    X_scaled = standardize_X(X)

    # 6️⃣ Régression linéaire
    model = LinearRegression()
    model.fit(X_scaled, y)

    # 7️⃣ Coefficients standardisés
    coef = pd.DataFrame({
        "variable": X.columns,
        "beta": model.coef_
    }).sort_values(by="beta", ascending=False)

    print("\n📊 Coefficients standardisés :")
    print(coef)

    # 8️⃣ Interprétation automatique simple
    print("\n🧠 Interprétation :")
    for _, row in coef.iterrows():
        var, beta = row["variable"], row["beta"]
        if beta > 0:
            print(f"✅ {var} : effet favorable (β={beta:.2f})")
        elif beta < 0:
            print(f"❌ {var} : effet défavorable (β={beta:.2f})")
        else:
            print(f"⚪ {var} : neutre (β≈0)")

    return coef, vif

if __name__ == "__main__":
    # Si aucune cible fournie, utilise target_score synthétique
    run_linear_regression()
