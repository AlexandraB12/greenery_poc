import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from linear import run_linear_regression
from utils import load_data, get_features

# 1️⃣ Charger les données
df = load_data()

# 2️⃣ Lancer la régression (target_score synthétique si besoin)
coef, vif = run_linear_regression()

# 3️⃣ Visualisation 1 : matrice de corrélation
features = get_features(df)
corr = features.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Corrélations pairwise entre métriques visuelles")
plt.show()

# 4️⃣ Visualisation 2 : coefficients β standardisés
plt.figure(figsize=(8,5))
sns.barplot(x="beta", y="variable", data=coef, palette="viridis")
plt.title("Coefficients β standardisés")
plt.xlabel("β")
plt.ylabel("Variable")
plt.show()

# 5️⃣ Visualisation 3 : VIF
plt.figure(figsize=(8,5))
sns.barplot(x="VIF", y="variable", data=vif, palette="magma")
plt.title("⚡ Variance Inflation Factor (VIF)")
plt.xlabel("VIF")
plt.ylabel("Variable")
plt.show()
