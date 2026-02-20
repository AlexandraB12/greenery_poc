import pandas as pd
import numpy as np

# --------------------------------------------------
# 1️⃣ CHARGEMENT DU CSV ACTUEL
# --------------------------------------------------
# ⚠️ Mets ici le chemin vers TON fichier
INPUT_CSV = "adresses_geocodees.csv"

# Le fichier propre final sera écrit ici
OUTPUT_CSV = "adresses_final_streetview.csv"

# Lecture du CSV
df = pd.read_csv(INPUT_CSV)

print(f"📂 {len(df)} lignes chargées")


# --------------------------------------------------
# 2️⃣ DÉFINITION DES LIMITES ÎLE-DE-FRANCE
# --------------------------------------------------
# Ces bornes sont volontairement larges
# pour éviter les faux négatifs
IDF_LAT_MIN = 48.20
IDF_LAT_MAX = 49.20
IDF_LON_MIN = 1.80
IDF_LON_MAX = 2.80


# --------------------------------------------------
# 3️⃣ DÉTECTION DES COORDONNÉES INVALIDES
# --------------------------------------------------
def is_outside_idf(row):
    """
    Retourne True si :
    - latitude ou longitude manquante
    - OU coordonnées hors Île-de-France
    """
    lat = row["latitude"]
    lon = row["longitude"]

    # Si coordonnées manquantes → invalide
    if pd.isna(lat) or pd.isna(lon):
        return True

    # Si hors bornes IDF → invalide
    if not (IDF_LAT_MIN <= lat <= IDF_LAT_MAX):
        return True

    if not (IDF_LON_MIN <= lon <= IDF_LON_MAX):
        return True

    return False


# Application ligne par ligne
df["invalid_geo"] = df.apply(is_outside_idf, axis=1)

nb_invalid = df["invalid_geo"].sum()
print(f"🚨 {nb_invalid} lignes détectées hors Île-de-France")


# --------------------------------------------------
# 4️⃣ NETTOYAGE DES LIGNES INVALIDES
# --------------------------------------------------
# Pour ces lignes :
# - on efface latitude / longitude
# - on marque geocoding_source = "ambiguous"

df.loc[df["invalid_geo"], "latitude"] = np.nan
df.loc[df["invalid_geo"], "longitude"] = np.nan
df.loc[df["invalid_geo"], "geocoding_source"] = "ambiguous"


# --------------------------------------------------
# 5️⃣ NETTOYAGE FINAL
# --------------------------------------------------
# On supprime la colonne technique temporaire
df.drop(columns=["invalid_geo"], inplace=True)


# --------------------------------------------------
# 6️⃣ EXPORT DU CSV FINAL
# --------------------------------------------------
df.to_csv(OUTPUT_CSV, index=False)

print("✅ CSV NETTOYÉ AVEC SUCCÈS")
print(f"📄 Fichier final : {OUTPUT_CSV}")

