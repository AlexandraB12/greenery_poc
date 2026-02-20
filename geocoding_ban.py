import pandas as pd
from time import sleep
from geopy.geocoders import Nominatim

# ----------------------------------
# Initialisation du géocodeur Nominatim
# ----------------------------------
geolocator_nominatim = Nominatim(user_agent="greenery_poc")  # Nominatim/OpenStreetMap

# Placeholder pour BAN (si tu veux ajouter plus tard)
def geocode_ban(address: str):
    # Ici tu peux appeler l'API BAN si nécessaire
    return None

def geocode_nominatim(address: str, retries: int = 3):
    """
    Tente de géocoder avec Nominatim, retry automatique.
    """
    for attempt in range(retries):
        try:
            location = geolocator_nominatim.geocode(address)
            sleep(1)  # respecter les limites de l'API
            if location:
                return location
        except Exception as e:
            print(f"⚠️ Exception sur tentative {attempt+1} pour {address}: {e}")
    return None

# ----------------------------------
# Lecture du CSV
# ----------------------------------
df = pd.read_csv("adresses.csv")  # ton CSV original

# Construction d'une adresse complète
df["adresse_complete"] = df["adresse"] + ", " + df["code_postal"].astype(str) + " " + df["commune"] + ", France"

# ----------------------------------
# Fonction de géocodage multi-services avec retry
# ----------------------------------
def geocode_address(address: str) -> pd.Series:
    """
    Géocode une adresse en essayant plusieurs services avec retry.
    Retourne latitude, longitude et la source du géocodage.
    """
    # -------------------------
    # Tentative 1 : BAN
    # -------------------------
    try:
        location = geocode_ban(address)
        if location:
            return pd.Series([
                location.latitude,
                location.longitude,
                "ban_ok"          # 👈 source explicitement déclarée
            ])
    except Exception:
        pass

    # -------------------------
    # Tentative 2 : Nominatim avec retry
    # -------------------------
    location = geocode_nominatim(address, retries=3)
    if location:
        return pd.Series([
            location.latitude,
            location.longitude,
            "nominatim_ok"    # 👈 source explicitement déclarée
        ])

    # -------------------------
    # Échec total
    # -------------------------
    return pd.Series([
        None,
        None,
        "ambiguous"             # 👈 on ASSUME l’ambiguïté si tout échoue
    ])

# ----------------------------------
# Application du géocodage
# ----------------------------------
df[["latitude", "longitude", "geocoding_source"]] = df["adresse_complete"].apply(geocode_address)

# ----------------------------------
# Vérification des adresses non géocodées
# ----------------------------------
df_missing = df[df["latitude"].isna() | df["longitude"].isna()]
print(f"Nombre d'adresses manquantes : {len(df_missing)}")
print(df_missing[['id', 'adresse', 'commune', 'categorie']])

# ----------------------------------
# 5️⃣ Manual fix si nécessaire
# ----------------------------------
# Correction Rue des prés à Carrières sur Seine
df.loc[42, "latitude"] = 48.90810          # lat trouvée sur Google Maps
df.loc[42, "longitude"] = 2.18697         # lon trouvée sur Google Maps
df.loc[42, "geocoding_source"] = "manual_fix"  # corrigé manuellement

# Correction Rue Roger Salengro à Champigny-sur-Marne
df.loc[43, "latitude"] = 48.81996          # lat trouvée sur Google Maps
df.loc[43, "longitude"] = 2.48647         # lon trouvée sur Google Maps
df.loc[43, "geocoding_source"] = "manual_fix"  # corrigé manuellement

# Correction Rue de la République à Montesson (Boulevard de la République)
df.loc[44, "latitude"] = 48.90644          # lat trouvée sur Google Maps
df.loc[44, "longitude"] = 2.14371        # lon trouvée sur Google Maps
df.loc[44, "geocoding_source"] = "manual_fix"  # corrigé manuellement

# Correction Avenue Jean Jaurès à Trappes
df.loc[45, "latitude"] = 48.77600          # lat trouvée sur Google Maps
df.loc[45, "longitude"] = 2.00031       # lon trouvée sur Google Maps
df.loc[45, "geocoding_source"] = "manual_fix"  # corrigé manuellement

# Correction Avenue de la République à Athis-Mons (Rue)
df.loc[46, "latitude"] = 48.71056          # lat trouvée sur Google Maps
df.loc[46, "longitude"] = 2.38860      # lon trouvée sur Google Maps
df.loc[46, "geocoding_source"] = "manual_fix"  # corrigé manuellement

# Correction Place Henri Barbusse à Grigny (Rue)
df.loc[47, "latitude"] = 48.65355          # lat trouvée sur Google Maps
df.loc[47, "longitude"] = 2.39265     # lon trouvée sur Google Maps
df.loc[47, "geocoding_source"] = "manual_fix"  # corrigé manuellement

# Correction Place de la Gare à Sainte-Geneviève-des-Bois (Rue)
df.loc[47, "latitude"] = 48.66997          # lat trouvée sur Google Maps
df.loc[47, "longitude"] = 2.33106    # lon trouvée sur Google Maps
df.loc[47, "geocoding_source"] = "manual_fix"  # corrigé manuellement

# Correction Boulevard Paul Vaillant-Couturier à Goussainville (Rue)
df.loc[48, "latitude"] = 48.03232         # lat trouvée sur Google Maps
df.loc[48, "longitude"] = 2.47067    # lon trouvée sur Google Maps
df.loc[48, "geocoding_source"] = "manual_fix"  # corrigé manuellement

# Correction Place de la République à Poissy (Avenue)
df.loc[49, "latitude"] = 48.92765        # lat trouvée sur Google Maps
df.loc[49, "longitude"] = 2.04285    # lon trouvée sur Google Maps
df.loc[49, "geocoding_source"] = "manual_fix"  # corrigé manuellement

# Correction Avenue de la Gare à Sucy-en-Brie (Rue)
df.loc[50, "latitude"] = 48.77108        # lat trouvée sur Google Maps
df.loc[50, "longitude"] = 2.50863    # lon trouvée sur Google Maps
df.loc[50, "geocoding_source"] = "manual_fix"  # corrigé manuellement

# Correction Boulevard Carnot à Alfortville (Avenue)
df.loc[51, "latitude"] = 48.79624       # lat trouvée sur Google Maps
df.loc[51, "longitude"] = 2.42831    # lon trouvée sur Google Maps
df.loc[51, "geocoding_source"] = "manual_fix"  # corrigé manuellement

# Correction Avenue Jean Jaurès à Arcueil (Rue)
df.loc[52, "latitude"] = 48.81101       # lat trouvée sur Google Maps
df.loc[52, "longitude"] = 2.33645    # lon trouvée sur Google Maps
df.loc[52, "geocoding_source"] = "manual_fix"  # corrigé manuellement

# ----------------------------------
# Sauvegarde du CSV final
# ----------------------------------
df.to_csv("adresses_geocodees.csv", index=False)
print("✅ Géocodage terminé ! Fichier adresses_geocodees.csv créé.")
