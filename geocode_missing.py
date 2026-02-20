import pandas as pd
from geopy.geocoders import Nominatim
from time import sleep

# Charger le CSV existant
df = pd.read_csv("adresses_geocodees.csv")

# Filtrer les adresses manquantes
df_missing = df[df['latitude'].isna() | df['longitude'].isna()]
print(f"Nombre d'adresses manquantes : {len(df_missing)}")
print(df_missing[['id','adresse','commune','categorie']])

# Initialisation du géocodeur
geolocator = Nominatim(user_agent="greenery_poc")

# Fonction pour retenter le géocodage
def geocode_missing(row):
    full_address = f"{row['adresse']}, {row['commune']}, France"  # code postal retiré
    try:
        location = geolocator.geocode(full_address)
        sleep(1)  # respecter les limites de l'API
        if location:
            return pd.Series([location.latitude, location.longitude])
        else:
            return pd.Series([None, None])
    except Exception as e:
        print(f"⚠️ Exception pour {row['adresse']}: {e}")
        return pd.Series([None, None])

# Retenter le géocodage
df_missing[['latitude','longitude']] = df_missing.apply(geocode_missing, axis=1)

# Mettre à jour le DataFrame principal (mais tu peux aussi garder df intact)
df_updated = df.copy()
df_updated.update(df_missing)

# Sauvegarder dans un nouveau fichier pour ne pas écraser l'ancien
df_updated.to_csv("adresses_geocodees_retente.csv", index=False)
print("✅ Fichier sauvegardé sous 'adresses_geocodees_retente.csv'")
