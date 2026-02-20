import os
print("Répertoire courant :", os.getcwd())
print("Fichiers présents :", os.listdir())


import pandas as pd
from geopy.geocoders import Nominatim    # geopy est une librairie Python qui interagit avec des services de géocodage.
# Nominatim = service de géocodage basé sur OpenStreetMap
from time import sleep   # sleep() sert à ralentir volontairement les requêtes vers l’API. Indispensable avec Nominatim (sinon blocage / ban).

# Initialisation du géocodeur
geolocator = Nominatim(user_agent="greenery_poc")    # Crée un objet “géocodeur”. Ce sera lui qui convertira texte → GPS
# user_agent est obligatoire pour travailler avec Nominatim

# Lecture du CSV
df = pd.read_csv("adresses.csv")

# Fonction de géocodage
def geocode_address(row, retries=3):
    full_address = f"{row['adresse']}, {row['commune']}, France"  # Construction de l’adresse complète

    for attempt in range(1, retries+1):  # Boucle de retry : tentative 1 à retries
        try:
            location = geolocator.geocode(full_address, timeout=10)  # Tentative de géocodage : Appel à l’API Nominatim
            sleep(2)  # respecter les limites de l'API : pause entre les requêtes
            if location:
                return pd.Series([location.latitude, location.longitude])  # Adresse trouvée : latitude / longitude
            else:
                print(f"⚠️ Tentative {attempt} : {full_address} non trouvée")  # Adresse non trouvée cette tentative
        except Exception as e:
            print(f"⚠️ Tentative {attempt} échouée pour {full_address}: {e}")  # Exception (ex. timeout)
            sleep(2)  # Attendre avant de retenter

    return pd.Series([None, None])  # Si aucune tentative réussit, renvoie None / None

# Application du géocodage à tout le detaframe
df[["latitude", "longitude"]] = df.apply(geocode_address, axis=1)   # df récupère latitude / longitude ligne par ligne (axis = 1)
# deux nouvelles colonnes sont créées

# Sauvegarde du résultat
df.to_csv("adresses_geocodees.csv", index=False)     # index=False : évite d’ajouter une colonne inutile (0,1,2…)

print("Géocodage terminé ! Fichier adresses_geocodees.csv créé.")
