"""
streetview_download.py
----------------------
Script final pour télécharger des images Google Street View à partir
d'un CSV géocodé (78 adresses validées pour le POC).

Fonctionnalités incluses :
- Filtrage des lignes valides (nominatim_ok + manual_fix)
- Retry automatique si l'image n'est pas téléchargée
- Gestion des logs et sauvegarde dans un dossier dédié
- Nommage clair des fichiers
"""

import os
import pandas as pd
import requests
import time

# ------------------------------
# CONFIGURATION
# ------------------------------

# Chemin vers le CSV final nettoyé
CSV_PATH = "adresses_final_streetview.csv"

# Clé API Google Street View (mettre ta clé ici)
API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GOOGLE_MAPS_API_KEY env var")

# Dossier où les images seront sauvegardées
OUTPUT_DIR = "streetview_images"

# Taille et format des images
IMG_SIZE = "640x640"  # max gratuit : 640x640

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # secondes entre chaque retry

# Crée le dossier si non existant
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# CHARGEMENT DU CSV ET FILTRAGE
# ------------------------------

df = pd.read_csv(CSV_PATH)

# On ne garde que les lignes valides
df_ok = df[df["geocoding_source"].isin(["nominatim_ok", "manual_fix"])].reset_index(drop=True)

print(f"📸 {len(df_ok)} adresses prêtes pour Street View")

# ------------------------------
# FONCTION DE TÉLÉCHARGEMENT
# ------------------------------

def download_streetview(lat, lon, filename, retry=0):
    """
    Télécharge une image Street View pour un point donné.

    Arguments :
    - lat, lon : coordonnées GPS
    - filename : chemin complet du fichier image
    - retry : compteur de retry en cas d'erreur
    """
    # Endpoint Google Street View API
    url = (
        "https://maps.googleapis.com/maps/api/streetview"
        f"?size={IMG_SIZE}&location={lat},{lon}&key={API_KEY}"
    )

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"✔ Image sauvegardée : {filename}")
            return True
        else:
            print(f"⚠ Erreur HTTP {response.status_code} pour {filename}")
            if retry < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
                return download_streetview(lat, lon, filename, retry + 1)
            else:
                print(f"❌ Échec après {MAX_RETRIES} tentatives pour {filename}")
                return False
    except Exception as e:
        print(f"⚠ Exception pour {filename} : {e}")
        if retry < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
            return download_streetview(lat, lon, filename, retry + 1)
        else:
            print(f"❌ Échec après {MAX_RETRIES} tentatives pour {filename}")
            return False

# ------------------------------
# BOUCLE PRINCIPALE
# ------------------------------

for idx, row in df_ok.iterrows():
    img_filename = os.path.join(OUTPUT_DIR, f"{idx+1:03d}.jpg")  # 001.jpg, 002.jpg, ...
    lat = row["latitude"]
    lon = row["longitude"]
    download_streetview(lat, lon, img_filename)

print("✅ Téléchargement terminé – images prêtes pour le pipeline CV")
