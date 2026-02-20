# prepare_dl_data.py
import pandas as pd
import os

# -------------------------------
# 1️⃣ Charger le CSV existant
# -------------------------------
csv_path = "metrics_output.csv"
df = pd.read_csv(csv_path)
print(f"ℹ️ CSV chargé : {len(df)} lignes")

# -------------------------------
# 2️⃣ Lister les images disponibles
# -------------------------------
img_dir = "streetview_images"
image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
print(f"ℹ️ {len(image_files)} images trouvées dans {img_dir}")

# -------------------------------
# 3️⃣ Vérifier correspondance CSV ↔ images
# -------------------------------
if len(image_files) == 0:
    raise ValueError("❌ Aucun fichier image trouvé !")

if len(image_files) < len(df):
    print(f"⚠️ Moins d’images ({len(image_files)}) que de lignes CSV ({len(df)})")
    print(f"Seules les {len(image_files)} premières lignes du CSV seront utilisées.")
    df = df.iloc[:len(image_files)]  # tronquer le CSV

elif len(image_files) > len(df):
    print(f"⚠️ Plus d’images ({len(image_files)}) que de lignes CSV ({len(df)})")
    print(f"Seules les {len(df)} premières images seront utilisées.")
    image_files = image_files[:len(df)]

# -------------------------------
# 4️⃣ Associer les images aux lignes
# -------------------------------
df["image_filename"] = image_files[:len(df)]
print(f"ℹ️ {len(image_files)} images trouvées dans {img_dir}")

# -------------------------------
# 5️⃣ Sauvegarder le nouveau CSV
# -------------------------------
output_csv = "metrics_output_with_images.csv"
df.to_csv("metrics_output_with_images.csv", index=False)
print(f"✅ CSV prêt pour le DL : {output_csv}")


