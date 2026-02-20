import os
import cv2
import numpy as np
import pandas as pd

from greenery_index import compute_greenery_ratio

# 1. Calcul de la luminosité (brightness)
def compute_brightness(image):
    """
    Luminosité moyenne de l'image (canal V de HSV)
    Proxy d'ouverture et de lisibilité de la rue
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2].mean() / 255.0
    return brightness

# Interprétation : faible → rue encaissée, ombragée; élevée → rue ouverte, façades claires

# 2. Calcul de l'ouverture du ciel (sky_ratio)
def compute_sky_ratio(image):
    """
    Approximation simple de l'ouverture du ciel
    Basée sur détection des zones bleues claires en haut de l'image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # plage approximative du ciel
    lower_blue = np.array([90, 20, 120])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # on se concentre sur la partie haute de l'image
    h = mask.shape[0]
    top_mask = mask[: int(h * 0.4), :]

    sky_ratio = np.count_nonzero(top_mask) / top_mask.size
    return sky_ratio

# Interprétation : faible → canyon urbain; élevé → ciel visible, respiration visuelle

# 3. Calcul de la complexité visuelle
def compute_visual_complexity(image):
    """
    Complexité visuelle basée sur la densité de contours
    Proxy de surcharge ou richesse visuelle
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    complexity = np.count_nonzero(edges) / edges.size
    return complexity

# Interprétation : trop faible → monotone; trop élevée → désordre perceptif
# = variable non linéairement désirable

# 4. Calcul de la régularité du bâti
def compute_building_regularity(image):
    """
    Régularité architecturale approximée
    Basée sur la dominance de lignes verticales / horizontales
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=50,
        maxLineGap=10
    )

    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    angles = np.array(angles)

    # proximité avec vertical / horizontal
    regularity = np.mean(
        (angles < 0.1) | (abs(angles - np.pi / 2) < 0.1)
    )

    return regularity

# Interprétation : élevé → façades alignées, répétitives; faible → hétérogénéité morphologique

# 5. Extraction globale (cœur de l'analyse)
def extract_metrics(image_path, image_id):
    image = cv2.imread(image_path)

    if image is None:
        return None

    return {
        "image_id": image_id,
        "greenery_ratio": compute_greenery_ratio(image),
        "brightness": compute_brightness(image),
        "sky_ratio": compute_sky_ratio(image),
        "visual_complexity": compute_visual_complexity(image),
        "building_regularity": compute_building_regularity(image),
    }

# Boucle principale + CSV
IMAGE_DIR = "streetview_images"
OUTPUT_CSV = "metrics_output.csv"

results = []

for filename in sorted(os.listdir(IMAGE_DIR)):
    if not filename.endswith(".jpg"):
        continue

    image_id = filename.replace(".jpg", "")
    image_path = os.path.join(IMAGE_DIR, filename)

    metrics = extract_metrics(image_path, image_id)
    if metrics:
        results.append(metrics)

df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ {len(df)} images traitées")
print(f"📄 Fichier généré : {OUTPUT_CSV}")
