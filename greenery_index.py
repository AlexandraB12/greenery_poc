# Convertir image → HSV
# Détecter la plage du vert
# Calculer le ratio

import cv2
import os
import pandas as pd

IMAGE_DIR = "streetview_images"
OUTPUT_CSV = "greenery_index.csv"


def compute_greenery_ratio(img) -> float:
    """
    Calcule le ratio de végétation visible à partir d'une image OpenCV.
    """
    if img is None:
        raise ValueError("Image invalide")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = (35, 40, 40)
    upper_green = (85, 255, 255)

    mask = cv2.inRange(hsv, lower_green, upper_green)

    green_pixels = cv2.countNonZero(mask)
    total_pixels = img.shape[0] * img.shape[1]

    return green_pixels / total_pixels


if __name__ == "__main__":
    results = []

    for filename in sorted(os.listdir(IMAGE_DIR)):
        if not filename.endswith(".jpg"):
            continue

        path = os.path.join(IMAGE_DIR, filename)

        img = cv2.imread(path)
        if img is None:
            continue

        try:
            greenery_ratio = compute_greenery_ratio(img)
        except Exception as e:
            print(f"⚠️ Erreur sur {filename} : {e}")
            continue

        results.append({
            "image_id": filename.replace(".jpg", ""),
            "greenery_ratio": round(greenery_ratio, 4)
        })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"🌿 Greenery index calculé pour {len(df)} images → {OUTPUT_CSV}")
