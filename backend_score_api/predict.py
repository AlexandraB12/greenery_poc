# fonction scoring

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import json, os

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def calibrate_charm(score_raw: float, calib_path="calibration.json") -> float:
    # fallback si pas de calibration: ancien *300
    if not os.path.exists(calib_path):
        return clamp(score_raw * 300)

    with open(calib_path, "r", encoding="utf-8") as f:
        calib = json.load(f)

    p05 = calib["p05"]
    p95 = calib["p95"]

    if p95 <= p05:
        return clamp(score_raw * 300)

    scaled = (score_raw - p05) / (p95 - p05) * 100.0
    return clamp(scaled)


def clamp(x, a=0, b=100):
    return max(a, min(b, x))

def compute_luminance_score(img: Image.Image) -> float:
    """
    Score luminosité simple (baseline) :
    - on convertit en niveaux de gris
    - on prend la moyenne (0..255) -> (0..100)
    """
    gray = img.convert("L")
    mean_val = float(np.array(gray).mean())  # 0..255
    return clamp((mean_val / 255.0) * 100.0)

def compute_greenery_score(img: Image.Image) -> float:
    """
    Score verdeur simple (baseline) :
    - on compte les pixels "plutôt verts" via un seuil RGB
    - ça marche correctement pour un MVP, mais ce n'est pas une segmentation propre
    """
    arr = np.array(img).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # pixels verts typiques : G dominant + pas trop sombre
    green_mask = (g > r + 15) & (g > b + 15) & (g > 60)
    ratio = float(green_mask.mean())  # 0..1
    return clamp(ratio * 100.0)

def predict_image(model, image_file):
    img = Image.open(image_file).convert("RGB")

    # --- Baselines "lisibles produit" ---
    luminance_score = compute_luminance_score(img)
    greenery_score = compute_greenery_score(img)

    # --- DL score (charme) ---
    device = next(model.parameters()).device
    model.eval()

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        score_raw = model(x).item()

    # calibration simple -> 0..100
    charm_score = calibrate_charm(score_raw)

    return {
        "dl_score_raw": float(score_raw),
        "charm_score": round(charm_score, 1),
        "greenery_score": round(greenery_score, 1),
        "luminance_score": round(luminance_score, 1),
    }

