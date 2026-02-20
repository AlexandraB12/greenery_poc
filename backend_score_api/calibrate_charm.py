# backend_score_api/calibrate_charm.py
import json, os
import numpy as np
from model import load_model
from predict import transform  # on réutilise la même transform
from PIL import Image
import torch

IMG_DIR = "../streetview_images"  # adapte si besoin
MODEL_PATH = "../models/streetview_model.pth"
OUT = "calibration.json"

def compute_raw_scores(model):
    device = next(model.parameters()).device
    model.eval()

    scores = []
    files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))])

    with torch.no_grad():
        for fn in files:
            img = Image.open(os.path.join(IMG_DIR, fn)).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            raw = float(model(x).item())
            scores.append(raw)
    return np.array(scores, dtype=np.float32)

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    raw = compute_raw_scores(model)

    p05 = float(np.percentile(raw, 5))
    p95 = float(np.percentile(raw, 95))

    payload = {"p05": p05, "p95": p95}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("✅ calibration.json créé:", payload)
