# Serveur API

from fastapi import FastAPI, UploadFile, File
from model import load_model
from predict import predict_image

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok", "message": "API Place To Be AI is running. Go to /docs"}

model = load_model("../models/streetview_model.pth")


def clamp(score, min_val=0, max_val=100):
    return max(min(score, max_val), min_val)


@app.post("/score")
async def score_image(file: UploadFile = File(...)):

    result = predict_image(model, file.file)

    # --- scores bruts ---
    dl_raw = result["dl_score_raw"]

    # --- score produit lisible ---
    charm_score = clamp(dl_raw * 300)   # ajuste si besoin

    # --- scores simples (provisoires mais utiles dès maintenant) ---
    greenery_score = clamp(result.get("greenery_score", charm_score * 0.6))
    luminance_score = clamp(result.get("luminance_score", charm_score * 1.1))

    return {
        "scores": {
            "charm": result["charm_score"],
            "greenery": result["greenery_score"],
            "luminance": result["luminance_score"]
        },
        "raw": {
            "dl_raw": result["dl_score_raw"]
        },
        "message": "OK"
    }
