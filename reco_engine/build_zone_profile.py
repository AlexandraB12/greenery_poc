# Fichier le plus important.
#
# Il fait : dossier d’images d’une zone → profil JSON
# un dossier d’images → signature de zone (embedding_mean + scores mean/std)

from __future__ import annotations
import os, glob, json
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ✅ IMPORTANT : on utilise le "vrai" modèle de ton backend (CNNScore)
# et on récupère à la fois : charme + embedding 512D
from .model_utils import extract_score_and_embedding
from .vision_features import luminance_score, greenery_ratio

# ZoneProfile (dataclass) = structure standard du profil
@dataclass
class ZoneProfile:
    zone_id: str     # zone_id : id de la zone
    embedding_mean: Optional[List[float]]     # embedding_mean : vecteur moyen (ou None si pas de modèle)
    scores_mean: Dict[str, Optional[float]]   # scores_mean : moyennes des scores
    scores_std: Dict[str, Optional[float]]    # scores_std : variabilité (homogène vs contrasté)
    n_images: int    # n_images : nombre d’images utilisées

# Fonctions utilitaires images
# def _load_image() : Charge l’image, la met en RGB (3 canaux), resize en carré (224×224)
def _load_image(path: str, size: int = 224) -> np.ndarray:    #
    img = Image.open(path).convert("RGB").resize((size, size))
    return np.array(img, dtype=np.uint8)    # Retourne un tableau NumPy (H, W, 3) en uint8 (0..255)

# _to_tensor() : Convertit l’image NumPy en tenseur PyTorch
def _to_tensor(rgb: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(rgb).float() / 255.0
    x = x.permute(2, 0, 1)  # (3,H,W)    # change l’ordre des dimensions : NumPy : (H, W, C), PyTorch : (C, H, W)

    # ImageNet normalization (souvent indispensable)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = (x - mean) / std

    return x.unsqueeze(0)   # (1,3,H,W)

# build_zone_profile() = fonction principale
def build_zone_profile(
    zone_dir: str,
    zone_id: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,   # model : si None, uniquement des scores heuristiques (verdure + luminance)
    device: str = "cpu",
    image_size: int = 224,
    max_images: Optional[int] = None,   # max_images : limite pour aller vite
) -> ZoneProfile:
    # Étape A — identifier la zone (Si zone_id non fourni → on prend le nom du dossier)
    zone_id = zone_id or os.path.basename(os.path.normpath(zone_dir))
    # Étape B — lister les images (récupère toutes les images du dossier)
    paths = sorted(glob.glob(os.path.join(zone_dir, "*.jpg")) + glob.glob(os.path.join(zone_dir, "*.jpeg")) + glob.glob(os.path.join(zone_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No images found in {zone_dir}")  # OK : zone vide = erreur explicite.

    if max_images is not None:    # Étape C — limiter le nombre d’images
        paths = paths[:max_images]
    # pratique pour : tests rapides, contrôle du coût, homogénéiser le nombre d’images par zone

    # Étape D — listes d’accumulation (calcul des valeurs par image, puis faire moyenne/std à la fin)
    greenery_vals, luminance_vals, charm_vals = [], [], []
    emb_list = []

    # Étape E — boucle sur chaque image (avec barre de progression)
    # tqdm = barre de progression (utile quand il y a beaucoup d’images)
    for p in tqdm(paths, desc=f"Profiling {zone_id}", leave=False):
        # Calcul des scores simples
        rgb = _load_image(p, size=image_size)
        # même sans modèle, on peut déjà profiler la zone
        greenery_vals.append(greenery_ratio(rgb))
        luminance_vals.append(luminance_score(rgb))

        # Si modèle est fourni : charm + embedding
        if model is not None:
            x = _to_tensor(rgb).to(device)

            # ✅ ICI la différence :
            # On récupère les 2 sorties "propres" :
            # - score charme (B,)
            # - embedding (B, 512)
            score, emb = extract_score_and_embedding(model, x)

            # Charm
            # (batch=1 donc on prend item())
            charm_vals.append(float(score.detach().cpu().item()))

            # Embedding
            emb_np = emb.detach().cpu().numpy().reshape(-1)   # on “flatten” l’embedding en vecteur 1D et on le stocke
            emb_list.append(emb_np)

    # Agrégation : moyenne & écart-type (fonction interne)
    def mean_std(arr: List[float]) -> (Optional[float], Optional[float]):
        if not arr:
            return None, None
        a = np.array(arr, dtype=np.float32)
        return float(a.mean()), float(a.std())   # si aucune valeur (pas de modèle donc pas de charm) → retourne None

    greenery_mean, greenery_std = mean_std(greenery_vals)
    luminance_mean, luminance_std = mean_std(luminance_vals)
    charm_mean, charm_std = mean_std(charm_vals)

    # Embedding moyen de la zone (si disponible)
    embedding_mean = None
    if emb_list:
        E = np.stack(emb_list, axis=0)   # E devient une matrice : (n_images, emb_dim)
        # on prend la moyenne sur les images, chaque zone devient un seul vecteur = sa “signature d’ambiance”
        embedding_mean = E.mean(axis=0).astype(np.float32).tolist()

    # Retour de l’objet ZoneProfile
    return ZoneProfile(
        zone_id=zone_id,
        embedding_mean=embedding_mean,
        scores_mean={"greenery": greenery_mean, "luminance": luminance_mean, "charm": charm_mean},
        scores_std={"greenery": greenery_std, "luminance": luminance_std, "charm": charm_std},
        n_images=len(paths),
    )    # sortie standard, propre, sérialisable

# save_zone_profile — sauvegarde en JSON
def save_zone_profile(profile: ZoneProfile, out_path: str) -> None:
    payload = {
        "zone_id": profile.zone_id,
        "embedding_mean": profile.embedding_mean,
        "scores_mean": profile.scores_mean,
        "scores_std": profile.scores_std,
        "n_images": profile.n_images,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
