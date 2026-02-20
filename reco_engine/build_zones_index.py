# parcourt toutes les zones dans data_zones/
# appelle build_zone_profile pour chaque zone
# produit :
# un dossier zone_profiles/ (1 json par zone)
# un fichier zones_index.csv (l’index global)
# C’est le “catalogue de zones”

from __future__ import annotations
import os, json, glob
from typing import Optional, List, Dict, Any

import pandas as pd
import torch

from .model_utils import load_torch_model
from .build_zone_profile import build_zone_profile, save_zone_profile

def build_zones_index(
    data_zones_dir: str,
    model_path: Optional[str],
    out_csv: str = "zones_index.csv",
    out_profiles_dir: str = "zone_profiles",
    device: str = "cpu",
    max_images_per_zone: Optional[int] = None,
) -> str:
    model = None     # Chargement du modèle
    if model_path:
        model = load_torch_model(model_path, device=device).to(device)

    # Découverte des zones
    zone_dirs = sorted([p for p in glob.glob(os.path.join(data_zones_dir, "*")) if os.path.isdir(p)])
    rows: List[Dict[str, Any]] = []

    os.makedirs(out_profiles_dir, exist_ok=True)

    for zd in zone_dirs:
        zone_id = os.path.basename(os.path.normpath(zd))   # nom du dossier
        prof = build_zone_profile(     # Création du profil de zone
            zone_dir=zd,
            zone_id=zone_id,
            model=model,
            device=device,
            max_images=max_images_per_zone,
        )

        prof_path = os.path.join(out_profiles_dir, f"{zone_id}.json")
        save_zone_profile(prof, prof_path)    # Sauvegarde du profil

        row = {           # Construction de la ligne d’index
            "zone_id": zone_id,
            "profile_path": prof_path,
            "n_images": prof.n_images,
            "greenery_mean": prof.scores_mean["greenery"],
            "luminance_mean": prof.scores_mean["luminance"],
            "charm_mean": prof.scores_mean["charm"],
            "greenery_std": prof.scores_std["greenery"],
            "luminance_std": prof.scores_std["luminance"],
            "charm_std": prof.scores_std["charm"],
        }

        if prof.embedding_mean is not None:      # Ajout de l’embedding
            for i, v in enumerate(prof.embedding_mean):
                row[f"emb_{i}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)     # Création de l’index final (index zones)
    df.to_csv(out_csv, index=False)
    return out_csv

if __name__ == "__main__":     # Mode script
    # Example CLI-like usage:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_zones_dir", default="data_zones")
    ap.add_argument("--model_path", default="models/streetview_model.pth")
    ap.add_argument("--out_csv", default="zones_index.csv")
    ap.add_argument("--out_profiles_dir", default="zone_profiles")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_images_per_zone", type=int, default=None)
    args = ap.parse_args()

    out = build_zones_index(
        data_zones_dir=args.data_zones_dir,
        model_path=args.model_path,
        out_csv=args.out_csv,
        out_profiles_dir=args.out_profiles_dir,
        device=args.device,
        max_images_per_zone=args.max_images_per_zone,
    )
    print(f"✅ wrote {out}")
