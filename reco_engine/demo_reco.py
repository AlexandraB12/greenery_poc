# Script “démo” : construit l’index si besoin
# calcule la zone de référence
# renvoie le top 10 + explications
# exporte un reco_results.csv
# c'est le bouton pour démontrer le concept

# reco_engine/demo_reco.py
from __future__ import annotations
import os
import pandas as pd

from .model_utils import load_torch_model
from .build_zone_profile import build_zone_profile, save_zone_profile
from .build_zones_index import build_zones_index
from .similarity_search import search_similar_zones, load_profile
from .explain import explain_zone

def run_demo(
    data_zones_dir: str = "data_zones",
    ref_zone_dir: str = "data_zones/ref_lille",
    model_path: str = "models/streetview_model.pth",
    device: str = "cpu",
    top_k: int = 10,
    max_images_per_zone: int | None = 30,
):
    # 1) Build index for all zones
    zones_csv = build_zones_index(
        data_zones_dir=data_zones_dir,
        model_path=model_path,
        out_csv="zones_index.csv",
        out_profiles_dir="zone_profiles",
        device=device,
        max_images_per_zone=30,
    )

    # 2) Build ref profile
    model = load_torch_model(model_path, device=device).to(device)

    ref_prof = build_zone_profile(
        ref_zone_dir,
        zone_id="ref_lille",
        model=model,
        device=device,
    )

    # 🔴 IMPORTANT : définir le path AVANT utilisation
    ref_prof_path = os.path.join("zone_profiles", "__ref_lille.json")

    save_zone_profile(ref_prof, ref_prof_path)

    # 3) Similarity search
    results = search_similar_zones(ref_prof_path, zones_csv, top_k=top_k)
    # results est un DataFrame (tableau) avec zone_id, similarity, profile_path
    # récupère les scores de référence
    ref_scores = load_profile(ref_prof_path).get("scores_mean", {})

    # 4) Explain results
    expl = []
    for _, row in results.iterrows():
        cand_profile_path = row["profile_path"]
        cand_scores = load_profile(cand_profile_path).get("scores_mean", {})
        expl.append(explain_zone(ref_scores, cand_scores))
    # Ajout au tableau
    results["explain"] = expl

    # 5) Print nice
    cols = ["zone_id", "similarity", "greenery_mean", "luminance_mean", "charm_mean", "explain"]
    print("\n🏆 Top similar zones\n")
    print(results[cols].to_string(index=False))

    # 6) Save output
    out_path = "reco_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\n✅ Saved: {out_path}")

# Mode script exécutable
if __name__ == "__main__":
    run_demo()
