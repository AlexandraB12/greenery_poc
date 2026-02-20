# Recherche par préférences utilisateur (ranking pondéré)

"""
🔜 RECO-23 — Recherche par préférences utilisateur (ranking pondéré)

But :
- On garde la reco "IA" (retrieval) basée sur l'embedding : reco_results.csv (top_k ou top_n)
- Puis on RE-RANKE ces candidats avec un score pondéré qui mélange :
    similarity (embedding) + greenery_mean + luminance_mean + charm_mean
- On exporte reco_weighted_results.csv avec score_final + explication adaptée.

Exécution (depuis la racine du projet greenery_poc/) :
    python -m reco_engine.weighted_ranker --help

* Structure globale du script

Le fichier reco_engine/weighted_ranker.py est organisé en 4 blocs :

    1. Imports + helpers (petites fonctions utilitaires)

    2. Explication pondérée (générer une phrase qui colle aux poids)

    3. run_weighted_rerank() : la fonction principale (pipeline)

    4. main() : lecture des arguments de commande et exécution

"""



from __future__ import annotations

import argparse   # permet de lancer le script en ligne de commande avec des paramètres (--sim_weight 0.7 etc.)
import os   # vérifier si un fichier existe
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import MinMaxScaler    # MinMaxScaler : normalisation en [0,1]
except ImportError as e:
    raise ImportError(
        "scikit-learn est requis pour ce module. Installe avec: pip install scikit-learn"
    ) from e


# -------------------------
# Helpers
# -------------------------

# fonction _safe_minmax : Calcule min/max, convertit la colonne en float et ignore les NaN (mask)
def _safe_minmax(col: pd.Series) -> pd.Series:
    """
    Normalise une colonne en [0,1] de façon robuste.
    - si la colonne est constante (min==max), renvoie 0.5 partout (ne discrimine pas)
    - ignore NaN (reste NaN)
    """
    x = col.astype(float)
    mask = x.notna()
    if mask.sum() == 0:
        return x  # tout NaN

    vmin = x[mask].min()
    vmax = x[mask].max()
    if np.isclose(vmin, vmax):
        out = pd.Series(np.nan, index=col.index, dtype=float)
        out[mask] = 0.5
        return out

    out = (x - vmin) / (vmax - vmin)
    return out

# Les poids : comment ils sont gérés -> calcul du score final, générer une explication orientée “ce qui a compté”
def _weights_dict(sim_weight: float, greenery_weight: float, luminance_weight: float, charm_weight: float) -> Dict[str, float]:
    return {
        "similarity": float(sim_weight),
        "greenery_mean": float(greenery_weight),
        "luminance_mean": float(luminance_weight),
        "charm_mean": float(charm_weight),
    }


def _dominant_preferences(weights: Dict[str, float]) -> List[Tuple[str, float]]:
    """
    Renvoie les préférences triées par poids décroissant (hors similarity si tu veux),
    utile pour écrire une explication orientée "préférence utilisateur".
    """
    items = [(k, v) for k, v in weights.items()]
    items.sort(key=lambda kv: kv[1], reverse=True)
    return items


def _pref_label(metric: str) -> str:
    if metric == "greenery_mean":
        return "🌿 verdure"
    if metric == "luminance_mean":
        return "☀️ luminosité"
    if metric == "charm_mean":
        return "✨ charme"
    if metric == "similarity":
        return "🧠 similarité IA"
    return metric


def _explain_weighted_row(
    row: pd.Series,
    weights: Dict[str, float],
    ref_row: pd.Series | None = None,
) -> str:
    """
    Crée une explication "pondérée" lisible.
    - row: la zone candidate (avec *_mean et similarity)
    - ref_row: la zone de référence si elle est présente (zone_id == ref_zone_id) dans le même CSV.
      Si absent -> on ne parle pas de delta, on parle surtout de "choisie car ..."
    """
    # On identifie ce qui a le plus de poids (préférence)
    prefs = _dominant_preferences(weights)
    top2 = [p for p in prefs if p[1] > 0][:2]
    if not top2:
        top2 = [("similarity", 1.0)]

    # Phrase de base sur la similarité (toujours utile)
    sim = row.get("similarity", np.nan)
    sim_txt = f"similarité IA {sim:.3f}" if pd.notna(sim) else "similarité IA n/a"

    # Si on a la ref, on peut exprimer des deltas
    if ref_row is not None:
        parts = [f"🎯 Re-rank pondéré ({sim_txt})."]
        # deltas bruts (non normalisés) pour rester métier
        for metric, w in top2:
            if metric == "similarity":
                continue
            v = row.get(metric, np.nan)
            r = ref_row.get(metric, np.nan)
            if pd.notna(v) and pd.notna(r):
                d = float(v) - float(r)
                sign = "+" if d >= 0 else ""
                parts.append(f"{_pref_label(metric)}: {sign}{d:.2f} vs ref")
            else:
                parts.append(f"{_pref_label(metric)}: n/a")
        # Conclusion en fonction de la préférence dominante (hors similarity si possible)
        dominant = top2[0][0]
        if dominant != "similarity":
            parts.append(f"👉 Favorisé car priorité = {_pref_label(dominant)} (poids {weights[dominant]:.2f}).")
        else:
            parts.append("👉 Favorisé car priorité = 🧠 similarité IA.")
        return " ".join(parts)

    # Sans ref : explication orientée "préférences"
    dominant = top2[0][0]
    if dominant == "similarity":
        return f"🎯 Re-rank pondéré : priorité 🧠 similarité IA → {sim_txt}."
    else:
        v = row.get(dominant, np.nan)
        v_txt = f"{v:.2f}" if pd.notna(v) else "n/a"
        return (
            f"🎯 Re-rank pondéré : priorité {_pref_label(dominant)} (poids {weights[dominant]:.2f}). "
            f"Zone retenue car {_pref_label(dominant)}={v_txt} tout en restant {sim_txt}."
        )


# -------------------------
# Core
# -------------------------

# fonction principale run_weighted_rerank()
def run_weighted_rerank(
    reco_csv: str = "reco_results.csv",
    zones_csv: str = "zones_index.csv",
    out_csv: str = "reco_weighted_results.csv",
    sim_weight: float = 0.55,
    greenery_weight: float = 0.25,
    luminance_weight: float = 0.10,
    charm_weight: float = 0.10,
    top_n: int = 200,
    keep_only_top_k: int = 50,
    ref_zone_id: str | None = None,
) -> pd.DataFrame:
    """
    Pipeline :
    1) Charge reco_results.csv (retrieval) → candidates
    2) Merge éventuel avec zones_index.csv (pour compléter les colonnes)
    3) Normalise (similarity + scores)
    4) Score pondéré → score_final
    5) Tri → export

    Paramètres :
    - top_n : garde au max top_n candidats du retrieval avant rerank (coûts)
    - keep_only_top_k : sortie finale (top_k du rerank)
    - ref_zone_id : si tu veux comparer explicitement au ref (sinon auto si zone_id==ref_lille existe)
    """
    if not os.path.exists(reco_csv):
        raise FileNotFoundError(f"Missing file: {reco_csv}")
    if not os.path.exists(zones_csv):
        # pas bloquant, mais on te le dit clairement
        print(f"⚠️ zones_csv introuvable: {zones_csv}. Je continue avec reco_csv uniquement.")
        zones_csv = ""

    reco = pd.read_csv(reco_csv)   # on récupère les zones triées par similarité (embedding) et leurs scores moyens

    # garde top_n si la reco_csv est déjà triée par similarity (souvent le cas)
    if top_n is not None and len(reco) > top_n:
        reco = reco.iloc[:top_n].copy()
        # Garder seulement top N candidats : on fait retrieval (embedding) → top 200
        # rerank (préférences) → top 10
        # C’est l’architecture classique des moteurs de recherche/reco.

    # Colonnes attendues
    needed = ["zone_id", "similarity", "greenery_mean", "luminance_mean", "charm_mean"]    # Vérifier que les colonnes nécessaires existent
    missing = [c for c in needed if c not in reco.columns]    # Si certaines manquent, on tente de les récupérer depuis zones_index.csv
    # Rend le script plus robuste : même si reco_results.csv est “minimal”, on peut compléter depuis l’index.

    # Si reco_results.csv ne contient pas tout, on complète depuis zones_index.csv
    if missing and zones_csv:
        zones = pd.read_csv(zones_csv)
        # on merge en gardant similarity venant de reco (retrieval)
        reco = reco.merge(
            zones[["zone_id", "greenery_mean", "luminance_mean", "charm_mean"]],
            on="zone_id",
            how="left",
            suffixes=("", "_zones"),
        )

    # Re-check colonnes
    for c in needed:
        if c not in reco.columns:
            raise ValueError(
                f"Colonne manquante: {c}. Assure-toi que {reco_csv} ou {zones_csv} contient {needed}."
            )

    # poids
    weights = _weights_dict(sim_weight, greenery_weight, luminance_weight, charm_weight)

    # Normalisation robuste (pas besoin de MinMaxScaler ici, on fait safe minmax pour gérer colonnes constantes)
    reco["similarity_norm"] = _safe_minmax(reco["similarity"])
    reco["greenery_norm"] = _safe_minmax(reco["greenery_mean"])
    reco["luminance_norm"] = _safe_minmax(reco["luminance_mean"])
    reco["charm_norm"] = _safe_minmax(reco["charm_mean"])
    # On obtient : similarity_norm ∈ [0,1], greenery_norm ∈ [0,1]. Donc les poids deviennent réellement comparables.

    # Calcul du score final pondéré : “ranking personnalisé”
    reco["score_final"] = (
        weights["similarity"] * reco["similarity_norm"].fillna(0.0)
        + weights["greenery_mean"] * reco["greenery_norm"].fillna(0.0)
        + weights["luminance_mean"] * reco["luminance_norm"].fillna(0.0)
        + weights["charm_mean"] * reco["charm_norm"].fillna(0.0)
    )
    # si sim_weight domine : on reste proche du ranking embedding
    # si greenery_weight monte : les zones les plus vertes remontent

    # Trouver la zone de référence (pour expliquer avec des deltas)
    # Il fait : si ref_lille existe → ref = ref_lille sinon → ref = la zone avec similarity la plus haute (souvent la ref)
    ref_row = None
    if ref_zone_id is None:
        # heuristique : si la ligne avec similarity==1.0 existe, c'est souvent la ref
        # sinon si "ref_lille" existe, on l'utilise
        if "ref_lille" in set(reco["zone_id"].astype(str).tolist()):
            ref_zone_id = "ref_lille"
        else:
            # similarity la plus haute
            ref_zone_id = str(reco.sort_values("similarity", ascending=False).iloc[0]["zone_id"])

    try:
        ref_row = reco[reco["zone_id"].astype(str) == str(ref_zone_id)].iloc[0]
    except Exception:
        ref_row = None

    # Explication pondérée (on garde l'explication existante aussi si elle existe)
    reco["weighted_explain"] = reco.apply(lambda r: _explain_weighted_row(r, weights, ref_row), axis=1)
    # Ce que fait _explain_weighted_row
        # regarde quels poids sont les plus grands
        # prend les 2 plus gros critères
        # construit un texte du genre : “Re-rank pondéré (similarité IA 0.97).” “🌿 verdure: +1.20 vs ref”
        # “👉 Favorisé car priorité = verdure (poids 0.40)”
    # On explique ce que le modèle a vraiment optimisé.

    # Tri par score_final (rerank)
    reco = reco.sort_values("score_final", ascending=False).reset_index(drop=True)

    # Sortie top_k
    if keep_only_top_k is not None and len(reco) > keep_only_top_k:
        reco_out = reco.iloc[:keep_only_top_k].copy()
    else:
        reco_out = reco.copy()

    # Colonnes de sortie (lisibles)
    cols = [
        "zone_id",
        "score_final",
        "similarity",
        "greenery_mean",
        "luminance_mean",
        "charm_mean",
        "weighted_explain",
    ]
    # bonus : conserver l'explain original si présent
    if "explain" in reco_out.columns:
        cols.insert(cols.index("weighted_explain"), "explain")

    # Ajout des poids (utile pour audit / traçabilité)
    reco_out["sim_weight"] = weights["similarity"]
    reco_out["greenery_weight"] = weights["greenery_mean"]
    reco_out["luminance_weight"] = weights["luminance_mean"]
    reco_out["charm_weight"] = weights["charm_mean"]

    reco_out[cols + ["sim_weight", "greenery_weight", "luminance_weight", "charm_weight"]].to_csv(out_csv, index=False)
    print(f"✅ Saved: {out_csv}")

    return reco_out


def main() -> None:
    ap = argparse.ArgumentParser(description="RECO-23 — Weighted reranking after embedding retrieval")
    ap.add_argument("--reco_csv", type=str, default="reco_results.csv")
    ap.add_argument("--zones_csv", type=str, default="zones_index.csv")
    ap.add_argument("--out_csv", type=str, default="reco_weighted_results.csv")

    ap.add_argument("--sim_weight", type=float, default=0.55)
    ap.add_argument("--greenery_weight", type=float, default=0.25)
    ap.add_argument("--luminance_weight", type=float, default=0.10)
    ap.add_argument("--charm_weight", type=float, default=0.10)

    ap.add_argument("--top_n", type=int, default=200, help="Nombre max de candidats à reranker (retrieval)")
    ap.add_argument("--top_k", type=int, default=50, help="Nombre de résultats finaux")
    ap.add_argument("--ref_zone_id", type=str, default=None, help="Optionnel: zone_id de référence (ex: ref_lille)")

    args = ap.parse_args()

    run_weighted_rerank(
        reco_csv=args.reco_csv,
        zones_csv=args.zones_csv,
        out_csv=args.out_csv,
        sim_weight=args.sim_weight,
        greenery_weight=args.greenery_weight,
        luminance_weight=args.luminance_weight,
        charm_weight=args.charm_weight,
        top_n=args.top_n,
        keep_only_top_k=args.top_k,
        ref_zone_id=args.ref_zone_id,
    )


if __name__ == "__main__":
    main()
