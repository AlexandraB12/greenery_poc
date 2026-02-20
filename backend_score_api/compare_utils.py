# compare_utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

def delta_label(d: float) -> str:
    """Retourne un qualificatif lisible selon la valeur absolue du delta."""
    ad = abs(d)
    if ad > 10:
        return "nettement"
    if ad >= 5:
        return "un peu"
    return "similaire"

def signed_delta(d: float) -> str:
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}"

def compare_sentence(a: Dict, b: Dict, name_a="A", name_b="B") -> str:
    """
    a,b: dict avec scores {charm, greenery, luminance}
    Retourne une phrase prête pour UI.
    """
    parts = []

    for metric, label_fr in [
        ("greenery", "verte"),
        ("luminance", "lumineuse"),
        ("charm", "charmante"),
    ]:
        da = a["scores"][metric]
        db = b["scores"][metric]
        d = db - da  # B - A

        qual = delta_label(d)
        if abs(d) < 1e-6:
            parts.append(f"{name_a} et {name_b} sont {qual} sur la dimension {label_fr}.")
            continue

        winner = name_b if d > 0 else name_a
        loser = name_a if d > 0 else name_b
        parts.append(
            f"{winner} est {qual} plus {label_fr} ({signed_delta(abs(d))}) que {loser}"
        )

    # petite mise en forme finale
    # Exemple: "B est nettement plus verte (+19.0) que A, ... "
    sentence = ", ".join(parts) + "."
    return sentence

# -------------------------------------------------------
# Ranking par objectif métier (pondérations)
# -------------------------------------------------------

WEIGHTS: Dict[str, Dict[str, float]] = {
    "coffee_shop": {"charm": 0.45, "greenery": 0.25, "luminance": 0.15, "regularity_penalty": 0.15},
    "premium_shop": {"charm": 0.55, "luminance": 0.20, "greenery": 0.15, "regularity_penalty": 0.10},
    "local_services": {"luminance": 0.35, "greenery": 0.30, "charm": 0.20, "regularity_penalty": 0.15},
    "coworking": {"luminance": 0.45, "greenery": 0.20, "charm": 0.10, "regularity_penalty": 0.25},
}

def compute_fit_score(item: Dict, objective: str) -> float:
    """
    item: sortie de scoring par image (contient scores + éventuellement building_regularity)
    objective: clé dans WEIGHTS
    Retourne un score d'adéquation 0..100
    """
    w = WEIGHTS[objective]
    s = item["scores"]

    # score principal
    fit = (
        w.get("charm", 0) * s.get("charm", 0)
        + w.get("greenery", 0) * s.get("greenery", 0)
        + w.get("luminance", 0) * s.get("luminance", 0)
    )

    # pénalité de monotonie (si tu l’as)
    # NB: building_regularity est dans tes métriques tabulaires, mais pas dans ton endpoint actuel.
    # On prévoit la compatibilité : si absent => pénalité = 0
    reg = item.get("metrics", {}).get("building_regularity", None)
    if reg is not None:
        # building_regularity ~ 0..1 => on pénalise jusqu’à 15–25 points selon objectif
        fit -= w.get("regularity_penalty", 0) * (reg * 100)

    # clamp
    return max(0.0, min(100.0, float(fit)))

def rank_items(items: Dict[str, Dict], objective: str) -> Dict:
    """
    items: dict { "A": resultA, "B": resultB, ... }
    retourne ranking + fit_scores
    """
    scored = []
    for name, res in items.items():
        fit = compute_fit_score(res, objective)
        scored.append((name, fit, res))

    scored.sort(key=lambda x: x[1], reverse=True)

    return {
        "objective": objective,
        "ranking": [
            {"name": name, "fit_score": round(fit, 1), "scores": res["scores"]}
            for name, fit, res in scored
        ],
    }
