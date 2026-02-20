# fait la partie “moteur de recommandation”
# lit zones_index.csv
# compare les embeddings
# renvoie Top K zones

# cosine similarity sur embeddings + explications

# reco_engine/similarity_search.py
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

from sklearn.metrics.pairwise import cosine_similarity

def load_profile(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _emb_from_row(row: pd.Series) -> Optional[np.ndarray]:
    emb_cols = [c for c in row.index if c.startswith("emb_")]
    if not emb_cols:
        return None
    emb_cols_sorted = sorted(emb_cols, key=lambda x: int(x.split("_")[1]))
    return row[emb_cols_sorted].to_numpy(dtype=np.float32)

def search_similar_zones(
    ref_profile_path: str,
    zones_index_csv: str,
    top_k: int = 10,
    exclude_same_zone: bool = True,
) -> pd.DataFrame:
    ref = load_profile(ref_profile_path)
    ref_emb = ref.get("embedding_mean", None)
    if ref_emb is None:
        raise RuntimeError("Reference profile has no embedding_mean. Your model must output embeddings for similarity search.")

    ref_vec = np.array(ref_emb, dtype=np.float32).reshape(1, -1)

    df = pd.read_csv(zones_index_csv)
    embs = []
    keep_rows = []
    for _, r in df.iterrows():
        emb = _emb_from_row(r)
        if emb is None:
            continue
        if exclude_same_zone and r["zone_id"] == ref.get("zone_id"):
            continue
        keep_rows.append(r)
        embs.append(emb)

    if not embs:
        raise RuntimeError("No embeddings found in zones_index.csv (no emb_* columns).")

    M = np.stack(embs, axis=0)  # (N, D)
    sims = cosine_similarity(ref_vec, M)[0]  # (N,)

    out = pd.DataFrame(keep_rows).copy()
    out["similarity"] = sims
    out = out.sort_values("similarity", ascending=False).head(top_k).reset_index(drop=True)

    # attach ref means for later explain
    out.attrs["ref_scores_mean"] = ref.get("scores_mean", {})
    return out
