# vérifie que changer les poids change bien l'ordre

import os
import pandas as pd

from reco_engine.weighted_ranker import run_weighted_rerank


def test_weight_change_affects_ranking(tmp_path):
    """
    Le top-1 peut rester identique (zone dominante),
    donc on teste que le ranking OU les scores changent au moins pour une zone.
    """

    out1 = tmp_path / "test_reco_1.csv"
    out2 = tmp_path / "test_reco_2.csv"

    df1 = run_weighted_rerank(
        reco_csv="reco_results.csv",
        zones_csv="zones_index.csv",
        out_csv=str(out1),
        sim_weight=0.9,
        greenery_weight=0.05,
        luminance_weight=0.03,
        charm_weight=0.02,
        top_n=200,
        keep_only_top_k=50,
    )

    df2 = run_weighted_rerank(
        reco_csv="reco_results.csv",
        zones_csv="zones_index.csv",
        out_csv=str(out2),
        sim_weight=0.2,
        greenery_weight=0.6,
        luminance_weight=0.15,
        charm_weight=0.05,
        top_n=200,
        keep_only_top_k=50,
    )

    # 1) même set de zones attendu (sauf si keep_only_top_k coupe différemment)
    # on compare sur l'intersection pour être robuste
    common = set(df1["zone_id"]).intersection(set(df2["zone_id"]))
    assert len(common) >= 2

    a = df1[df1["zone_id"].isin(common)].copy()
    b = df2[df2["zone_id"].isin(common)].copy()

    # 2) vérifier que les SCORE_FINAL changent pour au moins une zone
    a_scores = a.set_index("zone_id")["score_final"].sort_index()
    b_scores = b.set_index("zone_id")["score_final"].sort_index()

    score_diff = (a_scores - b_scores).abs()
    assert score_diff.max() > 1e-9, "Les scores finaux sont identiques malgré des poids différents."

    # 3) vérifier que le RANG change pour au moins une zone
    a_rank = a.set_index("zone_id")["score_final"].rank(ascending=False, method="min")
    b_rank = b.set_index("zone_id")["score_final"].rank(ascending=False, method="min")

    rank_diff = (a_rank - b_rank).abs()
    assert rank_diff.max() >= 1, "Aucun changement de rang détecté malgré des poids différents."

