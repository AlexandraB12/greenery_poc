# reco_engine/run_weighted_scenarios.py
from __future__ import annotations

import os

from .weighted_ranker import run_weighted_rerank
from .make_similarity_map import make_map


def run_all() -> None:
    os.makedirs("outputs", exist_ok=True)

    scenarios = [
        ("green_priority", dict(sim_weight=0.40, greenery_weight=0.40, luminance_weight=0.10, charm_weight=0.10)),
        ("charm_priority", dict(sim_weight=0.40, greenery_weight=0.10, luminance_weight=0.10, charm_weight=0.40)),
        ("light_priority", dict(sim_weight=0.40, greenery_weight=0.10, luminance_weight=0.40, charm_weight=0.10)),
        ("baseline", dict(sim_weight=0.55, greenery_weight=0.25, luminance_weight=0.10, charm_weight=0.10)),
    ]

    for name, w in scenarios:
        out_csv = f"outputs/reco_weighted_{name}.csv"
        run_weighted_rerank(
            reco_csv="reco_results.csv",
            zones_csv="zones_index.csv",
            out_csv=out_csv,
            top_n=200,
            keep_only_top_k=50,
            ref_zone_id="ref_lille",
            **w,
        )

        out_map = f"outputs/zones_map_{name}.html"
        make_map(
            zones_geo_path="zones_geo.csv",
            reco_results_path=out_csv,
            out_html=out_map,
        )

    print("✅ Done: outputs/reco_weighted_*.csv + outputs/zones_map_*.html")


if __name__ == "__main__":
    run_all()
