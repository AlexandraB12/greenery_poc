# reco_engine/weighted_ranker_cli.py
from __future__ import annotations

import argparse
import os

from .weighted_ranker import run_weighted_rerank


def main() -> None:
    ap = argparse.ArgumentParser(
        description="RECO-23 — Weighted reranking (CLI slider-friendly aliases)"
    )

    # Inputs/outputs
    ap.add_argument("--reco_csv", type=str, default="reco_results.csv",
                    help="retrieval file (reco_results.csv or any reco_*.csv with zone_id/similarity)")
    ap.add_argument("--zones_csv", type=str, default="zones_index.csv",
                    help="zones index (scores + emb_*)")
    ap.add_argument("--out_csv", type=str, default="reco_weighted_results.csv",
                    help="output CSV")
    ap.add_argument("--top_n", type=int, default=200,
                    help="max candidates kept from reco_csv before rerank")
    ap.add_argument("--top_k", type=int, default=50,
                    help="final number of rows in output")
    ap.add_argument("--ref_zone_id", type=str, default=None,
                    help="optional reference zone_id (ex: ref_lille) for delta explanations")

    # Canonical weights
    ap.add_argument("--sim_weight", type=float, default=0.55)
    ap.add_argument("--greenery_weight", type=float, default=0.25)
    ap.add_argument("--luminance_weight", type=float, default=0.10)
    ap.add_argument("--charm_weight", type=float, default=0.10)

    # ✅ Aliases (front slider style)
    ap.add_argument("--sim", dest="sim_weight", type=float,
                    help="alias of --sim_weight")
    ap.add_argument("--green", dest="greenery_weight", type=float,
                    help="alias of --greenery_weight")
    ap.add_argument("--light", dest="luminance_weight", type=float,
                    help="alias of --luminance_weight")
    ap.add_argument("--charm", dest="charm_weight", type=float,
                    help="alias of --charm_weight")
    ap.add_argument("--out", dest="out_csv", type=str,
                    help="alias of --out_csv")

    args = ap.parse_args()

    # Create output folder if user passes something like outputs/my.csv
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

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
