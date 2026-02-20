# Objectif : détecter automatiquement :
#
# zones pauvres en images
#
# variance extrême
#
# embedding manquant (emb_* absent / NaN)
#
# scores manquants
# Commande : python -m reco_engine.quality_monitor --min_images 15 --std_threshold 1.5



from __future__ import annotations

import argparse
import os
import pandas as pd


def monitor(
    zones_csv: str = "zones_index.csv",
    out_csv: str = "outputs/quality_report.csv",
    out_md: str = "outputs/quality_report.md",
    min_images: int = 15,
    std_threshold: float = 1.5,
) -> None:
    if not os.path.exists(zones_csv):
        raise FileNotFoundError(f"Missing {zones_csv}")

    df = pd.read_csv(zones_csv)

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if emb_cols:
        df["embedding_missing"] = df[emb_cols].isna().any(axis=1)
    else:
        df["embedding_missing"] = True

    if "n_images" not in df.columns:
        df["n_images"] = None

    flags = []
    for _, r in df.iterrows():
        reasons = []
        if pd.notna(r.get("n_images")) and float(r["n_images"]) < min_images:
            reasons.append(f"low_images<{min_images}")

        for m in ["greenery_std", "luminance_std", "charm_std"]:
            if m in df.columns and pd.notna(r.get(m)) and float(r[m]) > std_threshold:
                reasons.append(f"high_{m}>{std_threshold}")

        if bool(r.get("embedding_missing")):
            reasons.append("embedding_missing")

        for m in ["greenery_mean", "luminance_mean", "charm_mean"]:
            if m in df.columns and pd.isna(r.get(m)):
                reasons.append(f"missing_{m}")

        flags.append(";".join(reasons))

    out = df[["zone_id"]].copy()
    out["flags"] = flags
    out["has_issue"] = out["flags"].astype(str).str.len() > 0
    out = out.sort_values(["has_issue", "zone_id"], ascending=[False, True])

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"✅ Saved: {out_csv}")

    issues = out[out["has_issue"]]
    ok = out[~out["has_issue"]]

    md = []
    md.append("# Quality monitoring\n")
    md.append(f"- zones total: **{len(out)}**")
    md.append(f"- zones with issues: **{len(issues)}**\n")
    md.append("## Issues\n")
    md.append(issues.to_markdown(index=False) if not issues.empty else "_No issues found._")
    md.append("\n## OK\n")
    md.append(ok.to_markdown(index=False) if not ok.empty else "_None._")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"✅ Saved: {out_md}")


def main() -> None:
    ap = argparse.ArgumentParser(description="RECO-26 — Profile quality monitoring")
    ap.add_argument("--zones_csv", default="zones_index.csv")
    ap.add_argument("--min_images", type=int, default=15)
    ap.add_argument("--std_threshold", type=float, default=1.5)
    args = ap.parse_args()

    monitor(
        zones_csv=args.zones_csv,
        min_images=args.min_images,
        std_threshold=args.std_threshold,
    )


if __name__ == "__main__":
    main()
