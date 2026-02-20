# lit un fichier reco (standard ou pondéré)
#
# écrit un MD “business-ready”
#
# option : export PDF simple (si reportlab dispo)
# Commande : python -m reco_engine.make_reco_report --reco outputs/my_reco.csv --out_md outputs/my_reco_report.md --out_pdf outputs/my_reco_report.pdf --top_k 10

from __future__ import annotations

import argparse
import os
import pandas as pd

def make_md_report(reco_csv: str, out_md: str, top_k: int = 10) -> None:
    df = pd.read_csv(reco_csv)
    df = df.head(top_k).copy()

    lines = []
    lines.append("# Reco report\n")
    lines.append(f"- Source: `{reco_csv}`")
    lines.append(f"- Top K: **{top_k}**\n")

    cols = [c for c in ["zone_id", "score_final", "similarity", "greenery_mean", "luminance_mean", "charm_mean", "weighted_explain", "explain"] if c in df.columns]
    lines.append("## Résultats\n")
    lines.append(df[cols].to_markdown(index=False))

    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✅ Saved: {out_md}")

def md_to_pdf_simple(md_path: str, out_pdf: str) -> None:
    # PDF minimal : texte brut (suffit pour démo)
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)

    c = canvas.Canvas(out_pdf, pagesize=A4)
    width, height = A4
    y = height - 40
    for line in text.splitlines():
        if y < 40:
            c.showPage()
            y = height - 40
        c.drawString(40, y, line[:120])
        y -= 14
    c.save()
    print(f"✅ Saved: {out_pdf}")

def main() -> None:
    ap = argparse.ArgumentParser(description="RECO-25 — auto report (MD + optional PDF)")
    ap.add_argument("--reco", default="outputs/my_reco.csv", help="reco csv (weighted or not)")
    ap.add_argument("--out_md", default="outputs/reco_report.md")
    ap.add_argument("--out_pdf", default="", help="if set, export pdf too")
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    make_md_report(args.reco, args.out_md, top_k=args.top_k)
    if args.out_pdf:
        md_to_pdf_simple(args.out_md, args.out_pdf)

if __name__ == "__main__":
    main()
