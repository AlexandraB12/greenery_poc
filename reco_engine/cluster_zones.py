# reco_engine/cluster_zones.py
# - récupère matrice embeddings (N zones × 512)
# - standardiser
# - clustering (KMeans)
# - projection 2D (PCA)
# - export zone_clusters.csv
# - export zone_clusters_2d.csv
# - export clusters_pca.png
# - export cluster_summary.md

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


def cluster_zones(
    zones_csv: str = "zones_index.csv",
    out_csv: str = "zone_clusters.csv",
    out_plot_csv: str = "zone_clusters_2d.csv",
    out_plot_png: str = "clusters_pca.png",
    out_summary_md: str = "cluster_summary.md",
    k: int = 4,
    random_state: int = 42,
) -> None:
    if not os.path.exists(zones_csv):
        raise FileNotFoundError(f"Missing {zones_csv}. Run build_zones_index first.")

    df = pd.read_csv(zones_csv)
    emb_cols: List[str] = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("No embedding columns emb_* found in zones_index.csv")

    X = df[emb_cols].astype(float).values
    n = X.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 zones to cluster. Found n={n}.")

    # k adapté au nb de zones
    k_eff = min(int(k), n)
    if k_eff < 2:
        k_eff = 2

    # Standardisation
    Xs = StandardScaler().fit_transform(X)

    # KMeans
    km = KMeans(n_clusters=k_eff, n_init=10, random_state=random_state)
    labels = km.fit_predict(Xs)

    # CSV clusters
    out = df[["zone_id", "greenery_mean", "luminance_mean", "charm_mean"]].copy()
    out["cluster_id"] = labels
    out.to_csv(out_csv, index=False)
    print(f"✅ Saved: {out_csv} (k={k_eff}, n_zones={n})")

    # PCA 2D + CSV 2D
    pca = PCA(n_components=2, random_state=random_state)
    XY = pca.fit_transform(Xs)

    out2d = out[["zone_id", "cluster_id", "greenery_mean", "luminance_mean", "charm_mean"]].copy()
    out2d["x"] = XY[:, 0]
    out2d["y"] = XY[:, 1]
    out2d.to_csv(out_plot_csv, index=False)
    print(f"✅ Saved: {out_plot_csv} (PCA 2D)")

    # Plot PNG
    plt.figure()
    plt.scatter(out2d["x"], out2d["y"])
    for _, r in out2d.iterrows():
        plt.text(r["x"], r["y"], str(r["zone_id"]), fontsize=8)

    plt.title(f"PCA 2D of zone embeddings (k={k_eff})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_plot_png, dpi=200)
    plt.close()
    print(f"✅ Saved: {out_plot_png}")

    # Summary markdown (cluster_summary.md)
    grp = (
        out.groupby("cluster_id")
        .agg(
            n_zones=("zone_id", "count"),
            greenery_mean=("greenery_mean", "mean"),
            luminance_mean=("luminance_mean", "mean"),
            charm_mean=("charm_mean", "mean"),
        )
        .reset_index()
        .sort_values("cluster_id")
    )

    lines = []
    lines.append(f"# Cluster summary (k={k_eff}, n_zones={n})\n")
    lines.append("Moyennes calculées sur les scores *zone-level*.\n")

    for _, row in grp.iterrows():
        cid = int(row["cluster_id"])
        lines.append(f"## Cluster {cid} — {int(row['n_zones'])} zones")
        lines.append(f"- mean greenery: {row['greenery_mean']:.2f}")
        lines.append(f"- mean luminance: {row['luminance_mean']:.2f}")
        lines.append(f"- mean charm: {row['charm_mean']:.3f}")
        lines.append("")

    with open(out_summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Saved: {out_summary_md}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Cluster zones based on emb_* from zones_index.csv")
    ap.add_argument("--zones_csv", default="zones_index.csv")
    ap.add_argument("--out_csv", default="zone_clusters.csv")
    ap.add_argument("--out_plot_csv", default="zone_clusters_2d.csv")
    ap.add_argument("--out_plot_png", default="clusters_pca.png")
    ap.add_argument("--out_summary_md", default="cluster_summary.md")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    cluster_zones(
        zones_csv=args.zones_csv,
        out_csv=args.out_csv,
        out_plot_csv=args.out_plot_csv,
        out_plot_png=args.out_plot_png,
        out_summary_md=args.out_summary_md,
        k=args.k,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
