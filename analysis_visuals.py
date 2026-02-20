# analysis_visuals.py
# commande pour lancer : python analysis_visuals.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------
# Config
# ---------------------------
FIG_DIR = os.path.join("outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set(style="whitegrid")


# ---------------------------
# Helpers
# ---------------------------
def _find_cluster_col(df: pd.DataFrame) -> str:
    """Find a cluster column name in a dataframe."""
    candidates = ["cluster", "cluster_id", "cluster_label", "label", "kmeans_cluster", "cluster_kmeans"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Aucune colonne cluster trouvée. Colonnes dispo: {list(df.columns)}. "
        "Renomme ta colonne en 'cluster' ou ajoute-la dans candidates."
    )


def _ensure_xy(df: pd.DataFrame) -> None:
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(
            f"Colonnes PCA introuvables. Attendu: 'x' et 'y'. Reçu: {list(df.columns)}. "
            "Vérifie zone_clusters_2d.csv."
        )


def _save_close(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def radar_from_row(zones_df: pd.DataFrame, zone_id: str, title: str, filename: str):
    """
    Radar chart normalisé (0..1) pour éviter que luminance domine.
    """
    if zone_id not in set(zones_df["zone_id"].astype(str)):
        raise ValueError(f"zone_id '{zone_id}' introuvable. Exemples: {zones_df['zone_id'].astype(str).head().tolist()}")

    row = zones_df.loc[zones_df["zone_id"].astype(str) == str(zone_id)].iloc[0]

    labels = ["Greenery", "Luminance", "Charm"]
    cols = ["greenery_mean", "luminance_mean", "charm_mean"]

    # Normalisation min-max sur l'ensemble des zones
    mins = zones_df[cols].min()
    maxs = zones_df[cols].max()
    denom = (maxs - mins).replace(0, np.nan)
    norm = ((row[cols] - mins) / denom).fillna(0.0).values.tolist()

    # fermer le polygone
    values = norm + norm[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1)

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_ylim(0, 1)
    ax.set_title(title)

    _save_close(os.path.join(FIG_DIR, filename))


def compare_delta_bar(zones_df: pd.DataFrame, zone_a: str, zone_b: str, filename: str):
    """
    Graphique Δ (B - A) sur greenery/luminance/charm.
    """
    cols = ["greenery_mean", "luminance_mean", "charm_mean"]

    def _get(zid: str) -> pd.Series:
        return zones_df.loc[zones_df["zone_id"].astype(str) == str(zid), cols].iloc[0]

    a = _get(zone_a)
    b = _get(zone_b)
    delta = (b - a)

    labels = ["Δ Greenery", "Δ Luminance", "Δ Charm"]

    plt.figure(figsize=(7, 3))
    ax = plt.gca()
    ax.barh(labels, delta.values)
    ax.axvline(0, linewidth=1)
    ax.set_title(f"Delta (Zone {zone_b} - Zone {zone_a})")
    _save_close(os.path.join(FIG_DIR, filename))


# ---------------------------
# Load data
# ---------------------------
zones = pd.read_csv("zones_index.csv")
clusters = pd.read_csv("zone_clusters.csv")
clusters_2d = pd.read_csv("zone_clusters_2d.csv")
reco = pd.read_csv("reco_results.csv")

# Harmoniser zone_id en str (évite bugs merge si int/str)
for df in [zones, clusters, clusters_2d]:
    if "zone_id" not in df.columns:
        raise ValueError(f"zone_id manquant dans {df}")
    df["zone_id"] = df["zone_id"].astype(str)

# Détecter la colonne cluster dans zone_clusters.csv puis la renommer en 'cluster'
cluster_col = _find_cluster_col(clusters)
clusters = clusters.rename(columns={cluster_col: "cluster"})

# Merge
zones = zones.merge(clusters[["zone_id", "cluster"]], on="zone_id", how="left")
zones = zones.merge(clusters_2d[["zone_id", "x", "y"]], on="zone_id", how="left")

_ensure_xy(zones)

# Vérifs rapides
if zones["cluster"].isna().any():
    missing = zones.loc[zones["cluster"].isna(), "zone_id"].tolist()
    print(f"[WARN] cluster manquant pour zones: {missing}")

if "similarity" not in reco.columns:
    raise ValueError(f"Colonne 'similarity' absente de reco_results.csv. Colonnes: {list(reco.columns)}")


# ---------------------------
# 1) PCA Projection colorée
# ---------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(data=zones, x="x", y="y", hue="cluster", palette="tab10", s=120)
plt.title("PCA Projection of Urban Zones (colored by cluster)")
_save_close(os.path.join(FIG_DIR, "pca_projection.png"))

# ---------------------------
# 2) Heatmap cluster means
# ---------------------------
cluster_means = zones.groupby("cluster")[["greenery_mean", "luminance_mean", "charm_mean"]].mean()
plt.figure(figsize=(7, 4))
sns.heatmap(cluster_means, annot=True, cmap="viridis", fmt=".2f")
plt.title("Cluster Mean Profiles")
_save_close(os.path.join(FIG_DIR, "cluster_heatmap.png"))

# ---------------------------
# 3) Greenery vs Charm scatter
# ---------------------------
plt.figure(figsize=(6, 6))
sns.scatterplot(data=zones, x="greenery_mean", y="charm_mean", hue="cluster", palette="tab10", s=120)
plt.title("Greenery vs Charm (by cluster)")
_save_close(os.path.join(FIG_DIR, "greenery_vs_charm.png"))

# ---------------------------
# 4) Dispersion intra-zone (std)
# ---------------------------
if "greenery_std" in zones.columns:
    plt.figure(figsize=(9, 4))
    sns.barplot(data=zones, x="zone_id", y="greenery_std")
    plt.xticks(rotation=45)
    plt.title("Greenery Variability per Zone (greenery_std)")
    _save_close(os.path.join(FIG_DIR, "greenery_dispersion.png"))
else:
    print("[WARN] Colonne greenery_std absente => dispersion plot ignoré.")

# ---------------------------
# 5) Distribution similarity
# ---------------------------
plt.figure(figsize=(6, 4))
sns.histplot(reco["similarity"].dropna(), bins=10, kde=True)
plt.title("Similarity Score Distribution")
_save_close(os.path.join(FIG_DIR, "similarity_distribution.png"))

# ---------------------------
# 6) Radar chart (exemple)
# ---------------------------
# Choisis une zone existante (la première)
example_zone_id = zones["zone_id"].iloc[0]
radar_from_row(
    zones_df=zones,
    zone_id=example_zone_id,
    title=f"Zone {example_zone_id} — Urban Profile (normalized)",
    filename="radar_zone_example.png",
)

# ---------------------------
# 7) Delta comparison (exemple)
# ---------------------------
# si tu as au moins 2 zones, on compare la 1ère et la 2ème
if len(zones) >= 2:
    a = zones["zone_id"].iloc[0]
    b = zones["zone_id"].iloc[1]
    compare_delta_bar(zones, zone_a=a, zone_b=b, filename="delta_zone_comparison.png")

print(f"✅ All figures generated in: {FIG_DIR}")
