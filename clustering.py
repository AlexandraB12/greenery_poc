# clustering.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from utils import load_data, get_features


# --------------------------------------------------
# Standardisation
# --------------------------------------------------

def standardize_X(X):
    """Standardise les variables pour clustering."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)


# --------------------------------------------------
# Choix du nombre de clusters (méthode du coude)
# --------------------------------------------------

def choose_k(X_scaled, max_k=8):
    """Méthode du coude pour choisir le nombre optimal de clusters."""
    inertia = []

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(range(2, max_k + 1), inertia, marker='o')
    plt.xlabel("Nombre de clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Méthode du coude pour KMeans")

    plt.savefig("elbow_plot.png", bbox_inches="tight")
    print("📈 elbow_plot.png sauvegardé")

    plt.show()


# --------------------------------------------------
# KMeans
# --------------------------------------------------

def run_kmeans(X_scaled, n_clusters=4):
    """Applique KMeans et retourne les labels et les centres."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    centers = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=X_scaled.columns
    )

    return labels, centers


# --------------------------------------------------
# Visualisation PCA
# --------------------------------------------------

def visualize_pca(X_scaled, labels):
    """Projection 2D avec PCA pour visualiser les clusters."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))

    for cluster in sorted(set(labels)):
        plt.scatter(
            X_pca[labels == cluster, 0],
            X_pca[labels == cluster, 1],
            label=f"Cluster {cluster}"
        )

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Visualisation PCA des clusters")
    plt.legend()

    plt.savefig("pca_clusters.png", bbox_inches="tight")
    print("📊 pca_clusters.png sauvegardé")

    plt.show()


# --------------------------------------------------
# Pipeline complet
# --------------------------------------------------

def run_clustering(n_clusters=4):

    # 1️⃣ Charger les données
    df = load_data()

    # 2️⃣ Extraire features
    X = get_features(df)

    # 3️⃣ Standardisation
    X_scaled = standardize_X(X)

    # 4️⃣ Méthode du coude
    choose_k(X_scaled)

    # 5️⃣ KMeans
    labels, centers = run_kmeans(X_scaled, n_clusters=n_clusters)

    # 6️⃣ Ajouter cluster au dataframe
    df["cluster"] = labels

    # ✅ sauvegarde dataset clusterisé brut
    df.to_csv("clusters_output.csv", index=False)
    print("📦 clusters_output.csv sauvegardé")

    # 7️⃣ Construire luminance si absente
    if "luminance" not in df.columns:
        df["luminance"] = 0.5 * df["brightness"] + 0.5 * df["sky_ratio"]

    # 8️⃣ Profils moyens par cluster
    feature_cols = [
        "greenery_ratio",
        "luminance",
        "visual_complexity",
        "building_regularity"
    ]

    cluster_profiles = df.groupby("cluster")[feature_cols].mean()

    print("\n📊 Profils moyens par cluster :")
    print(cluster_profiles)

    cluster_profiles.to_csv("cluster_profiles.csv")
    print("📦 cluster_profiles.csv sauvegardé")

    # --------------------------------------------------
    # 🔥 9️⃣ Score perceptif par cluster
    # --------------------------------------------------

    cluster_score_map = {
        0: 80,
        1: 60,
        2: 45,
        3: 75
    }

    df["score_cluster"] = df["cluster"].map(cluster_score_map)

    df.to_csv("clusters_scored.csv", index=False)
    print("🏷️ clusters_scored.csv sauvegardé (avec score_cluster)")

    # --------------------------------------------------
    # 🔟 Visualisation PCA
    # --------------------------------------------------

    visualize_pca(X_scaled.values, labels)

    return df, cluster_profiles, centers


# --------------------------------------------------

if __name__ == "__main__":
    run_clustering(n_clusters=4)
