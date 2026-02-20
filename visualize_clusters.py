import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("cluster_profiles_DL.csv")

features = [
    "greenery_ratio",
    "luminance",
    "visual_complexity",
    "building_regularity"
]

# =========================
# Histogrammes
# =========================

for _, row in df.iterrows():

    plt.figure()
    plt.bar(features, row[features])
    plt.title(f"Profil visuel — Cluster {int(row['cluster'])}")
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(f"cluster_{int(row['cluster'])}_bars.png")
    plt.close()


# =========================
# Radar charts
# =========================

for _, row in df.iterrows():

    values = row[features].values.tolist()
    values += values[:1]

    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)

    ax.set_title(f"Radar profil — Cluster {int(row['cluster'])}")

    plt.savefig(f"cluster_{int(row['cluster'])}_radar.png")
    plt.close()


print("✅ Graphiques clusters générés")
