import os
import numpy as np
import pandas as pd

os.makedirs("outputs", exist_ok=True)

zones = pd.read_csv("zones_index.csv")
clusters = pd.read_csv("zone_clusters.csv")

zones["zone_id"] = zones["zone_id"].astype(str)
clusters["zone_id"] = clusters["zone_id"].astype(str)

# detect cluster column
cluster_col = next((c for c in ["cluster", "cluster_id", "label"] if c in clusters.columns), None)
if cluster_col is None:
    raise ValueError(f"No cluster column found in zone_clusters.csv. Columns: {list(clusters.columns)}")

clusters = clusters.rename(columns={cluster_col: "cluster_id"})
zones = zones.merge(clusters[["zone_id", "cluster_id"]], on="zone_id", how="left")

# --- cosine similarity on embeddings
emb_cols = [c for c in zones.columns if c.startswith("emb_")]
if not emb_cols:
    raise ValueError("No embedding columns found (emb_*) in zones_index.csv")

Z = zones.set_index("zone_id")
def cosine(a, b):
    a = np.asarray(a); b = np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

ref_id = "ref_lille"
if ref_id not in Z.index:
    raise ValueError(f"Reference zone '{ref_id}' not found in zones_index.csv")

ref_vec = Z.loc[ref_id, emb_cols].values

zones["similarity_to_ref"] = zones["zone_id"].apply(
    lambda zid: cosine(Z.loc[zid, emb_cols].values, ref_vec)
)

# --- normalize means
feat = ["greenery_mean", "luminance_mean", "charm_mean"]
mins = zones[feat].min()
maxs = zones[feat].max()
den = (maxs - mins).replace(0, np.nan)

for c in feat:
    zones[c + "_n"] = ((zones[c] - mins[c]) / den[c]).fillna(0.0)

# --- profile alignment score vs ref
wg, wl, wc = 1/3, 1/3, 1/3
ref = zones.loc[zones["zone_id"] == ref_id].iloc[0]

def profile_score(row):
    d = np.sqrt(
        wg*(row["greenery_mean_n"]-ref["greenery_mean_n"])**2
        + wl*(row["luminance_mean_n"]-ref["luminance_mean_n"])**2
        + wc*(row["charm_mean_n"]-ref["charm_mean_n"])**2
    )
    return float(1 - d)

zones["profile_score"] = zones.apply(profile_score, axis=1)
zones["same_cluster"] = (zones["cluster_id"] == ref["cluster_id"]).astype(int)

# --- match score
alpha = 0.7
beta = 0.05
zones["match_score"] = alpha*zones["similarity_to_ref"] + (1-alpha)*zones["profile_score"] + beta*zones["same_cluster"]

# --- stability (intra-zone) if std columns exist
std_cols = [c for c in ["greenery_std", "luminance_std", "charm_std"] if c in zones.columns]
if std_cols:
    mins_s = zones[std_cols].min()
    maxs_s = zones[std_cols].max()
    den_s = (maxs_s - mins_s).replace(0, np.nan)
    for c in std_cols:
        zones[c + "_n"] = ((zones[c] - mins_s[c]) / den_s[c]).fillna(0.0)

    zones["var_morph"] = zones[[c + "_n" for c in std_cols]].mean(axis=1)
    eps = 0.01
    zones["urban_stability"] = 1.0 / (eps + zones["var_morph"])
else:
    zones["var_morph"] = np.nan
    zones["urban_stability"] = np.nan
    print("[WARN] std columns not found => urban_stability not computed (need *_std).")

# --- export top recos
out = zones.sort_values("match_score", ascending=False).reset_index(drop=True)
out.to_csv("outputs/kpi_zones_ranked.csv", index=False)

print("✅ Saved outputs/kpi_zones_ranked.csv")
print(out[["zone_id","cluster_id","similarity_to_ref","profile_score","match_score","urban_stability"]].head(10))
