# dashboard_app.py
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk


# ============================================================
# Robust loading utilities
# ============================================================
def _drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    bad_cols = [c for c in df.columns if c is None or str(c).strip() == "" or str(c).startswith("Unnamed")]
    if bad_cols:
        df = df.drop(columns=bad_cols, errors="ignore")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _normalize_zone_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _to_float_series(s: pd.Series) -> pd.Series:
    # handle french decimals like "44,83"
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


@st.cache_data(show_spinner=False)
def load_csv_robust(csv_path: str) -> pd.DataFrame:
    """
    Robust CSV loader:
    - tries separators: comma, tab, semicolon
    - drops unnamed columns
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    last_err = None
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] >= 2:
                return _drop_unnamed_columns(df)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Could not read CSV with common separators. Last error: {last_err}")


def detect_embedding_cols(df: pd.DataFrame) -> List[str]:
    emb_cols = [c for c in df.columns if re.fullmatch(r"emb_\d+", str(c))]
    emb_cols.sort(key=lambda x: int(x.split("_")[1]))
    if not emb_cols:
        raise ValueError("No embedding columns found (expected columns like emb_0, emb_1, ...).")
    return emb_cols


def cosine_similarity_matrix(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    X_norm = np.linalg.norm(X, axis=1)
    ref_norm = np.linalg.norm(ref)
    denom = (X_norm * ref_norm)
    denom = np.where(denom == 0, np.nan, denom)
    sims = (X @ ref) / denom
    sims = np.nan_to_num(sims, nan=-1.0, posinf=-1.0, neginf=-1.0)
    return sims


def safe_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["1", "true", "yes", "y", "t"])


def build_match_score(
    df: pd.DataFrame,
    sim_col: str = "similarity_to_ref_calc",
    same_cluster_col: str = "same_cluster",
    stability_col: str = "urban_stability",
    n_images_col: str = "n_images",
    w_sim: float = 0.80,
    w_cluster_bonus: float = 0.10,
    w_stability: float = 0.07,
    w_images: float = 0.03,
) -> pd.Series:
    """
    Explainable score in [0,1]:
    - similarity: cosine in [-1,1] -> [0,1]
    - cluster bonus: 0/1
    - stability: assumed [0,1] (auto if looks like 0..100)
    - images: log-scaled confidence bonus
    """
    sim = pd.to_numeric(df.get(sim_col, np.nan), errors="coerce")
    sim01 = (sim + 1.0) / 2.0

    if same_cluster_col in df.columns:
        sc = safe_bool(df[same_cluster_col]).astype(float)
    else:
        sc = 0.0

    if stability_col in df.columns:
        stab = pd.to_numeric(df[stability_col], errors="coerce")
        if stab.max(skipna=True) > 1.5:
            stab = stab / 100.0
        stab = stab.clip(0, 1).fillna(0.0)
    else:
        stab = 0.0

    if n_images_col in df.columns:
        nimg = pd.to_numeric(df[n_images_col], errors="coerce").fillna(0.0)
        denom = np.log1p(max(1.0, float(np.nanmax(nimg))))
        img_bonus = np.log1p(nimg) / denom if denom > 0 else 0.0
        img_bonus = pd.Series(img_bonus, index=df.index).fillna(0.0).clip(0, 1)
    else:
        img_bonus = 0.0

    score = (w_sim * sim01) + (w_cluster_bonus * sc) + (w_stability * stab) + (w_images * img_bonus)
    return score.clip(0, 1)


# ============================================================
# Geo auto-fix helpers
# ============================================================
def _webmercator_to_wgs84(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # EPSG:3857 meters -> lon/lat degrees
    R = 6378137.0
    lon = (x / R) * (180.0 / np.pi)
    lat = (2.0 * np.arctan(np.exp(y / R)) - (np.pi / 2.0)) * (180.0 / np.pi)
    return lon, lat


def fix_geo_coords(geo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix common issues:
    - french decimals (comma)
    - swapped lat/lon
    - projected coords that look like EPSG:3857 (WebMercator) -> convert to WGS84
    """
    g = geo_df.copy()

    g["lat"] = _to_float_series(g["lat"])
    g["lon"] = _to_float_series(g["lon"])

    def in_france_bbox(lat: pd.Series, lon: pd.Series) -> pd.Series:
        return (lat.between(41, 51)) & (lon.between(-6, 10))

    n = int(len(g))
    fr_before = int(in_france_bbox(g["lat"], g["lon"]).sum())
    fr_swapped = int(in_france_bbox(g["lon"], g["lat"]).sum())

    if fr_swapped > fr_before:
        st.sidebar.warning("⚠️ zones_geo: lat/lon semblent inversés → swap automatique.")
        g[["lat", "lon"]] = g[["lon", "lat"]]
        fr_before = fr_swapped

    # Detect "projected" ranges
    med_abs_lat = float(np.nanmedian(np.abs(g["lat"].values)))
    med_abs_lon = float(np.nanmedian(np.abs(g["lon"].values)))

    looks_projected = (
        (med_abs_lat > 1000)
        or (med_abs_lon > 1000)
        or (g["lat"].abs().max(skipna=True) > 90)
        or (g["lon"].abs().max(skipna=True) > 180)
    )

    if looks_projected:
        st.sidebar.warning("⚠️ zones_geo: coords semblent projetées → conversion WebMercator (EPSG:3857) tentée.")
        # Common pattern: x=lon_col, y=lat_col (meters)
        x = g["lon"].to_numpy(dtype=float)
        y = g["lat"].to_numpy(dtype=float)
        lon2, lat2 = _webmercator_to_wgs84(x, y)
        g["lon"] = lon2
        g["lat"] = lat2

        fr_after = int(in_france_bbox(g["lat"], g["lon"]).sum())
        st.sidebar.info(f"📍 BBox France: {fr_before}/{n} → {fr_after}/{n} après conversion")

    return g


# ============================================================
# Cluster hull + colors
# ============================================================
def cluster_color(cid) -> List[int]:
    """Deterministic RGB per cluster_id."""
    try:
        cid_i = int(cid)
    except Exception:
        cid_i = 0
    rng = np.random.default_rng(cid_i + 12345)
    return [int(x) for x in rng.integers(50, 230, size=3).tolist()]


def convex_hull_lonlat(points: List[tuple]) -> List[tuple]:
    """
    Convex hull (Andrew monotonic chain).
    points: list of (lon, lat)
    returns hull as list of (lon, lat) in order (not closed).
    """
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Zone Similarity Recommender", layout="wide")
st.title("🔎 Zone Similarity Recommender (embeddings + scores)")

with st.sidebar:
    st.header("📁 Data")
    csv_path = st.text_input("Path to zones index CSV", value="data/zones_index.csv")
    geo_path = st.text_input("Path to zones geo CSV (optional)", value="data/zones_geo.csv")
    clusters_path = st.text_input("Path to clusters CSV (optional)", value="data/zone_clusters_2d.csv")

    st.header("⚙️ Ranking")
    top_k = st.slider("Top K", 5, 50, 10)
    min_images = st.slider("Min images per zone", 0, 200, 10)
    enforce_same_cluster = st.checkbox("Only same cluster as reference", value=False)

    st.header("🧮 Match score weights")
    w_sim = st.slider("Weight: similarity", 0.0, 1.0, 0.80, 0.01)
    w_cluster = st.slider("Weight: same-cluster bonus", 0.0, 1.0, 0.10, 0.01)
    w_stability = st.slider("Weight: stability", 0.0, 1.0, 0.07, 0.01)
    w_images = st.slider("Weight: n_images confidence", 0.0, 1.0, 0.03, 0.01)

    # Normalize weights to sum=1
    ssum = max(1e-9, (w_sim + w_cluster + w_stability + w_images))
    w_sim, w_cluster, w_stability, w_images = [x / ssum for x in (w_sim, w_cluster, w_stability, w_images)]

    st.caption(
        "💡 Les sliders contrôlent l'importance relative de chaque signal dans le score final "
        "(la somme est automatiquement normalisée à 1)."
    )


# -----------------------------
# Load index CSV
# -----------------------------
try:
    df = load_csv_robust(csv_path)
except Exception as e:
    st.error(f"Failed to load zones index CSV: {e}")
    st.info("➡️ Astuce: comme ton dashboard est dans `reco_engine/`, mets `../zones_index.csv` si le CSV est à la racine.")
    st.stop()

if "zone_id" not in df.columns:
    st.error("Missing required column: zone_id")
    st.write("Columns found:", list(df.columns))
    st.stop()

df["zone_id"] = _normalize_zone_id(df["zone_id"])


# -----------------------------
# Load clusters and merge (optional)
# -----------------------------
clusters_df = None
clusters_ok = False
clusters_msg = None

clusters_file = Path(clusters_path)
if clusters_path and clusters_file.exists():
    try:
        clusters_df = load_csv_robust(clusters_path)
        if "zone_id" not in clusters_df.columns:
            clusters_msg = "Clusters CSV loaded but missing column zone_id."
        else:
            clusters_df["zone_id"] = _normalize_zone_id(clusters_df["zone_id"])

            if "cluster_id" not in clusters_df.columns:
                clusters_msg = "Clusters CSV loaded but missing column cluster_id."
            else:
                clusters_df["cluster_id"] = pd.to_numeric(clusters_df["cluster_id"], errors="coerce")
                for c in ["x", "y"]:
                    if c in clusters_df.columns:
                        clusters_df[c] = pd.to_numeric(clusters_df[c], errors="coerce")
                clusters_ok = True
    except Exception as e:
        clusters_msg = f"Failed to load clusters CSV: {e}"
elif clusters_path:
    clusters_msg = f"Clusters CSV not found at: {clusters_path}"

if clusters_ok and clusters_df is not None:
    keep_cols = ["zone_id", "cluster_id"] + [c for c in ["x", "y"] if c in clusters_df.columns]
    df = df.merge(
        clusters_df[keep_cols],
        on="zone_id",
        how="left",
        suffixes=("", "_clusters"),
    )
else:
    if clusters_msg:
        st.sidebar.warning(clusters_msg)


# -----------------------------
# Detect embeddings + coerce types
# -----------------------------
try:
    emb_cols = detect_embedding_cols(df)
except Exception as e:
    st.error(str(e))
    st.write("Columns found:", list(df.columns))
    st.stop()

df = _coerce_numeric(
    df,
    [
        "n_images",
        "greenery_mean",
        "luminance_mean",
        "charm_mean",
        "greenery_std",
        "luminance_std",
        "charm_std",
        "urban_stability",
        "var_morph",
    ],
)
df = _coerce_numeric(df, emb_cols)

valid_mask = df[emb_cols].notna().all(axis=1)
if valid_mask.mean() < 0.9:
    st.warning(f"Some zones have missing embeddings and will be ignored: {(~valid_mask).sum()} rows.")
df_valid = df.loc[valid_mask].copy()


# -----------------------------
# Choose reference zone
# -----------------------------
zone_ids = df_valid["zone_id"].astype(str).tolist()
with st.sidebar:
    st.header("🎯 Reference")
    ref_zone_id = st.selectbox("Reference zone_id", zone_ids, index=0)

ref_row_df = df_valid[df_valid["zone_id"].astype(str) == str(ref_zone_id)]
if ref_row_df.empty:
    st.error("Reference zone not found after filtering.")
    st.stop()
ref_row = ref_row_df.iloc[0]


# -----------------------------
# Similarity + same_cluster + match_score
# -----------------------------
X = df_valid[emb_cols].to_numpy(dtype=np.float32)
ref_vec = ref_row[emb_cols].to_numpy(dtype=np.float32)
df_valid["similarity_to_ref_calc"] = cosine_similarity_matrix(X, ref_vec)

if "cluster_id" in df_valid.columns and pd.notna(ref_row.get("cluster_id")):
    df_valid["same_cluster"] = (df_valid["cluster_id"] == ref_row["cluster_id"])
else:
    df_valid["same_cluster"] = False

df_valid["match_score_calc"] = build_match_score(
    df_valid,
    sim_col="similarity_to_ref_calc",
    same_cluster_col="same_cluster",
    stability_col="urban_stability",
    n_images_col="n_images",
    w_sim=w_sim,
    w_cluster_bonus=w_cluster,
    w_stability=w_stability,
    w_images=w_images,
)

work = df_valid.copy()
if "n_images" in work.columns:
    work = work[work["n_images"].fillna(0) >= min_images]
if enforce_same_cluster:
    work = work[work["same_cluster"] == True]

work = work[work["zone_id"].astype(str) != str(ref_zone_id)]
work = work.sort_values(["match_score_calc", "similarity_to_ref_calc"], ascending=False).head(top_k)


# -----------------------------
# Load geo and merge (optional)
# -----------------------------
geo_df = None
geo_ok = False
geo_msg = None

geo_file = Path(geo_path)
if geo_path and geo_file.exists():
    try:
        geo_df = load_csv_robust(geo_path)
        if "zone_id" not in geo_df.columns:
            geo_msg = "zones_geo.csv loaded but missing column zone_id."
        else:
            geo_df["zone_id"] = _normalize_zone_id(geo_df["zone_id"])

            # Accept aliases
            if "lat" not in geo_df.columns or "lon" not in geo_df.columns:
                rename_map = {}
                if "latitude" in geo_df.columns and "lat" not in geo_df.columns:
                    rename_map["latitude"] = "lat"
                if "longitude" in geo_df.columns and "lon" not in geo_df.columns:
                    rename_map["longitude"] = "lon"
                if rename_map:
                    geo_df = geo_df.rename(columns=rename_map)

            if "lat" in geo_df.columns and "lon" in geo_df.columns:
                geo_df = fix_geo_coords(geo_df)
                geo_ok = True
            else:
                geo_msg = "zones_geo.csv loaded but missing lat/lon columns (or latitude/longitude)."
    except Exception as e:
        geo_msg = f"Failed to load zones geo CSV: {e}"
elif geo_path:
    geo_msg = f"zones_geo.csv not found at: {geo_path}"

if geo_msg:
    st.sidebar.info(geo_msg)

# Merge coords into work + reference
work_geo = work.copy()
ref_geo = pd.DataFrame([ref_row.to_dict()])

if geo_ok and geo_df is not None:
    work_geo = work_geo.merge(geo_df[["zone_id", "lat", "lon"]], on="zone_id", how="left")
    ref_geo = ref_geo.merge(geo_df[["zone_id", "lat", "lon"]], on="zone_id", how="left")
else:
    work_geo["lat"] = np.nan
    work_geo["lon"] = np.nan
    ref_geo["lat"] = np.nan
    ref_geo["lon"] = np.nan


# ============================================================
# Display (Tabs)
# ============================================================
tab1, tab2, tab3 = st.tabs(["🏆 Recommandations", "🗺️ Carte", "🧩 Clusters (2D)"])

# -----------------------------
# TAB 1: Recommendations
# -----------------------------
with tab1:
    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("📌 Reference zone summary")
        ref_cols = [
            "zone_id",
            "cluster_id",
            "n_images",
            "greenery_mean",
            "luminance_mean",
            "charm_mean",
            "greenery_std",
            "luminance_std",
            "charm_std",
            "urban_stability",
            "var_morph",
            "x",
            "y",
        ]
        ref_show = {c: ref_row.get(c, None) for c in ref_cols if c in df_valid.columns}
        if "lat" in ref_geo.columns and "lon" in ref_geo.columns:
            ref_show["lat"] = float(ref_geo["lat"].iloc[0]) if pd.notna(ref_geo["lat"].iloc[0]) else None
            ref_show["lon"] = float(ref_geo["lon"].iloc[0]) if pd.notna(ref_geo["lon"].iloc[0]) else None
        st.json(ref_show)

    with right:
        st.subheader("🏆 Top recommendations")
        show_cols = [
            "zone_id",
            "match_score_calc",
            "similarity_to_ref_calc",
            "same_cluster",
            "cluster_id",
            "n_images",
            "greenery_mean",
            "luminance_mean",
            "charm_mean",
            "urban_stability",
            "var_morph",
            "x",
            "y",
            "lat",
            "lon",
        ]
        show_cols = [c for c in show_cols if c in work_geo.columns]
        display_df = work_geo[show_cols].copy()

        for c in ["match_score_calc", "similarity_to_ref_calc"]:
            if c in display_df.columns:
                display_df[c] = pd.to_numeric(display_df[c], errors="coerce").round(4)

        for c in ["greenery_mean", "luminance_mean", "charm_mean", "urban_stability", "var_morph", "x", "y", "lat", "lon"]:
            if c in display_df.columns:
                display_df[c] = pd.to_numeric(display_df[c], errors="coerce").round(5)

        st.dataframe(display_df, width="stretch")

    st.divider()
    st.subheader("🔍 Comment lire le score (les sliders)")
    st.markdown(
        f"""
- **Similarity** (*poids {w_sim:.2f}*) : proximité visuelle basée sur les embeddings (cosine).
- **Same cluster bonus** (*poids {w_cluster:.2f}*) : petit boost si la zone est dans le même cluster que la référence (profil global similaire).
- **Stability** (*poids {w_stability:.2f}*) : favorise les zones plus “stables” via `urban_stability` (auto-normalisé si 0–100).
- **n_images confidence** (*poids {w_images:.2f}*) : légère préférence pour les zones avec plus d’images (info plus fiable).
"""
    )


# -----------------------------
# TAB 2: Map (CARTO free) + background + hulls on ALL zones
# -----------------------------
with tab2:
    st.subheader("🗺️ Carte des recommandations (clusters + hulls)")

    CARTO_POSITRON = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

    if (geo_df is None) or (not geo_ok):
        st.warning("⚠️ Aucun fichier geo valide chargé (zones_geo.csv). Impossible d'afficher la carte.")
        st.stop()

    with st.sidebar:
        st.header("🗺️ Carte (options)")
        show_background = st.checkbox("Afficher toutes les zones (fond)", value=True)
        show_convex_hulls = st.checkbox("Afficher convex hulls par cluster", value=True)

    # Recos = work_geo
    recos = work_geo.copy()
    recos["lat"] = pd.to_numeric(recos["lat"], errors="coerce")
    recos["lon"] = pd.to_numeric(recos["lon"], errors="coerce")
    recos = recos.dropna(subset=["lat", "lon"])

    # Reference coords from ref_geo
    ref_lat = pd.to_numeric(ref_geo["lat"].iloc[0], errors="coerce") if "lat" in ref_geo.columns else np.nan
    ref_lon = pd.to_numeric(ref_geo["lon"].iloc[0], errors="coerce") if "lon" in ref_geo.columns else np.nan
    if pd.isna(ref_lat) or pd.isna(ref_lon):
        st.error("❌ La zone de référence n'a pas de coordonnées lat/lon.")
        st.stop()

    # ALL zones with geo (for hull + background)
    all_geo = df_valid.copy()
    all_geo = all_geo.merge(geo_df[["zone_id", "lat", "lon"]], on="zone_id", how="left")
    all_geo["lat"] = pd.to_numeric(all_geo["lat"], errors="coerce")
    all_geo["lon"] = pd.to_numeric(all_geo["lon"], errors="coerce")
    all_geo = all_geo.dropna(subset=["lat", "lon"])

    if len(recos) == 0:
        st.warning("⚠️ Aucune recommandation n'a de lat/lon après merge.")
        st.stop()

    # Background points
    if "cluster_id" in all_geo.columns:
        all_geo["color"] = all_geo["cluster_id"].apply(cluster_color)
    else:
        all_geo["color"] = [[120, 120, 120]] * len(all_geo)
    all_geo["radius"] = 60

    # Reco points (color by cluster + size by score)
    if "cluster_id" in recos.columns:
        recos["color"] = recos["cluster_id"].apply(cluster_color)
    else:
        recos["color"] = [[255, 120, 0]] * len(recos)

    s = pd.to_numeric(recos.get("match_score_calc", 0.0), errors="coerce").fillna(0.0)
    recos["radius"] = (120 + 380 * s).astype(float)
    recos["type"] = "reco"

    ref_plot = pd.DataFrame(
        {
            "zone_id": [str(ref_row["zone_id"])],
            "lat": [float(ref_lat)],
            "lon": [float(ref_lon)],
            "color": [[30, 144, 255]],
            "radius": [520.0],
            "type": ["reference"],
            "match_score_calc": [1.0],
            "similarity_to_ref_calc": [1.0],
            "cluster_id": [ref_row.get("cluster_id", None)],
            "n_images": [ref_row.get("n_images", None)],
            "greenery_mean": [ref_row.get("greenery_mean", None)],
        }
    )

    layers = []

    # Hulls computed on ALL zones
    if show_convex_hulls and ("cluster_id" in all_geo.columns):
        polys = []
        for cid, g in all_geo.dropna(subset=["cluster_id"]).groupby("cluster_id"):
            pts = list(zip(g["lon"].astype(float).tolist(), g["lat"].astype(float).tolist()))
            pts = [p for p in pts if np.isfinite(p[0]) and np.isfinite(p[1])]
            if len(set(pts)) < 3:
                continue
            hull = convex_hull_lonlat(pts)
            if len(hull) < 3:
                continue
            hull = hull + [hull[0]]
            polys.append({"cluster_id": cid, "polygon": hull, "fill_color": cluster_color(cid)})

        if polys:
            poly_df = pd.DataFrame(polys)
            hull_layer = pdk.Layer(
                "PolygonLayer",
                data=poly_df,
                get_polygon="polygon",
                get_fill_color="fill_color",
                get_line_color=[30, 30, 30],
                line_width_min_pixels=1,
                pickable=True,
                opacity=0.12,
                stroked=True,
            )
            layers.append(hull_layer)

    # Background layer behind points
    if show_background:
        background_layer = pdk.Layer(
            "ScatterplotLayer",
            data=all_geo,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius="radius",
            pickable=False,
            opacity=0.15,
            stroked=False,
            radius_min_pixels=2,
            radius_max_pixels=6,
        )
        layers.append(background_layer)

    # Foreground points layer (ref + recos)
    plot_df = pd.concat([ref_plot, recos], ignore_index=True)

    points_layer = pdk.Layer(
        "ScatterplotLayer",
        data=plot_df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        opacity=0.9,
        stroked=True,
        get_line_color=[20, 20, 20],
        line_width_min_pixels=1,
        radius_min_pixels=7,
        radius_max_pixels=30,
    )
    layers.append(points_layer)

    tooltip = {
        "html": """
        <b>{type}</b><br/>
        <b>{zone_id}</b><br/>
        score: {match_score_calc}<br/>
        similarity: {similarity_to_ref_calc}<br/>
        cluster: {cluster_id}<br/>
        n_images: {n_images}<br/>
        greenery: {greenery_mean}
        """,
        "style": {"backgroundColor": "white", "color": "black"},
    }

    view_state = pdk.ViewState(
        latitude=float(ref_lat),
        longitude=float(ref_lon),
        zoom=6.5,
        pitch=0,
    )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=CARTO_POSITRON,
    )

    st.pydeck_chart(deck, width="stretch")

    with st.expander("🧪 Map debug"):
        st.write(
            {
                "ref_lat": float(ref_lat),
                "ref_lon": float(ref_lon),
                "n_recos": int(len(recos)),
                "n_all_geo": int(len(all_geo)),
                "all_geo_lat_range": (float(all_geo["lat"].min()), float(all_geo["lat"].max())),
                "all_geo_lon_range": (float(all_geo["lon"].min()), float(all_geo["lon"].max())),
            }
        )


# -----------------------------
# TAB 3: Clusters 2D (NOT a map)
# -----------------------------
with tab3:
    st.subheader("🧩 Clusters (2D) — projection x/y (pas une carte)")

    if ("x" not in df_valid.columns) or ("y" not in df_valid.columns) or ("cluster_id" not in df_valid.columns):
        st.warning("⚠️ Missing x/y/cluster_id. Vérifie que zone_clusters_2d.csv est bien chargé + mergé.")
    else:
        clusters_plot = df_valid[["zone_id", "cluster_id", "x", "y"]].copy()
        clusters_plot["x"] = pd.to_numeric(clusters_plot["x"], errors="coerce")
        clusters_plot["y"] = pd.to_numeric(clusters_plot["y"], errors="coerce")
        clusters_plot = clusters_plot.dropna(subset=["x", "y"])

        highlight_ids = set([str(ref_zone_id)] + work["zone_id"].astype(str).tolist())
        clusters_plot["highlight"] = clusters_plot["zone_id"].astype(str).isin(highlight_ids)

        st.caption("✅ x/y ne sont pas des coordonnées GPS → on affiche un nuage de points, pas une carte.")
        st.dataframe(
            clusters_plot.sort_values(["highlight", "cluster_id"], ascending=[False, True]),
            width="stretch",
        )

        st.scatter_chart(
            clusters_plot,
            x="x",
            y="y",
            color="cluster_id",
            size="highlight",
        )

        st.caption("💡 Highlighted = reference + top recommandations.")


# ============================================================
# Global debug
# ============================================================
with st.expander("🧪 Global debug (columns & healthchecks)"):
    st.write(f"Detected embedding columns: {len(emb_cols)} dims ({emb_cols[0]} → {emb_cols[-1]})")
    st.write("df_valid shape:", df_valid.shape)

    if "cluster_id" in df_valid.columns:
        st.write("cluster_id missing:", int(df_valid["cluster_id"].isna().sum()))
    else:
        st.write("cluster_id missing: no cluster_id col")

    if geo_df is not None:
        st.write("Geo columns:", list(geo_df.columns))
        st.write(
            "Geo ranges:",
            {
                "lat_min": float(pd.to_numeric(geo_df["lat"], errors="coerce").min()),
                "lat_max": float(pd.to_numeric(geo_df["lat"], errors="coerce").max()),
                "lon_min": float(pd.to_numeric(geo_df["lon"], errors="coerce").min()),
                "lon_max": float(pd.to_numeric(geo_df["lon"], errors="coerce").max()),
            },
        )

    if clusters_df is not None:
        st.write("Clusters columns:", list(clusters_df.columns))
