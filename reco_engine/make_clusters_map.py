from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd

try:
    import folium
except ImportError as e:
    raise ImportError(
        "Missing dependency: folium.\n"
        "Install it with:\n"
        "  pip install folium\n"
    ) from e


def _first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _resolve_latlon_cols(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    candidates = [
        ("lat", "lon"),
        ("lat", "lng"),
        ("latitude", "longitude"),
        ("center_lat", "center_lon"),
    ]
    for a, b in candidates:
        if a in df.columns and b in df.columns:
            return a, b
    return None


def _risk_tier_from_stability(stab: float) -> str:
    if stab is None or pd.isna(stab):
        return "Unknown"
    if stab > 4.0:
        return "Low"
    if stab >= 2.5:
        return "Medium"
    return "Higher"


def _cluster_label_map() -> Dict[int, str]:
    return {
        0: "Open & Bright Urban",
        1: "Balanced Historic Core",
        2: "Dense Traditional Fabric",
        3: "Transitional Mixed Areas",
        4: "Green Residential Pockets",
        5: "Modern Functional Corridors",
    }


def _resolve_stability_col(kpi: pd.DataFrame) -> Optional[str]:
    """
    Find a stability column in KPI file.
    Extend if your KPI uses another name.
    """
    for c in [
        "urban_stability",
        "stability",
        "stability_score",
        "stability_mean",
        "urban_stability_mean",
    ]:
        if c in kpi.columns:
            return c
    return None


def build_clusters_map(
    clusters2d_csv: str = "outputs/zone_clusters_2d.csv",
    zones_csv: str = "zones_index.csv",
    zones_geo_csv: str = "zones_geo.csv",
    kpi_csv: str = "outputs/kpi_zones_ranked.csv",
    out_html: str = "outputs/zones_clusters_map.html",
) -> str:
    clusters2d_path = _first_existing([clusters2d_csv, "zone_clusters_2d.csv"])
    zones_path = _first_existing([zones_csv, "outputs/zones_index.csv"])
    zones_geo_path = _first_existing([zones_geo_csv, "outputs/zones_geo.csv"])
    kpi_path = _first_existing([kpi_csv])

    if not clusters2d_path:
        raise FileNotFoundError(
            "Missing clusters 2D file. Expected outputs/zone_clusters_2d.csv.\n"
            "Run reco_engine/cluster_zones.py first."
        )
    if not zones_path:
        raise FileNotFoundError("Missing zones index. Expected zones_index.csv (or outputs/zones_index.csv).")

    c2d = pd.read_csv(clusters2d_path)
    zones = pd.read_csv(zones_path)

    for df in (c2d, zones):
        if "zone_id" in df.columns:
            df["zone_id"] = df["zone_id"].astype(str)

    # -----------------------------
    # Load zones_geo.csv and merge coords into zones
    # -----------------------------
    zones_geo = pd.DataFrame()
    if zones_geo_path and os.path.exists(zones_geo_path):
        zones_geo = pd.read_csv(zones_geo_path)
        if "zone_id" in zones_geo.columns:
            zones_geo["zone_id"] = zones_geo["zone_id"].astype(str)

    if not zones_geo.empty and "zone_id" in zones_geo.columns:
        zones = zones.merge(zones_geo, on="zone_id", how="left")

    # Resolve coordinates
    latlon = _resolve_latlon_cols(zones)
    if latlon is None:
        raise ValueError(
            "No geo coordinates found to build a map.\n"
            "I looked in zones_index.csv merged with zones_geo.csv.\n"
            "Expected columns like (lat, lon) or (lat, lng) or (latitude, longitude).\n"
        )
    lat_col, lon_col = latlon

    # Base: clusters + coords
    base = c2d.merge(zones[["zone_id", lat_col, lon_col]], on="zone_id", how="left")

    # Attach KPIs (optional)
    base["urban_stability"] = pd.NA
    base["risk_tier"] = "Unknown"

    if kpi_path and os.path.exists(kpi_path):
        kpi = pd.read_csv(kpi_path)
        if "zone_id" in kpi.columns:
            kpi["zone_id"] = kpi["zone_id"].astype(str)

        stab_col = _resolve_stability_col(kpi)
        if stab_col is not None:
            tmp = kpi[["zone_id", stab_col]].copy().rename(columns={stab_col: "urban_stability"})
            base = base.merge(tmp, on="zone_id", how="left")

            # ✅ robust: only compute if the column exists after merge
            if "urban_stability" in base.columns:
                base["risk_tier"] = base["urban_stability"].apply(_risk_tier_from_stability)

    # Cluster labels
    cmap = _cluster_label_map()
    if "cluster_id" in base.columns:
        base["cluster_id"] = pd.to_numeric(base["cluster_id"], errors="coerce")
        base["cluster_label"] = base["cluster_id"].apply(
            lambda x: cmap.get(int(x), f"Cluster {int(x)}") if pd.notna(x) else "Unknown"
        )
    else:
        base["cluster_label"] = "Unknown"

    # Center map
    base_valid = base.dropna(subset=[lat_col, lon_col])
    if base_valid.empty:
        raise ValueError("All zones are missing coordinates after merge. Check zone_id alignment in zones_geo.csv.")

    center_lat = float(base_valid[lat_col].iloc[0])
    center_lon = float(base_valid[lon_col].iloc[0])

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="cartodbpositron")

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
    ]

    def color_for_cluster(cid) -> str:
        if cid is None or pd.isna(cid):
            return "#7f7f7f"
        return palette[int(cid) % len(palette)]

    # Markers + tooltips
    for _, r in base.iterrows():
        lat = r.get(lat_col)
        lon = r.get(lon_col)
        if pd.isna(lat) or pd.isna(lon):
            continue

        tooltip_lines = [
            f"<b>Zone:</b> {r.get('zone_id', '—')}",
            f"<b>Cluster:</b> {r.get('cluster_label', '—')} (id={r.get('cluster_id', '—')})",
        ]

        # optional morphology metrics if present in c2d
        for k in ["greenery_mean", "luminance_mean", "charm_mean"]:
            if k in r and pd.notna(r[k]):
                tooltip_lines.append(f"<b>{k}:</b> {float(r[k]):.3f}")

        if "urban_stability" in base.columns and pd.notna(r.get("urban_stability")):
            tooltip_lines.append(f"<b>Urban stability:</b> {float(r['urban_stability']):.3f}")
        tooltip_lines.append(f"<b>Risk tier:</b> {r.get('risk_tier', 'Unknown')}")

        tooltip_html = "<br/>".join(tooltip_lines)

        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=7,
            color=color_for_cluster(r.get("cluster_id")),
            fill=True,
            fill_opacity=0.85,
            tooltip=folium.Tooltip(tooltip_html, sticky=True),
        ).add_to(m)

    Path(os.path.dirname(out_html) or ".").mkdir(parents=True, exist_ok=True)
    m.save(out_html)
    return out_html


def main() -> None:
    out = build_clusters_map()
    print(f"✅ Saved: {out}")


if __name__ == "__main__":
    main()
