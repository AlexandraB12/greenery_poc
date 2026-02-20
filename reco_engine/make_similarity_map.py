# reco_engine/make_similarity_map.py
from __future__ import annotations
import argparse
import pandas as pd
import folium

def make_map(
    zones_geo_path: str,
    reco_results_path: str,
    out_html: str,
    zoom_start: int = 6,
):
    geo = pd.read_csv(zones_geo_path)
    reco = pd.read_csv(reco_results_path)

    # merge coords
    df = reco.merge(geo, on="zone_id", how="left").dropna(subset=["lat", "lon"])
    if df.empty:
        raise RuntimeError(
            "No zones with lat/lon found after merge. "
            "Generate zones_geo.csv automatically (see update_zones_geo.py)."
        )

    # centre carte
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

    # normalise similarity pour couleur
    if "similarity" not in df.columns:
        raise ValueError("Missing column 'similarity' in reco file (needed for map color).")

    sim_min, sim_max = df["similarity"].min(), df["similarity"].max()

    def norm(x: float) -> float:
        if sim_max == sim_min:
            return 1.0
        return (x - sim_min) / (sim_max - sim_min)

    def color(sim: float) -> str:
        t = norm(sim)
        r = int(255 * (1 - t))
        g = int(200 * t)
        b = 60
        return f"#{r:02x}{g:02x}{b:02x}"

    for _, r in df.iterrows():
        tooltip = (
            f"<b>{r['zone_id']}</b><br>"
            f"Similarity: {float(r['similarity']):.3f}<br>"
            f"Score final: {float(r['score_final']):.3f}<br>" if "score_final" in df.columns else
            f"<b>{r['zone_id']}</b><br>Similarity: {float(r['similarity']):.3f}<br>"
        )
        # scores métier si dispo
        for col, label in [
            ("greenery_mean", "Greenery"),
            ("luminance_mean", "Luminance"),
            ("charm_mean", "Charm"),
        ]:
            if col in df.columns and pd.notna(r.get(col)):
                tooltip += f"{label}: {float(r[col]):.2f}<br>"

        # explication (weighted ou normal)
        if "weighted_explain" in df.columns and pd.notna(r.get("weighted_explain")):
            tooltip += f"{r['weighted_explain']}"
        elif "explain" in df.columns and pd.notna(r.get("explain")):
            tooltip += f"{r['explain']}"

        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=8,
            color=color(float(r["similarity"])),
            fill=True,
            fill_opacity=0.85,
            tooltip=tooltip,
        ).add_to(m)

    m.save(out_html)
    print(f"✅ Saved: {out_html}")

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate interactive similarity map (works with weighted reco too)")
    ap.add_argument("--geo", default="zones_geo.csv", help="zones_geo.csv with zone_id,lat,lon")
    ap.add_argument("--reco", default="reco_results.csv", help="reco file (reco_results.csv or reco_weighted_*.csv)")
    ap.add_argument("--out", default="zones_map.html", help="output html")
    ap.add_argument("--zoom", type=int, default=6, help="initial zoom")
    args = ap.parse_args()
    make_map(zones_geo_path=args.geo, reco_results_path=args.reco, out_html=args.out, zoom_start=args.zoom)

if __name__ == "__main__":
    main()
