# parcourir data_zones/<zone_id>/*.jpg
# extraire lat/lon depuis le filename
# faire moyenne par zone
# écrire zones_geo.csv

from __future__ import annotations
import os, re, glob
import pandas as pd

LATLON_RE = re.compile(r"_lat(-?\d+\.\d+)_lon(-?\d+\.\d+)")

def extract_latlon_from_name(fname: str):
    m = LATLON_RE.search(fname)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))

def build_zones_geo(data_zones_dir: str = "data_zones", out_path: str = "zones_geo.csv"):
    rows = []
    zone_dirs = [d for d in glob.glob(os.path.join(data_zones_dir, "*")) if os.path.isdir(d)]

    for zd in zone_dirs:
        zone_id = os.path.basename(zd)
        lats, lons = [], []
        for p in glob.glob(os.path.join(zd, "*.jpg")):
            latlon = extract_latlon_from_name(os.path.basename(p))
            if latlon:
                lat, lon = latlon
                lats.append(lat); lons.append(lon)

        if lats:
            rows.append({"zone_id": zone_id, "lat": sum(lats)/len(lats), "lon": sum(lons)/len(lons)})
        else:
            # si ton nom de fichier ne contient pas lat/lon => zone ignorée
            rows.append({"zone_id": zone_id, "lat": None, "lon": None})

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path} ({len(df)} zones)")
    return df

if __name__ == "__main__":
    build_zones_geo()