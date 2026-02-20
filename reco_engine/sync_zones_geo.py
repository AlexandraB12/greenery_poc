# reco_engine/sync_zones_geo.py:
# lit zones_index.csv (la liste des zones réelles)
# pour chaque zone, va chercher des fichiers JPG dans data_zones/<zone_id>/
# extrait lat/lon depuis le nom des fichiers (ex: _lat48.856600_lon2.352200)
# calcule un lat/lon représentatif (médiane)
# écrit zones_geo.csv complet
from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Optional, Tuple, List

import pandas as pd


LATLON_RE = re.compile(r"_lat(-?\d+\.\d+)_lon(-?\d+\.\d+)", re.IGNORECASE)


def _extract_latlon_from_filename(path: str) -> Optional[Tuple[float, float]]:
    m = LATLON_RE.search(os.path.basename(path))
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _zone_latlon(zone_dir: str, max_files: int = 50) -> Optional[Tuple[float, float]]:
    # prends jusqu'à max_files images, extrait lat/lon, renvoie médiane
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(zone_dir, pat)))
    files = sorted(files)[:max_files]

    coords = []
    for f in files:
        ll = _extract_latlon_from_filename(f)
        if ll is not None:
            coords.append(ll)

    if not coords:
        return None

    lats = sorted([c[0] for c in coords])
    lons = sorted([c[1] for c in coords])

    mid = len(coords) // 2
    if len(coords) % 2 == 1:
        return lats[mid], lons[mid]
    return (lats[mid - 1] + lats[mid]) / 2.0, (lons[mid - 1] + lons[mid]) / 2.0


def sync_zones_geo(
    zones_index_csv: str = "zones_index.csv",
    data_zones_dir: str = "data_zones",
    out_csv: str = "zones_geo.csv",
    max_files_per_zone: int = 50,
) -> str:
    if not os.path.exists(zones_index_csv):
        raise FileNotFoundError(f"Missing {zones_index_csv}. Run build_zones_index first.")

    df = pd.read_csv(zones_index_csv)
    if "zone_id" not in df.columns:
        raise ValueError("zones_index.csv must contain a zone_id column.")

    rows = []
    missing = []

    for zid in df["zone_id"].astype(str).tolist():
        zdir = os.path.join(data_zones_dir, zid)
        if not os.path.isdir(zdir):
            missing.append((zid, "missing_zone_dir"))
            continue

        ll = _zone_latlon(zdir, max_files=max_files_per_zone)
        if ll is None:
            missing.append((zid, "no_latlon_in_filenames"))
            continue

        lat, lon = ll
        rows.append({"zone_id": zid, "lat": lat, "lon": lon})

    out = pd.DataFrame(rows).sort_values("zone_id")
    out.to_csv(out_csv, index=False)

    print(f"✅ wrote {out_csv} with {len(out)} zones")

    if missing:
        print("⚠️ zones missing coords:")
        for zid, reason in missing:
            print(f"   - {zid}: {reason}")

    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Build zones_geo.csv automatically from data_zones filenames.")
    ap.add_argument("--zones_index_csv", default="zones_index.csv")
    ap.add_argument("--data_zones_dir", default="data_zones")
    ap.add_argument("--out_csv", default="zones_geo.csv")
    ap.add_argument("--max_files_per_zone", type=int, default=50)
    args = ap.parse_args()

    sync_zones_geo(
        zones_index_csv=args.zones_index_csv,
        data_zones_dir=args.data_zones_dir,
        out_csv=args.out_csv,
        max_files_per_zone=args.max_files_per_zone,
    )


if __name__ == "__main__":
    main()
