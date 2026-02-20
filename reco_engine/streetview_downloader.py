# Ce script :
# génère des points aléatoires dans un rayon
# vérifie si Street View existe à cet endroit (metadata)
# télécharge les images en JPG
# les range dans data_zones/<zone_name>/

# reco_engine/streetview_downloader.py
from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests
from tqdm import tqdm


@dataclass
class DownloadConfig:
    zone_name: str
    center_lat: float
    center_lon: float
    radius_m: int = 450    # radius_m=450 : “quartier” sans trop diluer
    n_points: int = 6
    headings: Tuple[int, ...] = (0, 120, 240)
    fov: int = 90
    pitch: int = 0
    size: str = "640x640"
    out_dir: str = "data_zones"
    min_distance_m_between_points: int = 35   # min_distance=35 : évite points trop proches
    max_tries: int = 400
    sleep_s: float = 0.10    # sleep_s=0.10 : moins de burst (plus safe)
    use_metadata_check: bool = True



def _meters_to_lat(m: float) -> float:
    return m / 111_320.0  # ~ meters per degree latitude


def _meters_to_lon(m: float, lat: float) -> float:
    return m / (111_320.0 * math.cos(math.radians(lat)))


def _haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = p2 - p1
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(x))


def _random_point_in_radius(lat: float, lon: float, radius_m: float) -> Tuple[float, float]:
    # uniform in circle area
    r = radius_m * math.sqrt(random.random())
    theta = 2 * math.pi * random.random()

    dlat = _meters_to_lat(r * math.cos(theta))
    dlon = _meters_to_lon(r * math.sin(theta), lat)

    return lat + dlat, lon + dlon


def _streetview_metadata(api_key: str, lat: float, lon: float) -> dict:
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": f"{lat},{lon}", "key": api_key}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _streetview_image(api_key: str, lat: float, lon: float, heading: int, size: str, fov: int, pitch: int) -> bytes:
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "location": f"{lat},{lon}",
        "size": size,
        "heading": str(heading),
        "fov": str(fov),
        "pitch": str(pitch),
        "key": api_key,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.content


def _is_far_enough(candidate: Tuple[float, float], selected: List[Tuple[float, float]], min_dist_m: float) -> bool:
    return all(_haversine_m(candidate, p) >= min_dist_m for p in selected)


def create_zone_images(cfg: DownloadConfig, api_key: Optional[str] = None) -> str:
    api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Set GOOGLE_MAPS_API_KEY env var or pass api_key=...")

    zone_dir = os.path.join(cfg.out_dir, cfg.zone_name)
    os.makedirs(zone_dir, exist_ok=True)

    # 🚫 Skip téléchargement si la zone contient déjà des images
    existing_imgs = [
        f for f in os.listdir(zone_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(existing_imgs) > 0:
        print(f"⏭️ Skipping {cfg.zone_name} — {len(existing_imgs)} images already exist")
        return zone_dir

    # 1) pick points
    points: List[Tuple[float, float]] = []
    tries = 0

    pbar = tqdm(total=cfg.n_points, desc=f"Selecting points for {cfg.zone_name}")
    while len(points) < cfg.n_points and tries < cfg.max_tries:
        tries += 1
        cand = _random_point_in_radius(cfg.center_lat, cfg.center_lon, cfg.radius_m)
        if not _is_far_enough(cand, points, cfg.min_distance_m_between_points):
            continue

        if cfg.use_metadata_check:
            md = _streetview_metadata(api_key, cand[0], cand[1])
            if md.get("status") != "OK":
                time.sleep(cfg.sleep_s)
                continue

        points.append(cand)
        pbar.update(1)
        time.sleep(cfg.sleep_s)

    pbar.close()

    if len(points) < cfg.n_points:
        print(f"⚠️ Only found {len(points)}/{cfg.n_points} valid Street View points. Proceeding anyway.")

    # 2) download images
    img_count = 0
    for i, (lat, lon) in enumerate(tqdm(points, desc=f"Downloading images for {cfg.zone_name}")):
        for h in cfg.headings:
            fname = f"{cfg.zone_name}_{i:03d}_h{h}_lat{lat:.6f}_lon{lon:.6f}.jpg"
            fpath = os.path.join(zone_dir, fname)
            if os.path.exists(fpath):
                continue

            try:
                content = _streetview_image(api_key, lat, lon, h, cfg.size, cfg.fov, cfg.pitch)
                # Basic sanity: Street View sometimes returns non-image for errors;
                # still usually jpg bytes, but we keep it simple here.
                with open(fpath, "wb") as f:
                    f.write(content)
                img_count += 1
            except Exception as e:
                # skip on error
                print(f"⚠️ Download failed for point {i} heading {h}: {e}")
            time.sleep(cfg.sleep_s)

    print(f"✅ Saved {img_count} images in: {zone_dir}")
    return zone_dir


if __name__ == "__main__":
    zones = [
        ("ref_lille", 50.6292, 3.0573),
        ("paris_zone_03", 48.8566, 2.3522),
        ("lyon", 45.7640, 4.8357),
        ("bordeaux", 44.8378, -0.5792),
        ("strasbourg", 48.5734, 7.7521),
        ("grenoble", 45.1885, 5.7245),
        ("pau", 43.2951, -0.3708),
        ("nantes", 47.2184, -1.5536),
        ("rennes", 48.1173, -1.6778),
        ("toulouse", 43.6047, 1.4442),
        ("marseille", 43.2965, 5.3698),
        ("nice", 43.7102, 7.2620),
    ]

    for name, lat, lon in zones:
        cfg = DownloadConfig(
            zone_name=name,
            center_lat=lat,
            center_lon=lon,
        )
        create_zone_images(cfg)


