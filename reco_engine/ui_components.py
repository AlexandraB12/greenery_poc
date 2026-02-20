from __future__ import annotations

import os
import json
from typing import Optional, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_csv(path: str) -> pd.DataFrame:
    """Robust CSV loader: auto-detect delimiter (, ; tab) + safe fallbacks."""
    if not path or not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        pass

    for sep in [",", "\t", ";"]:
        try:
            return pd.read_csv(path, sep=sep)
        except Exception:
            continue

    return pd.DataFrame()


def ensure_zone_id(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure zone_id exists as clean string (strip spaces) to allow merges."""
    if df is None or df.empty:
        return df
    if "zone_id" in df.columns:
        df["zone_id"] = df["zone_id"].astype(str).str.strip()
    return df


def first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def load_profiles(profile_dir: str) -> pd.DataFrame:
    """
    Load zone_profiles/*.json for audit (n_images + mean scores).
    Robust:
      - accepts profile_dir at root
      - accepts nested zone_profiles/zone_profiles
      - recursively finds *.json
    """
    if not profile_dir:
        return pd.DataFrame()

    if not os.path.exists(profile_dir):
        return pd.DataFrame()

    rows = []
    for root, _, files in os.walk(profile_dir):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            p = os.path.join(root, fn)
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                continue

            scores = d.get("scores_mean") or {}
            rows.append(
                {
                    "zone_id": str(d.get("zone_id")).strip(),
                    "n_images": d.get("n_images"),
                    "greenery": scores.get("greenery"),
                    "luminance": scores.get("luminance"),
                    "charm": scores.get("charm"),
                }
            )

    df = pd.DataFrame(rows)
    return ensure_zone_id(df)


def robustness_badge(n_images: float) -> str:
    n = float(n_images) if n_images is not None and not pd.isna(n_images) else 0.0
    if n >= 80:
        return "🟢 Strong"
    if n >= 30:
        return "🟡 Medium"
    return "🔴 Weak"


def risk_tier_from_stability(stab: float) -> str:
    if stab is None or pd.isna(stab):
        return "Unknown"
    stab = float(stab)
    if stab > 4.0:
        return "Low"
    if stab >= 2.5:
        return "Medium"
    return "Higher"


def nice_number(x, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def style_score_df(df: pd.DataFrame):
    """Colorize key numeric columns when possible."""
    if df is None or df.empty:
        return df

    candidates = [
        "investment_attractiveness",
        "match_score",
        "profile_score",
        "similarity",
        "similarity_to_ref",
        "urban_stability",
        "greenery_mean",
        "luminance_mean",
        "charm_mean",
    ]
    cols = [c for c in candidates if c in df.columns]
    try:
        styler = df.style
        for c in cols:
            styler = styler.background_gradient(subset=[c])
        return styler
    except Exception:
        return df


def make_radar_fig(labels, a_values, b_values, a_label="A", b_label="B"):
    vals = np.array([a_values, b_values], dtype=float)
    vals = np.where(np.isfinite(vals), vals, np.nan)

    mins = np.nanmin(vals, axis=0)
    maxs = np.nanmax(vals, axis=0)
    den = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
    norm = (vals - mins) / den
    norm = np.nan_to_num(norm, nan=0.0)

    a = norm[0].tolist()
    b = norm[1].tolist()
    a += a[:1]
    b += b[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1)

    fig = plt.figure(figsize=(5.6, 5.6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, a, linewidth=2, label=a_label)
    ax.fill(angles, a, alpha=0.15)
    ax.plot(angles, b, linewidth=2, label=b_label)
    ax.fill(angles, b, alpha=0.15)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
    ax.set_title("Radar (normalized per metric)")
    fig.tight_layout()
    return fig


def make_delta_bar_fig(delta_df: pd.DataFrame):
    df = delta_df.copy()
    fig = plt.figure(figsize=(6.6, 3.5))
    ax = plt.gca()

    metrics = df["metric"].astype(str).tolist()
    deltas = []
    for x in df["delta"].tolist():
        if x is None or (isinstance(x, float) and np.isnan(x)):
            deltas.append(0.0)
        else:
            deltas.append(float(x))

    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
    ax.bar(metrics, deltas, color=colors)
    ax.axhline(0, linewidth=1)
    ax.set_title("Δ (B − A)")
    ax.set_ylabel("Difference")
    fig.tight_layout()
    return fig
