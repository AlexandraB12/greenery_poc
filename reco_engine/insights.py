from __future__ import annotations

import pandas as pd


def strategy_summary_from_weights(sim, green, light, charm) -> str:
    weights = {"Similarity": sim, "Greenery": green, "Brightness": light, "Charm": charm}
    top = max(weights, key=weights.get)

    if top == "Similarity":
        return "🧠 **Similarity-first**: safest transfer of urban DNA (risk-controlled)."
    if top == "Greenery":
        return "🌿 **Green-first**: prioritizes greenery perception (QoL + desirability)."
    if top == "Brightness":
        return "☀️ **Brightness-first**: favors open/light morphology (premium feel, visibility)."
    return "✨ **Charm-first**: prioritizes perceived character (positioning, storytelling)."


def _score_col(df: pd.DataFrame):
    for c in ["match_score", "score", "weighted_score", "similarity", "similarity_to_ref"]:
        if c in df.columns:
            return c
    return None


def top_movers_vs_baseline(baseline: pd.DataFrame, selected: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    if baseline is None or selected is None or baseline.empty or selected.empty:
        return pd.DataFrame()
    if "zone_id" not in baseline.columns or "zone_id" not in selected.columns:
        return pd.DataFrame()

    bcol = _score_col(baseline)
    scol = _score_col(selected)
    if bcol is None or scol is None:
        return pd.DataFrame()

    b = baseline[["zone_id", bcol]].rename(columns={bcol: "baseline_score"}).copy()
    s = selected[["zone_id", scol]].rename(columns={scol: "selected_score"}).copy()

    b["baseline_rank"] = b["baseline_score"].rank(ascending=False, method="min")
    s["selected_rank"] = s["selected_score"].rank(ascending=False, method="min")

    m = b.merge(s, on="zone_id", how="inner")
    m["delta_rank"] = m["baseline_rank"] - m["selected_rank"]  # + = moved up
    m = m.sort_values("delta_rank", ascending=False)

    up = m.head(max(1, top // 2))
    down = m.sort_values("delta_rank", ascending=True).head(max(1, top // 2))
    out = pd.concat([up, down], axis=0).sort_values("delta_rank", ascending=False)

    return out[["zone_id", "baseline_rank", "selected_rank", "delta_rank", "baseline_score", "selected_score"]]


def badge_from_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "zone_id" not in df.columns:
        return pd.DataFrame()

    score_col = _score_col(df)
    if score_col is None:
        return pd.DataFrame()

    d = df.copy().sort_values(score_col, ascending=False).reset_index(drop=True)
    d["rank"] = d.index + 1
    d["badge"] = ""

    d.loc[d["rank"] <= 5, "badge"] = "🏆 Top Performer"
    d.loc[(d["rank"] > 5) & (d["rank"] <= 15), "badge"] = "🚀 High Potential"

    cols = ["zone_id", "rank", score_col, "badge"]
    if "risk_tier" in d.columns:
        cols.append("risk_tier")
    if "lat" in d.columns and "lon" in d.columns:
        cols += ["lat", "lon"]

    return d[cols].head(25)


def compare_insight(a, b, d_green, d_light, d_charm, kpi_df: pd.DataFrame | None) -> str:
    def fmt(name, val):
        if val is None:
            return None
        sign = "+" if val >= 0 else ""
        return f"{name} {sign}{val:.3f}"

    parts = [fmt("Greenery", d_green), fmt("Brightness", d_light), fmt("Charm", d_charm)]
    parts = [p for p in parts if p]
    delta_txt = ", ".join(parts) if parts else "Δ metrics not available."

    improvements = sum([(d_green is not None and d_green > 0), (d_light is not None and d_light > 0), (d_charm is not None and d_charm > 0)])
    downsides = sum([(d_green is not None and d_green < 0), (d_light is not None and d_light < 0), (d_charm is not None and d_charm < 0)])

    if improvements >= 2 and downsides <= 1:
        rec = f"✅ **{b} looks stronger** on most dimensions."
    elif downsides >= 2:
        rec = f"✅ **{a} looks more balanced** vs {b}."
    else:
        rec = "🟡 **Trade-off**: choose based on your strategy (green vs charm vs brightness)."

    return f"{rec}\n\n**Δ (B−A):** {delta_txt}"
