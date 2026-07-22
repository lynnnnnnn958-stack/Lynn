#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 29 — Fix / Diagnose Empty Options Gamma Table

Purpose:
If the Options/Gamma table in Dashboard v5 is empty, this script checks:
- Whether options_chain_snapshot.csv actually has rows
- Whether gamma / IV / OI / volume fields are missing
- Whether gamma_squeeze_candidates.csv is empty
- If possible, regenerates a more robust gamma_squeeze_candidates.csv from options_chain_snapshot.csv

Principles:
- Does not fabricate option chains
- No live orders
- If OI is missing, will explicitly mark as LOW_CONFIDENCE_VOLUME_FALLBACK
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

ROOT = Path.cwd()

CHAIN = ROOT / "options_chain_snapshot.csv"
CAND = ROOT / "gamma_squeeze_candidates.csv"
REPORT = ROOT / "options_gamma_diagnostics.md"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def num(s):
    return pd.to_numeric(s, errors="coerce")


def pct(x):
    try:
        return f"{float(x):.2%}"
    except Exception:
        return str(x)


def clean_chain(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    d = df.copy()

    for c in ["ticker", "expiration_date", "option_type"]:
        if c not in d.columns:
            d[c] = ""
        d[c] = d[c].astype(str)

    d["ticker"] = d["ticker"].str.upper().str.strip()
    d["option_type"] = d["option_type"].str.lower().str.strip()

    for c in ["strike", "spot", "gamma", "delta", "implied_volatility", "open_interest", "volume", "bid", "ask", "dte"]:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Keep rows that have at least strike/spot/type.
    d = d[
        d["ticker"].ne("")
        & d["option_type"].isin(["call", "put"])
        & d["strike"].notna()
        & d["spot"].notna()
    ].copy()

    # If gamma missing but IV exists, we still cannot accurately compute here; Step23B should do that.
    d["open_interest_missing"] = d["open_interest"].isna()
    d["volume_missing"] = d["volume"].isna()

    d["open_interest_filled"] = d["open_interest"].fillna(0)
    d["volume_filled"] = d["volume"].fillna(0)

    # If OI mostly missing, fallback pressure uses volume. Low confidence.
    d["oi_or_volume"] = np.where(d["open_interest_filled"] > 0, d["open_interest_filled"], d["volume_filled"])
    d["uses_volume_fallback"] = d["open_interest_filled"] <= 0

    d["gamma"] = d["gamma"].fillna(0)
    d["dte"] = d["dte"].fillna(9999)

    d["moneyness"] = d["strike"] / d["spot"] - 1.0
    d["near_otm_call"] = (
        d["option_type"].eq("call")
        & (d["strike"] > d["spot"])
        & (d["strike"] <= d["spot"] * 1.15)
        & (d["dte"] <= 21)
    )
    d["near_otm_put"] = (
        d["option_type"].eq("put")
        & (d["strike"] < d["spot"])
        & (d["strike"] >= d["spot"] * 0.85)
        & (d["dte"] <= 21)
    )

    sign = np.where(d["option_type"].eq("call"), 1.0, -1.0)
    d["gex_1pct_proxy"] = sign * d["gamma"] * d["oi_or_volume"] * 100 * (d["spot"] ** 2) * 0.01
    d["abs_gex_1pct_proxy"] = d["gex_1pct_proxy"].abs()

    return d


def wall(d: pd.DataFrame, ticker: str, opt_type: str, side: str) -> dict:
    x = d[(d["ticker"].eq(ticker)) & (d["option_type"].eq(opt_type))].copy()
    if x.empty:
        return {}

    spot = float(x["spot"].dropna().iloc[0])
    if side == "above":
        x = x[x["strike"] >= spot]
    elif side == "below":
        x = x[x["strike"] <= spot]
    if x.empty:
        return {}

    g = x.groupby("strike").agg(
        pressure=("abs_gex_1pct_proxy", "sum"),
        oi=("open_interest_filled", "sum"),
        volume=("volume_filled", "sum"),
        avg_dte=("dte", "mean"),
    ).reset_index()
    g = g.sort_values("pressure", ascending=False)
    r = g.iloc[0]
    return {
        "strike": float(r["strike"]),
        "distance": float(r["strike"] / spot - 1),
        "pressure": float(r["pressure"]),
        "oi": float(r["oi"]),
        "volume": float(r["volume"]),
        "avg_dte": float(r["avg_dte"]),
    }


def summarize_ticker(d: pd.DataFrame, ticker: str) -> dict:
    x = d[d["ticker"].eq(ticker)].copy()
    if x.empty:
        return {}

    spot = float(x["spot"].dropna().iloc[0])
    call_wall = wall(d, ticker, "call", "above")
    put_wall = wall(d, ticker, "put", "below")

    calls = x[x["option_type"].eq("call")]
    puts = x[x["option_type"].eq("put")]
    near_calls = x[x["near_otm_call"]]
    near_puts = x[x["near_otm_put"]]

    net_gex = float(x["gex_1pct_proxy"].sum())
    call_gex = float(calls["gex_1pct_proxy"].sum())
    put_gex = float(puts["gex_1pct_proxy"].sum())

    near_call_oi = float(near_calls["open_interest_filled"].sum())
    near_call_vol = float(near_calls["volume_filled"].sum())
    near_put_oi = float(near_puts["open_interest_filled"].sum())

    call_vol_oi = near_call_vol / near_call_oi if near_call_oi > 0 else np.nan
    put_call_near_oi_ratio = near_put_oi / near_call_oi if near_call_oi > 0 else np.nan

    call_wall_distance = call_wall.get("distance", np.nan) if call_wall else np.nan

    oi_rows = int((x["open_interest_filled"] > 0).sum())
    vol_fallback_rows = int(x["uses_volume_fallback"].sum())
    confidence = "NORMAL_OI_BASED" if oi_rows > 0 else "LOW_CONFIDENCE_VOLUME_FALLBACK"

    score = 0
    reasons = []

    if near_call_oi > 0 or near_call_vol > 0:
        score += 15
        reasons.append("near-term OTM call interest exists")

    if np.isfinite(call_vol_oi) and call_vol_oi >= 0.20:
        score += 20
        reasons.append("near-term call volume/OI elevated")
    elif near_call_oi == 0 and near_call_vol > 0:
        score += 10
        reasons.append("volume fallback: near-term call volume exists but OI missing")

    if np.isfinite(call_wall_distance) and 0 <= call_wall_distance <= 0.05:
        score += 20
        reasons.append("spot close to call wall")

    if abs(call_gex) > abs(put_gex) * 1.2:
        score += 15
        reasons.append("call-side pressure dominates put-side")

    if net_gex < 0:
        score += 15
        reasons.append("net GEX proxy negative / unstable hedging proxy")
    else:
        reasons.append("net GEX proxy positive / more pinning-stabilizing proxy")

    if np.isfinite(put_call_near_oi_ratio) and put_call_near_oi_ratio < 0.7:
        score += 10
        reasons.append("near-term put/call OI ratio low")

    if confidence.startswith("LOW"):
        score = max(0, score - 15)
        reasons.append("low confidence because OI is missing/zero")

    score = int(max(0, min(100, score)))

    if score >= 70:
        label = "HIGH_GAMMA_SQUEEZE_WATCH"
    elif score >= 45:
        label = "MEDIUM_GAMMA_WATCH"
    elif score >= 25:
        label = "LOW_GAMMA_WATCH"
    else:
        label = "NO_GAMMA_SETUP"

    return {
        "ticker": ticker,
        "spot": spot,
        "contracts": len(x),
        "data_confidence": confidence,
        "oi_rows": oi_rows,
        "volume_fallback_rows": vol_fallback_rows,
        "net_gex_1pct_proxy": net_gex,
        "call_gex_1pct_proxy": call_gex,
        "put_gex_1pct_proxy": put_gex,
        "call_wall_strike": call_wall.get("strike", np.nan) if call_wall else np.nan,
        "call_wall_distance": call_wall_distance,
        "put_wall_strike": put_wall.get("strike", np.nan) if put_wall else np.nan,
        "put_wall_distance": put_wall.get("distance", np.nan) if put_wall else np.nan,
        "near_otm_call_oi": near_call_oi,
        "near_otm_call_volume": near_call_vol,
        "near_call_volume_oi_ratio": call_vol_oi,
        "near_put_call_oi_ratio": put_call_near_oi_ratio,
        "gamma_squeeze_score": score,
        "gamma_squeeze_label": label,
        "reasons": "; ".join(reasons),
    }


def build_report(raw: pd.DataFrame, clean: pd.DataFrame, cand: pd.DataFrame) -> str:
    md = []
    md.append("# Canyon v9 Step 29 — Options Gamma Diagnostics")
    md.append("")
    md.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    md.append("")
    md.append("## File Status")
    md.append("")
    md.append(f"- options_chain_snapshot.csv exists: **{CHAIN.exists()}**")
    md.append(f"- raw rows: **{len(raw)}**")
    md.append(f"- clean rows: **{len(clean)}**")
    md.append(f"- candidate rows regenerated: **{len(cand)}**")
    md.append("")
    if not raw.empty:
        md.append("## Raw Columns")
        md.append("")
        md.append(", ".join(raw.columns))
        md.append("")
    if not clean.empty:
        md.append("## Data Quality")
        md.append("")
        q = pd.DataFrame([{
            "tickers": ", ".join(sorted(clean["ticker"].unique())),
            "gamma_nonzero_rows": int((clean["gamma"].abs() > 0).sum()),
            "oi_positive_rows": int((clean["open_interest_filled"] > 0).sum()),
            "volume_positive_rows": int((clean["volume_filled"] > 0).sum()),
            "rows_using_volume_fallback": int(clean["uses_volume_fallback"].sum()),
        }])
        md.append(q.to_markdown(index=False))
        md.append("")
    md.append("## Regenerated Gamma Candidates")
    md.append("")
    if cand.empty:
        md.append("_No candidate rows could be generated._")
    else:
        show = cand.copy()
        for c in show.columns:
            if "distance" in c or "ratio" in c:
                show[c] = pd.to_numeric(show[c], errors="coerce").map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
            if "gex" in c:
                show[c] = pd.to_numeric(show[c], errors="coerce").map(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
        md.append(show.to_markdown(index=False))
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append("- If data_confidence is NORMAL_OI_BASED, the gamma layer is using open interest.")
    md.append("- If data_confidence is LOW_CONFIDENCE_VOLUME_FALLBACK, Yahoo did not provide enough OI, so the result is weaker and should be treated only as a screen.")
    md.append("- This still does not prove true dealer long/short gamma.")
    md.append("")
    return "\n".join(md)


def main():
    print("=" * 88)
    print("🏔 CANYON v9 Step 29")
    print("Fix / Diagnose Empty Options Gamma Table")
    print("=" * 88)

    raw = read_csv(CHAIN)
    clean = clean_chain(raw)

    if clean.empty:
        pd.DataFrame().to_csv(CAND, index=False)
        REPORT.write_text(build_report(raw, clean, pd.DataFrame()), encoding="utf-8")
        print("No usable option chain rows.")
        print(f"Open: {REPORT}")
        return

    rows = []
    for t in sorted(clean["ticker"].unique()):
        rows.append(summarize_ticker(clean, t))
    cand = pd.DataFrame([r for r in rows if r])
    if not cand.empty:
        cand = cand.sort_values("gamma_squeeze_score", ascending=False)

    cand.to_csv(CAND, index=False)
    REPORT.write_text(build_report(raw, clean, cand), encoding="utf-8")

    print(f"Raw rows: {len(raw)}")
    print(f"Clean rows: {len(clean)}")
    print(f"Candidate rows: {len(cand)}")
    if not cand.empty:
        print(cand[["ticker", "data_confidence", "gamma_squeeze_label", "gamma_squeeze_score", "call_wall_strike", "call_wall_distance"]].to_string(index=False))

    print("\nFiles generated:")
    print(f"  {CAND}")
    print(f"  {REPORT}")
    print("\nNext:")
    print("  python3 -u canyon_final_v9_step24_option_kill_zone.py")
    print("  streamlit run canyon_final_v9_step26_dashboard_v5_options.py")


if __name__ == "__main__":
    main()
