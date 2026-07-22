#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 25 — Clean Options Daily Runner

Purpose:
1. Fetch option chains using Step 23B if available.
2. Build Gamma Squeeze candidates directly from options_chain_snapshot.csv.
3. Run Step 24 Option Kill Zone if available.
4. Never trade. Never connect to broker.
"""

from pathlib import Path
from datetime import datetime, date
import subprocess
import sys
import pandas as pd
import numpy as np

ROOT = Path.cwd()
LOG = ROOT / "options_daily_runner_log.md"

FETCHER = "canyon_final_v9_step23b_yfinance_options_fetcher.py"
KILL_ZONE = "canyon_final_v9_step24_option_kill_zone.py"

CHAIN = ROOT / "options_chain_snapshot.csv"
CAND = ROOT / "gamma_squeeze_candidates.csv"
GAMMA_REPORT = ROOT / "options_gamma_report.md"


def run_optional(script_name):
    path = ROOT / script_name
    if not path.exists():
        msg = f"SKIP missing {script_name}"
        print(msg)
        return 0, msg

    print()
    print("=" * 88)
    print(f"Running {script_name}")
    print("=" * 88)

    p = subprocess.run(
        [sys.executable, "-u", script_name],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
    print(out)
    return p.returncode, out


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def pct(x):
    try:
        return f"{float(x):.2%}"
    except Exception:
        return str(x)


def normalise_chain(df):
    if df.empty:
        return df

    d = df.copy()

    for c in ["ticker", "expiration_date", "option_type"]:
        if c not in d.columns:
            d[c] = ""
        d[c] = d[c].astype(str)

    d["ticker"] = d["ticker"].str.upper().str.strip()
    d["option_type"] = d["option_type"].str.lower().str.strip()

    for c in [
        "strike", "spot", "gamma", "delta", "implied_volatility",
        "open_interest", "volume", "bid", "ask", "dte"
    ]:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = to_num(d[c])

    if "dte" not in d.columns or d["dte"].isna().all():
        exp = pd.to_datetime(d["expiration_date"], errors="coerce")
        d["dte"] = (exp - pd.Timestamp(date.today())).dt.days

    d = d[
        d["ticker"].ne("")
        & d["option_type"].isin(["call", "put"])
        & d["strike"].notna()
        & d["spot"].notna()
    ].copy()

    if d.empty:
        return d

    d["open_interest"] = d["open_interest"].fillna(0)
    d["volume"] = d["volume"].fillna(0)
    d["gamma"] = d["gamma"].fillna(0)
    d["dte"] = d["dte"].fillna(9999)

    d["oi_or_volume"] = np.where(d["open_interest"] > 0, d["open_interest"], d["volume"])
    d["uses_volume_fallback"] = d["open_interest"] <= 0

    sign = np.where(d["option_type"].eq("call"), 1.0, -1.0)
    d["gex_1pct_proxy"] = sign * d["gamma"] * d["oi_or_volume"] * 100 * (d["spot"] ** 2) * 0.01
    d["abs_gex_1pct_proxy"] = d["gex_1pct_proxy"].abs()

    d["moneyness"] = d["strike"] / d["spot"] - 1
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
    return d


def wall(d, ticker, option_type, side):
    x = d[(d["ticker"].eq(ticker)) & (d["option_type"].eq(option_type))].copy()
    if x.empty:
        return {}

    spot = float(x["spot"].dropna().iloc[0])

    if side == "above":
        x = x[x["strike"] >= spot]
    if side == "below":
        x = x[x["strike"] <= spot]

    if x.empty:
        return {}

    x["pressure"] = np.where(
        x["abs_gex_1pct_proxy"] > 0,
        x["abs_gex_1pct_proxy"],
        x["oi_or_volume"]
    )

    g = x.groupby("strike").agg(
        pressure=("pressure", "sum"),
        oi=("open_interest", "sum"),
        volume=("volume", "sum"),
        avg_dte=("dte", "mean"),
    ).reset_index().sort_values("pressure", ascending=False)

    r = g.iloc[0]
    return {
        "strike": float(r["strike"]),
        "distance": float(r["strike"] / spot - 1),
        "pressure": float(r["pressure"]),
        "oi": float(r["oi"]),
        "volume": float(r["volume"]),
        "avg_dte": float(r["avg_dte"]),
    }


def summarise_ticker(d, ticker):
    x = d[d["ticker"].eq(ticker)].copy()
    if x.empty:
        return {}

    spot = float(x["spot"].dropna().iloc[0])

    calls = x[x["option_type"].eq("call")]
    puts = x[x["option_type"].eq("put")]
    near_calls = x[x["near_otm_call"]]
    near_puts = x[x["near_otm_put"]]

    call_wall = wall(d, ticker, "call", "above")
    put_wall = wall(d, ticker, "put", "below")

    net_gex = float(x["gex_1pct_proxy"].sum())
    call_gex = float(calls["gex_1pct_proxy"].sum())
    put_gex = float(puts["gex_1pct_proxy"].sum())

    near_call_oi = float(near_calls["open_interest"].sum())
    near_call_vol = float(near_calls["volume"].sum())
    near_put_oi = float(near_puts["open_interest"].sum())

    call_vol_oi = near_call_vol / near_call_oi if near_call_oi > 0 else np.nan
    put_call_ratio = near_put_oi / near_call_oi if near_call_oi > 0 else np.nan

    call_wall_distance = call_wall.get("distance", np.nan) if call_wall else np.nan

    oi_rows = int((x["open_interest"] > 0).sum())
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
        reasons.append("volume fallback: call activity exists but OI weak")

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
        reasons.append("net GEX proxy positive / more pinning-stabilising proxy")

    if np.isfinite(put_call_ratio) and put_call_ratio < 0.7:
        score += 10
        reasons.append("near-term put/call OI ratio low")

    if confidence.startswith("LOW"):
        score = max(0, score - 15)
        reasons.append("low confidence because OI is weak/missing")

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
        "near_put_call_oi_ratio": put_call_ratio,
        "gamma_squeeze_score": score,
        "gamma_squeeze_label": label,
        "reasons": "; ".join(reasons),
    }


def build_gamma_layer():
    raw = read_csv(CHAIN)
    d = normalise_chain(raw)

    if d.empty:
        pd.DataFrame().to_csv(CAND, index=False)
        GAMMA_REPORT.write_text(
            "# Options Gamma Report\n\nNo usable options chain rows.\n",
            encoding="utf-8"
        )
        return pd.DataFrame(), "No usable options chain rows."

    rows = []
    for t in sorted(d["ticker"].unique()):
        rows.append(summarise_ticker(d, t))

    cand = pd.DataFrame([r for r in rows if r])
    if not cand.empty:
        cand = cand.sort_values("gamma_squeeze_score", ascending=False)

    cand.to_csv(CAND, index=False)

    md = []
    md.append("# Canyon v9 Step 25 — Options Gamma Report")
    md.append("")
    md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    md.append("## Important caveat")
    md.append("")
    md.append("This is an OI/volume-based gamma pressure proxy, not true dealer positioning.")
    md.append("")
    md.append("## Gamma Squeeze Candidates")
    md.append("")
    if cand.empty:
        md.append("_No candidates generated._")
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
    md.append("- HIGH/MEDIUM gamma watch is not a buy signal.")
    md.append("- If Kill Zone is also high, short-dated options can be pinned or IV-crushed.")
    md.append("- Use this as a screen before paper trading only.")
    md.append("")

    GAMMA_REPORT.write_text("\n".join(md), encoding="utf-8")
    return cand, f"Generated {len(cand)} gamma candidate rows."


def main():
    print("=" * 88)
    print("CANYON v9 Step 25 — CLEAN OPTIONS RUNNER")
    print("=" * 88)

    log = []
    log.append("# Canyon v9 Step 25 — Options Daily Runner Log")
    log.append("")
    log.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.append("")

    code, out = run_optional(FETCHER)
    log.append("## Step 23B Fetcher")
    log.append("```text")
    log.append(out[-5000:])
    log.append("```")

    cand, msg = build_gamma_layer()
    print(msg)
    log.append("## Internal Gamma Layer")
    log.append(msg)

    code2, out2 = run_optional(KILL_ZONE)
    log.append("## Step 24 Kill Zone")
    log.append("```text")
    log.append(out2[-5000:])
    log.append("```")

    LOG.write_text("\n".join(log), encoding="utf-8")

    print()
    print("=" * 88)
    print("Options runner: OK")
    print("=" * 88)
    print("Open:")
    print("  open options_daily_runner_log.md")
    print("  open options_gamma_report.md")
    print("  open option_kill_zone_report.md")


if __name__ == "__main__":
    main()
