#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 24 — Option Pin / Dealer "Kill Zone" Risk Layer

Purpose:
Converts “market-maker option kill / premium kill / pinning / IV crush / theta bleed”
into a pre-trade risk check layer.

Important notes:
- This does not mean market makers intentionally “kill retail options.”
- More precisely: options market structure, dealer hedging, expiry dates, OI concentration, IV decline,
  bid-ask and time-value decay create an environment where long option buyers frequently lose.

- This script performs risk identification only; no order submission, no broker connection.
- Without options_chain_snapshot.csv, generates an explanation report rather than fabricating data.

Inputs:
- options_chain_snapshot.csv  from Step 23 or manual options_chain_input.csv
- gamma_squeeze_candidates.csv
- pre_trade_checklist.csv
- position_sizing_recommendations.csv

Outputs:
- option_kill_zone_report.md
- option_kill_zone_risk.csv
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, date
import pandas as pd
import numpy as np

ROOT = Path.cwd()

CHAIN_FILE = ROOT / "options_chain_snapshot.csv"
GAMMA_FILE = ROOT / "gamma_squeeze_candidates.csv"
PRETRADE_FILE = ROOT / "pre_trade_checklist.csv"
SIZING_FILE = ROOT / "position_sizing_recommendations.csv"

OUT_CSV = ROOT / "option_kill_zone_risk.csv"
OUT_MD = ROOT / "option_kill_zone_report.md"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def fnum(x, default=np.nan):
    try:
        s = str(x).replace("$", "").replace(",", "").strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return default
        return float(s)
    except Exception:
        return default


def pct(x):
    try:
        return f"{float(x):.2%}"
    except Exception:
        return str(x)


def normalize_chain(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    for col in ["ticker", "expiration_date", "option_type"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(str)

    out["ticker"] = out["ticker"].str.upper().str.strip()
    out["option_type"] = out["option_type"].str.lower().str.strip()

    for col in ["strike", "spot", "gamma", "delta", "implied_volatility", "open_interest", "volume", "bid", "ask", "mid"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "dte" not in out.columns:
        exp = pd.to_datetime(out["expiration_date"], errors="coerce")
        now = pd.Timestamp(date.today())
        out["dte"] = (exp - now).dt.days
    else:
        out["dte"] = pd.to_numeric(out["dte"], errors="coerce")

    out = out[
        out["ticker"].ne("")
        & out["option_type"].isin(["call", "put"])
        & out["strike"].notna()
        & out["spot"].notna()
        & out["open_interest"].notna()
    ].copy()

    out["spread_pct"] = np.where(
        (out["bid"].notna()) & (out["ask"].notna()) & ((out["bid"] + out["ask"]) > 0),
        (out["ask"] - out["bid"]) / ((out["ask"] + out["bid"]) / 2),
        np.nan
    )

    out["moneyness"] = out["strike"] / out["spot"] - 1
    out["abs_moneyness"] = out["moneyness"].abs()
    out["near_expiry"] = out["dte"].fillna(9999) <= 7
    out["monthly_or_weekly_expiry_zone"] = out["dte"].fillna(9999) <= 10

    return out


def candidate_tickers() -> list[str]:
    tickers = []
    pre = read_csv(PRETRADE_FILE)
    if not pre.empty and "ticker" in pre.columns:
        df = pre.copy()
        if "final_status" in df.columns:
            df = df[df["final_status"].astype(str).str.upper().isin(["PENDING_MANUAL_CHECKS", "ALREADY_OPEN_PAPER"])]
        tickers = df["ticker"].astype(str).str.upper().str.strip().tolist()

    if not tickers:
        sizing = read_csv(SIZING_FILE)
        if not sizing.empty and "ticker" in sizing.columns:
            tickers = sizing["ticker"].astype(str).str.upper().str.strip().tolist()

    out = []
    for t in tickers:
        if t and t not in {"CASH", "TACTICAL_CASH"} and t not in out:
            out.append(t)
    return out


def max_pain_proxy(df: pd.DataFrame, ticker: str) -> dict:
    """
    Max pain proxy:
    For each candidate expiry, calculate aggregate option payout by settlement price at all strikes.
    The settlement price with the minimum total payout is "max pain" proxy.

    This is heuristic and OI-based; it is not proof of manipulation.
    """
    d = df[df["ticker"].eq(ticker)].copy()
    if d.empty:
        return {}

    # choose nearest expiry with enough OI
    exp_groups = []
    for exp, g in d.groupby("expiration_date"):
        total_oi = g["open_interest"].sum()
        dte = g["dte"].dropna().min() if "dte" in g.columns else np.nan
        exp_groups.append((exp, dte, total_oi))
    exp_df = pd.DataFrame(exp_groups, columns=["expiration_date", "dte", "total_oi"])
    exp_df = exp_df[exp_df["total_oi"] > 0].sort_values(["dte", "total_oi"], ascending=[True, False])
    if exp_df.empty:
        return {}

    exp = exp_df.iloc[0]["expiration_date"]
    g = d[d["expiration_date"].eq(exp)].copy()
    strikes = np.sort(g["strike"].dropna().unique())
    if len(strikes) == 0:
        return {}

    payouts = []
    calls = g[g["option_type"].eq("call")]
    puts = g[g["option_type"].eq("put")]

    for settle in strikes:
        call_payoff = ((settle - calls["strike"]).clip(lower=0) * calls["open_interest"]).sum()
        put_payoff = ((puts["strike"] - settle).clip(lower=0) * puts["open_interest"]).sum()
        total = call_payoff + put_payoff
        payouts.append((settle, total))

    pain = pd.DataFrame(payouts, columns=["settlement_proxy", "total_payout_proxy"])
    pain = pain.sort_values("total_payout_proxy")
    r = pain.iloc[0]

    spot = float(g["spot"].dropna().iloc[0])
    max_pain = float(r["settlement_proxy"])
    distance = max_pain / spot - 1

    return {
        "nearest_expiration": exp,
        "max_pain_proxy": max_pain,
        "max_pain_distance": distance,
        "min_total_payout_proxy": float(r["total_payout_proxy"]),
    }


def oi_wall(df: pd.DataFrame, ticker: str, option_type: str, side: str) -> dict:
    d = df[(df["ticker"].eq(ticker)) & (df["option_type"].eq(option_type))].copy()
    if d.empty:
        return {}

    spot = float(d["spot"].dropna().iloc[0])
    if side == "above":
        d = d[d["strike"] >= spot]
    elif side == "below":
        d = d[d["strike"] <= spot]

    if d.empty:
        return {}

    g = d.groupby("strike").agg(
        oi=("open_interest", "sum"),
        volume=("volume", "sum"),
        avg_dte=("dte", "mean"),
        avg_iv=("implied_volatility", "mean"),
    ).reset_index()

    g = g.sort_values("oi", ascending=False)
    r = g.iloc[0]

    return {
        "strike": float(r["strike"]),
        "distance": float(r["strike"] / spot - 1),
        "oi": float(r["oi"]),
        "volume": float(r["volume"]),
        "avg_dte": float(r["avg_dte"]) if pd.notna(r["avg_dte"]) else np.nan,
        "avg_iv": float(r["avg_iv"]) if pd.notna(r["avg_iv"]) else np.nan,
    }


def summarize_ticker(df: pd.DataFrame, gamma: pd.DataFrame, ticker: str) -> dict:
    d = df[df["ticker"].eq(ticker)].copy()
    if d.empty:
        return {}

    spot = float(d["spot"].dropna().iloc[0])
    near = d[d["dte"].fillna(9999) <= 7].copy()
    near_10 = d[d["dte"].fillna(9999) <= 10].copy()

    call_wall = oi_wall(df, ticker, "call", "above")
    put_wall = oi_wall(df, ticker, "put", "below")
    pain = max_pain_proxy(df, ticker)

    near_oi = float(near["open_interest"].sum()) if not near.empty else 0.0
    total_oi = float(d["open_interest"].sum())
    near_oi_ratio = near_oi / total_oi if total_oi > 0 else np.nan

    atm = d[d["abs_moneyness"] <= 0.03].copy()
    atm_oi_ratio = float(atm["open_interest"].sum()) / total_oi if total_oi > 0 and not atm.empty else 0.0

    avg_spread = float(d["spread_pct"].replace([np.inf, -np.inf], np.nan).dropna().median()) if d["spread_pct"].notna().any() else np.nan

    # Pull gamma summary if available.
    gamma_label = ""
    gamma_score = np.nan
    net_gex = np.nan
    if not gamma.empty and "ticker" in gamma.columns:
        gg = gamma[gamma["ticker"].astype(str).str.upper().eq(ticker)]
        if not gg.empty:
            gamma_label = str(gg.iloc[0].get("gamma_squeeze_label", ""))
            gamma_score = fnum(gg.iloc[0].get("gamma_squeeze_score", np.nan))
            net_gex = fnum(gg.iloc[0].get("net_gex_1pct_proxy", np.nan))

    score = 0
    reasons = []

    # Pin risk: max pain or OI wall close to spot near expiry.
    max_pain_distance = pain.get("max_pain_distance", np.nan) if pain else np.nan
    if np.isfinite(max_pain_distance) and abs(max_pain_distance) <= 0.02:
        score += 25
        reasons.append("spot close to max pain proxy")

    if np.isfinite(call_wall.get("distance", np.nan) if call_wall else np.nan) and 0 <= call_wall["distance"] <= 0.03:
        score += 20
        reasons.append("spot close below call OI wall")

    if np.isfinite(put_wall.get("distance", np.nan) if put_wall else np.nan) and -0.03 <= put_wall["distance"] <= 0:
        score += 15
        reasons.append("spot close above put OI wall")

    if np.isfinite(near_oi_ratio) and near_oi_ratio >= 0.35:
        score += 20
        reasons.append("large share of OI expires within 7 days")

    if atm_oi_ratio >= 0.25:
        score += 15
        reasons.append("ATM OI concentration can increase pinning/chop risk")

    # Liquidity / spread risk.
    if np.isfinite(avg_spread) and avg_spread > 0.12:
        score += 15
        reasons.append("median option spread is wide")

    # Gamma conflict: high squeeze watch but also pin/max pain risk.
    if np.isfinite(gamma_score) and gamma_score >= 45:
        score += 10
        reasons.append("gamma watch active; direction can accelerate or fail violently")

    score = max(0, min(100, int(score)))

    if score >= 70:
        label = "HIGH_OPTION_KILL_ZONE"
    elif score >= 45:
        label = "MEDIUM_OPTION_KILL_ZONE"
    elif score >= 25:
        label = "LOW_OPTION_KILL_ZONE"
    else:
        label = "LOW_PIN_RISK"

    return {
        "ticker": ticker,
        "spot": spot,
        "option_kill_zone_score": score,
        "option_kill_zone_label": label,
        "nearest_expiration": pain.get("nearest_expiration", "") if pain else "",
        "max_pain_proxy": pain.get("max_pain_proxy", np.nan) if pain else np.nan,
        "max_pain_distance": max_pain_distance,
        "call_oi_wall": call_wall.get("strike", np.nan) if call_wall else np.nan,
        "call_oi_wall_distance": call_wall.get("distance", np.nan) if call_wall else np.nan,
        "put_oi_wall": put_wall.get("strike", np.nan) if put_wall else np.nan,
        "put_oi_wall_distance": put_wall.get("distance", np.nan) if put_wall else np.nan,
        "near_expiry_oi_ratio": near_oi_ratio,
        "atm_oi_ratio": atm_oi_ratio,
        "median_spread_pct": avg_spread,
        "gamma_squeeze_label": gamma_label,
        "gamma_squeeze_score": gamma_score,
        "net_gex_1pct_proxy": net_gex,
        "interpretation": "; ".join(reasons),
        "action_rule": action_rule(label),
    }


def action_rule(label: str) -> str:
    if label == "HIGH_OPTION_KILL_ZONE":
        return "Avoid buying short-dated options; prefer stock paper trade or wait for breakout confirmation."
    if label == "MEDIUM_OPTION_KILL_ZONE":
        return "Do not chase options; reduce size, avoid weekly OTM contracts, check IV/spread."
    if label == "LOW_OPTION_KILL_ZONE":
        return "Proceed only with manual checks; avoid overpaying IV."
    return "No major pin/kill-zone signal from available chain data."


def analyze() -> pd.DataFrame:
    chain = normalize_chain(read_csv(CHAIN_FILE))
    gamma = read_csv(GAMMA_FILE)
    tickers = candidate_tickers()

    if chain.empty:
        return pd.DataFrame()

    rows = []
    for t in tickers:
        if t in set(chain["ticker"].unique()):
            rows.append(summarize_ticker(chain, gamma, t))

    # If no current candidates overlap chain, analyze all chain tickers.
    if not rows:
        for t in sorted(chain["ticker"].unique()):
            rows.append(summarize_ticker(chain, gamma, t))

    out = pd.DataFrame([r for r in rows if r])
    if not out.empty:
        out = out.sort_values("option_kill_zone_score", ascending=False)
    return out


def md_table(df: pd.DataFrame, max_rows=30) -> str:
    if df.empty:
        return "_No data._"
    d = df.head(max_rows).copy()
    for c in d.columns:
        if "distance" in c or "ratio" in c or "spread" in c:
            d[c] = d[c].map(lambda x: pct(x) if pd.notna(x) and str(x) != "" else "")
        if "gex" in c:
            d[c] = pd.to_numeric(d[c], errors="coerce").map(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
    try:
        return d.to_markdown(index=False)
    except Exception:
        return d.to_string(index=False)


def no_data_report() -> str:
    return """# Canyon v9 Step 24 — Option Pin / Dealer Kill Zone Risk

## Status: NO OPTIONS CHAIN DATA

No `options_chain_snapshot.csv` found; cannot compute:

- max pain proxy
- call OI wall
- put OI wall
- pin risk
- IV crush / theta bleed risk
- option kill zone score

This is not a code failure — the system has no option chain data. Run Step 23 first and provide a Polygon API key or manual `options_chain_input.csv`.

## What does this layer solve?

What is commonly called “market-maker option kill” breaks down into several distinct mechanisms:

1. **Pinning / Max pain effect**  
   Before expiry, price may oscillate around high-OI strikes while short-dated option buyers lose to theta decay.

2. **Call wall / Put wall**  
   Near high-OI strikes, resistance/support may form or acceleration may occur after a breakout.

3. **IV crush**  
   After earnings, events, FOMC, or CPI, even a correct directional call can lose if IV collapses too fast.

4. **Spread kill**  
   Wide bid-ask spreads mean an immediate loss on entry.

5. **Theta bleed**
   Weekly OTM options bleed time value if there is no rapid breakout.

6. **Dealer hedging feedback**
   If dealers are forced to dynamically hedge, it may cause pinning or trend acceleration. OI alone cannot prove actual dealer positioning.

## Current action

Do not state gamma/market-maker logic as definitive conclusions. Wait for real option chain data from Step 23 before scoring.
"""


def build_report(df: pd.DataFrame) -> str:
    md = []
    md.append("# Canyon v9 Step 24 — Option Pin / Dealer Kill Zone Risk")
    md.append("")
    md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    md.append("## Concept")
    md.append("")
    md.append(“Do not treat 'market-maker option kill' as conspiratorial manipulation. A more tradeable framing: options structure, OI distribution, expiry dates, IV, bid-ask, theta, and dealer hedging collectively create an environment where long option buyers frequently lose.”)
    md.append("")
    md.append("## Risk Table")
    md.append("")
    md.append(md_table(df, max_rows=50))
    md.append("")
    md.append("## Rules")
    md.append("")
    md.append("- HIGH_OPTION_KILL_ZONE: avoid chasing short-dated options; wait for breakout confirmation or equities paper only.")
    md.append("- MEDIUM_OPTION_KILL_ZONE: reduce position size, avoid weekly OTM call/put, check IV and spread.")
    md.append("- Max pain is an OI-based proxy only, not a guarantee of where price goes.")
    md.append("- OI cannot prove actual dealer long/short gamma positioning.")
    md.append(“- If both Gamma Squeeze Watch and Kill Zone appear simultaneously, interpret as 'may accelerate or may be pinned' — wait for price confirmation.”)
    md.append("")
    return "\n".join(md)


def main():
    print("=" * 88)
    print("🏔 CANYON v9 Step 24")
    print("Option Pin / Dealer Kill Zone Risk")
    print("=" * 88)

    if not CHAIN_FILE.exists() or read_csv(CHAIN_FILE).empty:
        OUT_CSV.write_text("", encoding="utf-8")
        OUT_MD.write_text(no_data_report(), encoding="utf-8")
        print("No options_chain_snapshot.csv found or file is empty.")
        print(f"Generated: {OUT_MD}")
        print("Next: run Step 23 with real options chain data.")
        return

    df = analyze()
    df.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(build_report(df), encoding="utf-8")

    print(f"Rows: {len(df)}")
    if not df.empty:
        cols = ["ticker", "option_kill_zone_label", "option_kill_zone_score", "max_pain_proxy", "call_oi_wall", "put_oi_wall", "action_rule"]
        print(df[cols].head(10).to_string(index=False))

    print("\nFiles generated:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_MD}")
    print("\nNext: open option_kill_zone_report.md")


if __name__ == "__main__":
    main()
