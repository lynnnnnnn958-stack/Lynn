#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 23 — Options Gamma / Dealer Hedging Layer

Purpose:
Adds the missing “options distribution, Gamma squeeze, dealer hedging pressure” module to the system.

Important principles:
- No fabricated options data.
- Without POLYGON_API_KEY, does not fake data — generates instructions and input template only.
- OI-based gamma exposure is a proxy only, not actual dealer positioning.
- True dealer long/short gamma requires trade direction, client/dealer classification, and open/close info — plain open interest cannot fully infer this.
- This module is for research only; no order submission.

Data source modes:
1. Polygon API mode:
   export POLYGON_API_KEY="your_key"
   python3 -u canyon_final_v9_step23_options_gamma_layer.py

2. Manual CSV mode:
   Prepare options_chain_input.csv with:
   ticker, expiration_date, option_type, strike, spot, gamma, delta, implied_volatility, open_interest, volume, bid, ask
   Then run the same script.

Outputs:
- options_chain_snapshot.csv
- gamma_squeeze_candidates.csv
- options_gamma_report.md
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, date
import os
import json
import math
import time
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np


ROOT = Path.cwd()

CHECKLIST_FILE = ROOT / "pre_trade_checklist.csv"
SIZING_FILE = ROOT / "position_sizing_recommendations.csv"
MANUAL_INPUT_FILE = ROOT / "options_chain_input.csv"

OUT_CHAIN = ROOT / "options_chain_snapshot.csv"
OUT_CANDIDATES = ROOT / "gamma_squeeze_candidates.csv"
OUT_REPORT = ROOT / "options_gamma_report.md"
OUT_TEMPLATE = ROOT / "options_chain_input_template.csv"


CONTRACT_SIZE = 100
MAX_TICKERS = 8
MAX_CONTRACTS_PER_TICKER = 250


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def fnum(x, default=np.nan) -> float:
    try:
        s = str(x).replace("$", "").replace(",", "").strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return default
        return float(s)
    except Exception:
        return default


def pct(x) -> str:
    try:
        return f"{float(x):.2%}"
    except Exception:
        return str(x)


def make_template() -> None:
    cols = [
        "ticker", "expiration_date", "option_type", "strike", "spot",
        "gamma", "delta", "implied_volatility", "open_interest", "volume", "bid", "ask"
    ]
    sample = pd.DataFrame([{
        "ticker": "SPY",
        "expiration_date": "2026-06-19",
        "option_type": "call",
        "strike": "650",
        "spot": "640",
        "gamma": "0.012",
        "delta": "0.45",
        "implied_volatility": "0.22",
        "open_interest": "10000",
        "volume": "2000",
        "bid": "8.10",
        "ask": "8.30",
    }], columns=cols)
    sample.to_csv(OUT_TEMPLATE, index=False)


def choose_tickers() -> list[str]:
    """
    Prefer current actionable checklist, then sizing recommendations.
    Avoid closed rows and blocked rows where possible.
    """
    tickers = []

    checklist = read_csv(CHECKLIST_FILE)
    if not checklist.empty and "ticker" in checklist.columns:
        df = checklist.copy()
        if "final_status" in df.columns:
            df = df[df["final_status"].astype(str).str.upper().isin(["PENDING_MANUAL_CHECKS", "ALREADY_OPEN_PAPER"])]
        if "suggested_weight" in df.columns:
            df["_w"] = pd.to_numeric(df["suggested_weight"], errors="coerce").fillna(0)
            df = df.sort_values("_w", ascending=False)
        tickers = df["ticker"].astype(str).str.upper().str.strip().dropna().tolist()

    if not tickers:
        sizing = read_csv(SIZING_FILE)
        if not sizing.empty and "ticker" in sizing.columns:
            if "suggested_weight" in sizing.columns:
                sizing["_w"] = pd.to_numeric(sizing["suggested_weight"], errors="coerce").fillna(0)
                sizing = sizing.sort_values("_w", ascending=False)
            tickers = sizing["ticker"].astype(str).str.upper().str.strip().dropna().tolist()

    # Deduplicate, remove blanks/cash labels.
    out = []
    for t in tickers:
        if not t or t in {"CASH", "TACTICAL_CASH"}:
            continue
        if t not in out:
            out.append(t)
    return out[:MAX_TICKERS]


def polygon_get_json(url: str, api_key: str) -> dict:
    sep = "&" if "?" in url else "?"
    full_url = url + f"{sep}apiKey={urllib.parse.quote(api_key)}"
    req = urllib.request.Request(full_url, headers={"User-Agent": "CanyonOptionsGamma/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read().decode("utf-8")
    return json.loads(data)


def parse_polygon_contract(ticker: str, item: dict) -> dict:
    details = item.get("details", {}) or {}
    greeks = item.get("greeks", {}) or {}
    day = item.get("day", {}) or {}
    last_quote = item.get("last_quote", {}) or {}
    underlying = item.get("underlying_asset", {}) or {}

    bid = last_quote.get("bid", np.nan)
    ask = last_quote.get("ask", np.nan)
    spot = underlying.get("price", np.nan)

    return {
        "ticker": ticker,
        "contract_ticker": details.get("ticker", ""),
        "expiration_date": details.get("expiration_date", ""),
        "option_type": details.get("contract_type", ""),
        "strike": details.get("strike_price", np.nan),
        "spot": spot,
        "gamma": greeks.get("gamma", np.nan),
        "delta": greeks.get("delta", np.nan),
        "theta": greeks.get("theta", np.nan),
        "vega": greeks.get("vega", np.nan),
        "implied_volatility": item.get("implied_volatility", np.nan),
        "open_interest": item.get("open_interest", np.nan),
        "volume": day.get("volume", np.nan),
        "bid": bid,
        "ask": ask,
        "mid": (fnum(bid, np.nan) + fnum(ask, np.nan)) / 2 if np.isfinite(fnum(bid, np.nan)) and np.isfinite(fnum(ask, np.nan)) else np.nan,
    }


def fetch_polygon_chain(tickers: list[str], api_key: str) -> pd.DataFrame:
    rows = []

    for ticker in tickers:
        print(f"[Options] Fetching Polygon option chain snapshot for {ticker} ...")
        base = f"https://api.polygon.io/v3/snapshot/options/{urllib.parse.quote(ticker)}"
        params = urllib.parse.urlencode({"limit": MAX_CONTRACTS_PER_TICKER})
        url = base + "?" + params

        try:
            data = polygon_get_json(url, api_key)
        except Exception as e:
            print(f"[WARN] {ticker} Polygon fetch failed: {e}")
            continue

        results = data.get("results", []) or []
        for item in results:
            rows.append(parse_polygon_contract(ticker, item))

        # Keep it polite.
        time.sleep(0.3)

    return pd.DataFrame(rows)


def load_manual_chain() -> pd.DataFrame:
    df = read_csv(MANUAL_INPUT_FILE)
    if df.empty:
        return df

    rename = {}
    for col in df.columns:
        c = col.lower().strip()
        if c in {"symbol", "underlying"}:
            rename[col] = "ticker"
        elif c in {"expiry", "expiration", "exp"}:
            rename[col] = "expiration_date"
        elif c in {"type", "right", "cp"}:
            rename[col] = "option_type"
        elif c in {"iv", "implied_vol"}:
            rename[col] = "implied_volatility"
        elif c in {"oi", "openinterest"}:
            rename[col] = "open_interest"
    df = df.rename(columns=rename)

    required = ["ticker", "expiration_date", "option_type", "strike", "spot", "gamma", "open_interest"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"options_chain_input.csv missing columns: {missing}")

    return df


def normalize_chain(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    for col in ["ticker", "option_type", "expiration_date"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(str)

    out["ticker"] = out["ticker"].str.upper().str.strip()
    out["option_type"] = out["option_type"].str.lower().str.strip()

    for col in ["strike", "spot", "gamma", "delta", "theta", "vega", "implied_volatility", "open_interest", "volume", "bid", "ask", "mid"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Days to expiration
    exp = pd.to_datetime(out["expiration_date"], errors="coerce")
    now = pd.Timestamp(date.today())
    out["dte"] = (exp - now).dt.days

    out = out[
        out["ticker"].ne("")
        & out["option_type"].isin(["call", "put"])
        & out["strike"].notna()
        & out["spot"].notna()
        & out["gamma"].notna()
        & out["open_interest"].notna()
    ].copy()

    out["open_interest"] = out["open_interest"].clip(lower=0)
    out["volume"] = out["volume"].fillna(0).clip(lower=0)
    out["dte"] = out["dte"].fillna(9999)

    # Dollar gamma exposure proxy per 1% move.
    # GEX ≈ gamma * OI * contract_size * S^2 * 1%
    # Sign convention here is proxy only: calls positive, puts negative.
    sign = np.where(out["option_type"].eq("call"), 1.0, -1.0)
    out["gex_1pct_proxy"] = sign * out["gamma"] * out["open_interest"] * CONTRACT_SIZE * (out["spot"] ** 2) * 0.01
    out["abs_gex_1pct_proxy"] = out["gex_1pct_proxy"].abs()

    out["moneyness"] = out["strike"] / out["spot"] - 1.0
    out["otm_call_near"] = (out["option_type"].eq("call")) & (out["strike"] > out["spot"]) & (out["strike"] <= out["spot"] * 1.15) & (out["dte"] <= 21)
    out["otm_put_near"] = (out["option_type"].eq("put")) & (out["strike"] < out["spot"]) & (out["strike"] >= out["spot"] * 0.85) & (out["dte"] <= 21)

    out["volume_oi_ratio"] = np.where(out["open_interest"] > 0, out["volume"] / out["open_interest"], np.nan)

    return out


def top_strike(df: pd.DataFrame, ticker: str, option_type: str, side_filter: str) -> dict:
    d = df[(df["ticker"].eq(ticker)) & (df["option_type"].eq(option_type))].copy()
    if d.empty:
        return {}

    spot = float(d["spot"].dropna().iloc[0])
    if side_filter == "above":
        d = d[d["strike"] >= spot]
    elif side_filter == "below":
        d = d[d["strike"] <= spot]

    if d.empty:
        return {}
    d = d.sort_values("abs_gex_1pct_proxy", ascending=False)
    r = d.iloc[0]
    return {
        "strike": float(r["strike"]),
        "dte": int(r["dte"]) if pd.notna(r["dte"]) else None,
        "gex": float(r["gex_1pct_proxy"]),
        "abs_gex": float(r["abs_gex_1pct_proxy"]),
        "distance": float(r["strike"] / spot - 1.0),
    }


def summarize_ticker(df: pd.DataFrame, ticker: str) -> dict:
    d = df[df["ticker"].eq(ticker)].copy()
    if d.empty:
        return {}

    spot = float(d["spot"].dropna().iloc[0])

    total_gex = float(d["gex_1pct_proxy"].sum())
    call_gex = float(d[d["option_type"].eq("call")]["gex_1pct_proxy"].sum())
    put_gex = float(d[d["option_type"].eq("put")]["gex_1pct_proxy"].sum())

    near_calls = d[d["otm_call_near"]].copy()
    near_puts = d[d["otm_put_near"]].copy()

    call_wall = top_strike(df, ticker, "call", "above")
    put_wall = top_strike(df, ticker, "put", "below")

    near_call_oi = float(near_calls["open_interest"].sum()) if not near_calls.empty else 0.0
    near_call_vol = float(near_calls["volume"].sum()) if not near_calls.empty else 0.0
    near_put_oi = float(near_puts["open_interest"].sum()) if not near_puts.empty else 0.0

    call_vol_oi = near_call_vol / near_call_oi if near_call_oi > 0 else np.nan
    put_call_near_oi_ratio = near_put_oi / near_call_oi if near_call_oi > 0 else np.nan

    if not near_calls.empty:
        top_call_oi = near_calls.sort_values("open_interest", ascending=False).iloc[0]
        top_near_call_strike = float(top_call_oi["strike"])
        top_near_call_oi = float(top_call_oi["open_interest"])
        top_call_distance = top_near_call_strike / spot - 1
    else:
        top_near_call_strike = np.nan
        top_near_call_oi = 0.0
        top_call_distance = np.nan

    distance_to_call_wall = call_wall.get("distance", np.nan) if call_wall else np.nan

    # Heuristic gamma squeeze score.
    # This is a proxy, not a prediction.
    score = 0
    reasons = []

    if near_call_oi > 0:
        score += 15
        reasons.append("near-term OTM call OI exists")

    if np.isfinite(call_vol_oi) and call_vol_oi >= 0.20:
        score += 20
        reasons.append("near-term OTM call volume/OI elevated")

    if np.isfinite(distance_to_call_wall) and 0 <= distance_to_call_wall <= 0.05:
        score += 20
        reasons.append("spot close to call gamma wall")

    if call_gex > abs(put_gex) * 1.2:
        score += 15
        reasons.append("call-side gamma proxy dominates put-side")

    if total_gex < 0:
        score += 15
        reasons.append("net GEX proxy negative / potentially unstable hedging regime")
    else:
        reasons.append("net GEX proxy positive / more pinning-stabilizing regime")

    if np.isfinite(put_call_near_oi_ratio) and put_call_near_oi_ratio < 0.7:
        score += 10
        reasons.append("near-term put/call OI ratio low")

    if not np.isfinite(distance_to_call_wall) or abs(distance_to_call_wall) > 0.10:
        score -= 10
        reasons.append("call wall too far from spot")

    score = max(0, min(100, int(score)))

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
        "contracts": len(d),
        "net_gex_1pct_proxy": total_gex,
        "call_gex_1pct_proxy": call_gex,
        "put_gex_1pct_proxy": put_gex,
        "call_wall_strike": call_wall.get("strike", np.nan) if call_wall else np.nan,
        "call_wall_distance": distance_to_call_wall,
        "put_wall_strike": put_wall.get("strike", np.nan) if put_wall else np.nan,
        "put_wall_distance": put_wall.get("distance", np.nan) if put_wall else np.nan,
        "near_otm_call_oi": near_call_oi,
        "near_otm_call_volume": near_call_vol,
        "near_call_volume_oi_ratio": call_vol_oi,
        "near_put_call_oi_ratio": put_call_near_oi_ratio,
        "top_near_call_strike": top_near_call_strike,
        "top_near_call_distance": top_call_distance,
        "top_near_call_oi": top_near_call_oi,
        "gamma_squeeze_score": score,
        "gamma_squeeze_label": label,
        "reasons": "; ".join(reasons),
    }


def analyze_chain(chain: pd.DataFrame) -> pd.DataFrame:
    if chain.empty:
        return pd.DataFrame()
    rows = []
    for ticker in sorted(chain["ticker"].unique()):
        rows.append(summarize_ticker(chain, ticker))
    out = pd.DataFrame([r for r in rows if r])
    if not out.empty:
        out = out.sort_values("gamma_squeeze_score", ascending=False)
    return out


def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No data._"
    d = df.head(max_rows).copy()
    for col in d.columns:
        if "distance" in col or "ratio" in col:
            d[col] = d[col].map(lambda x: pct(x) if pd.notna(x) and x != "" else "")
        if "gex" in col:
            d[col] = pd.to_numeric(d[col], errors="coerce").map(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
    try:
        return d.to_markdown(index=False)
    except Exception:
        return d.to_string(index=False)


def build_no_data_report(tickers: list[str]) -> str:
    md = []
    md.append("# Canyon v9 Step 23 — Options Gamma / Dealer Hedging Layer")
    md.append("")
    md.append("## Status: NO OPTIONS DATA")
    md.append("")
    md.append("No option chain data available; system will not generate gamma squeeze conclusions.")
    md.append("")
    md.append("### You are correct")
    md.append("")
    md.append("The prior Canyon v9 already has:")
    md.append("")
    md.append("- price momentum")
    md.append("- relative strength")
    md.append("- sector exposure")
    md.append("- stress test")
    md.append("- paper ledger")
    md.append("- learning attribution")
    md.append("")
    md.append("But has not yet implemented:")
    md.append("")
    md.append("- option chain distribution")
    md.append("- strike-level open interest")
    md.append("- gamma exposure by strike")
    md.append("- call wall / put wall")
    md.append("- gamma squeeze watch")
    md.append("- market maker hedging pressure proxy")
    md.append("")
    md.append("### Required data")
    md.append("")
    md.append("Choose one:")
    md.append("")
    md.append("1. Set Polygon API key:")
    md.append("")
    md.append("```bash")
    md.append("export POLYGON_API_KEY='your_key'")
    md.append("python3 -u canyon_final_v9_step23_options_gamma_layer.py")
    md.append("```")
    md.append("")
    md.append("2. Manually fill `options_chain_input.csv`; column format is in `options_chain_input_template.csv`.")
    md.append("")
    md.append("### Tickers to be analyzed")
    md.append("")
    md.append(", ".join(tickers) if tickers else "_No tickers found from current candidates._")
    md.append("")
    md.append("### Important limitations")
    md.append("")
    md.append("Open interest only shows where positions are concentrated; it cannot directly reveal whether dealers are long or short gamma. True dealer positioning requires trade direction, client/dealer classification, and open/close information.")
    md.append("")
    return "\n".join(md)


def build_report(chain: pd.DataFrame, candidates: pd.DataFrame, source: str) -> str:
    md = []
    md.append("# Canyon v9 Step 23 — Options Gamma / Dealer Hedging Layer")
    md.append("")
    md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"Data source: **{source}**")
    md.append("")
    md.append("## What this adds")
    md.append("")
    md.append("This layer estimates option-chain pressure using strike-level open interest, gamma, volume, and distance from spot.")
    md.append("")
    md.append("It adds the missing logic for:")
    md.append("")
    md.append("- option distribution by strike")
    md.append("- call wall / put wall")
    md.append("- net gamma exposure proxy")
    md.append("- near-term OTM call crowding")
    md.append("- gamma squeeze watch label")
    md.append("")
    md.append("## Important caveat")
    md.append("")
    md.append("This is **not** true dealer positioning. Open interest does not reveal whether dealers are long or short each option. This is an OI-based gamma pressure proxy.")
    md.append("")
    md.append("## Gamma Squeeze Candidates")
    md.append("")
    md.append(md_table(candidates, max_rows=30))
    md.append("")
    md.append("## Interpretation Rules")
    md.append("")
    md.append("- HIGH_GAMMA_SQUEEZE_WATCH: do not automatically buy; check news, liquidity, IV, and underlying momentum.")
    md.append("- Positive net GEX proxy often implies more pinning/stabilizing pressure.")
    md.append("- Negative net GEX proxy can imply more unstable hedging feedback, but only as a proxy.")
    md.append("- A nearby call wall can become a magnet/resistance zone; if spot breaks through with volume, hedging flows may accelerate.")
    md.append("- High call volume/OI ratio can show fresh call demand, but can also mean expensive IV and poor risk/reward.")
    md.append("")
    md.append("## Chain Coverage")
    md.append("")
    if not chain.empty:
        summary = chain.groupby("ticker").agg(
            contracts=("contract_ticker", "count") if "contract_ticker" in chain.columns else ("strike", "count"),
            min_dte=("dte", "min"),
            max_dte=("dte", "max"),
            total_oi=("open_interest", "sum"),
            total_volume=("volume", "sum"),
        ).reset_index()
        md.append(md_table(summary, max_rows=50))
    else:
        md.append("_No chain rows._")
    md.append("")
    return "\n".join(md)


def main():
    print("=" * 88)
    print("🏔 CANYON v9 Step 23")
    print("Options Gamma / Dealer Hedging Layer")
    print("=" * 88)

    make_template()

    tickers = choose_tickers()
    api_key = os.getenv("POLYGON_API_KEY", "").strip()

    source = "NONE"
    raw_chain = pd.DataFrame()

    if MANUAL_INPUT_FILE.exists():
        print("[Options] Loading manual options_chain_input.csv")
        raw_chain = load_manual_chain()
        source = "manual options_chain_input.csv"
    elif api_key:
        print("[Options] POLYGON_API_KEY found.")
        print(f"[Options] Tickers: {', '.join(tickers)}")
        raw_chain = fetch_polygon_chain(tickers, api_key)
        source = "Polygon option chain snapshot"
    else:
        print("[Options] No POLYGON_API_KEY and no options_chain_input.csv.")
        OUT_REPORT.write_text(build_no_data_report(tickers), encoding="utf-8")
        pd.DataFrame().to_csv(OUT_CHAIN, index=False)
        pd.DataFrame().to_csv(OUT_CANDIDATES, index=False)
        print("\nNo data mode. Files generated:")
        print(f"  {OUT_TEMPLATE}")
        print(f"  {OUT_REPORT}")
        print("\nNext: set POLYGON_API_KEY or fill options_chain_input.csv.")
        return

    chain = normalize_chain(raw_chain)
    if chain.empty:
        OUT_REPORT.write_text(build_no_data_report(tickers), encoding="utf-8")
        pd.DataFrame().to_csv(OUT_CHAIN, index=False)
        pd.DataFrame().to_csv(OUT_CANDIDATES, index=False)
        print("[Options] Chain empty after normalization.")
        print(f"Report: {OUT_REPORT}")
        return

    candidates = analyze_chain(chain)

    chain.to_csv(OUT_CHAIN, index=False)
    candidates.to_csv(OUT_CANDIDATES, index=False)
    OUT_REPORT.write_text(build_report(chain, candidates, source), encoding="utf-8")

    print(f"Chain rows: {len(chain)}")
    print(f"Candidate rows: {len(candidates)}")

    if not candidates.empty:
        print("\nTop gamma watch:")
        cols = ["ticker", "gamma_squeeze_label", "gamma_squeeze_score", "call_wall_strike", "call_wall_distance", "near_call_volume_oi_ratio"]
        print(candidates[cols].head(10).to_string(index=False))

    print("\nFiles generated:")
    print(f"  {OUT_CHAIN}")
    print(f"  {OUT_CANDIDATES}")
    print(f"  {OUT_REPORT}")
    print(f"  {OUT_TEMPLATE}")


if __name__ == "__main__":
    main()
