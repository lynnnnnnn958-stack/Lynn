#!/usr/bin/env python3
"""
Canyon — Extended FRED Macro Data Sync
========================================
Fetches 14 key economic time series from FRED (St. Louis Fed).

No API key required — uses the free public CSV endpoint.
Optional: set FRED_API_KEY env var to use the official JSON API (10K calls/day, faster).

Complements step_macro_regime_outlook.py (5-indicator composite model).
This step saves the raw series to CSV for pipeline-wide use and
produces fred_macro_latest.json for dashboard integration.

New series beyond macro_outlook:
  ICSA       — Initial jobless claims (weekly, leading labor market)
  DGS10      — 10Y nominal Treasury yield (daily)
  DGS2       — 2Y nominal Treasury yield (daily)
  DFII10     — 10Y TIPS real yield (daily)
  BAMLC0A0CM — IG corporate OAS (daily, vs. HY OAS in macro_outlook)
  CPILFESL   — Core CPI YoY (monthly)
  PAYEMS     — Nonfarm payrolls (monthly)
  UMCSENT    — Michigan Consumer Sentiment (monthly)
  FEDFUNDS   — Effective federal funds rate (monthly)
  M2SL       — M2 money supply (monthly)

Output:
  fred_macro_data.csv    — all series as columns, date index, last 2 years
  fred_macro_latest.json — most recent values + 4-week / 3-month changes
"""

from __future__ import annotations

import json
import os
import time
import warnings
from datetime import datetime, date, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
OUT_CSV  = ROOT / "fred_macro_data.csv"
OUT_JSON = ROOT / "fred_macro_latest.json"

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
UA = {"User-Agent": "CanyonQuant research lynnnnnnn958@gmail.com"}

SERIES = {
    # Already in macro_outlook (overlap intentional for cross-check)
    "T10Y2Y":         "Yield Curve 10Y-2Y",
    "BAMLH0A0HYM2":   "HY OAS (High Yield Spread)",
    "UNRATE":         "Unemployment Rate",
    # New — yields
    "DGS10":          "10Y Treasury Yield",
    "DGS2":           "2Y Treasury Yield",
    "DFII10":         "10Y Real Yield (TIPS)",
    # New — credit
    "BAMLC0A0CM":     "IG Corporate OAS",
    # New — labor
    "ICSA":           "Initial Jobless Claims (weekly, K)",
    "PAYEMS":         "Nonfarm Payrolls (M)",
    # New — inflation
    "CPILFESL":       "Core CPI (index)",
    # New — monetary
    "FEDFUNDS":       "Fed Funds Rate",
    "M2SL":           "M2 Money Supply ($B)",
    # New — sentiment / activity
    "UMCSENT":        "Michigan Consumer Sentiment",
    "VIXCLS":         "VIX (CBOE, official FRED copy)",
}


def _fetch_csv(series_id: str, days: int = 730) -> pd.Series:
    """Fetch via free public FRED CSV endpoint — no API key needed."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, headers=UA, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text), parse_dates=["observation_date"])
        df = df.set_index("observation_date")
        col = series_id
        if col not in df.columns:
            col = df.columns[0]
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        return s[s.index >= cutoff].rename(series_id)
    except Exception as e:
        print(f"    FRED CSV {series_id}: {e}")
        return pd.Series(dtype=float, name=series_id)


def _fetch_api(series_id: str, days: int = 730) -> pd.Series:
    """Fetch via official FRED JSON API — requires FRED_API_KEY."""
    start = (date.today() - timedelta(days=days)).isoformat()
    url = (f"https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={FRED_API_KEY}"
           f"&file_type=json&observation_start={start}&sort_order=asc")
    try:
        r = requests.get(url, headers=UA, timeout=20)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        dates  = [o["date"] for o in obs if o["value"] != "."]
        values = [float(o["value"]) for o in obs if o["value"] != "."]
        s = pd.Series(values, index=pd.to_datetime(dates), name=series_id)
        return s
    except Exception as e:
        print(f"    FRED API {series_id}: {e}")
        return pd.Series(dtype=float, name=series_id)


def fetch(series_id: str, days: int = 730) -> pd.Series:
    if FRED_API_KEY:
        s = _fetch_api(series_id, days)
        if not s.empty:
            return s
    return _fetch_csv(series_id, days)


def _pct_change(s: pd.Series, periods: int) -> float | None:
    """Return % change over last N observations, or None if insufficient data."""
    if len(s) < periods + 1:
        return None
    old = float(s.iloc[-(periods + 1)])
    new = float(s.iloc[-1])
    if old == 0:
        return None
    return round((new - old) / abs(old) * 100, 3)


def _abs_change(s: pd.Series, periods: int) -> float | None:
    if len(s) < periods + 1:
        return None
    return round(float(s.iloc[-1]) - float(s.iloc[-(periods + 1)]), 4)


def build_latest_snapshot(all_series: dict[str, pd.Series]) -> dict:
    """Build fred_macro_latest.json: latest value + recent changes."""
    result = {}
    for sid, s in all_series.items():
        if s.empty:
            result[sid] = {"ok": False, "label": SERIES.get(sid, sid)}
            continue
        last_val  = float(s.iloc[-1])
        last_date = str(s.index[-1].date())

        # Approximate # of obs per period (daily vs weekly vs monthly series)
        n_total = len(s)
        if n_total > 400:
            # daily-ish: 21 obs ≈ 1 month, 63 ≈ 3 months
            p1m, p3m = 21, 63
        elif n_total > 80:
            # weekly: 4 obs ≈ 1 month, 13 ≈ 3 months
            p1m, p3m = 4, 13
        else:
            # monthly: 1 obs ≈ 1 month, 3 ≈ 3 months
            p1m, p3m = 1, 3

        result[sid] = {
            "ok":        True,
            "label":     SERIES.get(sid, sid),
            "value":     last_val,
            "as_of":     last_date,
            "chg_1m":    _abs_change(s, p1m),
            "chg_3m":    _abs_change(s, p3m),
            "pct_1m":    _pct_change(s, p1m),
            "pct_3m":    _pct_change(s, p3m),
        }
    return result


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"Canyon FRED Data Sync — {ts}")
    print(f"  Mode: {'API key (' + FRED_API_KEY[:4] + '...)' if FRED_API_KEY else 'public CSV (no key)'}")
    print("=" * 60)

    all_series: dict[str, pd.Series] = {}
    for sid, label in SERIES.items():
        print(f"  {sid:<20} {label} … ", end="", flush=True)
        t0 = time.time()
        s = fetch(sid)
        elapsed = time.time() - t0
        if s.empty:
            print("SKIP (no data)")
        else:
            all_series[sid] = s
            print(f"{len(s)} obs  latest={s.iloc[-1]:.3g}  ({elapsed:.1f}s)")
        time.sleep(0.3)  # be polite to FRED servers

    if not all_series:
        print("ERROR: No series fetched. Check connectivity.")
        return

    # Save wide CSV (date × series)
    print(f"\n  Assembling {len(all_series)} series into wide CSV …")
    df = pd.concat(all_series.values(), axis=1)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.to_csv(OUT_CSV, date_format="%Y-%m-%d")
    print(f"  Saved: {OUT_CSV.name}  ({df.shape[0]} rows × {df.shape[1]} cols)")

    # Save latest snapshot JSON
    snap = build_latest_snapshot(all_series)
    snap["_meta"] = {"as_of": ts, "n_series": len(all_series)}
    OUT_JSON.write_text(json.dumps(snap, indent=2, default=str))
    print(f"  Saved: {OUT_JSON.name}")

    # Print summary table
    print()
    print(f"  {'Series':<20} {'Latest':>10}  {'1M chg':>8}  {'3M chg':>8}  As-of")
    print(f"  {'-'*20} {'-'*10}  {'-'*8}  {'-'*8}  {'------'}")
    for sid, info in snap.items():
        if sid.startswith("_"):
            continue
        if not info.get("ok"):
            print(f"  {sid:<20} {'N/A':>10}")
            continue
        v    = f"{info['value']:.3g}"
        c1m  = f"{info['chg_1m']:+.3g}" if info.get("chg_1m") is not None else "  -"
        c3m  = f"{info['chg_3m']:+.3g}" if info.get("chg_3m") is not None else "  -"
        print(f"  {sid:<20} {v:>10}  {c1m:>8}  {c3m:>8}  {info['as_of']}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
