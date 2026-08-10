#!/usr/bin/env python3
"""
step_edgar_quarterly_fundamentals.py — PIT QUARTERLY quality fundamentals
=========================================================================
The annual quality test had only 15 data points (t=0.93). Quarterly data gives
~4x more INDEPENDENT observations → far tighter statistics. Fetches the pieces of
the strongest quality factor (Novy-Marx gross-profits-to-assets) plus ROA, each
with its EDGAR filed_date (know_date) for zero look-ahead.

Concepts (us-gaap), quarterly (~90-day frames), earliest filing per period:
  GrossProfit, Revenues, Assets, NetIncomeLoss

Output: quarterly_fundamentals.csv
  ticker, period_end, filed_date, gross_profit, revenues, assets, net_income
"""
from __future__ import annotations
import time
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).parent
UA = {"User-Agent": "canyon-quant research contact@example.com"}
OUT = ROOT / "quarterly_fundamentals.csv"
RATE = 0.11
CONCEPTS = {
    "gross_profit": ["GrossProfit"],
    "revenues":     ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "assets":       ["Assets"],
    "net_income":   ["NetIncomeLoss"],
}


def _series(facts: dict, names: list[str], quarterly: bool) -> pd.DataFrame:
    us = facts.get("facts", {}).get("us-gaap", {})
    for nm in names:
        if nm not in us:
            continue
        units = us[nm]["units"]
        key = "USD" if "USD" in units else next(iter(units), None)
        if not key:
            continue
        df = pd.DataFrame(units[key])
        if df.empty or "end" not in df.columns:
            continue
        df["end"] = pd.to_datetime(df["end"], errors="coerce")
        df["filed"] = pd.to_datetime(df.get("filed"), errors="coerce")
        if quarterly and "start" in df.columns:
            df["start"] = pd.to_datetime(df["start"], errors="coerce")
            dur = (df["end"] - df["start"]).dt.days
            df = df[(dur >= 80) & (dur <= 100)]            # flow → one quarter
        df = df.dropna(subset=["end", "filed", "val"])
        if df.empty:
            continue
        df = df.sort_values("filed").drop_duplicates("end", keep="first")
        return df[["end", "filed", "val"]]
    return pd.DataFrame()


def fundamentals(cik: str) -> pd.DataFrame:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        r = requests.get(url, headers=UA, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        facts = r.json()
    except Exception:
        return pd.DataFrame()

    # Assets is a stock (point-in-time), others are flows (quarterly)
    parts = {}
    for name, names in CONCEPTS.items():
        s = _series(facts, names, quarterly=(name != "assets"))
        if not s.empty:
            parts[name] = s.set_index("end")["val"].rename(name)
            parts[name + "_filed"] = s.set_index("end")["filed"].rename(name + "_filed")
    if "gross_profit" not in parts or "assets" not in parts:
        return pd.DataFrame()

    idx = parts["gross_profit"].index
    df = pd.DataFrame(index=idx)
    for c in ("gross_profit", "revenues", "assets", "net_income"):
        if c in parts:
            df[c] = parts[c].reindex(idx)
    # know_date = latest filing among the components we actually use
    filed_cols = [parts[c + "_filed"].reindex(idx) for c in ("gross_profit", "assets", "net_income") if c + "_filed" in parts]
    df["filed_date"] = pd.concat(filed_cols, axis=1).max(axis=1) if filed_cols else pd.NaT
    df["period_end"] = idx
    return df.dropna(subset=["filed_date"]).reset_index(drop=True)


def main():
    import step_edgar_eps_pit as E
    tks = E.universe()
    cmap = E.cik_map()
    print(f"Fetching PIT quarterly fundamentals for {len(tks)} tickers …")
    parts, got = [], 0
    for i, tk in enumerate(tks, 1):
        cik = cmap.get(tk)
        if not cik:
            continue
        f = fundamentals(cik)
        if not f.empty:
            f.insert(0, "ticker", tk); parts.append(f); got += 1
        time.sleep(RATE)
        if i % 100 == 0:
            print(f"  {i}/{len(tks)} … {got} with data")
    if not parts:
        print("no data"); return
    df = pd.concat(parts, ignore_index=True).sort_values(["ticker", "period_end"])
    df.to_csv(OUT, index=False)
    print(f"✓ {OUT.name}: {len(df):,} rows, {got} tickers "
          f"({df['period_end'].min()} → {df['period_end'].max()})")


if __name__ == "__main__":
    main()
