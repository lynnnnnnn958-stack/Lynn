#!/usr/bin/env python3
"""
step_edgar_eps_pit.py — point-in-time QUARTERLY EPS from SEC EDGAR (for PEAD)
============================================================================
The existing download_edgar_fundamentals.py fetches ANNUAL 10-K data. PEAD /
SUE needs QUARTERLY EPS with the exact date each report became public (know_date).
This fetches diluted EPS per fiscal quarter with its earliest filing date, so the
seasonal-random-walk surprise (SUE) can be computed with zero look-ahead.

Output: eps_pit.csv  columns: ticker, period_end, filed_date, eps, form
  - period_end : fiscal quarter end (fundamentals reference date)
  - filed_date : when the market first learned it (know_date) — PIT anchor
  - eps        : diluted EPS for the quarter

Free, no key. SEC rate limit ~10 req/s — we stay under.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).parent
UA = {"User-Agent": "canyon-quant research contact@example.com"}
OUT = ROOT / "eps_pit.csv"
CIK_CACHE = ROOT / "edgar_cik_cache.json"
RATE = 0.11                      # ~9 req/s
EPS_CONCEPTS = ["EarningsPerShareDiluted", "EarningsPerShareBasic"]


def cik_map() -> dict:
    if CIK_CACHE.exists():
        try:
            return json.load(open(CIK_CACHE))
        except Exception:
            pass
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=UA, timeout=20)
    m = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in r.json().values()}
    json.dump(m, open(CIK_CACHE, "w"))
    return m


def universe() -> list[str]:
    for f in ("sp500_price_history_deep.csv", "sp500_price_cache.csv"):
        p = ROOT / f
        if p.exists():
            cols = pd.read_csv(p, index_col=0, nrows=1).columns.tolist()
            return [c for c in cols if c not in ("SPY", "Date") and str(c).isalpha()]
    return []


def quarterly_eps(cik: str) -> pd.DataFrame:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        r = requests.get(url, headers=UA, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        facts = r.json().get("facts", {}).get("us-gaap", {})
    except Exception:
        return pd.DataFrame()

    for concept in EPS_CONCEPTS:
        if concept not in facts:
            continue
        units = facts[concept]["units"]
        key = next((k for k in units if "shares" in k.lower() or k == "USD/shares"), None)
        if not key:
            continue
        df = pd.DataFrame(units[key])
        if df.empty or "end" not in df.columns:
            continue
        # quarterly records: a ~3-month period (start→end ≈ 90d) OR fp in Q1-Q4
        df["end"] = pd.to_datetime(df["end"], errors="coerce")
        df["filed"] = pd.to_datetime(df.get("filed"), errors="coerce")
        if "start" in df.columns:
            df["start"] = pd.to_datetime(df["start"], errors="coerce")
            dur = (df["end"] - df["start"]).dt.days
            df = df[(dur >= 80) & (dur <= 100)]           # ~one quarter
        elif "fp" in df.columns:
            df = df[df["fp"].isin(["Q1", "Q2", "Q3", "Q4"])]
        df = df.dropna(subset=["end", "filed", "val"])
        if df.empty:
            continue
        # PIT: earliest filing per fiscal-quarter end
        df = df.sort_values("filed").drop_duplicates(subset="end", keep="first")
        out = df[["end", "filed", "val", "form"]].rename(
            columns={"end": "period_end", "filed": "filed_date", "val": "eps"})
        return out.reset_index(drop=True)
    return pd.DataFrame()


def main(max_tickers: int = 500):
    tickers = universe()[:max_tickers]
    cmap = cik_map()
    print(f"Fetching quarterly PIT EPS for {len(tickers)} tickers …")
    parts, got, miss = [], 0, 0
    for i, tk in enumerate(tickers, 1):
        cik = cmap.get(tk)
        if not cik:
            miss += 1; continue
        q = quarterly_eps(cik)
        if not q.empty:
            q.insert(0, "ticker", tk)
            parts.append(q); got += 1
        else:
            miss += 1
        time.sleep(RATE)
        if i % 50 == 0:
            print(f"  {i}/{len(tickers)} … {got} with data")
    if not parts:
        print("No EPS data fetched — aborting"); return
    df = pd.concat(parts, ignore_index=True).sort_values(["ticker", "period_end"])
    df.to_csv(OUT, index=False)
    print(f"✓ {OUT.name}: {len(df):,} quarterly EPS rows, {got} tickers "
          f"({df['period_end'].min()} → {df['period_end'].max()})  [{miss} missing]")


if __name__ == "__main__":
    main()
