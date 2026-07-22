#!/usr/bin/env python3
"""
Canyon Ticker Universe Builder
================================
Fetches tickers for different universes and saves to CSV.
Used by step_short_scanner.py and step_dcf_valuation.py to expand
coverage beyond S&P 500.

Supported universes:
  sp500       ~503 tickers  (from Wikipedia)
  russell1000 ~1000 tickers (S&P 500 + Russell MidCap via Wikipedia)
  sp400       ~400 tickers  (S&P MidCap 400 via Wikipedia)

Output: universe_<name>.csv with columns: ticker, name, sector, gics

Usage:
  .venv/bin/python step_get_universe.py              # default: russell1000
  .venv/bin/python step_get_universe.py sp500
  .venv/bin/python step_get_universe.py sp400
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent

WIKIPEDIA_URLS = {
    "sp500":  "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400":  "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600":  "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}


def _fetch_wikipedia_tickers(url: str, table_index: int = 0) -> pd.DataFrame:
    """Read ticker table from Wikipedia. Returns df with ticker/name/sector."""
    try:
        tables = pd.read_html(url, header=0)
        t = tables[table_index]
        # Normalize column names
        t.columns = [c.lower().replace(" ", "_") for c in t.columns]
        # Map common column name variants
        rename = {}
        for c in t.columns:
            if "symbol" in c or "ticker" in c:
                rename[c] = "ticker"
            elif "security" in c or "company" in c or "name" in c:
                rename[c] = "name"
            elif "sector" in c or "gics_sector" in c:
                rename[c] = "sector"
            elif "sub" in c and "industry" in c:
                rename[c] = "industry"
        t = t.rename(columns=rename)
        # Clean tickers (Wikipedia uses periods for BRK.B etc.)
        if "ticker" in t.columns:
            t["ticker"] = t["ticker"].astype(str).str.replace(".", "-", regex=False).str.strip()
        return t[["ticker"] + [c for c in ["name", "sector", "industry"] if c in t.columns]]
    except Exception as e:
        print(f"  Wikipedia fetch error ({url}): {e}")
        return pd.DataFrame()


def build_sp500() -> pd.DataFrame:
    print("  Fetching S&P 500 from Wikipedia …")
    df = _fetch_wikipedia_tickers(WIKIPEDIA_URLS["sp500"])
    print(f"  → {len(df)} tickers")
    return df


def build_sp400() -> pd.DataFrame:
    print("  Fetching S&P MidCap 400 from Wikipedia …")
    df = _fetch_wikipedia_tickers(WIKIPEDIA_URLS["sp400"])
    print(f"  → {len(df)} tickers")
    return df


def build_sp600() -> pd.DataFrame:
    print("  Fetching S&P SmallCap 600 from Wikipedia …")
    df = _fetch_wikipedia_tickers(WIKIPEDIA_URLS["sp600"])
    print(f"  → {len(df)} tickers")
    return df


def build_russell1000() -> pd.DataFrame:
    """Russell 1000 = S&P 500 + S&P MidCap 400 (approximate, publicly available)."""
    sp500 = build_sp500()
    sp400 = build_sp400()
    combined = pd.concat([sp500, sp400], ignore_index=True)
    combined = combined.drop_duplicates(subset="ticker")
    print(f"  Russell 1000 proxy: {len(combined)} unique tickers (S&P 500 + MidCap 400)")
    return combined


UNIVERSE_BUILDERS = {
    "sp500":       build_sp500,
    "sp400":       build_sp400,
    "sp600":       build_sp600,
    "russell1000": build_russell1000,
}


def main(universe: str = "russell1000"):
    if universe not in UNIVERSE_BUILDERS:
        print(f"Unknown universe '{universe}'. Options: {list(UNIVERSE_BUILDERS)}")
        sys.exit(1)

    print("=" * 60)
    print(f"  Canyon Universe Builder — {universe} — {date.today()}")
    print("=" * 60)

    df = UNIVERSE_BUILDERS[universe]()
    if df.empty:
        print("  ERROR: No tickers fetched. Check internet connection.")
        sys.exit(1)

    out_path = ROOT / f"universe_{universe}.csv"
    df["as_of"] = date.today().isoformat()
    df.to_csv(out_path, index=False)
    print(f"\n  Saved {len(df)} tickers → {out_path.name}")
    print(f"  Short Scanner and DCF will use this file automatically.")
    print(f"\n  To run Short Scanner on full {universe} universe:")
    print(f"    CANYON_UNIVERSE={universe} .venv/bin/python step_short_scanner.py")
    print(f"  To run DCF on full {universe} universe:")
    print(f"    CANYON_UNIVERSE={universe} .venv/bin/python step_dcf_valuation.py")


if __name__ == "__main__":
    universe = sys.argv[1] if len(sys.argv) > 1 else "russell1000"
    main(universe)
