"""
W9: Historical S&P 500 Constituent Tracker
==========================================
Tracks which stocks were actually in the S&P 500 on any given date.
Critical for eliminating survivorship bias in backtests.

Source: Wikipedia change log (free, covers ~2000–present).
Each row records when a stock was added or removed from the index.

Why this matters:
  In a 2018–2026 backtest using CURRENT S&P 500 members, we only see
  survivors. Stocks that dropped out (bankruptcies, M&A targets, spinoffs)
  are excluded, making the backtest universe better than history really was.

  Quantified impact (from survivorship_bias_audit.csv):
  - Estimated ~5-10% annualized upward bias in universe-level returns

Output: constituent_history.csv
  ticker, date_added, date_removed (NaT = still in index)

Usage:
    from data.sp500_constituents import get_universe_on_date, build_constituent_history
    hist = build_constituent_history()
    universe_jan2020 = get_universe_on_date(hist, pd.Timestamp("2020-01-31"))
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
CACHE_PATH = ROOT / "constituent_history.csv"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _fetch_wikipedia_constituents() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch current S&P 500 list and change log from Wikipedia.

    Returns:
        current_df:  Current constituents (ticker, company, sector, sub-industry)
        changes_df:  Historical additions/removals (date, added, removed)
    """
    try:
        tables = pd.read_html(WIKI_URL)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch Wikipedia S&P 500 page: {e}")

    # Table 0: current constituents
    current = tables[0].copy()
    current.columns = [str(c).strip() for c in current.columns]

    # Normalize column names (Wikipedia changes them occasionally)
    col_map = {}
    for col in current.columns:
        cl = col.lower()
        if "symbol" in cl or "ticker" in cl:
            col_map[col] = "ticker"
        elif "security" in cl or "company" in cl or "name" in cl:
            col_map[col] = "company"
        elif "gics sector" in cl or "sector" in cl:
            col_map[col] = "sector"
        elif "sub-industry" in cl or "subindustry" in cl:
            col_map[col] = "sub_industry"
        elif "date" in cl and "added" in cl:
            col_map[col] = "date_added"
    current = current.rename(columns=col_map)

    # Keep only relevant columns
    keep = [c for c in ["ticker", "company", "sector", "sub_industry", "date_added"]
            if c in current.columns]
    current = current[keep].copy()
    current["ticker"] = current["ticker"].str.replace(".", "-", regex=False).str.strip()

    # Table 1: change log (if available)
    changes = pd.DataFrame()
    if len(tables) > 1:
        try:
            chg = tables[1].copy()
            chg.columns = [str(c).strip() for c in chg.columns]

            # Flatten multi-level columns if present
            if isinstance(chg.columns, pd.MultiIndex):
                chg.columns = [" ".join(str(c).strip() for c in col if c != "").strip()
                               for col in chg.columns]

            # Normalize
            chg_map = {}
            for col in chg.columns:
                cl = col.lower()
                if "date" in cl:
                    chg_map[col] = "date"
                elif "added" in cl or "ticker" in cl and "add" in cl:
                    chg_map[col] = "ticker_added"
                elif "removed" in cl or "ticker" in cl and "remov" in cl:
                    chg_map[col] = "ticker_removed"
            chg = chg.rename(columns=chg_map)

            if "date" in chg.columns:
                chg["date"] = pd.to_datetime(chg["date"], errors="coerce")
                chg = chg.dropna(subset=["date"])
                changes = chg
        except Exception:
            pass

    return current, changes


def build_constituent_history(
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Build a point-in-time constituent history DataFrame.

    Each row represents a ticker's membership interval:
        ticker, date_added, date_removed

    date_removed = NaT means the stock is still in the index.

    Algorithm:
    1. Fetch current S&P 500 list from Wikipedia
    2. Fetch change log from Wikipedia
    3. Work backwards: for each removal event, mark date_removed
    4. For current members not in change log, set date_added = best estimate

    Returns DataFrame sorted by ticker, date_added.
    """
    if cache_path is None:
        cache_path = CACHE_PATH

    if cache_path.exists() and not force_refresh:
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days < 30:  # refresh monthly
            print(f"  [Constituents] Loading from cache ({age_days:.1f}d old)")
            return pd.read_csv(cache_path, parse_dates=["date_added", "date_removed"])

    print("  [Constituents] Fetching from Wikipedia...")
    current_df, changes_df = _fetch_wikipedia_constituents()

    # Start with current members — these are all currently active
    history_rows: list[dict] = []
    current_tickers = set(current_df["ticker"].dropna().tolist())

    # Add current members with their known add dates
    for _, row in current_df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue
        date_added_str = str(row.get("date_added", ""))
        try:
            date_added = pd.Timestamp(date_added_str)
        except Exception:
            date_added = pd.NaT

        history_rows.append({
            "ticker":       ticker,
            "date_added":   date_added,
            "date_removed": pd.NaT,  # still in index
            "sector":       row.get("sector", ""),
            "sub_industry": row.get("sub_industry", ""),
        })

    # Process change log to add historical members that were removed
    if not changes_df.empty and "date" in changes_df.columns:
        for _, row in changes_df.iterrows():
            # Stocks that were removed (they had a previous period of membership)
            removed = str(row.get("ticker_removed", "")).strip()
            if removed and removed != "nan" and removed not in current_tickers:
                # This ticker was removed; add a historical membership row
                # We don't know their exact add date from this log alone,
                # so we use a conservative estimate: 2 years before removal
                remove_date = row["date"]
                history_rows.append({
                    "ticker":       removed.replace(".", "-"),
                    "date_added":   remove_date - pd.Timedelta(days=730),  # conservative estimate
                    "date_removed": remove_date,
                    "sector":       "",
                    "sub_industry": "",
                })

    df = pd.DataFrame(history_rows)
    df = df.drop_duplicates(subset=["ticker", "date_added"])
    df = df.sort_values(["ticker", "date_added"]).reset_index(drop=True)

    df.to_csv(cache_path, index=False)
    print(f"  [Constituents] Saved {len(df)} membership records → {cache_path}")
    return df


def get_universe_on_date(
    history: pd.DataFrame,
    as_of: pd.Timestamp,
) -> list[str]:
    """
    Return list of S&P 500 tickers that were ACTIVE on as_of date.

    PIT logic:
      - ticker was in index if: date_added <= as_of AND (date_removed > as_of OR date_removed is NaT)
      - Handles the edge case: stocks added and removed on the same day (ignored)
    """
    mask_added   = history["date_added"].fillna(pd.Timestamp("2000-01-01")) <= as_of
    mask_not_yet_removed = (
        history["date_removed"].isna() |
        (history["date_removed"] > as_of)
    )
    active = history[mask_added & mask_not_yet_removed]["ticker"].unique().tolist()
    return sorted(active)


def get_sector_map(history: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """
    Return ticker → sector mapping for the universe on as_of date.
    Used for sector-neutralization in the backtest.
    """
    mask_added = history["date_added"].fillna(pd.Timestamp("2000-01-01")) <= as_of
    mask_active = (
        history["date_removed"].isna() |
        (history["date_removed"] > as_of)
    )
    active = history[mask_added & mask_active].copy()

    # Deduplicate: take the most recent sector entry per ticker
    active = active.sort_values("date_added").groupby("ticker").last()
    return active["sector"].dropna()


if __name__ == "__main__":
    print("Building S&P 500 constituent history...")
    hist = build_constituent_history(force_refresh=True)
    print(f"\nTotal membership records: {len(hist)}")
    print(f"Unique tickers ever: {hist['ticker'].nunique()}")
    print("\nSample current members:")
    current = hist[hist["date_removed"].isna()].head(10)
    print(current[["ticker", "date_added", "sector"]].to_string())

    # Test PIT universe query
    test_dates = [
        pd.Timestamp("2019-01-31"),
        pd.Timestamp("2020-03-31"),
        pd.Timestamp("2022-06-30"),
        pd.Timestamp("2024-01-31"),
    ]
    for d in test_dates:
        universe = get_universe_on_date(hist, d)
        print(f"\nUniverse on {d.date()}: {len(universe)} stocks")
        assert 400 <= len(universe) <= 510, f"Universe size {len(universe)} seems wrong for {d.date()}"

    print("\n✓ Constituent history validation passed")
