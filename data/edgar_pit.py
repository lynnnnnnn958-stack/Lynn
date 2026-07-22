"""
W4-W5: EDGAR Point-in-Time Fundamental Data
============================================
Fetches financial statement data from SEC EDGAR Company Facts API.
Critical: uses `filed` date as know_date (not period end date).
This is what makes the data truly point-in-time — we only use data
that was publicly available on a given signal date.

SEC API is free, no key needed.
Rate limit: ~10 requests/second per their policy.

Usage:
    from data.edgar_pit import fetch_pit_fundamentals, load_pit_fundamentals
    df = fetch_pit_fundamentals(["AAPL", "MSFT", "GOOGL"])
    df.to_csv("edgar_pit_fundamentals.csv", index=False)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent.parent

# SEC EDGAR Company Facts API — free, no auth needed
EDGAR_BASE  = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_TICKERS = "https://www.sec.gov/files/company_tickers.json"
HEADERS = {"User-Agent": "canyon_quant research lynnnnnnn958@gmail.com"}

# XBRL concept → local name mapping
# Each entry: (gaap_key, fallback_gaap_key, unit, our_name)
CONCEPTS = [
    # EPS
    ("EarningsPerShareBasic",                 "EarningsPerShareDiluted",     "USD/shares", "eps_basic"),
    # Revenue — multiple possible XBRL names
    ("Revenues",                              "SalesRevenueNet",             "USD",        "revenue"),
    # Operating cash flow
    ("NetCashProvidedByUsedInOperatingActivities", None,                     "USD",        "op_cf"),
    # Net income
    ("NetIncomeLoss",                         "ProfitLoss",                  "USD",        "net_income"),
    # Total assets
    ("Assets",                                None,                          "USD",        "total_assets"),
    # Total liabilities
    ("Liabilities",                           None,                          "USD",        "total_liabilities"),
    # Shares outstanding (for market cap estimation)
    ("CommonStockSharesOutstanding",          "EntityCommonStockSharesOutstanding", "shares", "shares_out"),
]


def _get_cik_map() -> dict[str, str]:
    """Download ticker → CIK mapping from SEC. Returns {TICKER: '0001234567'}."""
    cache = ROOT / "edgar_cik_map.json"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 7 * 86400:
        return json.loads(cache.read_text())

    try:
        r = requests.get(EDGAR_TICKERS, headers=HEADERS, timeout=30)
        r.raise_for_status()
        raw = r.json()
        mapping = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in raw.values()}
        cache.write_text(json.dumps(mapping))
        return mapping
    except Exception as e:
        print(f"  [EDGAR] CIK map fetch failed: {e}")
        return {}


def _fetch_company_facts(cik: str) -> Optional[dict]:
    """Fetch all XBRL facts for one company. Returns raw JSON dict."""
    url = EDGAR_BASE.format(cik=cik)
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _extract_concept(facts: dict, gaap_key: str, unit: str) -> list[dict]:
    """
    Extract all filings for one XBRL concept.
    Returns list of {period_end, know_date, val, form} dicts.
    """
    try:
        entries = (
            facts["facts"]["us-gaap"]
            .get(gaap_key, {})
            .get("units", {})
            .get(unit, [])
        )
    except (KeyError, TypeError):
        return []

    rows = []
    for e in entries:
        # Only keep annual (10-K) and quarterly (10-Q) filings
        form = e.get("form", "")
        if form not in ("10-K", "10-Q"):
            continue
        try:
            period_end = pd.Timestamp(e["end"])
            know_date  = pd.Timestamp(e["filed"])   # PIT: when market learned this
            val        = float(e["val"])
        except (KeyError, ValueError, TypeError):
            continue
        # Sanity: know_date must be AFTER period_end (filings are always delayed)
        if know_date <= period_end:
            continue
        rows.append({
            "period_end": period_end,
            "know_date":  know_date,
            "val":        val,
            "form":       form,
        })
    return rows


def fetch_pit_fundamentals(
    tickers: list[str],
    start_year: int = 2017,
    sleep_sec: float = 0.12,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch PIT fundamental data for a list of tickers.

    Returns DataFrame with columns:
        ticker, concept, period_end, know_date, val, form

    The key invariant: for any signal date D, you must only use rows
    where know_date <= D. This eliminates fundamental lookahead bias.

    Args:
        tickers:    List of ticker symbols (uppercase)
        start_year: Ignore filings before this year (reduces memory)
        sleep_sec:  Polite delay between SEC requests (10 req/s limit)
        cache_path: If provided, load from cache if fresh (< 7 days)
    """
    if cache_path and cache_path.exists():
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days < 7:
            print(f"  [EDGAR] Loading from cache ({age_days:.1f}d old): {cache_path}")
            return pd.read_csv(cache_path, parse_dates=["period_end", "know_date"])

    cik_map = _get_cik_map()
    all_rows: list[dict] = []
    start_ts = pd.Timestamp(f"{start_year}-01-01")
    n = len(tickers)

    for i, ticker in enumerate(tickers):
        cik = cik_map.get(ticker.upper())
        if not cik:
            continue

        facts = _fetch_company_facts(cik)
        if not facts:
            continue

        for gaap_key, fallback_key, unit, our_name in CONCEPTS:
            rows = _extract_concept(facts, gaap_key, unit)
            if not rows and fallback_key:
                rows = _extract_concept(facts, fallback_key, unit)

            for row in rows:
                if row["know_date"] >= start_ts:
                    all_rows.append({
                        "ticker":     ticker.upper(),
                        "concept":    our_name,
                        **row,
                    })

        if (i + 1) % 20 == 0:
            print(f"  [EDGAR] Fetched {i+1}/{n} tickers...")
        time.sleep(sleep_sec)

    if not all_rows:
        return pd.DataFrame(columns=["ticker", "concept", "period_end", "know_date", "val", "form"])

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["ticker", "concept", "know_date"]).reset_index(drop=True)

    if cache_path:
        df.to_csv(cache_path, index=False)
        print(f"  [EDGAR] Saved {len(df):,} rows → {cache_path}")

    return df


def load_pit_fundamentals(path: Optional[Path] = None) -> pd.DataFrame:
    """Load cached EDGAR PIT fundamentals. Returns empty DF if not found."""
    if path is None:
        path = ROOT / "edgar_pit_fundamentals.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["period_end", "know_date"])


def get_pit_snapshot(
    df: pd.DataFrame,
    concept: str,
    as_of: pd.Timestamp,
    lookback_quarters: int = 4,
) -> pd.Series:
    """
    Get the most recent known value of `concept` for each ticker,
    as of `as_of` date. This is the core PIT query.

    Only uses rows where know_date <= as_of (strict PIT).

    Args:
        df:                 Output of load_pit_fundamentals()
        concept:            One of: eps_basic, revenue, op_cf, net_income,
                            total_assets, total_liabilities, shares_out
        as_of:              Signal date (use only data known by this date)
        lookback_quarters:  Only look back this many quarters for recency

    Returns:
        pd.Series indexed by ticker, values = most recent known value
    """
    subset = df[(df["concept"] == concept) & (df["know_date"] <= as_of)].copy()
    if subset.empty:
        return pd.Series(dtype=float)

    # Take most recent known filing per ticker
    latest = subset.sort_values("know_date").groupby("ticker").last()
    # Drop stale data (older than lookback_quarters × 90 days)
    cutoff = as_of - pd.Timedelta(days=lookback_quarters * 92)
    latest = latest[latest["period_end"] >= cutoff]

    return latest["val"]


def compute_pit_accruals(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """
    Sloan (1996) Accruals = -(Net Income - Operating CF) / Total Assets
    Negative = high cash earnings = better quality (more negative = better)
    PIT: uses only data known by as_of.
    """
    ni     = get_pit_snapshot(df, "net_income",    as_of)
    op_cf  = get_pit_snapshot(df, "op_cf",         as_of)
    assets = get_pit_snapshot(df, "total_assets",  as_of)

    common = ni.index.intersection(op_cf.index).intersection(assets.index)
    if common.empty:
        return pd.Series(dtype=float)

    ni_c  = ni[common]
    cf_c  = op_cf[common]
    ast_c = assets[common].replace(0, np.nan)

    accruals = -(ni_c - cf_c) / ast_c
    # Winsorize at 1%/99% to remove extreme outliers
    lo, hi = accruals.quantile(0.01), accruals.quantile(0.99)
    return accruals.clip(lo, hi)


def compute_pit_revenue_growth(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """
    Year-over-year revenue growth using PIT data.
    Returns (revenue_now - revenue_1yr_ago) / |revenue_1yr_ago|
    """
    rev_now = get_pit_snapshot(df, "revenue", as_of, lookback_quarters=1)
    rev_1yr = get_pit_snapshot(df, "revenue", as_of - pd.Timedelta(days=365), lookback_quarters=1)

    common = rev_now.index.intersection(rev_1yr.index)
    if common.empty:
        return pd.Series(dtype=float)

    growth = (rev_now[common] - rev_1yr[common]) / rev_1yr[common].abs().replace(0, np.nan)
    lo, hi = growth.quantile(0.01), growth.quantile(0.99)
    return growth.clip(lo, hi)


def compute_pit_leverage(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Debt-to-Assets ratio = Total Liabilities / Total Assets (PIT)."""
    liab   = get_pit_snapshot(df, "total_liabilities", as_of)
    assets = get_pit_snapshot(df, "total_assets",      as_of)

    common = liab.index.intersection(assets.index)
    if common.empty:
        return pd.Series(dtype=float)

    leverage = liab[common] / assets[common].replace(0, np.nan)
    lo, hi = leverage.quantile(0.01), leverage.quantile(0.99)
    return leverage.clip(lo, hi)


if __name__ == "__main__":
    # Quick test: fetch 10 large-cap tickers
    import sys
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "UNH", "JNJ", "V"]
    print(f"Fetching EDGAR PIT fundamentals for {len(test_tickers)} tickers...")
    df = fetch_pit_fundamentals(
        test_tickers,
        start_year=2020,
        cache_path=ROOT / "edgar_pit_test.csv",
    )
    print(f"\nResult: {len(df):,} rows")
    print(df[["ticker", "concept", "period_end", "know_date", "val"]].head(20).to_string())

    # Verify PIT constraint: know_date always > period_end
    violations = df[df["know_date"] <= df["period_end"]]
    if violations.empty:
        print("\n✓ PIT check passed: all know_dates are after period_end")
    else:
        print(f"\n✗ PIT VIOLATION: {len(violations)} rows with know_date <= period_end")
        sys.exit(1)

    # Test snapshot query
    as_of = pd.Timestamp("2024-01-01")
    accruals = compute_pit_accruals(df, as_of)
    print(f"\nSloan Accruals as of {as_of.date()} ({len(accruals)} tickers):")
    print(accruals.sort_values().head(5).to_string())
