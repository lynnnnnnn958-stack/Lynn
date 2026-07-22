"""
W8: Extend Price History to 8 Years (2018-2026)
=================================================
Current cache: sp500_price_cache.csv covers 2023-06-12 to 2026-06-09 (3 years).
This module extends it back to 2018-01-01, giving 8+ years of history.

Why 8 years matters:
  - IC t-statistic ∝ sqrt(N_periods). 12 periods → t≈2.0; 40 periods → t≈4.0
  - Walk-forward backtest needs >5 years to show statistical significance
  - Longer history improves LightGBM training stability
  - Covers COVID (2020), Fed hike cycle (2022), post-GFC recovery (2018-2019)

Approach:
  - Yahoo Finance free tier supports 8+ years of daily adjusted closes
  - Download in chunks of 50 tickers (Yahoo rate limit: ~2000 req/day)
  - Deduplicate and join with existing cache
  - Save as sp500_price_cache_8yr.csv (keeps original cache intact)

Usage:
    python data/extend_history.py
    # or from code:
    from data.extend_history import build_8yr_cache
    build_8yr_cache()
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
CACHE_8YR = ROOT / "sp500_price_cache_8yr.csv"
CACHE_ORIG = ROOT / "sp500_price_cache.csv"
START_DATE = "2018-01-01"
CHUNK_SIZE = 50
SLEEP_SEC = 2.0  # polite delay between Yahoo chunks


def _load_existing_cache() -> pd.DataFrame:
    """Load the current 3-year price cache."""
    if not CACHE_ORIG.exists():
        raise FileNotFoundError(f"Original cache not found: {CACHE_ORIG}")
    df = pd.read_csv(CACHE_ORIG, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df


def _download_history(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance.
    Handles rate limits and partial failures gracefully.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance required: pip install yfinance")

    all_frames: list[pd.DataFrame] = []
    n_chunks = (len(tickers) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for i in range(n_chunks):
        chunk = tickers[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
        try:
            raw = yf.download(
                tickers=chunk,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                continue

            # yfinance returns multi-level columns when >1 ticker
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"]
            else:
                prices = raw[["Close"]].rename(columns={"Close": chunk[0]})

            all_frames.append(prices)
            print(f"  Chunk {i+1}/{n_chunks}: {len(chunk)} tickers downloaded")
        except Exception as e:
            print(f"  Chunk {i+1}/{n_chunks}: FAILED — {e}")

        if i < n_chunks - 1:
            time.sleep(SLEEP_SEC)

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, axis=1)
    combined.index = pd.to_datetime(combined.index)
    combined = combined.sort_index()
    return combined


def build_8yr_cache(
    force_refresh: bool = False,
    start: str = START_DATE,
) -> pd.DataFrame:
    """
    Build the 8-year price cache by extending the existing 3-year cache.

    Steps:
    1. Load existing cache (2023-2026)
    2. Download 2018-2023 history from Yahoo
    3. Concatenate (no overlap — Yahoo ends at existing cache start)
    4. Save to sp500_price_cache_8yr.csv

    Critically: does NOT replace the original cache. You must explicitly
    pass sp500_price_cache_8yr.csv to canyon_v11_full.py.

    Returns the full 8-year DataFrame.
    """
    if CACHE_8YR.exists() and not force_refresh:
        age_days = (time.time() - CACHE_8YR.stat().st_mtime) / 86400
        if age_days < 7:
            print(f"  [History] Loading 8yr cache ({age_days:.1f}d old): {CACHE_8YR}")
            df = pd.read_csv(CACHE_8YR, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index)
            return df

    print(f"  [History] Loading existing 3-year cache...")
    existing = _load_existing_cache()
    existing_start = existing.index.min()
    tickers = existing.columns.tolist()

    print(f"  [History] Existing: {existing.index.min().date()} → {existing.index.max().date()}, {len(tickers)} tickers")
    print(f"  [History] Downloading historical data from {start} to {existing_start.date()}...")

    # Download historical period (before existing cache)
    historical = _download_history(
        tickers=tickers,
        start=start,
        end=existing_start.strftime("%Y-%m-%d"),
    )

    if historical.empty:
        print("  [History] Download failed — returning original 3yr cache")
        return existing

    # Align columns: keep only tickers present in existing cache
    shared_cols = [c for c in existing.columns if c in historical.columns]
    historical = historical[shared_cols]
    existing_aligned = existing[shared_cols]

    print(f"  [History] Historical: {historical.index.min().date()} → {historical.index.max().date()}")
    print(f"  [History] Shared tickers: {len(shared_cols)}")

    # Concatenate: historical rows + existing rows (no overlap by design)
    combined = pd.concat([historical, existing_aligned], axis=0)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    # Fill small gaps (holidays, data issues) with forward-fill up to 3 days
    combined = combined.ffill(limit=3)

    # Report coverage quality
    coverage = combined.notna().mean()
    low_cov = coverage[coverage < 0.7]
    if len(low_cov) > 0:
        print(f"  [History] Warning: {len(low_cov)} tickers have <70% coverage")

    combined.to_csv(CACHE_8YR)
    print(f"  [History] Saved: {combined.shape} → {CACHE_8YR}")
    print(f"  [History] Date range: {combined.index.min().date()} → {combined.index.max().date()}")

    return combined


def validate_8yr_cache(df: pd.DataFrame) -> dict:
    """
    Validate the 8-year cache for common data quality issues.
    Returns a dict of validation results.
    """
    results = {}

    # Check date range
    results["start_date"]    = df.index.min().date()
    results["end_date"]      = df.index.max().date()
    results["n_days"]        = len(df)
    results["n_tickers"]     = len(df.columns)
    results["years_covered"] = (df.index.max() - df.index.min()).days / 365.25

    # Missing data
    missing_pct = df.isna().mean()
    results["avg_missing_pct"] = float(missing_pct.mean())
    results["worst_ticker"]    = str(missing_pct.idxmax())
    results["worst_missing"]   = float(missing_pct.max())

    # Price sanity: no negative prices, no extreme jumps
    daily_ret = df.pct_change()
    extreme_jumps = (daily_ret.abs() > 0.5).sum().sum()
    results["extreme_jumps_gt50pct"] = int(extreme_jumps)

    # VIX/SPY presence
    results["has_SPY"] = "SPY" in df.columns

    return results


if __name__ == "__main__":
    print("Building 8-year price cache...")
    df = build_8yr_cache()

    print("\nValidation results:")
    stats = validate_8yr_cache(df)
    for k, v in stats.items():
        print(f"  {k:30s}: {v}")

    # Assert minimum quality
    assert stats["years_covered"] >= 5.0, f"Only {stats['years_covered']:.1f} years — expected 8+"
    assert stats["avg_missing_pct"] < 0.15, f"Too many missing: {stats['avg_missing_pct']:.1%}"
    assert stats["has_SPY"], "SPY missing from price cache!"

    print("\n✓ 8-year cache validation passed")
