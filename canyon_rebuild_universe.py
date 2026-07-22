#!/usr/bin/env python3
"""
Canyon v9 — S&P 500 Universe Rebuild
======================================
Expands universe_master.csv from 20 tickers to full S&P 500 (~495 tickers).

Sources (all free):
  1. sp500_price_cache.csv  — already downloaded, 495 tickers, 751 days
  2. Wikipedia S&P 500 list — live constituent list, free
  3. yfinance               — sector/industry metadata per ticker

What this script does:
  1. Load tickers from existing price cache (495 stocks, data already there)
  2. Fetch fresh S&P 500 constituent list from Wikipedia (backup: use price cache)
  3. Add sector, industry, market cap metadata from yfinance
  4. Write universe_master.csv (replaces old 20-ticker version)
  5. Write sp500_sectors.csv (sector lookup table for L3 rotation)
  6. Print summary statistics

After running this:
  - All layer reports will pick up 495 tickers automatically
  - canyon_factor_deep_research.py loads STOCK_UNIVERSE from universe_master.csv
  - Action readiness, tonight plan, data quality report all use the full universe

Usage:
  python3 canyon_rebuild_universe.py              # full run with metadata
  python3 canyon_rebuild_universe.py --fast       # skip yfinance metadata fetch
"""
from __future__ import annotations

import argparse
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT  = Path(__file__).parent
TODAY = datetime.today().strftime("%Y-%m-%d")

ETF_TICKERS = {
    "SPY","QQQ","IWM","DIA","MDY","IWD","IWF","IWN","IWO","IWV",
    "XLK","XLV","XLF","XLE","XLI","XLY","XLP","XLU","XLC","XLB","IYR",
    "GLD","SLV","TLT","IEF","SHY","HYG","LQD","EMB","AGG","BND","UUP",
    "VNQ","ARKK","SMH","SOXX","MTUM","QUAL","USMV","VLUE",
    "VBR","VBK","VTV","VUG","VTI","VO","VOO","VEA","VWO",
}


# =============================================================================
# 1. GET S&P 500 CONSTITUENT LIST
# =============================================================================

def get_sp500_from_wikipedia() -> list[str]:
    """
    Pull current S&P 500 constituents from Wikipedia.
    Wikipedia table is updated within days of index changes.
    Completely free, no API key needed.
    """
    print("  [Wikipedia] fetching S&P 500 constituent list...")
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            header=0
        )
        sp500_df = tables[0]
        # Column is 'Symbol' or 'Ticker symbol'
        col = next((c for c in sp500_df.columns if "symbol" in c.lower() or "ticker" in c.lower()), None)
        if col is None:
            col = sp500_df.columns[0]
        tickers = sp500_df[col].str.strip().str.replace(".", "-", regex=False).dropna().tolist()
        tickers = [t for t in tickers if t and len(t) <= 5]
        print(f"  [Wikipedia] found {len(tickers)} S&P 500 constituents")
        return tickers
    except Exception as e:
        print(f"  [Wikipedia] failed: {e} — falling back to price cache")
        return []


def get_tickers_from_price_cache() -> list[str]:
    """Load ticker list from existing sp500_price_cache.csv."""
    cache = ROOT / "sp500_price_cache.csv"
    if not cache.exists():
        print("  [!] sp500_price_cache.csv not found")
        return []
    df = pd.read_csv(cache, index_col=0, nrows=1)
    tickers = [c for c in df.columns if c not in ETF_TICKERS]
    print(f"  [cache] {len(tickers)} tickers from sp500_price_cache.csv")
    return tickers


# =============================================================================
# 2. FETCH SECTOR / INDUSTRY METADATA
# =============================================================================

def fetch_metadata_batch(tickers: list[str], delay: float = 0.25) -> pd.DataFrame:
    """
    Download sector, industry, market cap for each ticker via yfinance.
    Batched with delay to avoid rate limits.
    Takes ~2-3 minutes for 495 tickers.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    print(f"\n  [yfinance] fetching metadata for {len(tickers)} tickers...")
    print(f"  Estimated time: {len(tickers) * delay / 60:.1f} minutes\n")

    records = []
    for i, tkr in enumerate(tickers):
        try:
            info = yf.Ticker(tkr).info
            records.append({
                "ticker":      tkr,
                "name":        info.get("longName") or info.get("shortName", tkr),
                "sector":      info.get("sector", "Unknown"),
                "industry":    info.get("industry", "Unknown"),
                "market_cap":  info.get("marketCap", np.nan),
                "currency":    info.get("currency", "USD"),
                "exchange":    info.get("exchange", ""),
                "country":     info.get("country", "US"),
                "asset_type":  "ETF" if tkr in ETF_TICKERS else "STOCK",
            })
            if (i + 1) % 50 == 0:
                pct = (i + 1) / len(tickers) * 100
                print(f"  Progress: {i+1}/{len(tickers)} ({pct:.0f}%) — last: {tkr}")
        except Exception as e:
            records.append({
                "ticker":     tkr,
                "name":       tkr,
                "sector":     "Unknown",
                "industry":   "Unknown",
                "market_cap": np.nan,
                "currency":   "USD",
                "exchange":   "",
                "country":    "US",
                "asset_type": "ETF" if tkr in ETF_TICKERS else "STOCK",
            })
        time.sleep(delay)

    return pd.DataFrame(records)


def build_metadata_stub(tickers: list[str]) -> pd.DataFrame:
    """Fast fallback: build metadata without yfinance calls (sector = Unknown)."""
    records = []
    for tkr in tickers:
        records.append({
            "ticker":     tkr,
            "name":       tkr,
            "sector":     "Unknown",
            "industry":   "Unknown",
            "market_cap": np.nan,
            "currency":   "USD",
            "exchange":   "",
            "country":    "US",
            "asset_type": "ETF" if tkr in ETF_TICKERS else "STOCK",
        })
    return pd.DataFrame(records)


# =============================================================================
# 3. COMPUTE DATA COVERAGE FROM PRICE CACHE
# =============================================================================

def compute_coverage_stats(tickers: list[str]) -> pd.DataFrame:
    """
    For each ticker: how many days of price data exist, when last updated,
    how many are missing or stale.
    """
    cache = ROOT / "sp500_price_cache.csv"
    if not cache.exists():
        return pd.DataFrame({"ticker": tickers, "data_days": 0, "last_date": None, "stale": True})

    prices = pd.read_csv(cache, index_col=0, parse_dates=True)
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    latest = prices.index.max()
    stale_threshold = pd.Timestamp.today() - pd.Timedelta(days=5)

    stats = []
    for tkr in tickers:
        if tkr in prices.columns:
            col = prices[tkr].dropna()
            last_date = col.index.max() if not col.empty else None
            stats.append({
                "ticker":    tkr,
                "data_days": len(col),
                "last_date": last_date.strftime("%Y-%m-%d") if last_date else None,
                "stale":     last_date < stale_threshold if last_date else True,
                "in_cache":  True,
            })
        else:
            stats.append({
                "ticker":    tkr,
                "data_days": 0,
                "last_date": None,
                "stale":     True,
                "in_cache":  False,
            })
    return pd.DataFrame(stats)


# =============================================================================
# 4. REFRESH PRICE CACHE FOR MISSING TICKERS
# =============================================================================

def refresh_missing_prices(missing_tickers: list[str], years: int = 5):
    """
    Download price data for tickers not yet in the cache.
    Appends to sp500_price_cache.csv.
    """
    if not missing_tickers:
        print("  [prices] No missing tickers — cache is complete.")
        return

    try:
        import yfinance as yf
    except ImportError:
        print("  [prices] yfinance not installed — skipping price refresh")
        return

    from datetime import timedelta
    start = (datetime.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    print(f"  [prices] Downloading {len(missing_tickers)} missing tickers from {start}...")

    batch_size = 50
    new_frames = []
    for i in range(0, len(missing_tickers), batch_size):
        batch = missing_tickers[i:i + batch_size]
        try:
            raw = yf.download(batch, start=start, auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw["Close"]
            else:
                closes = raw
            new_frames.append(closes)
            print(f"    Batch {i//batch_size + 1}: {len(batch)} tickers downloaded")
        except Exception as e:
            print(f"    Batch {i//batch_size + 1} error: {e}")
        time.sleep(0.5)

    if not new_frames:
        return

    new_prices = pd.concat(new_frames, axis=1)
    cache_path = ROOT / "sp500_price_cache.csv"
    if cache_path.exists():
        existing = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        combined = pd.concat([existing, new_prices], axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        combined.to_csv(cache_path)
        print(f"  [prices] Updated cache: now {combined.shape[1]} tickers × {combined.shape[0]} days")
    else:
        new_prices.to_csv(cache_path)
        print(f"  [prices] Created new cache: {new_prices.shape[1]} tickers × {new_prices.shape[0]} days")


# =============================================================================
# 5. WRITE OUTPUTS
# =============================================================================

def write_universe_master(metadata: pd.DataFrame, coverage: pd.DataFrame):
    """
    Write the new universe_master.csv.
    Format matches what step47 (L1 data integrity) expects:
      ticker, asset_type_guess, sector, industry, market_cap,
      data_days, last_date, stale, source_files, source_count, total_rows
    """
    meta = metadata.copy()
    cov  = coverage.copy()

    merged = meta.merge(cov, on="ticker", how="left")
    merged["asset_type_guess"] = merged["asset_type"]
    merged["source_files"]     = "sp500_price_cache.csv"
    merged["source_count"]     = 1
    merged["total_rows"]       = merged["data_days"].fillna(0).astype(int)

    # Sort by market cap descending (largest first), unknown at end
    merged = merged.sort_values("market_cap", ascending=False, na_position="last")

    out_cols = [
        "ticker", "asset_type_guess", "sector", "industry",
        "market_cap", "data_days", "last_date", "stale",
        "source_files", "source_count", "total_rows", "name",
    ]
    out_cols = [c for c in out_cols if c in merged.columns]
    merged[out_cols].to_csv(ROOT / "universe_master.csv", index=False)
    print(f"\n  [universe_master.csv] {len(merged)} tickers written")


def write_sector_lookup(metadata: pd.DataFrame):
    """Write sp500_sectors.csv — sector/industry lookup for L3 rotation."""
    sector_df = metadata[["ticker", "sector", "industry", "market_cap"]].copy()
    sector_df = sector_df[sector_df["sector"] != "Unknown"].dropna(subset=["sector"])
    sector_df.to_csv(ROOT / "sp500_sectors.csv", index=False)
    print(f"  [sp500_sectors.csv] {len(sector_df)} tickers with sector data")


def print_summary(metadata: pd.DataFrame, coverage: pd.DataFrame):
    """Print a summary of the new universe."""
    stocks = metadata[metadata["asset_type"] == "STOCK"]
    etfs   = metadata[metadata["asset_type"] == "ETF"]

    print(f"\n{'='*60}")
    print(f"  UNIVERSE REBUILD COMPLETE — {TODAY}")
    print(f"{'='*60}")
    print(f"  Total tickers: {len(metadata)}")
    print(f"    Stocks:  {len(stocks)}")
    print(f"    ETFs:    {len(etfs)}")

    if "data_days" in coverage.columns:
        in_cache = coverage[coverage["in_cache"] == True]
        stale    = coverage[coverage["stale"] == True]
        print(f"\n  Price Data:")
        print(f"    In cache:    {len(in_cache)} tickers ({len(in_cache)/len(coverage)*100:.0f}%)")
        print(f"    Avg days:    {in_cache['data_days'].mean():.0f}")
        print(f"    Stale/miss:  {len(stale)}")

    if "sector" in metadata.columns:
        sector_counts = stocks["sector"].value_counts().head(11)
        print(f"\n  Sector Breakdown:")
        for sector, count in sector_counts.items():
            if sector != "Unknown":
                print(f"    {sector:<35} {count:>3} stocks")

    if "market_cap" in metadata.columns:
        mc = stocks["market_cap"].dropna()
        if not mc.empty:
            print(f"\n  Market Cap:")
            print(f"    Largest:  {stocks.loc[mc.idxmax(), 'ticker']} (${mc.max()/1e12:.2f}T)")
            print(f"    Smallest: {stocks.loc[mc.idxmin(), 'ticker']} (${mc.min()/1e9:.1f}B)")
            print(f"    Median:   ${mc.median()/1e9:.1f}B")


# =============================================================================
# MAIN
# =============================================================================

def main(fast: bool = False):
    print(f"\nCanyon v9 — Universe Rebuild ({TODAY})")
    print(f"{'='*60}")

    # ── 1. Get ticker list ─────────────────────────────────────────────────────
    wiki_tickers = get_sp500_from_wikipedia()
    cache_tickers = get_tickers_from_price_cache()

    # Merge: Wikipedia is authoritative for membership, cache has actual data
    all_tickers = sorted(set(wiki_tickers) | set(cache_tickers))
    stock_tickers = [t for t in all_tickers if t not in ETF_TICKERS]
    print(f"\n  Combined universe: {len(all_tickers)} tickers ({len(stock_tickers)} stocks)")

    # ── 2. Check what's missing from price cache ───────────────────────────────
    coverage = compute_coverage_stats(all_tickers)
    missing = coverage[~coverage["in_cache"]]["ticker"].tolist()
    if missing:
        print(f"\n  [!] {len(missing)} tickers not in price cache: {missing[:10]}{'...' if len(missing)>10 else ''}")
        if not fast:
            refresh_missing_prices(missing, years=5)
            coverage = compute_coverage_stats(all_tickers)  # recompute after refresh
    else:
        print(f"  [✓] All {len(all_tickers)} tickers have price data in cache")

    # ── 3. Get metadata ────────────────────────────────────────────────────────
    if fast:
        print("\n  [fast mode] Skipping yfinance metadata — using stubs")
        metadata = build_metadata_stub(all_tickers)
    else:
        metadata = fetch_metadata_batch(stock_tickers, delay=0.20)
        # Add ETFs back with minimal metadata
        etf_records = [{"ticker": t, "name": t, "sector": "ETF", "industry": "ETF",
                        "market_cap": np.nan, "currency": "USD",
                        "exchange": "", "country": "US", "asset_type": "ETF"}
                       for t in ETF_TICKERS if t in all_tickers]
        if etf_records:
            metadata = pd.concat([metadata, pd.DataFrame(etf_records)], ignore_index=True)

    # ── 4. Write outputs ───────────────────────────────────────────────────────
    write_universe_master(metadata, coverage)
    write_sector_lookup(metadata)
    print_summary(metadata, coverage)

    # ── 5. Quick validation ────────────────────────────────────────────────────
    new_master = pd.read_csv(ROOT / "universe_master.csv")
    print(f"\n  [validation] universe_master.csv: {len(new_master)} rows")
    stocks_in = new_master[new_master["asset_type_guess"] == "STOCK"]
    print(f"  [validation] Stocks: {len(stocks_in)}")
    print(f"\n  Done. All layer reports will now use {len(new_master)}-ticker universe.")
    print(f"  Next step: run canyon_factor_deep_research.py to compute factors on full universe.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Skip yfinance metadata (no sector/industry labels)")
    args = parser.parse_args()
    main(fast=args.fast)
