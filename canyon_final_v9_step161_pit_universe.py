#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canyon v9 — Step 161: Point-in-Time S&P 500 Universe
=====================================================
Corrects survivorship bias by maintaining a historical constituent database.

At each historical rebalance date, only tickers that were ACTUALLY in the
S&P 500 on that date are included in the investable universe.  Current
backtest uses 42 current S&P survivors → all of them "won."  A proper
PIT universe includes historical failures (Enron, Lehman, WorldCom, etc.)

Sources:
  - S&P 500 historical changes: manually curated from public records /
    Wikipedia (https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
  - ~100 historically significant tickers from 1998-2026

Outputs:
  sp500_pit_universe.csv    ticker, added_date, removed_date, reason, is_current
  pit_price_cache.csv       price history for all PIT tickers

Usage:
  python3 canyon_final_v9_step161_pit_universe.py           # full run
  python3 canyon_final_v9_step161_pit_universe.py --skip-prices  # universe only
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

ROOT = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# Historical S&P 500 constituent database
# Format: (ticker, added_date, removed_date_or_None, removal_reason)
# Dates are best-estimate from public records.  None = still active 2026.
# ─────────────────────────────────────────────────────────────────────────────
HISTORICAL_CONSTITUENTS: list[tuple[str, str, str | None, str]] = [

    # ── Mega-cap tech (current members, with true entry dates) ──────────────
    ("AAPL",  "1982-11-30", None,         ""),
    ("MSFT",  "1994-06-01", None,         ""),
    ("AMZN",  "2005-11-18", None,         ""),
    ("NVDA",  "2001-11-30", None,         ""),
    ("GOOGL", "2006-04-03", None,         ""),
    ("META",  "2013-12-23", None,         ""),
    ("TSLA",  "2020-12-21", None,         ""),
    ("ADBE",  "1997-05-05", None,         ""),
    ("CRM",   "2020-09-21", None,         ""),
    ("INTC",  "1976-12-01", None,         ""),
    ("QCOM",  "1999-07-22", None,         ""),
    ("TXN",   "1975-01-01", None,         ""),
    ("AVGO",  "2014-08-18", None,         ""),
    ("AMD",   "2017-03-20", None,         ""),   # re-added; was removed 2013
    ("MU",    "1994-07-01", None,         ""),
    ("CSCO",  "1993-12-01", None,         ""),
    ("ORCL",  "1989-01-01", None,         ""),
    ("IBM",   "1957-03-04", None,         ""),
    ("HPQ",   "1957-03-04", None,         ""),

    # ── Financials (current) ─────────────────────────────────────────────────
    ("JPM",   "1975-01-01", None,         ""),
    ("V",     "2008-03-19", None,         ""),
    ("MA",    "2006-06-07", None,         ""),
    ("WFC",   "1976-01-01", None,         ""),
    ("BAC",   "1976-01-01", None,         ""),
    ("C",     "1976-01-01", None,         ""),
    ("GS",    "1999-05-04", None,         ""),
    ("MS",    "1986-03-01", None,         ""),
    ("AIG",   "1980-01-01", None,         ""),   # survived, still active
    ("BRK-B", "2010-02-16", None,         ""),

    # ── Healthcare (current) ────────────────────────────────────────────────
    ("JNJ",   "1973-06-01", None,         ""),
    ("UNH",   "1994-07-01", None,         ""),
    ("LLY",   "1970-01-01", None,         ""),
    ("MRK",   "1957-03-04", None,         ""),
    ("ABBV",  "2013-01-02", None,         ""),   # spun off from ABT
    ("TMO",   "2002-01-02", None,         ""),
    ("ABT",   "1957-03-04", None,         ""),
    ("PFE",   "1957-03-04", None,         ""),
    ("GILD",  "2002-08-01", None,         ""),
    ("BMY",   "1957-03-04", None,         ""),
    ("AMGN",  "1992-03-31", None,         ""),

    # ── Consumer (current) ──────────────────────────────────────────────────
    ("WMT",   "1982-08-31", None,         ""),
    ("COST",  "1993-10-01", None,         ""),
    ("HD",    "1988-03-01", None,         ""),
    ("KO",    "1957-03-04", None,         ""),
    ("PEP",   "1957-03-04", None,         ""),
    ("PG",    "1957-03-04", None,         ""),
    ("NFLX",  "2010-12-20", None,         ""),
    ("PYPL",  "2015-07-20", None,         ""),   # spun off from EBAY
    ("EBAY",  "2002-07-01", None,         ""),

    # ── Energy (current) ────────────────────────────────────────────────────
    ("XOM",   "1957-03-04", None,         ""),
    ("CVX",   "1957-03-04", None,         ""),
    ("COP",   "2002-08-31", None,         ""),
    ("SLB",   "1957-03-04", None,         ""),
    ("HAL",   "1957-03-04", None,         ""),

    # ── ETFs (always available — not S&P members, but tradeable) ────────────
    ("SPY",   "1993-01-29", None,         "etf"),
    ("QQQ",   "1999-03-10", None,         "etf"),
    ("XLK",   "1998-12-16", None,         "etf"),
    ("XLE",   "1998-12-16", None,         "etf"),
    ("XLF",   "1998-12-16", None,         "etf"),
    ("XLV",   "1998-12-16", None,         "etf"),
    ("XLU",   "1998-12-16", None,         "etf"),
    ("XLP",   "1998-12-16", None,         "etf"),
    ("SMH",   "2000-05-05", None,         "etf"),
    ("SOXX",  "2001-07-10", None,         "etf"),

    # ══════════════════════════════════════════════════════════════════════════
    # HISTORICAL REMOVALS — the survivorship bias corrections
    # These are the stocks step100 never included.  Including them exposes the
    # model to actual S&P 500 losers from each era.
    # ══════════════════════════════════════════════════════════════════════════

    # ── Financial crisis 2008-2009 ───────────────────────────────────────────
    ("LEH",   "1994-01-03", "2008-09-15", "bankruptcy"),          # Lehman Brothers — zero
    ("BSC",   "1994-01-03", "2008-03-17", "acquired_JPM"),        # Bear Stearns — $2/sh
    ("WB",    "1998-01-02", "2008-10-03", "acquired_WFC"),        # Wachovia — near zero
    ("MER",   "1994-01-03", "2009-01-01", "acquired_BAC"),        # Merrill Lynch
    ("WM",    "2004-01-02", "2008-09-26", "bankruptcy"),          # Washington Mutual — zero
    ("CFC",   "2004-01-02", "2008-07-01", "acquired_BAC"),        # Countrywide Financial
    ("DSL",   "2001-01-02", "2008-11-01", "bankruptcy"),          # Downey Financial
    ("CIT",   "2002-01-02", "2009-11-01", "bankruptcy"),          # CIT Group

    # ── Dot-com bust 2001-2003 ───────────────────────────────────────────────
    ("ENRN",  "1996-11-26", "2001-11-28", "fraud_bankruptcy"),    # Enron — zero
    ("WCOM",  "1997-01-13", "2002-06-25", "fraud_bankruptcy"),    # WorldCom — zero
    ("LU",    "1999-01-04", "2002-07-01", "distress_delisted"),   # Lucent -99%
    ("CPQ",   "1997-01-02", "2002-05-03", "acquired_HPQ"),        # Compaq
    ("GLW",   "1994-01-03", None,         ""),                    # Corning survived
    ("JDS",   "2000-07-10", "2002-01-14", "distress_delisted"),   # JDS Uniphase
    ("AOL",   "2001-01-11", "2002-09-17", "spinoff_distress"),    # AOL Time Warner
    ("WBVN",  "1998-01-02", "2001-06-01", "delisted"),            # webMD precursor
    ("XCIT",  "1999-07-01", "2001-08-01", "bankruptcy"),          # Excite@Home

    # ── Industrial decline ───────────────────────────────────────────────────
    ("EK",    "1964-01-02", "2004-04-01", "declining_removed"),   # Eastman Kodak -96%
    ("GM",    "1957-03-04", "2009-06-01", "bankruptcy"),          # General Motors — zero
    ("DAL",   "1994-01-03", "2005-09-14", "bankruptcy"),          # Delta Air Lines
    ("UAL",   "1994-01-03", "2002-12-09", "bankruptcy"),          # United parent
    ("F",     "1957-03-04", None,         ""),                    # Ford survived
    ("X",     "1957-03-04", "2014-06-01", "removed_reweight"),    # US Steel
    ("AA",    "1957-03-04", "2016-11-01", "spinoff"),             # Alcoa split

    # ── Retail collapse ──────────────────────────────────────────────────────
    ("SHLD",  "2005-03-24", "2018-10-15", "bankruptcy"),          # Sears Holdings — zero
    ("JCP",   "2002-01-02", "2020-05-15", "bankruptcy"),          # JCPenney
    ("KM",    "1994-01-03", "2002-01-22", "bankruptcy"),          # Kmart
    ("CC",    "2004-01-02", "2008-11-10", "bankruptcy"),          # Circuit City — zero
    ("BBI",   "2004-01-02", "2010-06-14", "bankruptcy"),          # Blockbuster
    ("RSH",   "2002-01-02", "2014-09-22", "distress_removed"),    # RadioShack
    ("M",     "2000-01-03", None,         ""),                    # Macy's survived (barely)

    # ── Telecom consolidation ─────────────────────────────────────────────────
    ("S",     "2005-08-12", "2020-04-01", "acquired_TMUS"),       # Sprint
    ("TWX",   "2001-01-11", "2018-06-14", "acquired_T"),          # Time Warner
    ("T",     "1983-01-03", None,         ""),                    # AT&T
    ("VZ",    "2000-07-03", None,         ""),                    # Verizon
    ("WIN",   "2000-01-03", "2019-02-25", "bankruptcy"),          # Windstream
    ("FTR",   "2010-10-01", "2019-04-14", "bankruptcy"),          # Frontier Communications

    # ── Energy bust 2015-2016 ─────────────────────────────────────────────────
    ("CHK",   "2012-01-02", "2020-06-08", "bankruptcy"),          # Chesapeake Energy
    ("RIG",   "2002-01-02", "2020-11-18", "bankruptcy"),          # Transocean
    ("DNR",   "2003-01-02", "2020-09-01", "bankruptcy"),          # Denbury Resources
    ("WLL",   "2003-01-02", "2020-07-01", "bankruptcy"),          # Whiting Petroleum
    ("CBL",   "2004-01-02", "2020-11-01", "bankruptcy"),          # CBL Properties (REIT)

    # ── Tech evolution / acquisitions ─────────────────────────────────────────
    ("YHOO",  "1999-12-08", "2017-06-13", "acquired_VZ"),         # Yahoo
    ("DELL",  "1996-06-01", "2013-10-29", "went_private"),        # Dell (re-listed 2018)
    ("NOK",   "1999-06-01", "2012-08-01", "distress_removed"),    # Nokia ADR
    ("EMC",   "2002-01-02", "2016-09-07", "acquired_DELL"),       # EMC
    ("CA",    "1994-01-03", "2018-11-06", "acquired_BRD"),        # CA Technologies
    ("SUNW",  "1998-01-02", "2010-01-26", "acquired_ORCL"),       # Sun Microsystems
    ("MOT",   "1994-01-03", "2012-01-04", "acquired_GOOG"),       # Motorola Solutions split
    ("PALM",  "2000-03-01", "2010-07-01", "acquired_HPQ"),        # Palm
    ("BRCM",  "2002-01-02", "2016-02-01", "acquired_AVGO"),       # Broadcom old
    ("ALTR",  "1994-01-03", "2015-12-28", "acquired_INTC"),       # Altera
    ("LSI",   "1994-01-03", "2014-05-06", "acquired_AVGO"),       # LSI Logic

    # ── Healthcare removed ─────────────────────────────────────────────────
    ("SGP",   "1994-01-03", "2009-11-03", "acquired_MRK"),        # Schering-Plough
    ("WYE",   "1994-01-03", "2009-10-15", "acquired_PFE"),        # Wyeth
    ("MGI",   "2003-01-02", "2014-11-01", "acquired"),            # MGI Pharma
    ("COV",   "2007-07-01", "2015-01-26", "acquired_MDT"),        # Covidien

    # ── Consumer removed ──────────────────────────────────────────────────
    ("KFT",   "2001-06-01", "2012-10-01", "split_MDLZ_KHC"),     # Kraft Foods
    ("MDLZ",  "2012-10-01", None,         ""),                    # Mondelez (post-split)
    ("HNZ",   "1994-01-03", "2013-06-07", "acquired_BRK"),        # Heinz
    ("DF",    "2002-01-02", "2016-09-07", "bankruptcy"),          # Dean Foods
    ("RAD",   "2007-01-02", "2015-07-01", "distress_removed"),    # Rite Aid
]


# ─────────────────────────────────────────────────────────────────────────────
# Build and save universe CSV
# ─────────────────────────────────────────────────────────────────────────────

def build_universe_csv() -> pd.DataFrame:
    rows = []
    for ticker, added, removed, reason in HISTORICAL_CONSTITUENTS:
        rows.append({
            "ticker":         ticker.upper(),
            "added_date":     pd.Timestamp(added),
            "removed_date":   pd.Timestamp(removed) if removed else pd.NaT,
            "removal_reason": reason,
            "is_current":     removed is None,
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["ticker", "added_date"])
    df = df.sort_values(["is_current", "added_date"], ascending=[False, True])

    out = ROOT / "sp500_pit_universe.csv"
    df.to_csv(out, index=False)
    print(f"[step161] Saved {len(df)} constituents → {out.name}")
    print(f"          Current: {df['is_current'].sum()}  |  Historical: {(~df['is_current']).sum()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Point-in-time lookup
# ─────────────────────────────────────────────────────────────────────────────

def get_universe_at_date(date: pd.Timestamp, df: pd.DataFrame | None = None) -> list[str]:
    """Return list of tickers active in S&P 500 (or as ETFs) on `date`."""
    if df is None:
        path = ROOT / "sp500_pit_universe.csv"
        if not path.exists():
            raise FileNotFoundError("Run step161 first to build sp500_pit_universe.csv")
        df = pd.read_csv(path, parse_dates=["added_date", "removed_date"])

    mask = (df["added_date"] <= date) & (
        df["removed_date"].isna() | (df["removed_date"] > date)
    )
    return sorted(df.loc[mask, "ticker"].tolist())


# ─────────────────────────────────────────────────────────────────────────────
# Price download for all PIT tickers
# ─────────────────────────────────────────────────────────────────────────────

def download_pit_prices(universe_df: pd.DataFrame) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        print("[step161] yfinance not installed — skipping price download")
        return pd.DataFrame()

    all_tickers = universe_df["ticker"].unique().tolist()
    # Skip tickers that yfinance almost certainly won't have (truly dead)
    skip_tickers = {"ENRN", "WCOM", "LU", "JDS", "AOL", "WBVN", "XCIT",
                    "EK", "XCIT", "DSL", "CHK", "RIG", "DNR", "WLL", "CBL",
                    "SUNW", "PALM", "SGP", "WYE", "MGI", "HNZ", "DF",
                    "RAD", "WIN", "FTR", "BRCM", "ALTR", "LSI", "CA",
                    "SHLD", "JCP", "KM", "CC", "BBI", "RSH", "X"}

    tickers_to_try = [t for t in all_tickers if t not in skip_tickers]

    cache_path = ROOT / "pit_price_cache.csv"
    # Load existing cache if fresh enough
    if cache_path.exists():
        try:
            existing = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            age_h = (pd.Timestamp.now() - pd.Timestamp(existing.index[-1])).total_seconds() / 3600
            if age_h < 48:
                print(f"[step161] Using existing pit_price_cache.csv "
                      f"({existing.shape[1]} tickers, last date {existing.index[-1].date()})")
                return existing
        except Exception:
            pass

    print(f"[step161] Downloading prices for {len(tickers_to_try)} PIT tickers …")
    start = "1993-01-01"
    end   = datetime.today().strftime("%Y-%m-%d")

    # Download in batches to avoid yfinance throttling
    BATCH = 30
    frames = []
    for i in range(0, len(tickers_to_try), BATCH):
        batch = tickers_to_try[i: i + BATCH]
        try:
            raw = yf.download(
                batch, start=start, end=end,
                auto_adjust=True, progress=False,
            )
            closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            frames.append(closes)
            print(f"  batch {i//BATCH + 1}: {closes.shape[1]} tickers OK")
            time.sleep(1)
        except Exception as e:
            print(f"  batch {i//BATCH + 1}: error — {e}")

    if not frames:
        print("[step161] No price data downloaded.")
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.dropna(how="all")

    prices.to_csv(cache_path)
    print(f"[step161] Saved pit_price_cache.csv: {prices.shape[1]} tickers × {len(prices)} days")
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# Coverage report
# ─────────────────────────────────────────────────────────────────────────────

def coverage_report(universe_df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    For each year from 2000 to 2025, compute:
      - how many S&P 500 members were in the PIT universe
      - how many have actual price data
      - coverage %
      - how many are 'failures' (removed_date not null)
    """
    years = range(2000, 2026)
    rows = []
    for yr in years:
        dt = pd.Timestamp(f"{yr}-06-01")
        universe = get_universe_at_date(dt, universe_df)
        n_universe = len(universe)

        # Separate ETFs from stocks
        etf_tickers = set(universe_df[universe_df["removal_reason"] == "etf"]["ticker"])
        stocks_in_universe = [t for t in universe if t not in etf_tickers]

        # How many have price data
        if not prices.empty:
            has_prices = [t for t in universe if t in prices.columns
                          and prices[t].notna().any()]
        else:
            has_prices = []

        # Failure tickers in this year's universe
        mask_fail = (
            (universe_df["added_date"] <= dt) &
            (universe_df["removed_date"].notna()) &
            (universe_df["removed_date"] > dt) &
            (universe_df["removal_reason"].str.contains("bankruptcy|fraud|zero|distress"))
        )
        failures_active = universe_df.loc[mask_fail, "ticker"].tolist()

        rows.append({
            "year":             yr,
            "pit_stocks":       len(stocks_in_universe),
            "pit_total_w_etf":  n_universe,
            "has_price_data":   len(has_prices),
            "coverage_pct":     round(len(has_prices) / max(n_universe, 1) * 100, 1),
            "active_failures":  len(failures_active),
            "failure_names":    ", ".join(failures_active[:5]),
        })

    df = pd.DataFrame(rows)
    out = ROOT / "pit_coverage_report.csv"
    df.to_csv(out, index=False)
    print(f"\n[step161] PIT coverage report:")
    print(df[["year", "pit_stocks", "has_price_data", "coverage_pct",
              "active_failures", "failure_names"]].to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-prices", action="store_true",
                        help="Build universe CSV only, skip price download")
    args = parser.parse_args()

    print("=" * 60)
    print("  Canyon v9 — Step 161: Point-in-Time Universe")
    print("=" * 60)

    universe_df = build_universe_csv()

    if args.skip_prices:
        print("[step161] Skipping price download (--skip-prices)")
        prices = pd.DataFrame()
    else:
        prices = download_pit_prices(universe_df)

    coverage_report(universe_df, prices)

    print("\n[step161] Done.")
    print("  sp500_pit_universe.csv — PIT universe definition")
    print("  pit_price_cache.csv    — price history for PIT tickers")
    print("  pit_coverage_report.csv — year-by-year data coverage")
    print("\n  Next: run step163 to quantify survivorship bias impact.")


if __name__ == "__main__":
    main()
