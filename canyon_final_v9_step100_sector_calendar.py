#!/usr/bin/env python3
"""
canyon_final_v9_step100_sector_calendar.py
==========================================
Canyon v9 — Step 100: Sector Seasonality Calendar

Compute monthly sector seasonality scores from 10 years of sector ETF
history. Identify which sectors historically outperform in each calendar
month. Output a seasonal bias for the current month to feed into step87's
alpha aggregation.

Outputs:
  sector_seasonality.csv          — full 11×12 monthly stats table
  current_month_sector_bias.json  — current-month top/bottom sectors
  sector_etf_monthly_cache.csv    — raw monthly price cache (7-day TTL)

Usage:
  python canyon_final_v9_step100_sector_calendar.py
  python canyon_final_v9_step100_sector_calendar.py --years 10
  python canyon_final_v9_step100_sector_calendar.py --refresh
  python canyon_final_v9_step100_sector_calendar.py --month 3
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SECTOR_ETFS: dict[str, str] = {
    "Technology":              "XLK",
    "Financials":              "XLF",
    "Health Care":             "XLV",
    "Consumer Discretionary":  "XLY",
    "Communication Services":  "XLC",
    "Industrials":             "XLI",
    "Consumer Staples":        "XLP",
    "Energy":                  "XLE",
    "Utilities":               "XLU",
    "Real Estate":             "XLRE",
    "Materials":               "XLB",
}

SPY_TICKER      = "SPY"
YEARS_HISTORY   = 10
CACHE_TTL_DAYS  = 7

CACHE_FILE      = ROOT / "sector_etf_monthly_cache.csv"
SEASONALITY_CSV = ROOT / "sector_seasonality.csv"
BIAS_JSON       = ROOT / "current_month_sector_bias.json"

MONTH_NAMES = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# ─────────────────────────────────────────────────────────────────────────────
# [1/4] Fetch / load monthly prices
# ─────────────────────────────────────────────────────────────────────────────

def _cache_is_fresh() -> bool:
    """Return True if the cache file exists and is less than CACHE_TTL_DAYS old."""
    if not CACHE_FILE.exists():
        return False
    mtime = datetime.fromtimestamp(CACHE_FILE.stat().st_mtime)
    return (datetime.now() - mtime) < timedelta(days=CACHE_TTL_DAYS)


def fetch_sector_prices(years: int = YEARS_HISTORY, force_refresh: bool = False) -> pd.DataFrame:
    """
    Download monthly close prices for all sector ETFs + SPY.

    Uses yf.download with interval='1mo', period=f'{years}y'.
    Returns DataFrame with columns = ETF tickers, index = monthly dates.
    Caches results to sector_etf_monthly_cache.csv (refreshes if > 7 days old).
    """
    if not force_refresh and _cache_is_fresh():
        print(f"  Loading prices from cache: {CACHE_FILE.name}")
        df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        return df

    tickers = list(SECTOR_ETFS.values()) + [SPY_TICKER]
    print(f"  Downloading monthly prices for {len(tickers)} tickers "
          f"({years}y history)...")

    try:
        import yfinance as yf

        raw = yf.download(
            tickers,
            interval="1mo",
            period=f"{years}y",
            auto_adjust=True,
            progress=False,
        )

        # yfinance MultiIndex: top level = price type, second = ticker
        # Use raw["Close"] to get ticker columns reliably
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]].copy()
            close.columns = tickers[:1]

        # Keep only the tickers we asked for (drop any missing)
        available = [t for t in tickers if t in close.columns]
        close = close[available].copy()

        # Drop rows where all values are NaN
        close.dropna(how="all", inplace=True)

        if close.empty:
            raise ValueError("Downloaded price data is empty.")

        close.to_csv(CACHE_FILE)
        print(f"  Cached {len(close)} monthly rows to {CACHE_FILE.name}")
        return close

    except Exception as exc:
        print(f"  WARNING: yfinance download failed — {exc}")
        if CACHE_FILE.exists():
            print(f"  Falling back to existing cache: {CACHE_FILE.name}")
            return pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        # Return empty DataFrame — downstream will produce neutral scores
        print("  No cache available. Returning empty prices.")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# [2/4] Compute monthly statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_monthly_stats(prices: pd.DataFrame) -> pd.DataFrame:
    """
    For each sector ETF, compute avg_return, avg_alpha (vs SPY), hit_rate,
    std, t_stat, and n_years for each calendar month (1–12).

    Returns DataFrame with columns:
      etf, sector, month, month_name, avg_return_pct, avg_alpha_pct,
      std_pct, hit_rate, t_stat, n_years
    """
    if prices.empty:
        # Return neutral stats for all sectors / months
        rows = []
        for sector, etf in SECTOR_ETFS.items():
            for m in range(1, 13):
                rows.append({
                    "etf": etf,
                    "sector": sector,
                    "month": m,
                    "month_name": MONTH_NAMES[m],
                    "avg_return_pct": 0.0,
                    "avg_alpha_pct": 0.0,
                    "std_pct": 0.0,
                    "hit_rate": 0.5,
                    "t_stat": 0.0,
                    "n_years": 0,
                })
        return pd.DataFrame(rows)

    # Compute monthly returns (percent change month-over-month)
    returns = prices.pct_change().dropna(how="all")

    # Build reverse lookup: ticker → sector
    etf_to_sector = {v: k for k, v in SECTOR_ETFS.items()}

    spy_ret = returns[SPY_TICKER].copy() if SPY_TICKER in returns.columns else pd.Series(dtype=float)

    rows = []
    for sector, etf in SECTOR_ETFS.items():
        if etf not in returns.columns:
            # Missing ETF — fill with neutral
            for m in range(1, 13):
                rows.append({
                    "etf": etf,
                    "sector": sector,
                    "month": m,
                    "month_name": MONTH_NAMES[m],
                    "avg_return_pct": 0.0,
                    "avg_alpha_pct": 0.0,
                    "std_pct": 0.0,
                    "hit_rate": 0.5,
                    "t_stat": 0.0,
                    "n_years": 0,
                })
            continue

        etf_ret = returns[etf].copy()

        for m in range(1, 13):
            mask = etf_ret.index.month == m
            monthly_vals = etf_ret[mask].dropna()
            n = len(monthly_vals)

            if n == 0:
                avg_ret = 0.0
                avg_alpha = 0.0
                std = 0.0
                hit_rate = 0.5
                t_stat = 0.0
            else:
                avg_ret = float(monthly_vals.mean())

                # SPY avg for the same months
                if not spy_ret.empty:
                    spy_mask = spy_ret.index.month == m
                    spy_vals = spy_ret[spy_mask].dropna()
                    spy_avg = float(spy_vals.mean()) if len(spy_vals) > 0 else 0.0
                else:
                    spy_avg = 0.0

                avg_alpha = avg_ret - spy_avg
                std = float(monthly_vals.std(ddof=1)) if n > 1 else 0.0
                hit_rate = float((monthly_vals > 0).sum()) / n
                t_stat = (avg_ret / (std / math.sqrt(n))) if (std > 0 and n > 1) else 0.0

            rows.append({
                "etf": etf,
                "sector": sector,
                "month": m,
                "month_name": MONTH_NAMES[m],
                "avg_return_pct": round(avg_ret * 100, 4),
                "avg_alpha_pct": round(avg_alpha * 100, 4),
                "std_pct": round(std * 100, 4),
                "hit_rate": round(hit_rate, 4),
                "t_stat": round(t_stat, 4),
                "n_years": n,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def seasonal_score(row: pd.Series) -> float:
    """
    Convert avg_alpha_pct and hit_rate to a 0-100 seasonal score.

    Score = 50 + clip(avg_alpha_pct * 10, -40, 40) + (hit_rate - 0.5) * 20
    Clipped to [0, 100].
    """
    alpha_contrib = float(np.clip(row["avg_alpha_pct"] * 10, -40, 40))
    hit_contrib   = (float(row["hit_rate"]) - 0.5) * 20.0
    score = 50.0 + alpha_contrib + hit_contrib
    return float(np.clip(score, 0.0, 100.0))


# ─────────────────────────────────────────────────────────────────────────────
# Current-month bias
# ─────────────────────────────────────────────────────────────────────────────

def current_month_bias(stats_df: pd.DataFrame, target_month: int | None = None) -> dict:
    """
    For the target calendar month (default = today's month), return a bias
    dict with top 3 and bottom 3 sectors plus all-sector scores.
    """
    if target_month is None:
        target_month = datetime.now().month

    month_name = MONTH_NAMES[target_month]
    month_df = stats_df[stats_df["month"] == target_month].copy()

    if month_df.empty:
        return {
            "month": target_month,
            "month_name": month_name,
            "top_sectors": [],
            "bottom_sectors": [],
            "all_sectors": {},
        }

    month_df = month_df.sort_values("seasonal_score", ascending=False).reset_index(drop=True)

    def _row_to_dict(r: pd.Series) -> dict:
        return {
            "sector":        r["sector"],
            "etf":           r["etf"],
            "avg_alpha_pct": round(float(r["avg_alpha_pct"]), 2),
            "hit_rate":      round(float(r["hit_rate"]), 2),
            "score":         round(float(r["seasonal_score"]), 1),
        }

    top_sectors    = [_row_to_dict(month_df.iloc[i]) for i in range(min(3, len(month_df)))]
    bottom_sectors = [_row_to_dict(month_df.iloc[i]) for i in range(max(0, len(month_df) - 3), len(month_df))]
    bottom_sectors = list(reversed(bottom_sectors))

    all_sectors = {
        row["sector"]: round(float(row["seasonal_score"]), 1)
        for _, row in month_df.iterrows()
    }

    return {
        "month":          target_month,
        "month_name":     month_name,
        "top_sectors":    top_sectors,
        "bottom_sectors": bottom_sectors,
        "all_sectors":    all_sectors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# [3/4] Print functions
# ─────────────────────────────────────────────────────────────────────────────

def _alpha_str(val: float) -> str:
    """Format alpha value as '+1.2%' or '-0.8%' with fixed width."""
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.1f}%"


def print_calendar(stats_df: pd.DataFrame) -> None:
    """
    Print a readable sector seasonality calendar:
    rows = sectors, columns = Jan..Dec, values = avg_alpha_pct.
    """
    col_w = 7   # column width per month

    header_months = "".join(f"{m:>{col_w}}" for m in MONTH_NAMES[1:])
    print(f"\n{'Sector Seasonality Calendar (10Y avg alpha vs SPY)':}")
    print(f"{'':26}{header_months}")
    print("-" * (26 + 12 * col_w))

    for sector in SECTOR_ETFS:
        row_vals = []
        for m in range(1, 13):
            cell = stats_df[(stats_df["sector"] == sector) & (stats_df["month"] == m)]
            if cell.empty:
                row_vals.append(f"{'N/A':>{col_w}}")
            else:
                val = float(cell.iloc[0]["avg_alpha_pct"])
                row_vals.append(f"{_alpha_str(val):>{col_w}}")
        print(f"{sector:<26}" + "".join(row_vals))

    print()


def print_current_month_summary(bias: dict) -> None:
    """Print current month top/bottom sectors."""
    month_name = bias["month_name"]
    month_num  = bias["month"]

    print(f"Current Month Seasonal Bias — {month_name} (month {month_num})")
    print("=" * 52)

    print("  Top sectors (seasonal tailwinds):")
    for i, s in enumerate(bias["top_sectors"], 1):
        bar_len = max(0, int((s["score"] - 50) / 2))
        bar = "#" * bar_len
        print(f"    {i}. {s['sector']:<26} ({s['etf']})  "
              f"alpha={_alpha_str(s['avg_alpha_pct'])}  "
              f"hit={s['hit_rate']:.0%}  "
              f"score={s['score']:.0f}  {bar}")

    print("  Bottom sectors (seasonal headwinds):")
    for i, s in enumerate(bias["bottom_sectors"], 1):
        bar_len = max(0, int((50 - s["score"]) / 2))
        bar = "v" * bar_len
        print(f"    {i}. {s['sector']:<26} ({s['etf']})  "
              f"alpha={_alpha_str(s['avg_alpha_pct'])}  "
              f"hit={s['hit_rate']:.0%}  "
              f"score={s['score']:.0f}  {bar}")

    print("\n  All sector scores this month:")
    sorted_all = sorted(bias["all_sectors"].items(), key=lambda x: x[1], reverse=True)
    for sector, score in sorted_all:
        bar_len = max(0, int((score - 50) / 2))
        bar = "#" * bar_len if score >= 50 else "v" * max(0, int((50 - score) / 2))
        print(f"    {sector:<26}  score={score:5.1f}  {bar}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# [4/4] Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_sector_calendar(
    years: int = YEARS_HISTORY,
    force_refresh: bool = False,
    target_month: int | None = None,
) -> None:
    """
    Full pipeline:
      1. Fetch/load monthly sector prices
      2. Compute monthly stats for all 12 months x 11 sectors
      3. Add seasonal_score column
      4. Write sector_seasonality.csv
      5. Write current_month_sector_bias.json
      6. Print calendar table
      7. Print current month top/bottom sectors
    """
    print("\n[1/4] Fetching monthly sector ETF prices...")
    prices = fetch_sector_prices(years=years, force_refresh=force_refresh)

    if prices.empty:
        print("  WARNING: Price data unavailable. Producing neutral output.")

    print(f"\n[2/4] Computing monthly statistics ({years}Y, 11 sectors x 12 months)...")
    stats_df = compute_monthly_stats(prices)

    print("\n[3/4] Scoring seasonality and writing output files...")
    stats_df["seasonal_score"] = stats_df.apply(seasonal_score, axis=1).round(1)

    # Write full seasonality table
    out_cols = [
        "etf", "sector", "month", "month_name",
        "avg_return_pct", "avg_alpha_pct", "std_pct",
        "hit_rate", "t_stat", "n_years", "seasonal_score",
    ]
    out_df = stats_df[out_cols].sort_values(["month", "seasonal_score"], ascending=[True, False])
    out_df.to_csv(SEASONALITY_CSV, index=False)
    print(f"  Wrote {SEASONALITY_CSV.name}  ({len(out_df)} rows)")

    # Write current-month bias JSON
    bias = current_month_bias(stats_df, target_month=target_month)
    with open(BIAS_JSON, "w") as f:
        json.dump(bias, f, indent=2)
    print(f"  Wrote {BIAS_JSON.name}  (month={bias['month_name']})")

    print("\n[4/4] Displaying results...")
    print_calendar(stats_df)
    print_current_month_summary(bias)

    # Summary line
    if bias["top_sectors"]:
        top_names = ", ".join(s["sector"] for s in bias["top_sectors"])
        bot_names = ", ".join(s["sector"] for s in bias["bottom_sectors"])
        print(f"  Seasonal tailwinds this month: {top_names}")
        print(f"  Seasonal headwinds this month: {bot_names}")
    print(f"\n  Done. Files written to {ROOT}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canyon v9 Step 100 — Sector Seasonality Calendar",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--years", type=int, default=YEARS_HISTORY,
        help="Years of monthly history to use",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Force re-download of ETF prices (ignore cache)",
    )
    parser.add_argument(
        "--month", type=int, default=None,
        metavar="N",
        help="Show bias for this calendar month (1-12). Default = current month.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.month is not None and not (1 <= args.month <= 12):
        print(f"ERROR: --month must be between 1 and 12, got {args.month}")
        raise SystemExit(0)

    run_sector_calendar(
        years=args.years,
        force_refresh=args.refresh,
        target_month=args.month,
    )


if __name__ == "__main__":
    main()
