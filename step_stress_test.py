#!/usr/bin/env python3
"""
Canyon — step_stress_test.py
============================
Stress-test the CURRENT alpha ranking applied to historical price data.

IMPORTANT DISCLAIMER
--------------------
This simulation uses TODAY's fixed alpha_score rankings applied retroactively.
This is NOT true out-of-sample testing. Alpha scores computed today would not
have been available in 2020 or 2022. Survivorship bias is present (universe
is current S&P 500 constituents). Use for directional intuition only.

Periods tested
--------------
a) COVID crash   : 2020-02-19 to 2020-03-23
b) 2022 bear     : 2022-01-01 to 2022-12-31
c) 2023-recovery : 2023-01-01 to present

Strategy: top-30 stocks by current alpha_score, equal weight, rebalanced monthly.

Saves: stress_test_results.json
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
TODAY = datetime.now().strftime("%Y-%m-%d")

DISCLAIMER = (
    "SIMULATION ONLY — uses current alpha rankings applied retroactively. "
    "NOT true out-of-sample. Survivorship bias present (current S&P 500 "
    "constituents only). Rankings available today were not available in 2020/2022."
)

def log(msg: str) -> None:
    print(f"  {msg}")

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ── helpers ───────────────────────────────────────────────────────────────────

def compute_max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown from peak."""
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return float(drawdown.min())

def compute_sharpe(rets: pd.Series, ann_factor: float = 252) -> float:
    """Annualized Sharpe (assumes daily returns)."""
    if rets.std() == 0 or len(rets) < 5:
        return float("nan")
    return float(rets.mean() / rets.std() * np.sqrt(ann_factor))

def compute_period_stats(
    price_df: pd.DataFrame,
    tickers: list[str],
    start: str,
    end: str,
    spy_prices: pd.Series,
) -> dict:
    """Compute strategy and SPY stats for a given period."""
    valid_tickers = [t for t in tickers if t in price_df.columns]
    if len(valid_tickers) < 5:
        return {"error": "Insufficient tickers", "n_tickers": len(valid_tickers)}

    # Clip to period
    mask = (price_df.index >= start) & (price_df.index <= end)
    period_prices = price_df.loc[mask, valid_tickers].copy()
    spy_mask = (spy_prices.index >= start) & (spy_prices.index <= end)
    spy_period = spy_prices.loc[spy_mask].copy()

    if len(period_prices) < 5:
        return {"error": "Not enough price data in period", "n_days": len(period_prices)}

    # Forward fill, then daily returns
    period_prices = period_prices.ffill().bfill()
    daily_rets = period_prices.pct_change().dropna(how="all")

    # Equal-weight strategy daily returns
    strat_daily = daily_rets.mean(axis=1)

    # SPY daily returns
    spy_period = spy_period.ffill().bfill()
    spy_daily = spy_period.pct_change().dropna()

    # Align
    common_idx = strat_daily.index.intersection(spy_daily.index)
    strat_aligned = strat_daily.loc[common_idx]
    spy_aligned   = spy_daily.loc[common_idx]

    # Compute total return
    strat_equity = (1 + strat_aligned).cumprod()
    spy_equity   = (1 + spy_aligned).cumprod()

    strat_total = float(strat_equity.iloc[-1] - 1)
    spy_total   = float(spy_equity.iloc[-1] - 1)

    strat_mdd = compute_max_drawdown(strat_equity)
    spy_mdd   = compute_max_drawdown(spy_equity)

    strat_sharpe = compute_sharpe(strat_aligned)
    spy_sharpe   = compute_sharpe(spy_aligned)

    return {
        "strategy_ret"   : round(strat_total, 6),
        "spy_ret"        : round(spy_total, 6),
        "strategy_mdd"   : round(strat_mdd, 6),
        "spy_mdd"        : round(spy_mdd, 6),
        "strategy_sharpe": round(strat_sharpe, 4) if not np.isnan(strat_sharpe) else None,
        "spy_sharpe"     : round(spy_sharpe, 4) if not np.isnan(spy_sharpe) else None,
        "n_tickers_used" : len(valid_tickers),
        "n_days"         : len(common_idx),
    }

# ── load alpha_scores to get top-30 ──────────────────────────────────────────

section("1. Loading alpha_scores.csv for top-30 current stocks")

alpha_path = ROOT / "alpha_scores.csv"
if not alpha_path.exists():
    print("  ERROR: alpha_scores.csv not found. Exiting.")
    raise SystemExit(1)

alpha_df = pd.read_csv(alpha_path)
alpha_df = alpha_df.sort_values("alpha_score", ascending=False)
top30 = alpha_df["ticker"].head(30).tolist()
top50 = alpha_df["ticker"].head(50).tolist()  # download extra in case some fail

log(f"Top-30 tickers: {', '.join(top30)}")

# ── download prices via yfinance ──────────────────────────────────────────────

section("2. Downloading historical prices via yfinance (2019-01-01 to present)")
log("This may take 1-2 minutes...")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    log("WARNING: yfinance not installed. Attempting pip install...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf
    HAS_YFINANCE = True

# Download SPY first
spy_prices = None
try:
    spy_data = yf.download("SPY", start="2019-01-01", end=TODAY,
                            auto_adjust=True, progress=False)
    if len(spy_data) > 0:
        spy_prices = spy_data["Close"].squeeze()
        spy_prices.name = "SPY"
        log(f"SPY downloaded: {len(spy_prices)} days ({spy_prices.index[0].date()} - {spy_prices.index[-1].date()})")
    else:
        log("WARNING: SPY download returned empty data.")
except Exception as e:
    log(f"WARNING: SPY download failed: {e}")

if spy_prices is None:
    print("  FATAL: Cannot run stress test without SPY prices.")
    raise SystemExit(1)

# Download stock prices in batches of 10
all_prices: dict[str, pd.Series] = {}
batch_size = 10
success_count = 0

for i in range(0, len(top50), batch_size):
    batch = top50[i : i + batch_size]
    log(f"Downloading batch {i//batch_size + 1}: {batch}")
    try:
        raw = yf.download(
            batch, start="2019-01-01", end=TODAY,
            auto_adjust=True, progress=False, threads=True
        )
        if raw.empty:
            log(f"  Batch returned empty data, skipping.")
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, :len(batch)]
        else:
            close = raw[["Close"]] if "Close" in raw.columns else raw

        for ticker in batch:
            if isinstance(close, pd.DataFrame) and ticker in close.columns:
                s = close[ticker].dropna()
                if len(s) > 100:
                    all_prices[ticker] = s
                    success_count += 1
            elif isinstance(close, pd.Series):
                # Single ticker
                s = close.dropna()
                if len(s) > 100:
                    all_prices[batch[0]] = s
                    success_count += 1
                break

    except Exception as e:
        log(f"  Batch failed: {e}. Trying individually...")
        for ticker in batch:
            try:
                raw_t = yf.download(ticker, start="2019-01-01", end=TODAY,
                                     auto_adjust=True, progress=False)
                if not raw_t.empty and len(raw_t) > 100:
                    all_prices[ticker] = raw_t["Close"].squeeze()
                    success_count += 1
            except Exception as e2:
                log(f"    {ticker} failed: {e2}")

log(f"Successfully downloaded {success_count} stocks out of {len(top50)} attempted.")

if success_count < 10:
    print("  FATAL: Fewer than 10 tickers downloaded. Cannot run stress test.")
    raise SystemExit(1)

# Build price DataFrame
price_df = pd.DataFrame(all_prices)
price_df.index = pd.to_datetime(price_df.index)
price_df = price_df.sort_index()
spy_prices.index = pd.to_datetime(spy_prices.index)

# Narrow to top-30 actually downloaded
available_top30 = [t for t in top30 if t in price_df.columns]
log(f"Top-30 tickers with prices: {len(available_top30)}/{len(top30)}")

# Use available top-30; if fewer than 20, use all available
use_tickers = available_top30 if len(available_top30) >= 10 else list(price_df.columns)[:30]
log(f"Using {len(use_tickers)} tickers for stress test.")

# ── define periods ────────────────────────────────────────────────────────────

section("3. Running stress test across periods")

PERIODS = [
    {
        "name" : "COVID Crash",
        "start": "2020-02-19",
        "end"  : "2020-03-23",
        "note" : "Peak to trough of COVID crash. Using CURRENT alpha rankings (retroactive).",
    },
    {
        "name" : "2022 Rate-Hike Bear Market",
        "start": "2022-01-01",
        "end"  : "2022-12-31",
        "note" : "Full bear year driven by Fed tightening cycle. Using CURRENT alpha rankings.",
    },
    {
        "name" : "2023-2026 Recovery",
        "start": "2023-01-01",
        "end"  : TODAY,
        "note" : "Post-bear recovery period. Using CURRENT alpha rankings (retroactive).",
    },
]

period_results = []
for p in PERIODS:
    log(f"Computing: {p['name']} ({p['start']} to {p['end']})")
    stats = compute_period_stats(
        price_df     = price_df,
        tickers      = use_tickers,
        start        = p["start"],
        end          = p["end"],
        spy_prices   = spy_prices,
    )
    stats.update({
        "name" : p["name"],
        "start": p["start"],
        "end"  : p["end"],
        "note" : p["note"],
    })
    period_results.append(stats)

    if "error" not in stats:
        log(f"  Strategy: {stats['strategy_ret']*100:.1f}%  |  "
            f"SPY: {stats['spy_ret']*100:.1f}%  |  "
            f"Strategy MDD: {stats['strategy_mdd']*100:.1f}%")
    else:
        log(f"  ERROR: {stats['error']}")

# ── save results ──────────────────────────────────────────────────────────────

section("4. Saving stress_test_results.json")

results = {
    "generated_at"   : pd.Timestamp.now().isoformat(),
    "disclaimer"     : DISCLAIMER,
    "n_tickers_used" : len(use_tickers),
    "tickers_used"   : use_tickers,
    "strategy"       : "Equal-weight top-30 by current alpha_score, retroactively applied",
    "periods"        : period_results,
}

out_path = ROOT / "stress_test_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
log(f"Saved to {out_path}")

# ── print summary ─────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  STRESS TEST SUMMARY")
print("="*60)
print(f"  {DISCLAIMER[:80]}...")
print()
for p in period_results:
    print(f"  [{p['name']}]  {p['start']} to {p['end']}")
    if "error" not in p:
        print(f"    Strategy : {p['strategy_ret']*100:+.1f}%  |  SPY: {p['spy_ret']*100:+.1f}%")
        print(f"    Max DD   : {p['strategy_mdd']*100:.1f}%  |  SPY MDD: {p['spy_mdd']*100:.1f}%")
        print(f"    Sharpe   : {p.get('strategy_sharpe', 'N/A')}")
    else:
        print(f"    ERROR: {p['error']}")
    print()
print("  => stress_test_results.json saved.")
