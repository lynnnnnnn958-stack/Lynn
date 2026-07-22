#!/usr/bin/env python3
"""
Canyon — step_backtest_extended.py
====================================
Download historical data and run extended backtest from 2019 to present.

IMPORTANT DISCLAIMER
--------------------
This uses CURRENT alpha_score rankings (today's fixed weights) applied to
historical price data from 2019 to present. This is NOT true out-of-sample.
Rankings available today were not available at historical rebalance dates.
Survivorship bias is present (current S&P 500 only). For directional use only.

Strategy
--------
- Universe: top-50 current alpha_score stocks (broader pool, hold top-20)
- Hold  : top-20 by current alpha_score at each monthly rebalance
- Weight: equal weight (1/20 each)
- TC    : 0.10% per round trip per monthly rebalance (10 bps one-way)
- Bench : SPY buy-and-hold

Saves: backtest_extended.json
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT  = Path(__file__).parent
TODAY = datetime.now().strftime("%Y-%m-%d")

DISCLAIMER = (
    "SIMULATION WITH LOOK-AHEAD BIAS — current alpha rankings applied retroactively "
    "from 2019 to present. NOT true out-of-sample. Survivorship bias present "
    "(current S&P 500 constituents). Do not treat this as actual backtest performance."
)

TOP_N         = 20      # number of stocks to hold
DOWNLOAD_N    = 50      # top-N to download (wider pool in case some fail)
TC_ONE_WAY_BP = 10      # bps per leg (buy or sell), 20 bps round trip
REBALANCE_FREQ = "MS"   # monthly start

def log(msg: str) -> None:
    print(f"  {msg}")

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def compute_max_drawdown(equity: pd.Series) -> float:
    peak   = equity.cummax()
    dd     = (equity - peak) / peak
    return float(dd.min()) if len(dd) > 0 else float("nan")

def compute_sharpe(daily_rets: pd.Series, ann: float = 252) -> float:
    mu  = daily_rets.mean()
    sig = daily_rets.std()
    if sig == 0 or len(daily_rets) < 10:
        return float("nan")
    return float(mu / sig * np.sqrt(ann))

def compute_cagr(equity: pd.Series, n_years: float) -> float:
    if equity.empty or n_years <= 0:
        return float("nan")
    total = float(equity.iloc[-1] / equity.iloc[0])
    return float(total ** (1.0 / n_years) - 1)

# ── load alpha rankings ───────────────────────────────────────────────────────

section("1. Loading current alpha rankings")

alpha_path = ROOT / "alpha_scores.csv"
if not alpha_path.exists():
    print("  ERROR: alpha_scores.csv not found.")
    raise SystemExit(1)

alpha_df = pd.read_csv(alpha_path).sort_values("alpha_score", ascending=False)
top_tickers = alpha_df["ticker"].head(DOWNLOAD_N).tolist()
hold_tickers = alpha_df["ticker"].head(TOP_N).tolist()

log(f"Top-{TOP_N} hold tickers : {', '.join(hold_tickers[:10])}...")
log(f"Downloading top-{DOWNLOAD_N} for price data")

# ── download prices ───────────────────────────────────────────────────────────

section("2. Downloading historical prices (2019-01-01 to present)")
log("Batched download — may take 1-2 minutes...")

try:
    import yfinance as yf
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf

# Download SPY
spy_prices = None
try:
    spy_data = yf.download("SPY", start="2019-01-01", end=TODAY,
                            auto_adjust=True, progress=False)
    if not spy_data.empty:
        spy_prices = spy_data["Close"].squeeze()
        spy_prices.index = pd.to_datetime(spy_prices.index)
        log(f"SPY: {len(spy_prices)} days ({spy_prices.index[0].date()} - {spy_prices.index[-1].date()})")
except Exception as e:
    log(f"SPY download failed: {e}")

if spy_prices is None:
    print("  FATAL: SPY prices unavailable.")
    raise SystemExit(1)

# Download stocks in batches
all_prices: dict[str, pd.Series] = {}
BATCH = 10
min_required = 10

for i in range(0, len(top_tickers), BATCH):
    batch = top_tickers[i: i + BATCH]
    log(f"  Batch {i//BATCH + 1}/{-(-len(top_tickers)//BATCH)}: {batch}")
    try:
        raw = yf.download(
            batch, start="2019-01-01", end=TODAY,
            auto_adjust=True, progress=False, threads=True
        )
        if raw.empty:
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            close_data = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
        else:
            close_data = raw

        if isinstance(close_data, pd.DataFrame):
            for t in batch:
                if t in close_data.columns:
                    s = close_data[t].dropna()
                    if len(s) >= 200:
                        all_prices[t] = s
        elif isinstance(close_data, pd.Series) and len(batch) == 1:
            s = close_data.dropna()
            if len(s) >= 200:
                all_prices[batch[0]] = s

    except Exception as e:
        log(f"  Batch failed ({e}). Trying individually...")
        for t in batch:
            try:
                r = yf.download(t, start="2019-01-01", end=TODAY,
                                  auto_adjust=True, progress=False)
                if not r.empty and len(r) >= 200:
                    all_prices[t] = r["Close"].squeeze()
            except Exception:
                pass

log(f"Successfully downloaded: {len(all_prices)} / {len(top_tickers)} tickers")

if len(all_prices) < min_required:
    print(f"  FATAL: Only {len(all_prices)} tickers downloaded, need {min_required}.")
    raise SystemExit(1)

# Build DataFrame
price_df = pd.DataFrame(all_prices)
price_df.index = pd.to_datetime(price_df.index)
price_df = price_df.sort_index()
spy_prices = spy_prices.reindex(price_df.index.union(spy_prices.index)).sort_index()

# Final hold tickers (subset of downloaded)
available_hold = [t for t in hold_tickers if t in price_df.columns]
if len(available_hold) < min_required:
    # Fall back to whatever we have
    available_hold = list(price_df.columns)[:TOP_N]
    log(f"Using top-{len(available_hold)} available tickers as proxy.")

log(f"Hold tickers available: {len(available_hold)}/{TOP_N}")

# ── run monthly backtest ──────────────────────────────────────────────────────

section("3. Running monthly backtest (2019-01-01 to present)")

# Generate monthly rebalance dates
start_date = "2019-01-01"
all_dates = pd.date_range(start=start_date, end=TODAY, freq=REBALANCE_FREQ)
# Filter to dates within price data
all_dates = [d for d in all_dates if d >= price_df.index[0] and d <= price_df.index[-1]]

log(f"Monthly rebalance dates: {len(all_dates)}")

# Portfolio simulation
portfolio_prices = price_df[available_hold].ffill().bfill()
spy_aligned = spy_prices.reindex(portfolio_prices.index).ffill().bfill()

# Daily returns
stk_rets = portfolio_prices.pct_change()
spy_rets = spy_aligned.pct_change()

TC_COST = TC_ONE_WAY_BP * 2 / 10_000  # round trip cost per stock per rebalance

# Simulate: at each rebalance date, hold top-N equally weighted
# Since rankings are FIXED (current), we always hold the same available_hold tickers
# TC is applied on all positions (assume full turnover each month for conservatism)

nav = 1.0
spy_nav = 1.0
nav_history: list[dict] = []

prev_rebal_idx = None

for i, date in enumerate(all_dates):
    # Find the index of this rebalance date in price data
    date_idx_list = portfolio_prices.index.searchsorted(date)
    if date_idx_list >= len(portfolio_prices):
        continue

    # Next rebalance date
    if i + 1 < len(all_dates):
        next_date = all_dates[i + 1]
        next_idx  = portfolio_prices.index.searchsorted(next_date)
    else:
        next_idx  = len(portfolio_prices)

    period_rets = stk_rets.iloc[date_idx_list : next_idx][available_hold]
    spy_period  = spy_rets.iloc[date_idx_list : next_idx]

    if len(period_rets) == 0:
        continue

    # Equal-weight portfolio return
    period_strat = period_rets.mean(axis=1).fillna(0)
    period_spy   = spy_period.fillna(0)

    strat_cum = (1 + period_strat).prod()
    spy_cum   = (1 + period_spy).prod()

    # Apply TC at start of period (round trip on all positions)
    tc_this_period = TC_COST * len(available_hold) / len(available_hold)
    # Equivalent to: cost = TC_ONE_WAY_BP * 2 bps on the whole portfolio
    portfolio_tc = TC_COST  # 0.1% round-trip on entire portfolio per rebalance

    nav     = nav * strat_cum * (1 - portfolio_tc)
    spy_nav = spy_nav * float(spy_cum)

    nav_history.append({
        "date"     : date.strftime("%Y-%m-%d"),
        "nav"      : round(nav, 6),
        "spy_nav"  : round(spy_nav, 6),
        "period_ret": round(float(strat_cum - 1), 6),
        "spy_ret"  : round(float(spy_cum - 1), 6),
        "tc_cost"  : round(portfolio_tc, 6),
    })

nav_df = pd.DataFrame(nav_history)
nav_df["date"] = pd.to_datetime(nav_df["date"])
nav_df = nav_df.set_index("date")

log(f"Simulation periods: {len(nav_df)}")

# ── compute per-year stats ────────────────────────────────────────────────────

section("4. Computing per-year statistics")

YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
CURRENT_YEAR = datetime.now().year

yearly_stats: list[dict] = []

for year in YEARS:
    year_mask = nav_df.index.year == year
    year_df   = nav_df[year_mask]

    if len(year_df) == 0:
        continue

    strat_ret_yr = float(year_df["period_ret"].add(1).prod() - 1)
    spy_ret_yr   = float(year_df["spy_ret"].add(1).prod() - 1)
    alpha_yr     = strat_ret_yr - spy_ret_yr

    # Intra-year max drawdown via NAV
    nav_yr = year_df["nav"] / year_df["nav"].iloc[0]
    mdd_yr = compute_max_drawdown(nav_yr)

    yearly_stats.append({
        "year"        : year,
        "label"       : f"{year}",
        "strategy_ret": round(strat_ret_yr, 6),
        "spy_ret"     : round(spy_ret_yr, 6),
        "alpha"       : round(alpha_yr, 6),
        "max_drawdown": round(mdd_yr, 6),
        "n_periods"   : int(len(year_df)),
        "note"        : "LOOK-AHEAD: current alpha rankings retroactively applied",
    })

# Current year (YTD)
ytd_mask = nav_df.index.year == CURRENT_YEAR
ytd_df   = nav_df[ytd_mask]
if len(ytd_df) > 0:
    strat_ret_ytd = float(ytd_df["period_ret"].add(1).prod() - 1)
    spy_ret_ytd   = float(ytd_df["spy_ret"].add(1).prod() - 1)
    ytd_mdd       = compute_max_drawdown(ytd_df["nav"] / ytd_df["nav"].iloc[0])
    yearly_stats.append({
        "year"        : CURRENT_YEAR,
        "label"       : f"{CURRENT_YEAR}-YTD",
        "strategy_ret": round(strat_ret_ytd, 6),
        "spy_ret"     : round(spy_ret_ytd, 6),
        "alpha"       : round(strat_ret_ytd - spy_ret_ytd, 6),
        "max_drawdown": round(ytd_mdd, 6),
        "n_periods"   : int(len(ytd_df)),
        "note"        : "Year-to-date. LOOK-AHEAD: current alpha rankings retroactively applied",
    })

log("Year-by-year results:")
for s in yearly_stats:
    log(f"  {s['label']:9s}  Strategy: {s['strategy_ret']*100:+6.1f}%  "
        f"SPY: {s['spy_ret']*100:+6.1f}%  "
        f"Alpha: {s['alpha']*100:+5.1f}%  "
        f"MDD: {s['max_drawdown']*100:5.1f}%")

# ── overall stats ─────────────────────────────────────────────────────────────

section("5. Overall statistics")

total_nav_series = nav_df["nav"]
total_spy_series = nav_df["spy_nav"]

n_years_total = len(nav_df) / 12.0 if len(nav_df) > 0 else 1.0

total_strat_ret = float(total_nav_series.iloc[-1] - 1)
total_spy_ret   = float(total_spy_series.iloc[-1] - 1)

cagr     = compute_cagr(total_nav_series, n_years_total)
spy_cagr = compute_cagr(total_spy_series, n_years_total)
mdd      = compute_max_drawdown(total_nav_series)

# Sharpe: use monthly returns
monthly_rets = nav_df["period_ret"]
ann_sharpe = compute_sharpe(monthly_rets, ann=12)

log(f"Total period: {len(nav_df)} months (~{n_years_total:.1f} years)")
log(f"Total return — Strategy: {total_strat_ret*100:.1f}%  SPY: {total_spy_ret*100:.1f}%")
log(f"CAGR         — Strategy: {cagr*100:.2f}%  SPY: {spy_cagr*100:.2f}%")
log(f"Strategy MDD : {mdd*100:.2f}%")
log(f"Monthly Sharpe (annualized): {ann_sharpe:.3f}")

total_period = {
    "start"           : nav_df.index[0].strftime("%Y-%m-%d"),
    "end"             : nav_df.index[-1].strftime("%Y-%m-%d"),
    "n_months"        : int(len(nav_df)),
    "strategy_total_ret": round(total_strat_ret, 6),
    "spy_total_ret"   : round(total_spy_ret, 6),
    "cagr"            : round(cagr, 6) if not np.isnan(cagr) else None,
    "spy_cagr"        : round(spy_cagr, 6) if not np.isnan(spy_cagr) else None,
    "mdd"             : round(mdd, 6) if not np.isnan(mdd) else None,
    "sharpe_monthly_ann": round(ann_sharpe, 4) if not np.isnan(ann_sharpe) else None,
}

# ── save results ──────────────────────────────────────────────────────────────

section("6. Saving backtest_extended.json")

results = {
    "generated_at"  : pd.Timestamp.now().isoformat(),
    "disclaimer"    : DISCLAIMER,
    "strategy"      : (
        f"Equal-weight top-{TOP_N} by current alpha_score. "
        f"Monthly rebalance. TC={TC_ONE_WAY_BP}bps one-way."
    ),
    "n_hold"        : TOP_N,
    "tc_one_way_bps": TC_ONE_WAY_BP,
    "tickers_used"  : available_hold,
    "total_period"  : total_period,
    "cagr"          : round(cagr, 6) if not np.isnan(cagr) else None,
    "spy_cagr"      : round(spy_cagr, 6) if not np.isnan(spy_cagr) else None,
    "sharpe"        : round(ann_sharpe, 4) if not np.isnan(ann_sharpe) else None,
    "mdd"           : round(mdd, 6) if not np.isnan(mdd) else None,
    "yearly_stats"  : yearly_stats,
}

out_path = ROOT / "backtest_extended.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
log(f"Saved to {out_path}")

# ── print summary ─────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  EXTENDED BACKTEST SUMMARY")
print("="*60)
print(f"  {DISCLAIMER[:80]}...")
print()
print(f"  Period: {total_period['start']} to {total_period['end']}")
print(f"  Strategy CAGR : {cagr*100:.2f}%" if not np.isnan(cagr) else "  CAGR: N/A")
print(f"  SPY CAGR      : {spy_cagr*100:.2f}%" if not np.isnan(spy_cagr) else "  SPY CAGR: N/A")
print(f"  Sharpe (ann)  : {ann_sharpe:.3f}" if not np.isnan(ann_sharpe) else "  Sharpe: N/A")
print(f"  Max Drawdown  : {mdd*100:.2f}%" if not np.isnan(mdd) else "  MDD: N/A")
print()
print("  Year-by-year:")
for s in yearly_stats:
    print(f"    {s['label']:9s}  Strat: {s['strategy_ret']*100:+6.1f}%  "
          f"SPY: {s['spy_ret']*100:+6.1f}%  "
          f"Alpha: {s['alpha']*100:+5.1f}%  "
          f"MDD: {s['max_drawdown']*100:5.1f}%")
print()
print("  => backtest_extended.json saved.")
