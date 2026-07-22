#!/usr/bin/env python3
"""
Canyon — step_beta_neutral.py
=============================
Compute market beta for each stock and create beta-neutralized alpha scores.

Methodology
-----------
1. Load sp500_price_cache.csv, use last 126 trading days.
2. Compute daily returns for all stocks and SPY.
3. For each stock: beta = cov(stock_ret, spy_ret) / var(spy_ret).
4. Load alpha_scores.csv, add market_beta column.
5. Compute residual alpha:
     beta_neutral_alpha  = alpha_score adjusted for beta exposure
     sector_neutral_alpha = alpha_score - sector_mean (within sector)
     combined_neutral_alpha = mean(beta_neutral_alpha, sector_neutral_alpha)
6. Print top-10 by combined_neutral_alpha vs raw alpha_score.

Saves: alpha_scores_beta_neutral.csv
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

LOOKBACK_DAYS = 126  # ~6 months of trading days

def log(msg: str) -> None:
    print(f"  {msg}")

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ── load price cache ──────────────────────────────────────────────────────────

section("1. Loading sp500_price_cache.csv")

price_path = ROOT / "sp500_price_cache.csv"
if not price_path.exists():
    print("  ERROR: sp500_price_cache.csv not found. Exiting.")
    raise SystemExit(1)

price_raw = pd.read_csv(price_path)
price_raw["Date"] = pd.to_datetime(price_raw["Date"], errors="coerce")
price_raw = price_raw.dropna(subset=["Date"])
price_raw = price_raw.set_index("Date").sort_index()

# Drop spurious non-ticker columns
for bad_col in ["1", "Unnamed: 0"]:
    if bad_col in price_raw.columns:
        price_raw = price_raw.drop(columns=[bad_col])

price_raw = price_raw.astype(float)
log(f"Price matrix: {price_raw.shape}")
log(f"Date range  : {price_raw.index[0].date()} to {price_raw.index[-1].date()}")
log(f"SPY in cache: {'SPY' in price_raw.columns}")

# Use last 126 trading days
price_window = price_raw.iloc[-LOOKBACK_DAYS:]
log(f"Using last {LOOKBACK_DAYS} trading days: "
    f"{price_window.index[0].date()} to {price_window.index[-1].date()}")

# ── get SPY prices ────────────────────────────────────────────────────────────

section("2. Getting SPY prices for the window period")

spy_available_in_cache = "SPY" in price_window.columns

if spy_available_in_cache:
    spy_prices = price_window["SPY"].dropna()
    log(f"SPY loaded from cache: {len(spy_prices)} days")
else:
    log("SPY not in price cache. Downloading from yfinance...")
    try:
        import yfinance as yf
        start_str = price_window.index[0].strftime("%Y-%m-%d")
        end_str   = price_window.index[-1].strftime("%Y-%m-%d")
        spy_raw   = yf.download("SPY", start=start_str, end=end_str,
                                  auto_adjust=True, progress=False)
        spy_prices = spy_raw["Close"].squeeze()
        spy_prices.index = pd.to_datetime(spy_prices.index)
        log(f"SPY downloaded: {len(spy_prices)} days")
    except Exception as e:
        log(f"ERROR: Cannot get SPY prices: {e}")
        raise SystemExit(1)

# Daily returns for SPY
spy_ret = spy_prices.pct_change().dropna()
spy_var = float(spy_ret.var())
log(f"SPY daily return variance: {spy_var:.6f}")

if spy_var == 0:
    print("  FATAL: SPY variance is zero. Cannot compute betas.")
    raise SystemExit(1)

# ── compute daily returns for each stock ──────────────────────────────────────

section("3. Computing daily returns for all stocks")

# Exclude SPY from stock list
stock_tickers = [c for c in price_window.columns if c != "SPY"]
price_stocks  = price_window[stock_tickers].astype(float).ffill()
daily_rets    = price_stocks.pct_change().dropna(how="all")

# Align with SPY dates
common_dates = daily_rets.index.intersection(spy_ret.index)
daily_rets   = daily_rets.loc[common_dates]
spy_ret_aligned = spy_ret.loc[common_dates]

log(f"Common date window: {len(common_dates)} days")
log(f"Stocks with data: {daily_rets.notna().any().sum()}")

# ── compute betas ─────────────────────────────────────────────────────────────

section("4. Computing market betas")

betas: dict[str, float] = {}
min_obs = 30  # require at least 30 valid observations

for ticker in stock_tickers:
    if ticker not in daily_rets.columns:
        betas[ticker] = float("nan")
        continue

    stock_r = daily_rets[ticker].dropna()
    common  = stock_r.index.intersection(spy_ret_aligned.index)

    if len(common) < min_obs:
        betas[ticker] = float("nan")
        continue

    s_ret = stock_r.loc[common]
    m_ret = spy_ret_aligned.loc[common]

    cov = float(np.cov(s_ret.values, m_ret.values)[0, 1])
    var = float(m_ret.var())

    betas[ticker] = cov / var if var > 0 else float("nan")

beta_series = pd.Series(betas, name="market_beta")
valid_betas = beta_series.dropna()
log(f"Betas computed: {len(valid_betas)} / {len(stock_tickers)}")
log(f"Beta distribution — mean: {valid_betas.mean():.3f}, "
    f"std: {valid_betas.std():.3f}, "
    f"min: {valid_betas.min():.3f}, "
    f"max: {valid_betas.max():.3f}")

# ── load alpha_scores ─────────────────────────────────────────────────────────

section("5. Loading alpha_scores.csv")

alpha_path = ROOT / "alpha_scores.csv"
if not alpha_path.exists():
    print("  ERROR: alpha_scores.csv not found. Exiting.")
    raise SystemExit(1)

alpha_df = pd.read_csv(alpha_path)
log(f"Alpha scores: {len(alpha_df)} tickers")
log(f"Columns: {alpha_df.columns.tolist()}")

if "sector" not in alpha_df.columns:
    log("WARNING: 'sector' column not found. Sector neutralization will be skipped.")
    alpha_df["sector"] = "Unknown"

# ── merge betas ───────────────────────────────────────────────────────────────

section("6. Merging betas and computing neutralized alpha")

alpha_df = alpha_df.merge(
    beta_series.reset_index().rename(columns={"index": "ticker"}),
    on="ticker", how="left"
)

n_with_beta = alpha_df["market_beta"].notna().sum()
log(f"Tickers with beta computed: {n_with_beta} / {len(alpha_df)}")

# Fill missing betas with median
median_beta = float(valid_betas.median())
alpha_df["market_beta"] = alpha_df["market_beta"].fillna(median_beta)
alpha_df["market_beta"] = alpha_df["market_beta"].round(4)

log(f"Median beta (used for fills): {median_beta:.3f}")

# ── beta-neutral alpha ─────────────────────────────────────────────────────────
# Regress alpha_score on market_beta, take residual
# beta_neutral_alpha = alpha_score - (beta_coef * (market_beta - 1.0))
# where beta_coef is estimated via OLS

scores = alpha_df["alpha_score"].values
betas_arr = alpha_df["market_beta"].values

# Demean beta (relative to beta=1.0 = market neutral)
beta_demeaned = betas_arr - 1.0

# OLS: alpha_score ~ const + beta_coef * (beta - 1)
X = np.column_stack([np.ones(len(beta_demeaned)), beta_demeaned])
try:
    ols_coef, _, _, _ = np.linalg.lstsq(X, scores, rcond=None)
    mean_alpha = ols_coef[0]
    beta_coef  = ols_coef[1]
    log(f"OLS: mean_alpha = {mean_alpha:.3f}, beta_coef = {beta_coef:.3f}")
    log(f"Interpretation: each unit of beta above 1 adds {beta_coef:.2f} alpha score points")
except Exception as e:
    log(f"OLS failed: {e}. Using simple subtraction.")
    mean_alpha = float(np.mean(scores))
    beta_coef  = 0.0

alpha_df["beta_neutral_alpha"] = (
    alpha_df["alpha_score"] - beta_coef * (alpha_df["market_beta"] - 1.0)
).round(4)

# ── sector-neutral alpha ──────────────────────────────────────────────────────

sector_means = alpha_df.groupby("sector")["alpha_score"].transform("mean")
alpha_df["sector_neutral_alpha"] = (alpha_df["alpha_score"] - sector_means).round(4)

# ── combined neutral alpha ────────────────────────────────────────────────────

# Re-scale both to same range, then average
def min_max_scale(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return ((s - s.min()) / rng * 100) if rng > 0 else s - s.mean() + 50

bn_scaled  = min_max_scale(alpha_df["beta_neutral_alpha"])
sn_scaled  = min_max_scale(alpha_df["sector_neutral_alpha"])
alpha_df["combined_neutral_alpha"] = ((bn_scaled + sn_scaled) / 2).round(4)

# ── print top 10 comparison ───────────────────────────────────────────────────

section("7. Top-10 comparison: raw alpha_score vs combined_neutral_alpha")

top10_raw  = alpha_df.nlargest(10, "alpha_score")[["ticker", "alpha_score", "market_beta", "sector", "combined_neutral_alpha"]]
top10_neu  = alpha_df.nlargest(10, "combined_neutral_alpha")[["ticker", "alpha_score", "market_beta", "sector", "combined_neutral_alpha"]]

log("--- Top 10 by RAW alpha_score ---")
print(top10_raw.to_string(index=False))
print()
log("--- Top 10 by COMBINED_NEUTRAL_ALPHA ---")
print(top10_neu.to_string(index=False))
print()

# Compute overlap
raw_set = set(top10_raw["ticker"].tolist())
neu_set = set(top10_neu["ticker"].tolist())
overlap = raw_set.intersection(neu_set)
log(f"Overlap between top-10 lists: {len(overlap)} tickers — {sorted(overlap)}")

new_entrants = neu_set - raw_set
log(f"New in neutral top-10 (low-beta, high-alpha): {sorted(new_entrants)}")
dropped = raw_set - neu_set
log(f"Dropped from raw top-10 (high-beta): {sorted(dropped)}")

# ── save CSV ──────────────────────────────────────────────────────────────────

section("8. Saving alpha_scores_beta_neutral.csv")

out_path = ROOT / "alpha_scores_beta_neutral.csv"
alpha_df.to_csv(out_path, index=False)
log(f"Saved to {out_path}")
log(f"Columns: {alpha_df.columns.tolist()}")

# ── print summary ─────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  BETA NEUTRAL ALPHA SUMMARY")
print("="*60)
print(f"  Beta lookback        : {LOOKBACK_DAYS} trading days")
print(f"  Tickers with beta    : {n_with_beta}")
print(f"  Median market beta   : {median_beta:.3f}")
print(f"  OLS beta coefficient : {beta_coef:.3f}")
print(f"  Overlap (raw vs neutral top-10): {len(overlap)}/10 tickers")
print()
print("  New entries in neutral top-10 (low-beta, high-alpha):")
for t in sorted(new_entrants):
    row = alpha_df[alpha_df["ticker"] == t].iloc[0]
    print(f"    {t:6s}  raw={row['alpha_score']:.1f}  beta={row['market_beta']:.2f}  "
          f"neutral={row['combined_neutral_alpha']:.1f}  sector={row['sector']}")
print()
print("  => alpha_scores_beta_neutral.csv saved.")
