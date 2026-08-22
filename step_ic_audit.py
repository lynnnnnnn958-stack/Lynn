#!/usr/bin/env python3
"""
Canyon — step_ic_audit.py
=========================
Audit whether IC=0.37 is statistically valid.

Methodology
-----------
1. Load alpha_score_history.csv (ticker/date/alpha_score).
2. Load sp500_price_cache.csv (Date x ticker price matrix).
3. For each unique date T in history: rank stocks by alpha_score, then compute
   21-day forward return for each stock using price data.
4. Compute Spearman IC between alpha_score_rank and 21d forward return.
5. Compute 95% CI, verdict, look-ahead risk assessment.

Saves: ic_audit_report.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── helpers ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"  {msg}")

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ── load data ─────────────────────────────────────────────────────────────────

section("1. Loading alpha_score_history.csv")
hist_path = ROOT / "alpha_score_history.csv"
if not hist_path.exists():
    print("  ERROR: alpha_score_history.csv not found. Exiting.")
    raise SystemExit(1)

hist = pd.read_csv(hist_path, parse_dates=["date"])
log(f"History rows: {len(hist):,}")
log(f"Unique dates: {hist['date'].nunique()}")
log(f"Date range  : {hist['date'].min().date()} to {hist['date'].max().date()}")
log(f"Unique tickers: {hist['ticker'].nunique()}")

n_unique_dates = hist["date"].nunique()
unique_dates_sorted = sorted(hist["date"].unique())

# ── load price cache ──────────────────────────────────────────────────────────

section("2. Loading sp500_price_cache.csv")
price_path = ROOT / "sp500_price_cache.csv"
if not price_path.exists():
    print("  ERROR: sp500_price_cache.csv not found. Exiting.")
    raise SystemExit(1)

price_raw = pd.read_csv(price_path)
if "Date" not in price_raw.columns:            # 日期是无名 index 列 → 命名为 Date
    price_raw = price_raw.rename(columns={price_raw.columns[0]: "Date"})
price_raw["Date"] = pd.to_datetime(price_raw["Date"], errors="coerce")
price_raw = price_raw.dropna(subset=["Date"])
price_raw = price_raw.set_index("Date").sort_index()

# Drop non-ticker column named '1' if present
if "1" in price_raw.columns:
    price_raw = price_raw.drop(columns=["1"])

log(f"Price matrix shape : {price_raw.shape}")
log(f"Price date range   : {price_raw.index[0].date()} to {price_raw.index[-1].date()}")
log(f"SPY in price cache : {'SPY' in price_raw.columns}")

# ── compute IC per period ─────────────────────────────────────────────────────

section("3. Computing Spearman IC (alpha_score_rank vs 21d forward return)")

FWD_DAYS = 21
ic_values: list[float] = []
skipped = 0

price_dates = price_raw.index.to_list()

for date_t in unique_dates_sorted:
    # Get alpha scores at date T
    snap = hist[hist["date"] == date_t][["ticker", "alpha_score"]].copy()
    snap = snap.dropna(subset=["alpha_score"])
    if len(snap) < 10:
        skipped += 1
        continue

    # Find T+21 price date (nearest trading day >= T+21)
    target_date = date_t + pd.Timedelta(days=FWD_DAYS * 1.5)  # buffer for weekends
    future_dates = [d for d in price_dates if d >= target_date]
    if not future_dates:
        skipped += 1
        continue
    date_fwd = future_dates[0]

    # Get prices at T and T+21
    # Find nearest past price date for T
    past_dates = [d for d in price_dates if d <= date_t]
    if not past_dates:
        skipped += 1
        continue
    date_now = past_dates[-1]

    prices_now = price_raw.loc[date_now]
    prices_fwd = price_raw.loc[date_fwd]

    # Compute forward returns for each ticker
    tickers = snap["ticker"].tolist()
    valid_tickers = [t for t in tickers if t in prices_now.index and t in prices_fwd.index]
    if len(valid_tickers) < 10:
        skipped += 1
        continue

    p_now = prices_now[valid_tickers].astype(float)
    p_fwd = prices_fwd[valid_tickers].astype(float)
    fwd_ret = (p_fwd / p_now) - 1.0
    fwd_ret = fwd_ret.replace([np.inf, -np.inf], np.nan).dropna()

    snap_sub = snap[snap["ticker"].isin(fwd_ret.index)].copy()
    snap_sub = snap_sub.set_index("ticker")
    common = snap_sub.index.intersection(fwd_ret.index)
    if len(common) < 10:
        skipped += 1
        continue

    alpha_scores = snap_sub.loc[common, "alpha_score"]
    returns = fwd_ret.loc[common]

    # Rank alpha scores (already provided, but re-rank for robustness)
    alpha_rank = alpha_scores.rank(ascending=True)

    corr, pval = spearmanr(alpha_rank, returns)
    if not np.isnan(corr):
        ic_values.append(float(corr))

log(f"Valid IC observations : {len(ic_values)}")
log(f"Skipped periods       : {skipped}")

# ── compute statistics ────────────────────────────────────────────────────────

section("4. Computing IC statistics")

n_periods = len(ic_values)

if n_periods == 0:
    ic_mean    = float("nan")
    ic_se      = float("nan")
    ic_ci_low  = float("nan")
    ic_ci_high = float("nan")
    verdict    = "NO DATA"
else:
    ic_mean    = float(np.mean(ic_values))
    ic_se      = float(1.0 / np.sqrt(n_periods)) if n_periods > 1 else float("nan")
    ic_ci_low  = float(ic_mean - 1.96 * ic_se) if not np.isnan(ic_se) else float("nan")
    ic_ci_high = float(ic_mean + 1.96 * ic_se) if not np.isnan(ic_se) else float("nan")

    if n_periods < 60:
        verdict = "NOT RELIABLE"
    elif n_periods < 120:
        verdict = "LOW CONFIDENCE"
    else:
        verdict = "RELIABLE"

log(f"n_periods  : {n_periods}")
log(f"IC (mean)  : {ic_mean:.4f}" if not np.isnan(ic_mean) else "IC: N/A")
log(f"IC SE      : {ic_se:.4f}" if not np.isnan(ic_se) else "SE: N/A")
log(f"CI (95%)   : [{ic_ci_low:.4f}, {ic_ci_high:.4f}]" if not np.isnan(ic_ci_low) else "CI: N/A")
log(f"Verdict    : {verdict}")

# ── look-ahead bias assessment ────────────────────────────────────────────────

section("5. Look-ahead bias assessment")

# Check: all historical dates use the SAME alpha_score for each ticker?
# If a ticker appears on multiple dates with identical alpha_score, it's a red flag.
ticker_date_scores = hist.groupby("ticker")["alpha_score"].nunique()
pct_with_one_unique = (ticker_date_scores == 1).mean()

look_ahead_risk = bool(pct_with_one_unique > 0.5)

log(f"Tickers with only 1 unique alpha_score across all dates : "
    f"{(ticker_date_scores == 1).sum()} / {len(ticker_date_scores)} "
    f"({pct_with_one_unique*100:.1f}%)")
log(f"Look-ahead risk flagged : {look_ahead_risk}")

if look_ahead_risk:
    log("WARNING: Most tickers show the same alpha_score across all historical dates.")
    log("         This strongly suggests TODAY's signal file was used for all past dates.")
    log("         This makes any IC computed on this data MEANINGLESS (look-ahead bias).")

# ── recommendations ───────────────────────────────────────────────────────────

recommendations: list[str] = []

if n_unique_dates < 60:
    recommendations.append(
        f"Only {n_unique_dates} unique dates found (need 60+ for statistical reliability). "
        "Run the system daily for 3+ months to accumulate enough history."
    )

if look_ahead_risk:
    recommendations.append(
        "HIGH LOOK-AHEAD RISK: alpha_score_history.csv appears to use the current signal "
        "for all past dates. True IC requires storing daily scores as they were computed "
        "on each historical date. Implement a daily snapshot logger."
    )

if n_periods > 0 and not np.isnan(ic_ci_low) and ic_ci_low <= 0:
    recommendations.append(
        f"IC confidence interval [{ic_ci_low:.4f}, {ic_ci_high:.4f}] includes zero. "
        "Cannot reject null hypothesis of zero predictive power at 95% confidence."
    )

if n_periods > 0 and ic_mean > 0.2:
    recommendations.append(
        f"Computed IC={ic_mean:.4f} is suspiciously high for a composite factor. "
        "Typical literature values are 0.02-0.08. This may reflect look-ahead bias."
    )

recommendations.append(
    "To obtain a valid IC: (1) log daily alpha_scores to a time-stamped snapshot file, "
    "(2) accumulate 60+ trading periods (3+ months), (3) re-run this audit."
)

# ── save report ───────────────────────────────────────────────────────────────

section("6. Saving ic_audit_report.json")

report = {
    "generated_at"   : pd.Timestamp.now().isoformat(),
    "n_unique_dates_in_history" : int(n_unique_dates),
    "n_periods"      : int(n_periods),
    "history_date_range_start"  : str(min(unique_dates_sorted).date()),
    "history_date_range_end"    : str(max(unique_dates_sorted).date()),
    "ic_value"       : round(ic_mean, 6) if not np.isnan(ic_mean) else None,
    "ic_se"          : round(ic_se, 6)   if not np.isnan(ic_se)   else None,
    "ic_ci_low"      : round(ic_ci_low, 6)  if not np.isnan(ic_ci_low)  else None,
    "ic_ci_high"     : round(ic_ci_high, 6) if not np.isnan(ic_ci_high) else None,
    "verdict"        : verdict,
    "look_ahead_risk": look_ahead_risk,
    "pct_tickers_with_constant_alpha": round(float(pct_with_one_unique), 4),
    "recommendations": recommendations,
    "methodology"    : (
        f"Spearman IC between alpha_score_rank at date T and "
        f"{FWD_DAYS}-day forward return at T+{FWD_DAYS}. "
        "SE = 1/sqrt(n). CI = IC ± 1.96*SE. "
        "Verdict thresholds: n<60 → NOT RELIABLE, 60-120 → LOW CONFIDENCE, >120 → RELIABLE."
    ),
}

out_path = ROOT / "ic_audit_report.json"
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
log(f"Saved to {out_path}")

# ── print summary ─────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  IC AUDIT SUMMARY")
print("="*60)
print(f"  History dates   : {n_unique_dates} unique dates")
print(f"  IC observations : {n_periods}")
print(f"  Computed IC     : {ic_mean:.4f}" if not np.isnan(ic_mean) else "  Computed IC : N/A")
print(f"  95% CI          : [{ic_ci_low:.4f}, {ic_ci_high:.4f}]" if not np.isnan(ic_ci_low) else "  95% CI : N/A")
print(f"  VERDICT         : {verdict}")
print(f"  Look-ahead risk : {look_ahead_risk}")
print()
for i, r in enumerate(recommendations, 1):
    print(f"  [{i}] {r[:100]}...")
    if len(r) > 100:
        print(f"      {r[100:]}")
print()
print("  => ic_audit_report.json saved.")
