#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canyon v9 — Step 470: Accruals Quality Signal (Sloan 1996)
==========================================================
Classic anomaly discovered by Richard Sloan (1996 Accounting Review):

  Companies where accounting earnings > cash earnings → low earnings quality → future underperformance
  Companies where accounting earnings < cash earnings → high earnings quality → future outperformance

Core metric:

  Accrual Ratio = (Net Income - Operating Cash Flow) / Total Assets
  Higher = more "paper component" in earnings = lower quality = SHORT signal

Why it works:
  Institutions use cash flow for DCF valuation, not accounting earnings
  When retail investors/analysts are misled by high accounting earnings and overvalue companies
  → institutions short overvalued accrual stocks → price reverts to mean

OOS IC: Sloan original paper +0.083; replication studies +0.06~+0.12

Data sources:
  yfinance.Ticker.cashflow        → Operating Cash Flow
  yfinance.Ticker.income_stmt     → Net Income
  yfinance.Ticker.balance_sheet   → Total Assets

Usage:
  .venv/bin/python canyon_final_v9_step470_accruals.py
"""
from __future__ import annotations

import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
OOS_CUTOFF    = pd.Timestamp("2020-01-01")
REQUEST_DELAY = 0.40
ANN_FACTOR    = 252


# =============================================================================
# 1. Fetch Financial Data
# =============================================================================

def fetch_financials(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch annual: Operating Cash Flow, Net Income, Total Assets.
    Returns a long-format DataFrame with one row per (ticker, fiscal_year).
    """
    cache = ROOT / "accruals_financials.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["fiscal_year"])
        print(f"  Financials: loaded from cache ({len(df)} rows)")
        return df

    rows = []
    for tk in tickers:
        try:
            obj = yf.Ticker(tk)

            # Cash flow statement
            cf = obj.cashflow
            # Income statement
            inc = obj.income_stmt
            # Balance sheet
            bs  = obj.balance_sheet

            if cf is None or cf.empty:
                continue
            if inc is None or inc.empty:
                continue

            # Operating cash flow row
            ocf_row = None
            for label in ["Operating Cash Flow", "Cash From Operations",
                           "Total Cash From Operating Activities"]:
                if label in cf.index:
                    ocf_row = cf.loc[label]
                    break

            # Net income row
            ni_row = None
            for label in ["Net Income", "Net Income Common Stockholders",
                          "Net Income Applicable To Common Shares"]:
                if label in inc.index:
                    ni_row = inc.loc[label]
                    break

            # Total assets
            ta_row = None
            if bs is not None and not bs.empty:
                for label in ["Total Assets"]:
                    if label in bs.index:
                        ta_row = bs.loc[label]
                        break

            if ocf_row is None or ni_row is None:
                continue

            for col in ocf_row.index:
                try:
                    ocf = float(ocf_row[col])
                    ni  = float(ni_row[col]) if col in ni_row.index else np.nan
                    ta  = float(ta_row[col]) if (ta_row is not None and
                                                   col in ta_row.index) else np.nan

                    if np.isnan(ni) or np.isnan(ocf):
                        continue

                    accrual = ni - ocf
                    accrual_ratio = accrual / (abs(ta) + 1e-6) if not np.isnan(ta) \
                                    else accrual / (abs(ni) + 1e-6)

                    rows.append({
                        "ticker":        tk,
                        "fiscal_year":   pd.Timestamp(col),
                        "net_income":    round(ni,  0),
                        "op_cashflow":   round(ocf, 0),
                        "total_assets":  round(ta,  0) if not np.isnan(ta) else None,
                        "accrual":       round(accrual, 0),
                        "accrual_ratio": round(accrual_ratio, 4),
                    })
                except Exception:
                    continue

            time.sleep(REQUEST_DELAY)

        except Exception as e:
            print(f"  {tk}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(cache, index=False)
    return df


# =============================================================================
# 2. IC Evaluation
# =============================================================================

def evaluate_accruals_ic(
    fin_df:   pd.DataFrame,
    prices:   pd.DataFrame,
    reb_dates: list[str],
) -> tuple[pd.DataFrame, float, float]:
    """
    At each OOS rebalance date, use most-recent annual accrual ratio
    to predict next-period cross-sectional returns.
    """
    oos = [r for r in reb_dates if r >= str(OOS_CUTOFF.date())]
    ic_rows = []

    for i, reb in enumerate(oos[:-1]):
        next_reb = oos[i + 1]
        reb_ts   = pd.Timestamp(reb)

        # Most recent fiscal year available before rebalance
        avail = fin_df[fin_df["fiscal_year"] < reb_ts]
        if avail.empty:
            continue
        latest = (avail.sort_values("fiscal_year")
                       .groupby("ticker")["accrual_ratio"]
                       .last()
                       .reset_index())
        latest.columns = ["ticker", "accrual_ratio"]

        # Forward returns
        fwd = {}
        t0s = prices.index[prices.index >= reb_ts]
        t1s = prices.index[prices.index >= pd.Timestamp(next_reb)]
        if not len(t0s) or not len(t1s):
            continue

        for tk in latest["ticker"]:
            if tk not in prices.columns:
                continue
            p0 = prices[tk].loc[t0s[0]]
            p1 = prices[tk].loc[t1s[0]]
            if p0 > 0 and p1 > 0:
                fwd[tk] = float(p1 / p0) - 1

        merged = latest[latest["ticker"].isin(fwd)].copy()
        merged["fwd_ret"] = merged["ticker"].map(fwd)
        merged = merged.dropna()
        if len(merged) < 5:
            continue

        # Low accrual = high quality = positive IC, so negate accrual_ratio
        ic, pval = spearmanr(-merged["accrual_ratio"], merged["fwd_ret"])
        ic_rows.append({
            "rebalance_date": reb,
            "ic": round(ic, 4),
            "p_value": round(pval, 4),
            "n_stocks": len(merged),
        })

    if not ic_rows:
        return pd.DataFrame(), 0.0, 0.0
    df     = pd.DataFrame(ic_rows)
    mean_ic= df["ic"].mean()
    t_stat = mean_ic / (df["ic"].std() / np.sqrt(len(df)) + 1e-10)
    return df, float(mean_ic), float(t_stat)


# =============================================================================
# 3. Current Accruals Signal Snapshot
# =============================================================================

def build_accruals_snapshot(fin_df: pd.DataFrame) -> pd.DataFrame:
    """Most-recent accrual ratio per ticker, ranked."""
    if fin_df.empty:
        return pd.DataFrame()
    latest = (fin_df.sort_values("fiscal_year")
                    .groupby("ticker")
                    .last()
                    .reset_index()
                    [["ticker", "fiscal_year", "accrual_ratio",
                      "net_income", "op_cashflow"]])
    latest["accrual_quality"] = -latest["accrual_ratio"]  # higher = better quality
    return latest.sort_values("accrual_quality", ascending=False)


# =============================================================================
# main
# =============================================================================

def main() -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("=" * 70)
    print("Canyon v9 — Step 470: Accruals Quality Signal (Sloan 1996)")
    print("=" * 70)

    print("\n[1/4] Loading base data …")
    prices = pd.read_csv(ROOT / "backtest_price_cache.csv",
                         index_col=0, parse_dates=True)
    preds  = pd.read_csv(ROOT / "cpcv_predictions.csv",
                         parse_dates=["rebalance_date"])
    tickers   = [c for c in prices.columns if c != "SPY"]
    reb_dates = sorted(preds["rebalance_date"].dt.date.unique().astype(str).tolist())
    print(f"  {len(tickers)} tickers")

    print("\n[2/4] Fetching annual financials (OCF, NI, TA) …")
    fin_df = fetch_financials(tickers)
    print(f"  Records: {len(fin_df)} ({fin_df['ticker'].nunique()} tickers)")

    print("\n[3/4] Evaluating accruals IC (OOS) …")
    ic_df, mean_ic, t_stat = evaluate_accruals_ic(fin_df, prices, reb_dates)
    ic_df.to_csv(ROOT / "accruals_ic.csv", index=False)
    stars = "★★★" if abs(t_stat) > 2.0 else "★★" if abs(t_stat) > 1.5 else "★"
    print(f"  OOS Mean IC: {mean_ic:+.4f}  t = {t_stat:+.2f}  {stars}")

    print("\n[4/4] Current accruals signal snapshot …")
    snapshot = build_accruals_snapshot(fin_df)
    snapshot.to_csv(ROOT / "accruals_snapshot.csv", index=False)

    print(f"\n  HIGH QUALITY (low accruals = more cash earnings = LONG):")
    for _, r in snapshot.head(6).iterrows():
        ocf_ratio = float(r["op_cashflow"]) / (abs(float(r["net_income"])) + 1e-6)
        print(f"    {r['ticker']:<6}  accrual_ratio={r['accrual_ratio']:+.3f}  "
              f"OCF/NI={ocf_ratio:.2f}x  "
              f"FY={str(r['fiscal_year'])[:7]}")

    print(f"\n  LOW QUALITY (high accruals = paper earnings = SHORT):")
    for _, r in snapshot.tail(6).iloc[::-1].iterrows():
        print(f"    {r['ticker']:<6}  accrual_ratio={r['accrual_ratio']:+.3f}  "
              f"FY={str(r['fiscal_year'])[:7]}")

    # Updated signal stack
    print(f"\n  ── Signal Stack Update ──")
    print(f"  PEAD:            IC = +0.229  ★★★")
    print(f"  VIX:             IC = +0.223  ★★★")
    print(f"  Accruals:        IC = {mean_ic:+.3f}  {stars}  (Sloan 1996)")
    print(f"  Quality (ROE):   IC = +0.048  ★★")
    print(f"  Analyst Revision:IC = +0.038  ★★")

    print("\n" + "=" * 70)
    print("Step 470 Complete — Accruals Quality Signal")
    print("=" * 70)
    for f in ["accruals_ic.csv", "accruals_snapshot.csv",
              "accruals_financials.csv"]:
        p = ROOT / f
        sz = f"{p.stat().st_size/1024:.1f} KB" if p.exists() else "—"
        print(f"  {f:<44} {sz}")


if __name__ == "__main__":
    main()
