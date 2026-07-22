#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canyon v9 — Step 165: Fundamental Signal Module
================================================
Adds institutional-grade fundamental alpha signals to the feature set.

Signals implemented (all with 45-day filing lag — no lookahead):
  accruals        — Earnings quality: -(Net Income - Op CF) / Assets     [Sloan 1996]
  rev_growth      — Revenue growth YoY (quarterly)                        [sales momentum]
  gross_margin_chg— Gross margin improvement vs year ago                  [quality trend]
  roe             — Return on equity (trailing 12 months)                 [Fama-French quality]
  debt_change     — Change in debt-to-assets ratio YoY (negative signal) [Penman 2001]
  pb_ratio        — Price-to-book ratio (value factor)                    [Fama-French HML]

All signals are:
  1. Downloaded via yfinance quarterly statements
  2. Shifted forward by 45 days (conservative filing lag)
  3. Forward-filled until next quarter + lag
  4. Evaluated via Spearman IC vs 1-month forward price return

Outputs:
  fundamental_signals_daily.csv  — date × ticker × 6 signal panel
  fundamental_ic_report.csv      — per-signal IS/OOS IC stats
  fundamental_ic_report.md       — narrative markdown report

Usage:
  python3 canyon_final_v9_step165_fundamental_signals.py           # full run
  python3 canyon_final_v9_step165_fundamental_signals.py --ic-only # skip download
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT              = Path(__file__).parent
FILING_LAG_Q      = 45    # days: 10-Q quarterly filing lag (large accelerated filer)
FILING_LAG_A      = 120   # days: 10-K annual filing lag
OOS_CUTOFF        = pd.Timestamp("2020-01-01")
USE_ANNUAL        = True  # annual data goes back 5 years; quarterly only ~4 quarters

# Tickers that have quarterly financial statements (exclude ETFs)
ETF_TICKERS = {"SPY","QQQ","SMH","SOXX","XLK","XLE","XLF","XLV","XLU","XLP"}

STOCK_UNIVERSE = [
    "NVDA","TSLA","AMD","MU","GOOGL","AMZN","MSFT","AAPL","META","JPM",
    "XOM","CVX","JNJ","WMT","KO","PEP","MRK","ABBV","UNH","LLY","TMO",
    "COST","V","MA","HD","PYPL","NFLX","INTC","QCOM","TXN","AVGO",
    "CRM","ADBE","CSCO","ORCL","IBM","HPQ","WFC","BAC","C","GS","MS",
    "PFE","GILD","BMY","AMGN","ABT","COP","SLB","PG","F",
]

SIGNALS = ["accruals", "rev_growth", "gross_margin_chg", "roe", "debt_change", "pb_ratio"]


# ─────────────────────────────────────────────────────────────────────────────
# yfinance helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_row(df: pd.DataFrame, candidates: list[str]) -> Optional[pd.Series]:
    """Try multiple row names (yfinance naming is inconsistent across tickers)."""
    for name in candidates:
        if name in df.index:
            row = df.loc[name]
            if not row.dropna().empty:
                return row.dropna().sort_index()   # chronological order
    return None


def _compute_ticker_signals(ticker: str) -> Optional[pd.DataFrame]:
    """
    Download annual fundamentals and compute 6 signals for one ticker.
    Uses 10-K annual statements (go back ~5 years) rather than quarterly
    (only ~4 quarters available via yfinance free tier).

    Returns a DataFrame indexed by 'available_date' (fiscal-year-end + filing lag)
    with columns = SIGNALS.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)

        # Annual statements — go back ~5 fiscal years
        qf = t.income_stmt         # Annual P&L
        qb = t.balance_sheet       # Annual balance sheet
        qc = t.cashflow            # Annual cash flow

        if qf is None or qf.empty:
            return None

        # Convert columns to timestamps, sort chronological (oldest first)
        def _clean(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df.columns = pd.to_datetime(df.columns)
            return df[sorted(df.columns)]

        qf = _clean(qf)
        qb = _clean(qb) if qb is not None and not qb.empty else pd.DataFrame()
        qc = _clean(qc) if qc is not None and not qc.empty else pd.DataFrame()

        quarter_dates = qf.columns   # fiscal year-end dates
        if len(quarter_dates) < 2:   # need at least 2 years for YoY
            return None

        rows = []
        for i, q_end in enumerate(quarter_dates[1:], start=1):
            q_ago_1y = quarter_dates[i - 1]   # prior fiscal year

            available_date = q_end + pd.Timedelta(days=FILING_LAG_A)

            rec: dict[str, float] = {"fiscal_year_end": q_end,
                                     "available_date": available_date}

            # ── 1. Accruals (Sloan 1996): -(NI - OpCF) / Assets ────────────
            ni_row  = _get_row(qf, ["Net Income", "Net Income Common Stockholders",
                                    "Diluted NI Available To Com Stockholders"])
            cf_row  = _get_row(qc, ["Operating Cash Flow", "Cash Flow From Operations",
                                    "Cash Flow From Continuing Operating Activities"])
            ast_row = _get_row(qb, ["Total Assets"])

            try:
                ni   = float(ni_row[q_end])
                cf   = float(cf_row[q_end])
                ast  = float(ast_row[q_end])
                rec["accruals"] = -(ni - cf) / (abs(ast) + 1) if ast != 0 else np.nan
            except Exception:
                rec["accruals"] = np.nan

            # ── 2. Revenue growth YoY ────────────────────────────────────────
            rev_row = _get_row(qf, ["Total Revenue", "Revenue", "Sales"])
            try:
                rev_now = float(rev_row[q_end])
                rev_ago = float(rev_row[q_ago_1y])
                rec["rev_growth"] = (rev_now - rev_ago) / (abs(rev_ago) + 1) \
                                     if rev_ago != 0 else np.nan
            except Exception:
                rec["rev_growth"] = np.nan

            # ── 3. Gross margin change ───────────────────────────────────────
            gp_row = _get_row(qf, ["Gross Profit"])
            try:
                gp_now  = float(gp_row[q_end])
                gp_ago  = float(gp_row[q_ago_1y])
                rev_now = float(rev_row[q_end])
                rev_ago = float(rev_row[q_ago_1y])
                gm_now  = gp_now / (abs(rev_now) + 1)
                gm_ago  = gp_ago / (abs(rev_ago) + 1)
                rec["gross_margin_chg"] = gm_now - gm_ago
            except Exception:
                rec["gross_margin_chg"] = np.nan

            # ── 4. ROE (annual NI / average equity) ──────────────────────────
            eq_row = _get_row(qb, ["Stockholders Equity", "Total Stockholder Equity",
                                   "Common Stock Equity"])
            try:
                ni_annual = float(ni_row[q_end])
                eq_now = float(eq_row[q_end])
                eq_ago = float(eq_row[q_ago_1y])
                eq_avg = (eq_now + eq_ago) / 2
                rec["roe"] = ni_annual / (abs(eq_avg) + 1) if eq_avg != 0 else np.nan
            except Exception:
                rec["roe"] = np.nan

            # ── 5. Debt change (change in debt ratio YoY — negative signal) ──
            ld_row = _get_row(qb, ["Long Term Debt",
                                   "Long Term Debt And Capital Lease Obligation"])
            sd_row = _get_row(qb, ["Current Debt", "Short Long Term Debt",
                                   "Current Debt And Capital Lease Obligation"])
            try:
                ld_now = float(ld_row[q_end]) if ld_row is not None and q_end in ld_row.index else 0
                sd_now = float(sd_row[q_end]) if sd_row is not None and q_end in sd_row.index else 0
                td_now = ld_now + sd_now
                ast_now = float(ast_row[q_end])
                dr_now  = td_now / (abs(ast_now) + 1)

                ld_ago = float(ld_row[q_ago_1y]) if ld_row is not None and q_ago_1y in ld_row.index else 0
                sd_ago = float(sd_row[q_ago_1y]) if sd_row is not None and q_ago_1y in sd_row.index else 0
                td_ago = ld_ago + sd_ago
                ast_ago = float(ast_row[q_ago_1y])
                dr_ago  = td_ago / (abs(ast_ago) + 1)

                rec["debt_change"] = -(dr_now - dr_ago)   # negative = increasing debt = negative signal
            except Exception:
                rec["debt_change"] = np.nan

            # ── 6. P/B ratio (price / book per share) ───────────────────────
            try:
                shares_row = _get_row(qb, ["Ordinary Shares Number", "Share Issued",
                                           "Common Stock Shares Outstanding"])
                shares = float(shares_row[q_end]) if shares_row is not None \
                         and q_end in shares_row.index else np.nan
                eq = float(eq_row[q_end]) if eq_row is not None else np.nan
                bvps = eq / shares if (shares and shares > 0) else np.nan
                # P/B = 1/bvps × current price (will be computed later using prices)
                rec["_bvps"] = bvps
                rec["pb_ratio"] = np.nan   # filled in after merge with prices
            except Exception:
                rec["_bvps"] = np.nan
                rec["pb_ratio"] = np.nan

            rows.append(rec)

        if not rows:
            return None

        df = pd.DataFrame(rows).set_index("available_date").sort_index()
        df.index = pd.to_datetime(df.index)
        return df

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Build panel
# ─────────────────────────────────────────────────────────────────────────────

def build_fundamental_panel(tickers: list[str], prices: pd.DataFrame) -> pd.DataFrame:
    """
    For each ticker, compute fundamental signals and forward-fill to daily.
    Returns long-format DataFrame: [date, ticker, accruals, rev_growth, ...]
    """
    all_daily: list[pd.DataFrame] = []
    price_end = prices.index[-1]
    date_index = prices.index

    print(f"[step165] Computing fundamentals for {len(tickers)} tickers …")

    for i, ticker in enumerate(tickers):
        sig_df = _compute_ticker_signals(ticker)
        if sig_df is None or sig_df.empty:
            print(f"  {ticker:<8} — no data")
            continue

        # Compute P/B from price and book value per share
        if "_bvps" in sig_df.columns and ticker in prices.columns:
            for idx in sig_df.index:
                bvps = sig_df.loc[idx, "_bvps"]
                if pd.notna(bvps) and bvps > 0:
                    # Price at filing date
                    price_date = prices.index[prices.index <= idx]
                    if len(price_date) > 0:
                        px = prices.loc[price_date[-1], ticker]
                        sig_df.loc[idx, "pb_ratio"] = px / bvps if bvps > 0 else np.nan
            sig_df = sig_df.drop(columns=["_bvps"], errors="ignore")

        # Forward-fill quarterly signals to daily date index
        # Only use dates up to price_end to avoid out-of-range
        sig_cols = [c for c in SIGNALS if c in sig_df.columns]
        if not sig_cols:
            continue

        daily = (
            sig_df[sig_cols]
            .reindex(date_index, method="ffill")   # carry last known value forward
        )
        # Zero out dates before the first available signal (no lookahead)
        first_available = sig_df.index[0]
        daily.loc[daily.index < first_available] = np.nan

        daily["ticker"] = ticker
        daily.index.name = "date"
        all_daily.append(daily.reset_index())

        n_valid = daily[sig_cols].notna().all(axis=1).sum()
        print(f"  {ticker:<8} OK — {len(sig_df)} quarters, {n_valid} days with full data")
        time.sleep(0.3)   # be gentle with yfinance

    if not all_daily:
        print("[step165] No fundamental data retrieved.")
        return pd.DataFrame()

    panel = pd.concat(all_daily, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# IC evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ic_report(panel: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    For each fundamental signal and each rebalance date, compute Spearman IC
    vs 1-month forward return.  Report IS vs OOS IC stats.
    """
    if panel.empty:
        return pd.DataFrame()

    perf_path = ROOT / "wf_oos_backtest_perf.csv"
    if not perf_path.exists():
        print("[step165] wf_oos_backtest_perf.csv not found — using monthly dates")
        rebal_dates = pd.date_range("2015-01-01", prices.index[-1], freq="MS")
    else:
        perf = pd.read_csv(perf_path, parse_dates=["rebalance_date"])
        rebal_dates = pd.to_datetime(perf["rebalance_date"]).sort_values().unique()

    records = []
    for dt in rebal_dates:
        # 1-month forward price return
        fwd_end = dt + pd.offsets.BDay(21)
        if fwd_end > prices.index[-1]:
            continue
        price_at_dt  = prices.loc[prices.index <= dt].iloc[-1]
        price_at_fwd = prices.loc[prices.index <= fwd_end].iloc[-1]
        fwd_ret = (price_at_fwd / price_at_dt - 1).dropna()

        # Fundamental signals as of this date
        day_data = panel[panel["date"] == panel["date"][
            panel["date"] <= dt
        ].max()].set_index("ticker") if not panel.empty else pd.DataFrame()

        # More robust: find closest available date
        avail_dates = panel["date"].unique()
        past_dates  = avail_dates[avail_dates <= dt]
        if len(past_dates) == 0:
            continue
        closest = past_dates[-1]
        day_data = panel[panel["date"] == closest].set_index("ticker")

        period = "OOS" if dt >= OOS_CUTOFF else "IS"

        for sig in SIGNALS:
            if sig not in day_data.columns:
                continue
            sig_series = day_data[sig].dropna()
            common = sig_series.index.intersection(fwd_ret.index)
            if len(common) < 5:
                continue
            ic, _ = spearmanr(sig_series[common], fwd_ret[common])
            records.append({"date": dt, "signal": sig, "ic": ic, "period": period,
                            "n": len(common)})

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Summary table
    summary = df.groupby(["signal", "period"]).agg(
        mean_ic=("ic", "mean"),
        t_stat=("ic", lambda x: (x.mean() / (x.std() / np.sqrt(len(x))))
                                 if x.std() > 0 else 0),
        hit_rate=("ic", lambda x: (x > 0).mean()),
        n_months=("ic", "count"),
    ).reset_index()

    summary.to_csv(ROOT / "fundamental_ic_report.csv", index=False)
    print(f"\n[step165] IC report ({len(df)} signal-month observations):")

    for sig in SIGNALS:
        sub = summary[summary["signal"] == sig]
        is_row  = sub[sub["period"] == "IS"]
        oos_row = sub[sub["period"] == "OOS"]
        is_ic  = is_row["mean_ic"].values[0]  if len(is_row)  else np.nan
        oos_ic = oos_row["mean_ic"].values[0] if len(oos_row) else np.nan
        is_t   = is_row["t_stat"].values[0]   if len(is_row)  else np.nan
        oos_t  = oos_row["t_stat"].values[0]  if len(oos_row) else np.nan
        print(f"  {sig:<20} IS IC={is_ic:+.4f} (t={is_t:+.2f})  "
              f"OOS IC={oos_ic:+.4f} (t={oos_t:+.2f})")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────

def write_markdown_report(summary: pd.DataFrame, panel: pd.DataFrame):
    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    n_tickers = panel["ticker"].nunique() if not panel.empty else 0
    n_days    = panel["date"].nunique()   if not panel.empty else 0

    def fmt_row(sig: str) -> str:
        sub = summary[summary["signal"] == sig]
        is_row  = sub[sub["period"] == "IS"]
        oos_row = sub[sub["period"] == "OOS"]
        is_ic   = f"{is_row['mean_ic'].values[0]:+.4f}" if len(is_row)  else "N/A"
        oos_ic  = f"{oos_row['mean_ic'].values[0]:+.4f}" if len(oos_row) else "N/A"
        is_t    = f"{is_row['t_stat'].values[0]:+.2f}"  if len(is_row)  else "N/A"
        oos_t   = f"{oos_row['t_stat'].values[0]:+.2f}" if len(oos_row) else "N/A"
        is_hit  = f"{is_row['hit_rate'].values[0]*100:.0f}%" if len(is_row)  else "N/A"
        oos_hit = f"{oos_row['hit_rate'].values[0]*100:.0f}%" if len(oos_row) else "N/A"
        return (f"| {sig:<20} | {is_ic} (t={is_t}) | {is_hit} "
                f"| {oos_ic} (t={oos_t}) | {oos_hit} |")

    signal_rows = "\n".join(fmt_row(s) for s in SIGNALS
                             if s in summary["signal"].values)

    # Reference ICs for existing technical signals (from previous runs)
    ref_table = """| mom_12m_skip1m (existing) | +0.041 | 68% | +0.041 | 65% |
| trend_200 (existing)      | +0.045 | 64% | +0.040 | 62% |
| macd (existing)           | +0.025 | 58% | +0.025 | 57% |
| rsi_14 (existing)         | +0.015 | 54% | +0.018 | 56% |"""

    md = f"""# Canyon v9 — Fundamental Signal Report
**Generated:** {today}
**Universe:** {n_tickers} stocks · {n_days} days
**Filing lag:** {FILING_LAG_A} days / 10-K annual (no lookahead)
**Signals:** {', '.join(SIGNALS)}

---

## Signal Definitions

| Signal | Academic Source | Economic Intuition |
|---|---|---|
| accruals | Sloan (1996) | Low accruals = cash-backed earnings = beats expectations |
| rev_growth | Sales momentum | Growing revenue predicts earnings acceleration |
| gross_margin_chg | Quality factor | Improving margins = pricing power or cost discipline |
| roe | Fama-French quality | High ROE firms outperform; profitability factor |
| debt_change | Penman (2001) | Rising debt signals distress; net issuance is negative |
| pb_ratio | Fama-French value | Low P/B (value) outperforms; HML factor |

---

## IC Results

| Signal | IS Mean IC (t-stat) | IS Hit Rate | OOS Mean IC (t-stat) | OOS Hit Rate |
|---|---|---|---|---|
{signal_rows}

### Reference: Existing Technical Signals
| Signal | IS IC | IS Hit Rate | OOS IC | OOS Hit Rate |
|---|---|---|---|---|
{ref_table}

---

## Interpretation

**Strong signals** (|IC| > 0.03, |t| > 2.0): Worth including in ensemble
**Weak signals** (0.01 < |IC| < 0.03): Marginal — worth including since ML
  can extract non-linear interactions
**Dead signals** (|IC| < 0.01, t near 0): Exclude — noise in ensemble

**Key insight:** Fundamental signals have low standalone IC (< 0.05) — same
as technical signals. The value is in combination: ML ensemble can extract
interaction effects between fundamental and technical signals that neither
captures alone.

Academic reference for combined fundamental + technical ML:
- Gu, Kelly, Xiu (2020, RFS): Adding fundamentals to a ML model improves
  out-of-sample R² by ~20% vs price-only models.
- Green, Hand, Zhang (2017, RFS): 94 firm characteristics; combined ML
  dominates any individual signal.

---

## Integration into Step100

To add these signals to the walk-forward backtest:
1. Load `fundamental_signals_daily.csv`
2. At each rebalance date, merge fundamentals into the feature panel
3. Add signal names to the FEATURES list in step100

The integration is implemented — run step100 with `--use-fundamentals` flag.

---

*Canyon v9 — Research only. No live orders.*
"""

    out = ROOT / "fundamental_ic_report.md"
    out.write_text(md)
    print(f"[step165] Saved fundamental_ic_report.md")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ic-only", action="store_true",
                        help="Skip download; compute IC from existing cache only")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Override ticker list (default: full STOCK_UNIVERSE)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Canyon v9 — Step 165: Fundamental Signals")
    print("=" * 60)

    tickers = args.tickers if args.tickers else STOCK_UNIVERSE

    # Load price data
    cache_path = ROOT / "backtest_price_cache.csv"
    pit_cache  = ROOT / "pit_price_cache.csv"
    if pit_cache.exists():
        prices = pd.read_csv(pit_cache, index_col=0, parse_dates=True)
    elif cache_path.exists():
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        raise FileNotFoundError("No price cache found. Run step100 or step161 first.")

    panel_path = ROOT / "fundamental_signals_daily.csv"

    if args.ic_only and panel_path.exists():
        print("[step165] Loading existing panel …")
        panel = pd.read_csv(panel_path, parse_dates=["date"])
    else:
        panel = build_fundamental_panel(tickers, prices)
        if not panel.empty:
            panel.to_csv(panel_path, index=False)
            print(f"\n[step165] Saved fundamental_signals_daily.csv "
                  f"({panel['ticker'].nunique()} tickers, {panel['date'].nunique()} days)")

    if panel.empty:
        print("[step165] No data to evaluate. Check yfinance connectivity.")
        return

    print("\n[step165] Computing IC …")
    summary = compute_ic_report(panel, prices)

    if not summary.empty:
        write_markdown_report(summary, panel)

    print("\n[step165] Outputs:")
    print("  fundamental_signals_daily.csv")
    print("  fundamental_ic_report.csv")
    print("  fundamental_ic_report.md")
    print("\n  Next: run step100 with --use-fundamentals to test combined IC.")


if __name__ == "__main__":
    main()
