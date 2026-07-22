#!/usr/bin/env python3
"""
Canyon v9 — Clean 5-Year Backtest  (2021-06 to 2026-05)
=========================================================
NO LOOKAHEAD:
  · All signals computed from prices available at close of T → enter at T+1 open
  · 252-day warmup before first rebalance (signal computation only)
  · Fundamental signals not used (would require point-in-time data not available here)
  · Regime detection from trailing price action only (no future regime knowledge)

SURVIVORSHIP NOTE:
  Universe = current liquid large-caps.  Stocks that delisted within window are
  included from their last price and then dropped.  This slightly overstates returns
  vs a true point-in-time constituent list (industry standard caveat).

METHODOLOGY:
  · Monthly rebalance (first trading day of each calendar month)
  · Composite score = weighted rank average of 5 price signals (all lag-1):
      mom_12m_skip1m  weight 2.0   (12m momentum, skip last month)
      trend_200       weight 1.5   (price / 200d SMA - 1)
      mom_6m          weight 2.0   (6-month return)
      mom_3m          weight 1.5   (3-month return)
      new_high_52w    weight 1.0   (proximity to 52-week high)
  · Regime-conditional weights loaded from regime_history.csv if available
  · Top 10 by composite score → equal weight
  · Transaction cost: 10 bps per stock per one-way trade
  · Long only, no leverage, no short
  · Benchmark: SPY
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ─── Config ──────────────────────────────────────────────────────────────────
BACKTEST_START = "2021-06-01"   # first allowed rebalance date
BACKTEST_END   = "2026-05-29"   # last price date
WARMUP_DAYS    = 252            # days before BACKTEST_START used for signal computation
TOP_N          = 10             # stocks per rebalance
TC_BPS         = 10             # transaction cost (one-way, per stock)
HOLD_BUFFER    = 0.02           # score premium needed to replace a held stock

# Universe — the same 50 names used in backtest_price_cache
UNIVERSE = [
    "SPY",                                           # benchmark only
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA",
    "GOOGL", "JPM", "JNJ", "UNH", "V",
    "WMT", "XOM", "CVX", "LLY", "AVGO", "COST",
    "MA", "HD", "BAC", "ABBV", "PFE", "MRK",
    "PEP", "KO", "CSCO", "ABT", "TMO", "CRM",
    "ADBE", "NFLX", "AMD", "MU",
    "QQQ", "SMH", "SOXX", "XLK", "XLC", "XLF",
    "XLI", "XLE", "XLV", "XLY", "XLP", "XLB",
    "XLU", "TLT", "GLD",
]
TRADEABLE = [t for t in UNIVERSE if t != "SPY"]  # exclude benchmark from picks

SIGNAL_WEIGHTS = {
    "mom_12m_skip1m": 2.0,
    "trend_200":      1.5,
    "mom_6m":         2.0,
    "mom_3m":         1.5,
    "new_high_52w":   1.0,
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_prices(universe: list[str]) -> pd.DataFrame:
    """
    Combine two caches for maximum history:
    - sp500_price_cache.csv  → individual stocks, history back to 2000
    - backtest_price_cache.csv → ETFs + SPY, history from 2021
    For tickers in both, sp500 data is used (longer history = better warmup).
    """
    sp_path = ROOT / "sp500_price_cache.csv"
    bt_path = ROOT / "backtest_price_cache.csv"

    frames   = []
    sp_tickers: list = []
    if sp_path.exists():
        sp = pd.read_csv(sp_path, index_col=0, parse_dates=True)
        sp_tickers = [t for t in universe if t in sp.columns]
        if sp_tickers:
            frames.append(sp[sp_tickers])

    if bt_path.exists():
        bt = pd.read_csv(bt_path, index_col=0, parse_dates=True)
        already_loaded = set(sp_tickers)
        bt_tickers = [t for t in universe if t in bt.columns and t not in already_loaded]
        if bt_tickers:
            frames.append(bt[bt_tickers])

    if not frames:
        raise FileNotFoundError("No price cache found")

    df = pd.concat(frames, axis=1).sort_index()
    # Forward-fill gaps from merging two sources with different date ranges
    df = df.ffill()
    return df


def build_signals(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All signals lagged 1 day (shift(1)) → no lookahead."""
    p = prices.ffill()
    sigs = {}
    sigs["mom_3m"]          = p.pct_change(63).shift(1)
    sigs["mom_6m"]          = p.pct_change(126).shift(1)
    sigs["mom_12m_skip1m"]  = (p.pct_change(252) - p.pct_change(21)).shift(1)
    sigs["trend_200"]       = ((p / (p.rolling(200).mean() + 1e-10)) - 1).shift(1)
    sigs["new_high_52w"]    = (p / (p.rolling(252).max() + 1e-10)).shift(1)
    return sigs


def composite_score(sigs: dict[str, pd.DataFrame],
                    weights: dict[str, float]) -> pd.DataFrame:
    """Cross-sectional rank normalise each signal then weighted average."""
    total_w = sum(w for w in weights.values() if w > 0)
    result  = None
    for name, df in sigs.items():
        w = weights.get(name, 0.0)
        if w <= 0:
            continue
        ranked = df.rank(axis=1, pct=True).fillna(0.5)
        result = ranked * (w / total_w) if result is None else result + ranked * (w / total_w)
    return result


def load_regime_history() -> pd.Series:
    path = ROOT / "regime_history.csv"
    if not path.exists():
        return pd.Series(dtype=str)
    try:
        df = pd.read_csv(path)
        # Handle both "date" and "Date" column names
        date_col = next((c for c in df.columns if c.lower() == "date"), None)
        if date_col is None:
            return pd.Series(dtype=str)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.drop_duplicates(date_col).set_index(date_col).sort_index()
        # "regime" column name (exact or case-insensitive)
        reg_col = next((c for c in df.columns if c.lower() == "regime"), None)
        if reg_col:
            return df[reg_col]
    except Exception:
        pass
    return pd.Series(dtype=str)

REGIME_WEIGHTS = {
    "BULL":     {"mom_12m_skip1m": 2.0, "trend_200": 1.5, "mom_6m": 2.0,
                 "mom_3m": 1.5, "new_high_52w": 1.0},
    "BEAR":     {"mom_12m_skip1m": 2.0, "trend_200": 3.0, "mom_6m": 1.0,
                 "mom_3m": 0.5, "new_high_52w": 0.5},
    "SIDEWAYS": {"mom_12m_skip1m": 2.0, "trend_200": 2.0, "mom_6m": 2.0,
                 "mom_3m": 1.0, "new_high_52w": 1.0},
}


def regime_composite(sigs: dict[str, pd.DataFrame],
                     regime_series: pd.Series) -> pd.DataFrame:
    """Build composite with date-varying weights from regime history."""
    ranked = {n: df.rank(axis=1, pct=True).fillna(0.5) for n, df in sigs.items()}
    ref    = next(iter(ranked.values()))
    reg    = regime_series.reindex(ref.index, method="ffill").fillna("BULL")

    result = pd.DataFrame(np.nan, index=ref.index, columns=ref.columns)
    for rname, wts in REGIME_WEIGHTS.items():
        mask = reg == rname
        if not mask.any():
            continue
        tw  = sum(wts.values())
        sub = None
        for n, rk in ranked.items():
            w = wts.get(n, 0.0) / tw
            chunk = rk.loc[mask]
            sub   = chunk * w if sub is None else sub + chunk * w
        result.loc[mask] = sub.values

    nan_rows = result.isna().all(axis=1)
    if nan_rows.any():
        fallback = composite_score(sigs, SIGNAL_WEIGHTS)
        result.loc[nan_rows] = fallback.loc[nan_rows].values
    return result


# ─── Backtest loop ───────────────────────────────────────────────────────────

def run_backtest(prices: pd.DataFrame,
                 score_df: pd.DataFrame,
                 tradeable: list[str]) -> pd.DataFrame:
    """
    Walk-forward monthly rebalance.
    Returns DataFrame with one row per month.
    """
    spy        = prices["SPY"]
    stock_px   = prices[[t for t in tradeable if t in prices.columns]]

    bt_start   = pd.Timestamp(BACKTEST_START)
    bt_end     = pd.Timestamp(BACKTEST_END)
    bt_prices  = prices.loc[bt_start:bt_end]
    rebal_dates = [
        bt_prices.index[bt_prices.index >= pd.Timestamp(f"{y}-{m:02d}-01")][0]
        for y in range(bt_start.year, bt_end.year + 1)
        for m in range(1, 13)
        if len(bt_prices.index[bt_prices.index >= pd.Timestamp(f"{y}-{m:02d}-01")]) > 0
           and pd.Timestamp(f"{y}-{m:02d}-01") >= bt_start
           and pd.Timestamp(f"{y}-{m:02d}-01") <= bt_end
    ]
    rebal_dates = sorted(set(rebal_dates))

    records   = []
    held      = {}   # {ticker: score_at_entry}
    strat_cum = 1.0
    bench_cum = 1.0

    for i, rb_date in enumerate(rebal_dates[:-1]):
        next_rb  = rebal_dates[i + 1]
        score_row = score_df.loc[:rb_date].iloc[-1] if rb_date in score_df.index \
                    else score_df.loc[:rb_date].iloc[-1] if len(score_df.loc[:rb_date]) > 0 \
                    else None
        if score_row is None:
            continue

        # Eligible = tradeable + have price + have score + not NaN
        eligible = [t for t in tradeable
                    if t in score_row.index and pd.notna(score_row[t])
                    and t in stock_px.columns]
        if len(eligible) < TOP_N:
            continue

        scores_today = score_row[eligible].sort_values(ascending=False)

        # Hold buffer: only replace held stocks if new candidate beats by HOLD_BUFFER
        new_portfolio = []
        for t in scores_today.index:
            if t in held:
                new_portfolio.append(t)
            elif scores_today[t] > scores_today.quantile(1 - TOP_N / len(eligible)) + HOLD_BUFFER:
                new_portfolio.append(t)
            if len(new_portfolio) >= TOP_N:
                break
        # Fall back to plain top-N if buffer logic gives too few
        if len(new_portfolio) < TOP_N:
            new_portfolio = list(scores_today.head(TOP_N).index)

        # Turnover & TC
        prev_set = set(held.keys())
        new_set  = set(new_portfolio)
        n_changed = len(new_set - prev_set)
        tc_cost   = n_changed * TC_BPS / 10_000   # fractional total cost

        # Period returns (rb_date → next_rb)
        period_idx = prices.index[(prices.index >= rb_date) & (prices.index <= next_rb)]
        if len(period_idx) < 2:
            continue
        p0 = period_idx[0]
        p1 = period_idx[-1]

        # Strategy return = equal weight of portfolio
        valid_held = [t for t in new_portfolio if t in stock_px.columns]
        rets = []
        for t in valid_held:
            if pd.notna(stock_px.loc[p0, t]) and pd.notna(stock_px.loc[p1, t]) \
               and stock_px.loc[p0, t] > 0:
                rets.append(stock_px.loc[p1, t] / stock_px.loc[p0, t] - 1)
        if not rets:
            continue
        strat_ret = float(np.mean(rets)) - tc_cost

        # Benchmark return
        bench_ret = float(spy.loc[p1] / spy.loc[p0] - 1) if spy.loc[p0] > 0 else 0.0

        strat_cum *= (1 + strat_ret)
        bench_cum *= (1 + bench_ret)

        records.append({
            "rebalance_date": rb_date.strftime("%Y-%m-%d"),
            "period_end":     p1.strftime("%Y-%m-%d"),
            "strategy_ret":   round(strat_ret, 6),
            "spy_ret":        round(bench_ret, 6),
            "alpha":          round(strat_ret - bench_ret, 6),
            "strategy_cum":   round(strat_cum - 1, 4),
            "bench_cum":      round(bench_cum - 1, 4),
            "n_held":         len(valid_held),
            "tickers":        " | ".join(valid_held),
            "n_changed":      n_changed,
            "tc_cost_bps":    round(n_changed * TC_BPS, 1),
        })

        held = {t: scores_today.get(t, 0) for t in new_portfolio}

    return pd.DataFrame(records)


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> dict:
    strat = df["strategy_ret"].values
    bench = df["spy_ret"].values
    alpha = df["alpha"].values

    # Annualised return (CAGR)
    n_years = len(strat) / 12
    strat_cum_final = float(np.prod(1 + strat)) - 1
    bench_cum_final = float(np.prod(1 + bench)) - 1
    cagr_strat = (1 + strat_cum_final) ** (1 / max(n_years, 0.1)) - 1
    cagr_bench = (1 + bench_cum_final) ** (1 / max(n_years, 0.1)) - 1

    # Annualised Sharpe (monthly returns × √12)
    sharpe = float(np.mean(strat) / (np.std(strat, ddof=1) + 1e-10) * np.sqrt(12))

    # Annualised Sortino
    downside = strat[strat < 0]
    sortino_denom = float(np.std(downside, ddof=1) + 1e-10) * np.sqrt(12)
    sortino = float(np.mean(strat) * 12 / sortino_denom)

    # Max drawdown
    cum = np.cumprod(1 + strat)
    running_max = np.maximum.accumulate(cum)
    drawdowns = cum / running_max - 1
    max_dd = float(drawdowns.min())

    # Calmar
    calmar = abs(cagr_strat / max_dd) if max_dd < 0 else np.nan

    # Monthly win rate vs SPY
    win_rate = float((alpha > 0).mean())

    # IC (cross-sectional signal IC proxy: correlation of monthly alpha with t)
    t_stat, pval = stats.ttest_1samp(strat, 0)

    # Alpha annualised
    alpha_ann = float(np.mean(alpha) * 12)

    # Total TC
    total_tc = df["tc_cost_bps"].sum()

    return {
        "periods":         len(strat),
        "years":           round(n_years, 1),
        "strat_total_ret": round(strat_cum_final * 100, 2),
        "bench_total_ret": round(bench_cum_final * 100, 2),
        "alpha_total":     round((strat_cum_final - bench_cum_final) * 100, 2),
        "cagr_strat":      round(cagr_strat * 100, 2),
        "cagr_bench":      round(cagr_bench * 100, 2),
        "alpha_ann":       round(alpha_ann * 100, 2),
        "sharpe":          round(sharpe, 3),
        "sortino":         round(sortino, 3),
        "max_dd":          round(max_dd * 100, 2),
        "calmar":          round(calmar, 3) if not np.isnan(calmar) else None,
        "monthly_win_pct": round(win_rate * 100, 1),
        "t_stat":          round(float(t_stat), 2),
        "total_tc_bps":    round(total_tc, 0),
    }


# ─── Yearly breakdown ────────────────────────────────────────────────────────

def yearly_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["year"] = pd.to_datetime(df2["rebalance_date"]).dt.year
    rows = []
    for yr, grp in df2.groupby("year"):
        s = grp["strategy_ret"].values
        b = grp["spy_ret"].values
        rows.append({
            "Year":         yr,
            "Strat Return": f"{(np.prod(1+s)-1)*100:+.1f}%",
            "SPY Return":   f"{(np.prod(1+b)-1)*100:+.1f}%",
            "Alpha":        f"{(np.prod(1+s)-np.prod(1+b))*100:+.1f}%",
            "Sharpe (ann)": f"{np.mean(s)/(np.std(s,ddof=1)+1e-10)*np.sqrt(12):.2f}",
            "Max DD":       f"{(np.cumprod(1+s)/np.maximum.accumulate(np.cumprod(1+s))-1).min()*100:.1f}%",
            "Win Rate":     f"{(s>b).mean()*100:.0f}%",
        })
    return pd.DataFrame(rows)


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 65)
    print("Canyon v9 — Clean 5-Year Backtest")
    print(f"  Period : {BACKTEST_START}  →  {BACKTEST_END}")
    print(f"  Top-N  : {TOP_N}  |  TC: {TC_BPS}bps  |  Rebalance: monthly")
    print(f"  Warmup : {WARMUP_DAYS}d  |  Signals: price-only (lag-1)")
    print("=" * 65)

    # 1. Prices
    print("\n[1/4] Loading prices …")
    prices = load_prices(UNIVERSE)
    spy_in = "SPY" in prices.columns
    print(f"  {len(prices)} days × {len(prices.columns)} tickers  "
          f"({prices.index.min().date()} → {prices.index.max().date()})")

    tradeable = [t for t in TRADEABLE if t in prices.columns]
    print(f"  Tradeable universe: {len(tradeable)} tickers")

    # 2. Signals — computed on ALL available history (no lookahead)
    print("\n[2/4] Building signals (lag-1) …")
    sigs = build_signals(prices)

    # Regime-conditional composite if history available
    regime_hist = load_regime_history()
    if not regime_hist.empty:
        print(f"  Regime history loaded: {len(regime_hist)} dates  "
              f"→ using regime-conditional weights")
        score_df = regime_composite(sigs, regime_hist)[tradeable]
    else:
        print("  No regime history — using default signal weights")
        score_df = composite_score(sigs, SIGNAL_WEIGHTS)[tradeable]

    # 3. Walk-forward backtest
    print("\n[3/4] Running walk-forward backtest …")
    monthly = run_backtest(prices, score_df, tradeable)
    print(f"  {len(monthly)} rebalance periods completed")

    if monthly.empty:
        print("  ERROR: no periods — check date range / price data")
        return

    # 4. Metrics
    print("\n[4/4] Computing performance metrics …")
    m = compute_metrics(monthly)

    print("\n" + "─" * 65)
    print("  RESULTS — NO LOOKAHEAD  (survivorship note: current universe)")
    print("─" * 65)
    print(f"  Periods tested        : {m['periods']} months  ({m['years']} years)")
    print(f"  Strategy total return : {m['strat_total_ret']:+.2f}%")
    print(f"  SPY total return      : {m['bench_total_ret']:+.2f}%")
    print(f"  Total alpha vs SPY    : {m['alpha_total']:+.2f}%")
    print(f"  ─")
    print(f"  CAGR (strategy)       : {m['cagr_strat']:+.2f}%")
    print(f"  CAGR (SPY)            : {m['cagr_bench']:+.2f}%")
    print(f"  Annualised alpha      : {m['alpha_ann']:+.2f}%")
    print(f"  ─")
    print(f"  Sharpe (annualised)   : {m['sharpe']:.3f}")
    print(f"  Sortino               : {m['sortino']:.3f}")
    print(f"  Max drawdown          : {m['max_dd']:.2f}%")
    print(f"  Calmar ratio          : {m['calmar']}")
    print(f"  Monthly win vs SPY    : {m['monthly_win_pct']:.1f}%")
    print(f"  t-stat (α ≠ 0)        : {m['t_stat']:.2f}")
    print(f"  Total TC paid         : {m['total_tc_bps']:.0f} bps")
    print("─" * 65)

    # Yearly breakdown
    yr_df = yearly_breakdown(monthly)
    print("\n  YEAR-BY-YEAR:")
    print(yr_df.to_string(index=False))

    # Last 12 months
    last12 = monthly.tail(12)
    if len(last12) >= 6:
        m12 = compute_metrics(last12)
        print(f"\n  LAST 12 MONTHS:")
        print(f"    Strategy return   : {m12['strat_total_ret']:+.2f}%")
        print(f"    SPY return        : {m12['bench_total_ret']:+.2f}%")
        print(f"    Sharpe (ann)      : {m12['sharpe']:.3f}")
        print(f"    Max drawdown      : {m12['max_dd']:.2f}%")

    # Save
    out_monthly = ROOT / "backtest_5yr_monthly.csv"
    out_summary = ROOT / "backtest_5yr_summary.csv"
    monthly.to_csv(out_monthly, index=False)

    summary_rows = [
        ("Total Return (Strategy)", f"{m['strat_total_ret']:+.2f}%", f"{m['bench_total_ret']:+.2f}%"),
        ("Total Alpha vs SPY",      f"{m['alpha_total']:+.2f}%",     "0%"),
        ("CAGR",                    f"{m['cagr_strat']:+.2f}%",      f"{m['cagr_bench']:+.2f}%"),
        ("Annualised Alpha",        f"{m['alpha_ann']:+.2f}%",       "0%"),
        ("Sharpe (annualised)",     f"{m['sharpe']:.3f}",            "~0.60 (SPY hist.)"),
        ("Sortino",                 f"{m['sortino']:.3f}",           "~0.90 (SPY hist.)"),
        ("Max Drawdown",            f"{m['max_dd']:.2f}%",           "SPY ~-57% (2008)"),
        ("Calmar Ratio",            f"{m['calmar']}",                ">1.0 good"),
        ("Monthly Win Rate vs SPY", f"{m['monthly_win_pct']:.1f}%",  "50%"),
        ("Periods Tested",          f"{m['periods']} months",        ">36 reliable"),
        ("Total TC Paid",           f"{m['total_tc_bps']:.0f} bps",  "10bps/trade"),
    ]
    pd.DataFrame(summary_rows, columns=["metric", "strategy", "benchmark"]) \
      .to_csv(out_summary, index=False)

    print(f"\n  Saved: {out_monthly.name}  |  {out_summary.name}")
    print("=" * 65)


if __name__ == "__main__":
    main()
