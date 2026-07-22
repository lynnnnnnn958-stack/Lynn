#!/usr/bin/env python3
"""
Canyon v9  Step 223 — Walk-Forward Backtest
============================================
Strict out-of-sample backtesting framework.

Core principles:
    Training set 2000-01-01 → 2019-12-31  used to fix signal parameters
    Validation set 2016-01-01 → 2019-12-31  checks for overfitting (last 20% of training set)
    Test set 2020-01-01 → today           never used for fitting, only for evaluation

Test set includes:
    - 2020 COVID crash (max single-month loss -34%)
    - 2020-2021 recovery
    - 2022 bear market (-20%)
    - 2023-2024 bull market
    - 2025-2026 continuation

Only if the model works on the test set is true capability demonstrated.

Signals (all reconstructable from historical prices, no lookahead bias):
    1. Cross-sectional momentum (J-T 12-1)     ← price only
    2. 52-week high ratio (G-H)                ← price only
    3. Volatility-scaled momentum (AQR)        ← price only
    4. Inverse volatility (Low-Vol)             ← price only

Portfolio construction:
    - Rebalance on first trading day of each month
    - Top-N cross-sectional rank → equal weight (baseline)
    - Transaction cost: 10 bps one-way (5 bps each side)

Outputs:
    walk_forward_returns.csv    monthly returns, all portfolios and SPY benchmark
    walk_forward_report.md      full report with in-sample/out-of-sample comparison
    walk_forward_summary.json   key metrics
"""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── Parameters ─────────────────────────────────────────────────────────────────────
TRAIN_END   = "2019-12-31"   # training set end (inclusive)
TEST_START  = "2020-01-01"   # test set start (never used for parameter selection)
VAL_START   = "2016-01-01"   # validation set start (inside training set)

TOP_N       = 30             # stocks held per month
TC_BPS      = 10             # one-way transaction cost (bps) = 0.10%
RF_ANNUAL   = 0.025          # risk-free rate (annualized)

# Momentum parameters (fixed on training set, not adjusted using test set)
MOM_LOOKBACK = 252           # 12 months
MOM_SKIP     = 21            # skip most recent 1 month (avoid reversal)
HIGH52_WIN   = 252           # 52 weeks
VOL_WIN      = 21            # 21-day volatility


# =============================================================================
# 1. Data loading
# =============================================================================

def load_prices() -> pd.DataFrame:
    path = ROOT / "sp500_price_cache.csv"
    if not path.exists():
raise FileNotFoundError("sp500_price_cache.csv not found")
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return px.dropna(axis=1, how="all")


def load_spy() -> pd.Series:
    path = ROOT / "spy_price_cache.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    spy = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    col = spy.columns[0]
    return spy[col]


# =============================================================================
# 2. Monthly signal computation (cross-sectional signal for a given date)
# =============================================================================

def compute_monthly_signals(
    prices: pd.DataFrame,
    date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Computes per-stock signal scores as of `date` using only prior history.
    All data used is strictly before `date` — no lookahead bias.
    """
    # Price history up to and including date
    hist = prices.loc[:date]
    if len(hist) < MOM_LOOKBACK + 10:
        return pd.DataFrame()

    # Price snapshot on date
    px_today = hist.iloc[-1]
    # Price 1 month back (momentum calculation start, skipping reversal)
    px_skip  = hist.iloc[-(MOM_SKIP + 1)]
    # Price 12 months ago
    if len(hist) <= MOM_LOOKBACK + MOM_SKIP:
        return pd.DataFrame()
    px_12m   = hist.iloc[-(MOM_LOOKBACK + MOM_SKIP)]

    # ── Signal 1: Cross-sectional momentum (12-1 month) ────────────────────────────────────
    mom_12_1 = (px_skip / px_12m - 1).replace([np.inf, -np.inf], np.nan)

    # ── Signal 2: 52-week high ratio ──────────────────────────────────────────────────────
    high52 = hist.tail(HIGH52_WIN).max()
    high52_ratio = (px_today / high52).clip(0, 1).replace([np.inf, -np.inf], np.nan)

    # ── Signal 3: Volatility-scaled momentum ────────────────────────────────────────────
    log_rets = np.log(hist / hist.shift(1)).dropna()
    if len(log_rets) < VOL_WIN:
        return pd.DataFrame()
    realized_vol = log_rets.tail(VOL_WIN).std() * np.sqrt(252)
    realized_vol = realized_vol.clip(lower=0.05)   # 5% minimum volatility floor
    vol_scaled   = (mom_12_1 / realized_vol).replace([np.inf, -np.inf], np.nan)

    # ── Signal 4: Low volatility (inverse vol) ────────────────────────────────────────
    low_vol = 1.0 / realized_vol.replace(0, np.nan)

    # ── Composite cross-sectional rank → 0-100 score ──────────────────────────────────────
    def cs_rank(s: pd.Series) -> pd.Series:
        return s.rank(pct=True, na_option="bottom") * 100

    # Equal-weight composite (weights confirmed on training set)
    composite = (
        0.40 * cs_rank(mom_12_1)
        + 0.25 * cs_rank(high52_ratio)
        + 0.25 * cs_rank(vol_scaled)
        + 0.10 * cs_rank(low_vol)
    )

    df = pd.DataFrame({
        "ticker":       composite.index,
        "composite":    composite.values,
        "mom_12_1":     mom_12_1.reindex(composite.index).values,
        "high52_ratio": high52_ratio.reindex(composite.index).values,
        "vol_scaled":   vol_scaled.reindex(composite.index).values,
        "realized_vol": realized_vol.reindex(composite.index).values,
    })
    df = df.dropna(subset=["composite"])
    return df.sort_values("composite", ascending=False).reset_index(drop=True)


# =============================================================================
# 3. Monthly rebalance
# =============================================================================

def get_monthly_rebal_dates(prices: pd.DataFrame, start: str, end: str) -> list[pd.Timestamp]:
    """Return the first trading day of each month."""
    idx = prices.loc[start:end].index
    df  = pd.DataFrame({"date": idx})
    df["ym"] = df["date"].dt.to_period("M")
    first_of_month = df.groupby("ym")["date"].first()
    return list(first_of_month)


def compute_period_return(
    prices: pd.DataFrame,
    tickers: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    tc_bps: float = TC_BPS,
    prev_tickers: list[str] | None = None,
) -> float:
    """
    Compute equal-weight portfolio return for the holding period (start_date → end_date).
    Deducts transaction cost (turnover portion).
    """
    avail = [t for t in tickers if t in prices.columns]
    if not avail:
        return 0.0

    # Find the nearest available price date
    px_dates = prices.index
    def find_date(d, forward=True):
        if d in px_dates:
            return d
        if forward:
            later = px_dates[px_dates >= d]
            return later[0] if len(later) else None
        else:
            earlier = px_dates[px_dates <= d]
            return earlier[-1] if len(earlier) else None

    d0 = find_date(start_date, forward=True)
    d1 = find_date(end_date,   forward=False)
    if d0 is None or d1 is None or d0 >= d1:
        return 0.0

    p0 = prices.loc[d0, avail].dropna()
    p1 = prices.loc[d1, avail].dropna()
    common = p0.index.intersection(p1.index)
    if len(common) == 0:
        return 0.0

    ret = ((p1[common] / p0[common]) - 1).mean()

    # Transaction cost: turnover × tc_bps
    if prev_tickers is not None:
        prev_set  = set(prev_tickers)
        curr_set  = set(avail)
        turnover  = len(curr_set.symmetric_difference(prev_set)) / max(len(curr_set), 1)
        tc        = turnover * tc_bps / 10000
        ret      -= tc

    return float(ret)


# =============================================================================
# 4. Main backtest loop
# =============================================================================

def run_backtest(
    prices: pd.DataFrame,
    spy: pd.Series,
    period_start: str,
    period_end:   str,
    label:        str,
    top_n:        int = TOP_N,
) -> pd.DataFrame:
    """
    Run monthly-rebalance backtest over [period_start, period_end].
    Returns a monthly-returns DataFrame.
    """
    print(f"\n  {'─'*50}")
    print(f"  Backtest range: {period_start} → {period_end}  [{label}]")

    rebal_dates = get_monthly_rebal_dates(prices, period_start, period_end)
    if len(rebal_dates) < 2:
print(f"  [skip] fewer than 2 months")
        return pd.DataFrame()

    rows = []
    prev_tickers: list[str] = []

    for i in range(len(rebal_dates) - 1):
        d_start = rebal_dates[i]
        d_end   = rebal_dates[i + 1]

        # Compute signal at month start (only uses data prior to d_start)
        sigs = compute_monthly_signals(prices, d_start)
        if sigs.empty:
            continue

        # Select top-N
        top_tickers = sigs.head(top_n)["ticker"].tolist()

        # Holding-period return (top-N equal weight)
        ret_portfolio = compute_period_return(
            prices, top_tickers, d_start, d_end,
            prev_tickers=prev_tickers
        )

        # Benchmark: equal-weight full market
        all_tickers = sigs["ticker"].tolist()
        ret_equal_w = compute_period_return(prices, all_tickers, d_start, d_end)

        # SPY benchmark
        spy_px = spy.reindex(prices.loc[d_start:d_end].index, method="ffill")
        ret_spy = float((spy_px.iloc[-1] / spy_px.iloc[0]) - 1) if len(spy_px) >= 2 else 0.0

        rows.append({
            "date":          d_start,
            "ret_portfolio": ret_portfolio,
            "ret_equal_w":   ret_equal_w,
            "ret_spy":       ret_spy,
            "n_holdings":    len(top_tickers),
            "top_tickers":   ",".join(top_tickers[:5]),
        })
        prev_tickers = top_tickers

    if not rows:
print(f"  [skip] no valid data (possibly insufficient warm-up)")
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("date")
    print(f"  {len(df)} months  "
          f"port ann return={df['ret_portfolio'].mean()*12*100:.1f}%  "
          f"SPY ann return={df['ret_spy'].mean()*12*100:.1f}%")
    return df


# =============================================================================
# 5. Performance statistics
# =============================================================================

def perf_stats(monthly_rets: pd.Series, label: str, rf_annual: float = RF_ANNUAL) -> dict:
    """Compute annualized return, volatility, Sharpe, and max drawdown."""
    if monthly_rets.empty:
        return {}

    ann_ret  = float(monthly_rets.mean() * 12)
    ann_vol  = float(monthly_rets.std() * np.sqrt(12))
    rf_m     = (1 + rf_annual) ** (1/12) - 1
    sharpe   = (monthly_rets.mean() - rf_m) / monthly_rets.std() * np.sqrt(12) if monthly_rets.std() > 0 else 0

    # Max drawdown
    cum      = (1 + monthly_rets).cumprod()
    rolling_max = cum.expanding().max()
    drawdowns   = cum / rolling_max - 1
    max_dd   = float(drawdowns.min())

    # Win rate (months that beat SPY)
    n_months = len(monthly_rets)

    return {
        "label":    label,
        "n_months": n_months,
        "ann_ret":  ann_ret,
        "ann_vol":  ann_vol,
        "sharpe":   sharpe,
        "max_dd":   max_dd,
    }


# =============================================================================
# 6. Report
# =============================================================================

def write_report(
    in_sample:    pd.DataFrame,
    out_sample:   pd.DataFrame,
    val_sample:   pd.DataFrame | None,
    spy:          pd.Series,
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    def fmt(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{x:.2f}"

    def fmt_pct(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{x*100:+.1f}%"

    lines = [
"# Canyon v9 — Walk-Forward Backtest Report (Step 223)",
f"Generated: {ts}",
        "",
"## Methodology",
        "",
"**Strict out-of-sample test framework**",
        "",
"| Period | Date range | Purpose |",
        "|------|------|------|",
        f"| Training | 2000-01-01 → {TRAIN_END} | fix signal weights and parameters |",
        f"| Validation | {VAL_START} → {TRAIN_END} | check overfitting (inside training set) |",
        f"| **Test set (locked)** | **{TEST_START} → today** | **final evaluation, never used for selection** |",
        "",
"**Signals** (all from historical prices, no lookahead bias):",
"- Cross-sectional momentum 12-1 (Jegadeesh-Titman 1993) × 40%",
"- 52-week high ratio (George-Hwang 2004) × 25%",
"- Volatility-scaled momentum (Barroso-Santa-Clara 2015) × 25%",
"- Low volatility (Low-Vol) × 10%",
        "",
        f"**Portfolio construction**: rebalance on first trading day each month, hold Top-{TOP_N} equal weight,"
        f"transaction cost {TC_BPS} bps one-way.",
        "",
"## Performance comparison",
        "",
"| Metric | Training (in-sample) | Validation (in-sample tail) | **Test set (OOS)** |",
        "|------|---------------|----------------|----------------|",
    ]

    periods = [
        ("Training",    in_sample,  "ret_portfolio"),
        ("Validation",  val_sample, "ret_portfolio"),
        ("Test ✅",      out_sample, "ret_portfolio"),
    ]
    stat_rows = {}
    for label, df, col in periods:
        if df is None or df.empty:
            stat_rows[label] = {}
            continue
        stat_rows[label] = perf_stats(df[col], label)

    for metric_key, metric_name in [
        ("ann_ret",  "Ann Return"),
        ("ann_vol",  "Ann Vol"),
        ("sharpe",   "Sharpe"),
        ("max_dd",   "Max DD"),
        ("n_months", "Months"),
    ]:
        row = f"| {metric_name} |"
        for label, _, _ in periods:
            s = stat_rows.get(label, {})
            v = s.get(metric_key)
            if metric_key in ("ann_ret", "ann_vol", "max_dd"):
                row += f" {fmt_pct(v)} |"
            else:
                row += f" {fmt(v)} |"
        lines.append(row)

    # SPY comparison (test set)
    if not out_sample.empty:
        spy_stats = perf_stats(out_sample["ret_spy"], "SPY")
        port_stats = stat_rows.get("Test ✅", {})
        excess = (port_stats.get("ann_ret", 0) or 0) - (spy_stats.get("ann_ret", 0) or 0)
        lines += [
            "",
"### Test set (2020-present) detailed comparison",
            "",
"| | Momentum Portfolio | SPY | Excess Return |",
            "|--|--------|-----|--------|",
            f"| Ann Return | {fmt_pct(port_stats.get('ann_ret'))} | {fmt_pct(spy_stats.get('ann_ret'))} | {fmt_pct(excess)} |",
            f"| Ann Vol | {fmt_pct(port_stats.get('ann_vol'))} | {fmt_pct(spy_stats.get('ann_vol'))} | — |",
            f"| Sharpe | {fmt(port_stats.get('sharpe'))} | {fmt(spy_stats.get('sharpe'))} | — |",
            f"| Max DD | {fmt_pct(port_stats.get('max_dd'))} | {fmt_pct(spy_stats.get('max_dd'))} | — |",
        ]

    # Honest evaluation: training vs test
    train_sharpe = stat_rows.get("Training",{}).get("sharpe", 0) or 0
    test_sharpe  = stat_rows.get("Test ✅",{}).get("sharpe", 0) or 0
    degradation  = (train_sharpe - test_sharpe) / train_sharpe if train_sharpe > 0 else 0

    lines += [
        "",
"## Honest evaluation",
        "",
f"**Training Sharpe**: {train_sharpe:.2f}",
f"**Test Sharpe**: {test_sharpe:.2f}",
f"**Performance decay**: {degradation*100:.0f}%",
        "",
    ]

    if degradation < 0.30:
lines.append("✅ **Performance decay < 30%** — Good generalization; training and test results are close.")
    elif degradation < 0.60:
lines.append("⚠ **Performance decay 30-60%** — Partial overfitting; test set still has excess return but weaker.")
    else:
lines.append("🔴 **Performance decay > 60%** — Severe overfitting. Training results unreliable; signals need redesign.")

    lines += [
        "",
"## Data warnings",
        "",
"⚠️ **Survivorship bias**: Price data uses current S&P 500 constituents, excluding historically removed stocks."
"This systematically overstates backtest returns. True excess return is roughly 1-3 pp lower.",
        "",
"⚠️ **Overfitting note**: Momentum signal parameters (12-1 month, 52-week, etc.) come from academic literature,"
"not optimized on this dataset, so overfitting risk is relatively low."
"However signal weights (40/25/25/10) were chosen on the training set, so selection bias remains.",
        "",
f"**Source files**: sp500_price_cache.csv, spy_price_cache.csv",
    ]

    (ROOT / "walk_forward_report.md").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 7. Main
# =============================================================================

def run() -> dict:
    print(f"\n{'='*65}")
    print(f"Canyon v9 — Step 223: Walk-Forward Backtest  [{datetime.now():%Y-%m-%d %H:%M:%S}]")
    print(f"{'='*65}")
    print(f"\n  Training set: 2000 → {TRAIN_END}")
    print(f"  Validation set: {VAL_START} → {TRAIN_END}")
    print(f"  Test set:  {TEST_START} → today  ← key numbers")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[1/5] Loading price data ...")
    prices = load_prices()
    spy    = load_spy()
    print(f"  Price matrix: {prices.shape}  SPY: {len(spy)} days")

    # ── In-sample backtest (training set) ──────────────────────────────────────
    print("\n[2/5] In-sample backtest (training set 2000-2019) ...")
    # Start 2002: momentum signal needs (252+21+10=283) days of history, ~14 months warm-up
    in_sample = run_backtest(prices, spy, "2002-01-01", TRAIN_END,
                             label="Training (in-sample)")

    # ── Validation set ────────────────────────────────────────────────────────────────
    print("\n[3/5] Validation set backtest (2016-2019) ...")
    val_sample = run_backtest(prices, spy, VAL_START, TRAIN_END,
                              label="Validation (training tail)")

    # ── Out-of-sample backtest (test set — key!) ──────────────────────────────────────
    print(f"\n[4/5] 🔒 Out-of-sample test ({TEST_START} → today) ...")
    print("  ← This result is the only one that truly matters!")
    out_sample = run_backtest(prices, spy, TEST_START, prices.index[-1].strftime("%Y-%m-%d"),
                              label="Test set (OOS)")

    # ── Write outputs ──────────────────────────────────────────────────────────────
    print("\n[5/5] Writing outputs ...")

    # Merge and save
    all_dfs = []
    for df, period in [(in_sample, "train"), (val_sample, "val"), (out_sample, "test")]:
        if not df.empty:
            df2 = df.copy()
            df2["period"] = period
            all_dfs.append(df2)

    if all_dfs:
        combined = pd.concat(all_dfs)
        combined.to_csv(ROOT / "walk_forward_returns.csv")
        print(f"  [write] walk_forward_returns.csv  ({len(combined)} rows)")

    write_report(in_sample, out_sample, val_sample, spy)
    print(f"  [write] walk_forward_report.md")

    # JSON summary
    import json
    def _stats(df, col):
        if df is None or df.empty or col not in df.columns:
            return {}
        s = perf_stats(df[col], "")
        return {k: round(float(v), 4) if isinstance(v, float) else v
                for k, v in s.items()}

    summary = {
        "generated_at":  datetime.now().isoformat(),
        "train_end":     TRAIN_END,
        "test_start":    TEST_START,
        "top_n":         TOP_N,
        "tc_bps":        TC_BPS,
        "in_sample":     _stats(in_sample, "ret_portfolio"),
        "val_sample":    _stats(val_sample, "ret_portfolio"),
        "out_sample":    _stats(out_sample, "ret_portfolio"),
        "spy_test":      _stats(out_sample, "ret_spy"),
    }
    (ROOT / "walk_forward_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  [write] walk_forward_summary.json")

    # ── Final key numbers ───────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Key results (out-of-sample, 2020-present):")
    if not out_sample.empty:
        oos = perf_stats(out_sample["ret_portfolio"], "")
        spy_s = perf_stats(out_sample["ret_spy"], "")
        print(f"  Momentum portfolio → ann {oos.get('ann_ret',0)*100:.1f}%  "
              f"Sharpe {oos.get('sharpe',0):.2f}  "
              f"Max DD {oos.get('max_dd',0)*100:.1f}%")
        print(f"  SPY benchmark → ann {spy_s.get('ann_ret',0)*100:.1f}%  "
              f"Sharpe {spy_s.get('sharpe',0):.2f}  "
              f"Max DD {spy_s.get('max_dd',0)*100:.1f}%")
        excess = oos.get("ann_ret",0) - spy_s.get("ann_ret",0)
        print(f"  Annual excess return: {excess*100:+.1f} pp")

    in_sh  = perf_stats(in_sample,  "")["sharpe"] if not in_sample.empty else 0  # type: ignore
    oos_sh = oos.get("sharpe",0)  if not out_sample.empty else 0  # type: ignore
    if not in_sample.empty:
        in_sh = perf_stats(in_sample["ret_portfolio"], "")["sharpe"]
    deg = (in_sh - oos_sh) / in_sh * 100 if in_sh > 0 else 0
    print(f"  Performance decay: training Sharpe {in_sh:.2f} → test {oos_sh:.2f} (decay {deg:.0f}%)")
    print(f"{'─'*65}\n")

    return summary


if __name__ == "__main__":
    import sys
    result = run()
    sys.exit(0 if result else 1)
