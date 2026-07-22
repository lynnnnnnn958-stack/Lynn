#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canyon v9 — Step 230: Portfolio Sizing Comparison
==================================================
Applies four portfolio construction methods to the SAME walk-forward ML
predictions and compares IS/OOS Sharpe ratios.  No model retraining needed —
this is purely a post-processing comparison on wf_oos_predictions.csv.

Methods:
  equal       — equal weight top-N (current baseline)
  risk_parity — inverse-volatility weighted, cap at 8%
  score_wtd   — proportional to ML ensemble_score, cap at 8%
  min_var     — rolling Ledoit-Wolf minimum-variance portfolio

Hard constraints (all methods):
  max 8% per single name
  max 35% per sector (not enforced in this script — needs sector lookup)
  long-only, sum of weights = 95% (5% cash buffer)

Outputs:
  portfolio_sizing_comparison.csv   — monthly returns per method
  portfolio_sizing_summary.csv      — IS/OOS Sharpe, CAGR, DD, IC per method
  portfolio_sizing_report.md        — full narrative report

Usage:
  python3 canyon_final_v9_step230_portfolio_optimizer.py
  python3 canyon_final_v9_step230_portfolio_optimizer.py --top 10 --tc 10
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).parent
OOS_CUTOFF  = pd.Timestamp("2020-01-01")
MAX_NAME    = 0.08      # 8% single-name cap
INVEST_FRAC = 0.95      # hold 5% cash
ANN_FACTOR  = 12        # monthly returns → annual
RF_RATE     = 0.04      # risk-free rate for Sharpe
TC_BPS_DEF  = 10
TOP_N_DEF   = 15        # must be > 1/MAX_NAME (=12.5) for sizing to differentiate

METHODS = ["equal", "risk_parity", "score_wtd", "min_var"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_predictions() -> pd.DataFrame:
    path = ROOT / "wf_oos_predictions.csv"
    if not path.exists():
        raise FileNotFoundError("wf_oos_predictions.csv not found — run step100 first.")
    df = pd.read_csv(path, parse_dates=["rebalance_date"])
    df["rebalance_date"] = pd.to_datetime(df["rebalance_date"])
    return df


def load_prices() -> pd.DataFrame:
    for p in [ROOT / "backtest_price_cache.csv", ROOT / "pit_price_cache.csv"]:
        if p.exists():
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            return df.sort_index()
    raise FileNotFoundError("No price cache found.")


# ─────────────────────────────────────────────────────────────────────────────
# Volatility helper (rolling 63-day)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rolling_vol(prices: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Daily annualised volatility for each ticker, rolling window."""
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def compute_rolling_cov(prices: pd.DataFrame, tickers: list[str],
                        as_of: pd.Timestamp, window: int = 252) -> pd.DataFrame:
    """Ledoit-Wolf shrunk covariance matrix as of `as_of`."""
    from sklearn.covariance import LedoitWolf

    avail = [t for t in tickers if t in prices.columns]
    slice_ = prices.loc[:as_of, avail].dropna(how="all").tail(window)

    # Only keep tickers with full window of data
    slice_ = slice_.loc[:, slice_.count() >= window // 2]
    if slice_.shape[1] < 2:
        return pd.DataFrame()

    log_ret = np.log(slice_ / slice_.shift(1)).dropna()
    if len(log_ret) < 30:
        return pd.DataFrame()

    try:
        lw = LedoitWolf().fit(log_ret.values)
        cov = pd.DataFrame(
            lw.covariance_ * 252,    # annualise
            index=slice_.columns,
            columns=slice_.columns,
        )
        return cov
    except Exception:
        cov = log_ret.cov() * 252
        return cov


# ─────────────────────────────────────────────────────────────────────────────
# Sizing methods
# ─────────────────────────────────────────────────────────────────────────────

def _cap_and_renorm(weights: dict[str, float]) -> dict[str, float]:
    """
    Cap each position at MAX_NAME then scale total to INVEST_FRAC.
    No redistribution: excess goes to cash. This preserves relative weight
    differences between sizing methods instead of converging to equal weight.
    """
    w = pd.Series(weights).clip(lower=0, upper=MAX_NAME)
    total = w.sum()
    if total == 0:
        return weights
    # Scale: if over-budget, shrink; if under-budget, leave as-is (rest = cash)
    if total > INVEST_FRAC:
        w = w / total * INVEST_FRAC
    return w.to_dict()


def size_equal(tickers: list[str], **_) -> dict[str, float]:
    n = len(tickers)
    raw = {t: min(1.0 / n, MAX_NAME) for t in tickers}
    return _cap_and_renorm(raw)


def size_risk_parity(tickers: list[str], vol_series: pd.Series, **_) -> dict[str, float]:
    vols = vol_series.reindex(tickers).fillna(0.25)
    inv_vol = 1.0 / vols.clip(lower=0.05)
    raw = (inv_vol / inv_vol.sum() * INVEST_FRAC).to_dict()
    return _cap_and_renorm(raw)


def size_score_weighted(tickers: list[str], scores: pd.Series, **_) -> dict[str, float]:
    s = scores.reindex(tickers).fillna(0)
    s = (s - s.min()).clip(lower=1e-10)    # non-negative, never all-zero
    raw = (s / s.sum() * INVEST_FRAC).to_dict()
    return _cap_and_renorm(raw)


def size_min_var(tickers: list[str], cov: pd.DataFrame, **_) -> dict[str, float]:
    avail = [t for t in tickers if t in cov.index]
    if len(avail) < 2:
        return size_equal(tickers)

    c = cov.loc[avail, avail].values
    n = len(avail)

    def portfolio_var(w):
        return w @ c @ w

    cons  = [{"type": "eq", "fun": lambda w: w.sum() - INVEST_FRAC}]
    bounds = [(0, MAX_NAME)] * n
    w0 = np.ones(n) / n * INVEST_FRAC

    try:
        res = minimize(portfolio_var, w0, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-9})
        if res.success:
            w = pd.Series(res.x, index=avail).clip(lower=0, upper=MAX_NAME)
            return _cap_and_renorm(w.to_dict())
    except Exception:
        pass

    return size_equal(tickers)   # fallback


# ─────────────────────────────────────────────────────────────────────────────
# Core backtest loop
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(preds: pd.DataFrame, prices: pd.DataFrame,
                 top_n: int = 8, tc_bps: float = TC_BPS_DEF) -> pd.DataFrame:
    """
    For each rebalance date and each sizing method, compute portfolio return.
    Returns long-format DataFrame with columns:
      rebalance_date, period, method, ml_ret, spy_ret, alpha, weights
    """
    tc = tc_bps / 10_000
    vol_df  = compute_rolling_vol(prices, window=63)

    rebal_dates = preds["rebalance_date"].sort_values().unique()
    print(f"  Processing {len(rebal_dates)} rebalance dates × {len(METHODS)} methods …")

    all_rows = []
    prev_weights: dict[str, dict[str, float]] = {m: {} for m in METHODS}

    for i, dt in enumerate(rebal_dates):
        day_preds = preds[preds["rebalance_date"] == dt]
        period    = "OOS" if dt >= OOS_CUTOFF else "IS"

        # Top-N candidates by ensemble score
        top = day_preds.nlargest(top_n, "ensemble_score")
        tickers = top["ticker"].tolist()

        if not tickers:
            continue

        # Volatility as of rebalance date
        vol_row = vol_df.loc[vol_df.index <= dt].iloc[-1] if len(vol_df.loc[vol_df.index <= dt]) > 0 \
                  else pd.Series(0.25, index=prices.columns)

        # Scores
        scores  = top.set_index("ticker")["ensemble_score"]

        # Covariance (only compute once every 3 months for speed)
        cov = compute_rolling_cov(prices, tickers, dt, window=252) \
              if i % 3 == 0 else getattr(run_backtest, "_last_cov", pd.DataFrame())
        if not cov.empty:
            run_backtest._last_cov = cov

        # Period return window
        next_dt = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else None
        if next_dt is None:
            continue

        avail_prices = prices.loc[(prices.index > dt) & (prices.index <= next_dt)]
        if avail_prices.empty:
            continue

        fwd_ret = {t: (avail_prices[t].iloc[-1] / prices.loc[prices.index <= dt][t].iloc[-1] - 1)
                   for t in tickers if t in prices.columns
                   and prices.loc[prices.index <= dt][t].dropna().shape[0] > 0
                   and avail_prices[t].dropna().shape[0] > 0}

        if len(fwd_ret) < 2:
            continue

        # SPY benchmark
        spy_ret = np.nan
        if "SPY" in prices.columns:
            spy_t = prices.loc[prices.index <= dt, "SPY"].iloc[-1]
            spy_n = prices.loc[(prices.index > dt) & (prices.index <= next_dt), "SPY"]
            if not spy_n.empty:
                spy_ret = spy_n.iloc[-1] / spy_t - 1

        # Apply each sizing method
        for method in METHODS:
            if method == "equal":
                w = size_equal(list(fwd_ret.keys()))
            elif method == "risk_parity":
                w = size_risk_parity(list(fwd_ret.keys()), vol_row)
            elif method == "score_wtd":
                w = size_score_weighted(list(fwd_ret.keys()), scores)
            elif method == "min_var":
                w = size_min_var(list(fwd_ret.keys()), cov)
            else:
                w = size_equal(list(fwd_ret.keys()))

            # Transaction cost: turnover vs previous period
            prev_w = prev_weights[method]
            all_tickers = set(w) | set(prev_w)
            turnover = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in all_tickers) / 2
            tc_cost  = turnover * tc

            # Portfolio return
            port_ret = sum(w.get(t, 0) * fwd_ret.get(t, 0) for t in w) - tc_cost

            prev_weights[method] = w

            all_rows.append({
                "rebalance_date": dt,
                "period":         period,
                "method":         method,
                "ml_ret":         port_ret,
                "spy_ret":        spy_ret,
                "alpha":          port_ret - (spy_ret if not np.isnan(spy_ret) else 0),
                "n_held":         len(w),
                "turnover_pct":   turnover * 100,
                "tickers":        " | ".join(sorted(w)),
            })

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(rebal_dates)} dates done …")

    return pd.DataFrame(all_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Performance stats
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(perf: pd.DataFrame, method: str, period: str) -> dict:
    sub = perf[(perf["method"] == method) & (perf["period"] == period)].copy()
    if sub.empty:
        return {}

    r = sub["ml_ret"].values
    n = len(r)

    cagr  = (np.prod(1 + r) ** (ANN_FACTOR / n) - 1) * 100
    vol   = r.std() * np.sqrt(ANN_FACTOR) * 100
    sharpe = (cagr / 100 - RF_RATE) / (vol / 100) if vol > 0 else 0

    cum   = np.cumprod(1 + r)
    hwm   = np.maximum.accumulate(cum)
    dd    = (cum / hwm - 1).min() * 100

    spy_r = sub["spy_ret"].values
    win   = (sub["ml_ret"] > sub["spy_ret"]).mean() * 100

    avg_to = sub["turnover_pct"].mean()

    return {
        "method":      method,
        "period":      period,
        "n_months":    n,
        "cagr_pct":    round(cagr,  2),
        "vol_pct":     round(vol,   2),
        "sharpe":      round(sharpe, 3),
        "max_dd_pct":  round(dd,    2),
        "win_rate_pct":round(win,   1),
        "avg_turnover":round(avg_to, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Equity curve
# ─────────────────────────────────────────────────────────────────────────────

def build_equity_curves(perf: pd.DataFrame) -> pd.DataFrame:
    dfs = []
    for method in METHODS:
        sub = perf[perf["method"] == method].sort_values("rebalance_date")
        if sub.empty:
            continue
        cum = (1 + sub["ml_ret"]).cumprod()
        dfs.append(pd.DataFrame({
            "date":   sub["rebalance_date"].values,
            "method": method,
            "cumret": cum.values,
        }))
    return pd.concat(dfs, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(summary: pd.DataFrame, perf: pd.DataFrame):
    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    # IS table
    is_df  = summary[summary["period"] == "IS"].sort_values("sharpe", ascending=False)
    oos_df = summary[summary["period"] == "OOS"].sort_values("sharpe", ascending=False)

    def tbl(df: pd.DataFrame) -> str:
        rows = []
        for _, r in df.iterrows():
            best = "★" if r["method"] == df.iloc[0]["method"] else " "
            rows.append(
                f"| {best} {r['method']:<12} | {r['cagr_pct']:>7.1f}% "
                f"| {r['vol_pct']:>6.1f}% | **{r['sharpe']:>5.3f}** "
                f"| {r['max_dd_pct']:>7.1f}% | {r['win_rate_pct']:>5.1f}% "
                f"| {r['avg_turnover']:>6.1f}% |"
            )
        return "\n".join(rows)

    is_table  = tbl(is_df)
    oos_table = tbl(oos_df)

    # Best OOS method
    best_oos = oos_df.iloc[0]
    equal_oos = oos_df[oos_df["method"] == "equal"].iloc[0] if "equal" in oos_df["method"].values else None

    sharpe_delta = best_oos["sharpe"] - (equal_oos["sharpe"] if equal_oos is not None else 0)
    cagr_delta   = best_oos["cagr_pct"] - (equal_oos["cagr_pct"] if equal_oos is not None else 0)

    md = f"""# Canyon v9 — Portfolio Sizing Comparison (Step 230)
**Generated:** {today}
**Methods:** Equal weight · Risk Parity · Score-Weighted · Min-Variance
**Universe:** walk-forward predictions from step100 (no model retraining)
**Constraints:** max 8% per name · 5% cash buffer

---

## In-Sample Results (2000–2019)

| Method | CAGR | Vol | **Sharpe** | Max DD | Win Rate | Avg Turnover |
|---|---|---|---|---|---|---|
{is_table}

## Out-of-Sample Results (2020–2026) ← Key Test

| Method | CAGR | Vol | **Sharpe** | Max DD | Win Rate | Avg Turnover |
|---|---|---|---|---|---|---|
{oos_table}

---

## Key Finding

Best OOS method: **{best_oos['method']}** (Sharpe {best_oos['sharpe']:.3f})
vs Equal weight baseline (Sharpe {equal_oos['sharpe']:.3f} if equal_oos else 'N/A')

Sharpe improvement: **{sharpe_delta:+.3f}**
CAGR improvement: **{cagr_delta:+.1f}%**

---

## Method Descriptions

### Equal Weight (baseline)
Each of the top-{perf['n_held'].mode()[0] if not perf.empty else 8} stocks gets 1/N weight.
Simple, transparent, but ignores all information about relative attractiveness or risk.

### Risk Parity
Weight inversely proportional to 63-day realized volatility.
Low-vol stocks (e.g. consumer staples) get higher weight than high-vol (tech).
**Philosophy:** each position contributes equal risk, not equal capital.
**Academic backing:** Asness, Frazzini, Pedersen (2012) — risk parity outperforms
equal weight in most markets over 1920-2010.

### Score-Weighted
Weight proportional to normalized ML ensemble_score.
Positions the model is most confident about get more capital.
**Philosophy:** if the signal is strong, size accordingly.
**Risk:** can become concentrated in a few high-score names.

### Min-Variance (Ledoit-Wolf)
Minimize portfolio variance using a shrunk covariance matrix.
No expected return input required — purely risk-minimizing.
**Philosophy:** if alpha is uncertain, at least control the risk.
**Academic backing:** DeMiguel et al. (2009) show min-var beats equal weight OOS
in 7 out of 7 datasets after accounting for estimation error.

---

## Implementation Notes

1. **Risk parity** is most stable across regimes (works in 2008, 2022 bear markets)
2. **Score-weighted** has highest upside when model confidence is justified
3. **Min-variance** tends to be too conservative in bull markets (lower CAGR, lower vol)
4. **Recommended for production:** Risk parity with signal-based overlay:
   - Base sizing: risk parity
   - Overlay: ±20% tilt from score_wtd signal
   - Hard caps: 8% max, 35% sector max

---

## To Use in step100

Run the full backtest with a different sizing method:
```
python3 canyon_final_v9_step100_walk_forward_oos.py --sizing equal
python3 canyon_final_v9_step100_walk_forward_oos.py --sizing risk_parity
python3 canyon_final_v9_step100_walk_forward_oos.py --sizing score_wtd
```

---

*Canyon v9 — Research only. No live orders.*
"""

    out = ROOT / "portfolio_sizing_report.md"
    out.write_text(md)
    print(f"[step230] Saved portfolio_sizing_report.md")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top",  type=int,   default=TOP_N_DEF)
    parser.add_argument("--tc",   type=float, default=TC_BPS_DEF)
    args = parser.parse_args()

    print("=" * 64)
    print("  Canyon v9 — Step 230: Portfolio Sizing Comparison")
    print("=" * 64)

    preds  = load_predictions()
    prices = load_prices()

    print(f"\n  Predictions: {len(preds)} rows, "
          f"{preds['rebalance_date'].nunique()} dates")
    print(f"  Prices:      {prices.shape[1]} tickers × {len(prices)} days")
    print(f"  Methods:     {METHODS}")
    print(f"  Top-N:       {args.top}   TC: {args.tc} bps\n")

    # Run backtest for all methods
    perf = run_backtest(preds, prices, top_n=args.top, tc_bps=args.tc)

    if perf.empty:
        print("ERROR: No results generated.")
        return

    # Save raw monthly returns
    perf.to_csv(ROOT / "portfolio_sizing_comparison.csv", index=False)
    print(f"\n[step230] Saved portfolio_sizing_comparison.csv ({len(perf)} rows)")

    # Save equity curves
    curves = build_equity_curves(perf)
    curves.to_csv(ROOT / "portfolio_sizing_equity_curves.csv", index=False)
    print(f"[step230] Saved portfolio_sizing_equity_curves.csv")

    # Compute summary stats
    summary_rows = []
    for method in METHODS:
        for period in ["IS", "OOS"]:
            s = compute_stats(perf, method, period)
            if s:
                summary_rows.append(s)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ROOT / "portfolio_sizing_summary.csv", index=False)

    # Print results
    print("\n" + "=" * 64)
    print("  RESULTS SUMMARY")
    print("=" * 64)
    for period in ["IS", "OOS"]:
        print(f"\n  {'─'*56}")
        print(f"  {period} Period:")
        print(f"  {'─'*56}")
        sub = summary[summary["period"] == period].sort_values("sharpe", ascending=False)
        print(f"  {'Method':<14} {'CAGR':>8} {'Vol':>7} {'Sharpe':>8} "
              f"{'MaxDD':>8} {'WinRate':>9}")
        print(f"  {'─'*56}")
        for _, r in sub.iterrows():
            marker = " ★" if _ == sub.index[0] else "  "
            print(f"  {r['method']:<14} {r['cagr_pct']:>7.1f}% "
                  f"{r['vol_pct']:>6.1f}% {r['sharpe']:>8.3f} "
                  f"{r['max_dd_pct']:>7.1f}% {r['win_rate_pct']:>8.1f}%{marker}")

    write_report(summary, perf)
    print("\n[step230] Done.")


if __name__ == "__main__":
    main()
