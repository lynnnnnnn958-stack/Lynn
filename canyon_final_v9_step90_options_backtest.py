"""
canyon_final_v9_step90_options_backtest.py
Canyon v9 — Options Strategy Backtest (Step 90)

Simulates 3 options strategies over historical price data using Black-Scholes
with realized volatility as IV proxy. No paid options data required.

Strategies:
  1. Covered_Call   — sell 30-delta call, 30 DTE, monthly roll
  2. Cash_Secured_Put (CSP) — sell 30-delta put, 30 DTE, monthly roll
  3. Bull_Call_Spread — buy 50-delta call, sell 20-delta call, monthly roll

Inputs:
  sp500_price_cache.csv   — daily close prices (index=Date, columns=tickers)
  regime_ml_scores.csv    — ticker, predicted_score (universe selection)
  iv_history.csv          — optional: date, ticker, atm_iv (step82 output)

Outputs:
  options_backtest_results.csv
  options_backtest_report.md
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Black-Scholes primitives (no scipy — Abramowitz & Stegun approximation)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Normal CDF via Abramowitz & Stegun, max error < 1.5e-7."""
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (
        0.319381530
        + t * (
            -0.356563782
            + t * (
                1.781477937
                + t * (-1.821255978 + t * 1.330274429)
            )
        )
    )
    approx = 1.0 - (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x) * poly
    return approx if x >= 0.0 else 1.0 - approx


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put price."""
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_delta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call delta."""
    if T <= 0.0 or sigma <= 0.0:
        return 0.5
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)


# ---------------------------------------------------------------------------
# Realized volatility
# ---------------------------------------------------------------------------

def realized_vol(price_series: pd.Series, window: int = 30) -> pd.Series:
    """30-day annualized realized volatility from daily log returns."""
    log_ret = np.log(price_series / price_series.shift(1))
    return log_ret.rolling(window).std() * math.sqrt(252)


# ---------------------------------------------------------------------------
# Strike selection — find strike closest to target delta
# ---------------------------------------------------------------------------

_STRIKE_MULTIPLES = [
    0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.97,
    1.00, 1.03, 1.05, 1.08, 1.10, 1.15, 1.20,
]

def _snap_strike(strike: float) -> float:
    """Snap to nearest $2.50 increment (or $5 for strikes above $100)."""
    if strike >= 100.0:
        return round(strike / 5.0) * 5.0
    return round(strike / 2.5) * 2.5


def find_call_strike(
    spot: float,
    T: float,
    r: float,
    sigma: float,
    target_delta: float = 0.30,
) -> float:
    """Return the call strike closest to target_delta (snapped)."""
    best_strike = spot * 1.05
    best_diff = 999.0
    for mult in _STRIKE_MULTIPLES:
        K = _snap_strike(spot * mult)
        if K <= 0:
            continue
        delta = bs_delta_call(spot, K, T, r, sigma)
        diff = abs(delta - target_delta)
        if diff < best_diff:
            best_diff = diff
            best_strike = K
    return best_strike


def find_put_strike(
    spot: float,
    T: float,
    r: float,
    sigma: float,
    target_delta: float = 0.30,
) -> float:
    """Return the put strike closest to target_delta via put-call parity.

    For a put, delta_put = delta_call - 1, so we look for call delta = 1 - target.
    OTM puts are below spot (call_delta ~ 0.30 -> put below spot).
    """
    call_target = 1.0 - target_delta  # ~0.70 for 0.30-delta put
    # We actually want the OTM put: call delta on same strike ≈ 0.30 (below spot)
    # So target call delta = target_delta (same strike, put delta = call_delta - 1)
    best_strike = spot * 0.95
    best_diff = 999.0
    for mult in _STRIKE_MULTIPLES:
        K = _snap_strike(spot * mult)
        if K <= 0 or K >= spot:  # only OTM puts (below spot)
            continue
        delta_c = bs_delta_call(spot, K, T, r, sigma)
        # put delta = delta_c - 1 in magnitude → |put_delta| = 1 - delta_c
        put_delta_abs = 1.0 - delta_c
        diff = abs(put_delta_abs - target_delta)
        if diff < best_diff:
            best_diff = diff
            best_strike = K
    return best_strike


# ---------------------------------------------------------------------------
# Per-ticker simulation functions
# ---------------------------------------------------------------------------

RISK_FREE = 0.05
DTE = 30
T_OPTION = DTE / 365.0
ANNUAL_RF_DAILY = (1.0 + RISK_FREE) ** (1.0 / 252.0) - 1.0


def _safe_vol(vol_series: pd.Series, date: pd.Timestamp) -> Optional[float]:
    """Return realized vol on or before date; None if unavailable/zero."""
    try:
        v = vol_series.loc[:date].dropna().iloc[-1]
        if v > 0.01 and v < 5.0:  # sanity: 1% to 500%
            return float(v)
    except (IndexError, KeyError):
        pass
    return None


def sim_covered_call(
    prices: pd.Series,
    iv_series: Optional[pd.Series],
    rebal_dates: List[pd.Timestamp],
) -> Dict:
    """Simulate Covered Call strategy. Returns dict of per-month results."""
    monthly_pnl: List[float] = []
    premiums: List[float] = []
    stock_rets: List[float] = []

    rvol = realized_vol(prices)

    for i in range(len(rebal_dates) - 1):
        entry_date = rebal_dates[i]
        exit_date = rebal_dates[i + 1]

        # Prices at open and close of period
        try:
            S_entry = float(prices.loc[entry_date])
            S_exit = float(prices.loc[exit_date])
        except KeyError:
            continue

        if S_entry <= 0 or math.isnan(S_entry) or math.isnan(S_exit):
            continue

        # IV: prefer iv_history, fallback to realized vol
        sigma = None
        if iv_series is not None:
            sigma = _safe_vol(iv_series, entry_date)
        if sigma is None:
            sigma = _safe_vol(rvol, entry_date)
        if sigma is None:
            sigma = 0.20  # last resort default

        # Select ~0.30-delta call strike
        K_call = find_call_strike(S_entry, T_OPTION, RISK_FREE, sigma, 0.30)

        # Premium collected
        premium = bs_call(S_entry, K_call, T_OPTION, RISK_FREE, sigma)
        premiums.append(premium)

        # Stock P&L per share
        stock_pnl = S_exit - S_entry
        stock_rets.append((S_exit - S_entry) / S_entry)

        # Call payout at expiry (short call)
        call_payout = max(S_exit - K_call, 0.0)

        # Net monthly P&L (per share basis, virtual 100 shares but normalized to 1 share)
        pnl = stock_pnl + premium - call_payout
        monthly_pnl.append(pnl / S_entry)  # return-based

    return {
        "monthly_returns": monthly_pnl,
        "premiums": premiums,
        "stock_returns": stock_rets,
    }


def sim_csp(
    prices: pd.Series,
    iv_series: Optional[pd.Series],
    rebal_dates: List[pd.Timestamp],
) -> Dict:
    """Simulate Cash-Secured Put strategy."""
    monthly_pnl: List[float] = []
    premiums: List[float] = []

    rvol = realized_vol(prices)

    for i in range(len(rebal_dates) - 1):
        entry_date = rebal_dates[i]
        exit_date = rebal_dates[i + 1]

        try:
            S_entry = float(prices.loc[entry_date])
            S_exit = float(prices.loc[exit_date])
        except KeyError:
            continue

        if S_entry <= 0 or math.isnan(S_entry) or math.isnan(S_exit):
            continue

        sigma = None
        if iv_series is not None:
            sigma = _safe_vol(iv_series, entry_date)
        if sigma is None:
            sigma = _safe_vol(rvol, entry_date)
        if sigma is None:
            sigma = 0.20

        K_put = find_put_strike(S_entry, T_OPTION, RISK_FREE, sigma, 0.30)
        premium = bs_put(S_entry, K_put, T_OPTION, RISK_FREE, sigma)
        premiums.append(premium)

        # Put payout at expiry (short put)
        put_payout = max(K_put - S_exit, 0.0)

        # Cash earns risk-free on reserve (K_put * 100 shares, normalized)
        days = (exit_date - entry_date).days
        cash_interest = K_put * ((1.0 + RISK_FREE) ** (days / 365.0) - 1.0)

        pnl = premium - put_payout + cash_interest
        monthly_pnl.append(pnl / K_put)  # return on reserved capital

    return {
        "monthly_returns": monthly_pnl,
        "premiums": premiums,
    }


def sim_bull_call_spread(
    prices: pd.Series,
    iv_series: Optional[pd.Series],
    rebal_dates: List[pd.Timestamp],
) -> Dict:
    """Simulate Bull Call Spread: buy 0.50-delta call, sell 0.20-delta call."""
    monthly_pnl: List[float] = []
    debits: List[float] = []

    rvol = realized_vol(prices)

    for i in range(len(rebal_dates) - 1):
        entry_date = rebal_dates[i]
        exit_date = rebal_dates[i + 1]

        try:
            S_entry = float(prices.loc[entry_date])
            S_exit = float(prices.loc[exit_date])
        except KeyError:
            continue

        if S_entry <= 0 or math.isnan(S_entry) or math.isnan(S_exit):
            continue

        sigma = None
        if iv_series is not None:
            sigma = _safe_vol(iv_series, entry_date)
        if sigma is None:
            sigma = _safe_vol(rvol, entry_date)
        if sigma is None:
            sigma = 0.20

        # Long leg: ATM (~0.50 delta)
        K_long = find_call_strike(S_entry, T_OPTION, RISK_FREE, sigma, 0.50)
        # Short leg: OTM (~0.20 delta, ~15-20% above spot)
        K_short = find_call_strike(S_entry, T_OPTION, RISK_FREE, sigma, 0.20)

        # Ensure spread makes sense
        if K_short <= K_long:
            K_short = _snap_strike(K_long * 1.10)

        long_price = bs_call(S_entry, K_long, T_OPTION, RISK_FREE, sigma)
        short_price = bs_call(S_entry, K_short, T_OPTION, RISK_FREE, sigma)
        net_debit = long_price - short_price
        debits.append(net_debit)

        if net_debit <= 0:
            # Credit spread — skip (degenerate case)
            monthly_pnl.append(0.0)
            continue

        # Payoff at expiry
        long_payout = max(S_exit - K_long, 0.0)
        short_payout = max(S_exit - K_short, 0.0)
        net_payout = long_payout - short_payout  # capped at K_short - K_long

        pnl = net_payout - net_debit
        # Return on max risk (= net_debit)
        monthly_pnl.append(pnl / net_debit if net_debit > 0.001 else 0.0)

    return {
        "monthly_returns": monthly_pnl,
        "debits": debits,
    }


# ---------------------------------------------------------------------------
# Portfolio-level metrics
# ---------------------------------------------------------------------------

def compute_metrics(monthly_returns: List[float], rf_monthly: float = 0.05 / 12) -> Dict:
    """Compute annualized return, Sharpe, max drawdown, win rate."""
    if not monthly_returns:
        return {
            "ann_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "n_months": 0,
        }

    rets = np.array(monthly_returns, dtype=float)
    n = len(rets)

    ann_return = float((1.0 + rets.mean()) ** 12 - 1.0)

    excess = rets - rf_monthly
    sharpe = (
        float(excess.mean() / excess.std() * math.sqrt(12))
        if excess.std() > 1e-9
        else 0.0
    )

    # Max drawdown
    cum = np.cumprod(1.0 + rets)
    running_max = np.maximum.accumulate(cum)
    drawdowns = cum / running_max - 1.0
    max_dd = float(drawdowns.min())

    win_rate = float(np.mean(rets > 0))

    return {
        "ann_return": ann_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "n_months": n,
    }


def compute_bnh_metrics(prices: pd.Series, rebal_dates: List[pd.Timestamp]) -> Dict:
    """Buy-and-hold monthly returns over the same period."""
    monthly_rets = []
    for i in range(len(rebal_dates) - 1):
        try:
            s = float(prices.loc[rebal_dates[i]])
            e = float(prices.loc[rebal_dates[i + 1]])
            if s > 0 and not math.isnan(s) and not math.isnan(e):
                monthly_rets.append((e - s) / s)
        except KeyError:
            continue
    return compute_metrics(monthly_rets)


# ---------------------------------------------------------------------------
# Rebalance date generation
# ---------------------------------------------------------------------------

def get_rebal_dates(index: pd.DatetimeIndex, years: int) -> List[pd.Timestamp]:
    """Return list of monthly rebalance dates (every 21 trading days)."""
    cutoff = index[-1] - pd.DateOffset(years=years)
    window = index[index >= cutoff]
    if len(window) < 42:
        window = index[-min(len(index), 504):]  # fallback: 2 years ~
    dates = list(window[::21])
    if window[-1] not in dates:
        dates.append(window[-1])
    return dates


# ---------------------------------------------------------------------------
# Main simulation runner
# ---------------------------------------------------------------------------

def run_backtest(top_n: int = 20, years: int = 2) -> None:
    print(f"\n{'='*60}")
    print(f"  Canyon v9 — Options Strategy Backtest (Step 90)")
    print(f"  Top {top_n} tickers | {years}-year lookback")
    print(f"{'='*60}\n")

    # --- Load inputs ---
    price_path = ROOT / "sp500_price_cache.csv"
    scores_path = ROOT / "regime_ml_scores.csv"
    iv_path = ROOT / "iv_history.csv"

    if not price_path.exists():
        print(f"ERROR: {price_path} not found. Aborting.")
        sys.exit(1)
    if not scores_path.exists():
        print(f"ERROR: {scores_path} not found. Aborting.")
        sys.exit(1)

    print("Loading price data ...", end="", flush=True)
    prices_df = pd.read_csv(price_path, index_col=0, parse_dates=True)
    prices_df.sort_index(inplace=True)
    print(f" {prices_df.shape[0]} rows × {prices_df.shape[1]} tickers")

    print("Loading regime ML scores ...", end="", flush=True)
    scores_df = pd.read_csv(scores_path)
    # Handle both header variants
    if "predicted_score" not in scores_df.columns and "signal" in scores_df.columns:
        scores_df["predicted_score"] = scores_df.get(
            "rank_mom", scores_df.iloc[:, 1]
        )
    scores_df = scores_df.dropna(subset=["predicted_score"])
    scores_df = scores_df.sort_values("predicted_score", ascending=False)
    universe = scores_df["ticker"].head(top_n).tolist()
    # Keep only tickers actually in price data
    universe = [t for t in universe if t in prices_df.columns]
    print(f" {len(universe)} tickers selected from top {top_n}")
    print(f"  Tickers: {', '.join(universe[:10])}{'...' if len(universe) > 10 else ''}")

    # Optional IV history
    iv_df = None
    if iv_path.exists():
        print("Loading IV history ...", end="", flush=True)
        try:
            iv_df = pd.read_csv(iv_path, parse_dates=["date"])
            iv_df = iv_df.set_index(["date", "ticker"])["atm_iv"].unstack("ticker")
            print(f" loaded {iv_df.shape}")
        except Exception as exc:
            print(f" skipped ({exc})")
            iv_df = None
    else:
        print("iv_history.csv not found — using realized vol as IV proxy")

    # Rebalance dates (common to all tickers)
    rebal_dates = get_rebal_dates(prices_df.index, years)
    print(f"\nSimulation period: {rebal_dates[0].date()} → {rebal_dates[-1].date()}")
    print(f"Monthly rebalance dates: {len(rebal_dates) - 1} periods\n")

    # --- Per-ticker simulation ---
    strategy_names = ["Covered_Call", "Cash_Secured_Put", "Bull_Call_Spread"]

    # Collect monthly returns per strategy across tickers
    all_rets: Dict[str, Dict[str, List[float]]] = {
        s: {} for s in strategy_names
    }
    all_premiums: Dict[str, Dict[str, List[float]]] = {
        "Covered_Call": {},
        "Cash_Secured_Put": {},
        "Bull_Call_Spread": {},
    }
    bnh_rets: Dict[str, List[float]] = {}

    rows: List[Dict] = []

    for idx, ticker in enumerate(universe):
        print(f"  [{idx+1:2d}/{len(universe)}] {ticker:<8s}", end="", flush=True)

        px = prices_df[ticker].dropna()
        if len(px) < 63:  # need at least 3 months
            print(" SKIP (insufficient data)")
            continue

        # Align prices to available rebal dates
        avail = prices_df.index
        t_rebal = [
            d for d in rebal_dates
            if d in avail and not math.isnan(float(px.reindex([d]).iloc[0]) if d in px.index else float("nan"))
        ]
        # Use forward-fill to find nearest available price on rebal dates
        px_ff = px.reindex(prices_df.index).ffill()
        t_rebal = [d for d in rebal_dates if d in px_ff.index and px_ff[d] > 0]

        if len(t_rebal) < 3:
            print(" SKIP (too few rebalance dates with data)")
            continue

        # IV series for this ticker
        iv_ts = None
        if iv_df is not None and ticker in iv_df.columns:
            iv_ts = iv_df[ticker].dropna()

        try:
            # Strategy 1: Covered Call
            cc = sim_covered_call(px_ff, iv_ts, t_rebal)
            all_rets["Covered_Call"][ticker] = cc["monthly_returns"]
            all_premiums["Covered_Call"][ticker] = cc["premiums"]

            # Strategy 2: CSP
            csp = sim_csp(px_ff, iv_ts, t_rebal)
            all_rets["Cash_Secured_Put"][ticker] = csp["monthly_returns"]
            all_premiums["Cash_Secured_Put"][ticker] = csp["premiums"]

            # Strategy 3: Bull Call Spread
            bcs = sim_bull_call_spread(px_ff, iv_ts, t_rebal)
            all_rets["Bull_Call_Spread"][ticker] = bcs["monthly_returns"]
            all_premiums["Bull_Call_Spread"][ticker] = bcs.get("debits", [])

            # Buy and hold
            bnh = compute_bnh_metrics(px_ff, t_rebal)
            bnh_rets[ticker] = bnh

            n_months = len(cc["monthly_returns"])
            avg_prem = (
                float(np.mean(cc["premiums"])) if cc["premiums"] else 0.0
            )
            print(
                f" {n_months:2d} months | CC avg prem ${avg_prem:.2f} | B&H {bnh['ann_return']*100:+.1f}%"
            )

            # Per-ticker rows for results CSV
            for strat in strategy_names:
                m = compute_metrics(all_rets[strat].get(ticker, []))
                prems = all_premiums[strat].get(ticker, [])
                rows.append(
                    {
                        "ticker": ticker,
                        "strategy": strat,
                        "ann_return": round(m["ann_return"], 4),
                        "sharpe": round(m["sharpe"], 3),
                        "max_drawdown": round(m["max_drawdown"], 4),
                        "win_rate": round(m["win_rate"], 3),
                        "n_months": m["n_months"],
                        "avg_premium": round(float(np.mean(prems)) if prems else 0.0, 4),
                        "bnh_ann_return": round(bnh["ann_return"], 4),
                        "alpha_vs_bnh": round(m["ann_return"] - bnh["ann_return"], 4),
                    }
                )

        except Exception as exc:
            print(f" ERROR: {exc}")
            continue

    print(f"\nSimulated {len(set(r['ticker'] for r in rows))} tickers successfully.")

    # --- Portfolio-level aggregation ---
    print("\nAggregating portfolio returns (equal-weight across tickers) ...")

    def portfolio_returns(strat_rets: Dict[str, List[float]]) -> List[float]:
        """Equal-weight monthly returns across all tickers."""
        if not strat_rets:
            return []
        max_len = max(len(v) for v in strat_rets.values())
        combined = []
        for i in range(max_len):
            period_rets = [
                v[i] for v in strat_rets.values() if i < len(v)
            ]
            if period_rets:
                combined.append(float(np.mean(period_rets)))
        return combined

    portfolio: Dict[str, Dict] = {}
    for strat in strategy_names:
        port_rets = portfolio_returns(all_rets[strat])
        portfolio[strat] = compute_metrics(port_rets)
        portfolio[strat]["raw_monthly"] = port_rets

    # BnH portfolio
    bnh_port_rets = portfolio_returns(
        {t: [m["ann_return"] / 12.0] * m["n_months"] for t, m in bnh_rets.items()}
    )
    # Better: use actual monthly returns from stored data
    bnh_monthly_raw: Dict[str, List[float]] = {}
    for ticker in bnh_rets:
        px_ff = prices_df[ticker].dropna().reindex(prices_df.index).ffill()
        t_rebal = [d for d in rebal_dates if d in px_ff.index and px_ff[d] > 0]
        bnh_m = []
        for i in range(len(t_rebal) - 1):
            s = float(px_ff[t_rebal[i]])
            e = float(px_ff[t_rebal[i + 1]])
            if s > 0:
                bnh_m.append((e - s) / s)
        bnh_monthly_raw[ticker] = bnh_m

    bnh_port_rets2 = portfolio_returns(bnh_monthly_raw)
    portfolio["Buy_and_Hold"] = compute_metrics(bnh_port_rets2)

    # --- Save results CSV ---
    results_path = ROOT / "options_backtest_results.csv"
    if rows:
        results_df = pd.DataFrame(rows)
        results_df.to_csv(results_path, index=False)
        print(f"\nSaved: {results_path} ({len(rows)} rows)")
    else:
        print("\nWARNING: No results to save.")

    # --- Print comparison table ---
    print(f"\n{'='*72}")
    print(f"  PORTFOLIO STRATEGY COMPARISON")
    print(f"{'='*72}")
    header = f"{'Strategy':<22} {'Ann Ret':>9} {'Sharpe':>8} {'Max DD':>9} {'Win%':>7} {'N':>5}"
    print(header)
    print("-" * 72)

    all_strats = strategy_names + ["Buy_and_Hold"]
    for strat in all_strats:
        m = portfolio.get(strat, {})
        if not m:
            continue
        ar = m.get("ann_return", 0.0)
        sh = m.get("sharpe", 0.0)
        md = m.get("max_drawdown", 0.0)
        wr = m.get("win_rate", 0.0)
        n = m.get("n_months", 0)
        print(
            f"  {strat:<20} {ar*100:>+8.1f}%  {sh:>7.2f}  {md*100:>8.1f}%  {wr*100:>6.1f}%  {n:>5d}"
        )
    print("=" * 72)

    # Premium capture rate (selling strategies only)
    print("\n  PREMIUM ANALYSIS (selling strategies)")
    print("-" * 72)
    for strat in ["Covered_Call", "Cash_Secured_Put"]:
        all_p = []
        for t in all_premiums[strat]:
            all_p.extend(all_premiums[strat][t])
        if all_p:
            print(
                f"  {strat:<22}  Avg premium: ${np.mean(all_p):.2f}  "
                f"Total collected: ${sum(all_p)*100:.0f} (virtual 100 shares)"
            )

    # Alpha table
    if rows:
        r_df = pd.DataFrame(rows)
        print("\n  TOP TICKER ALPHA vs Buy-and-Hold (Covered Call)")
        cc_alpha = (
            r_df[r_df["strategy"] == "Covered_Call"]
            .sort_values("alpha_vs_bnh", ascending=False)
            .head(5)[["ticker", "ann_return", "bnh_ann_return", "alpha_vs_bnh", "sharpe"]]
        )
        print(cc_alpha.to_string(index=False))

    # --- Write markdown report ---
    _write_report(portfolio, rows, all_premiums, universe, years, top_n)
    print(f"\nReport saved: {ROOT / 'options_backtest_report.md'}")
    print("\nDone.\n")


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------

def _write_report(
    portfolio: Dict,
    rows: List[Dict],
    all_premiums: Dict,
    universe: List[str],
    years: int,
    top_n: int,
) -> None:
    r_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    lines: List[str] = []
    lines.append("# Canyon v9 — Options Strategy Backtest Report\n")
    lines.append(f"**Step 90 | Top {top_n} tickers | {years}-year lookback**\n")
    lines.append(
        "_Black-Scholes pricing with 30-day realized vol as IV proxy. "
        "No paid options data required. Approximate but directionally valid for premium-selling strategies._\n"
    )
    lines.append(f"**Universe ({len(universe)} tickers):** {', '.join(universe)}\n")
    lines.append("\n---\n")

    # Summary table
    lines.append("## Portfolio-Level Results\n")
    lines.append(
        "| Strategy | Ann Return | Sharpe | Max Drawdown | Win Rate | N Months |\n"
        "|:---------|----------:|-------:|-------------:|---------:|---------:|\n"
    )
    all_strats = ["Covered_Call", "Cash_Secured_Put", "Bull_Call_Spread", "Buy_and_Hold"]
    bnh_ret = portfolio.get("Buy_and_Hold", {}).get("ann_return", 0.0)
    for strat in all_strats:
        m = portfolio.get(strat, {})
        if not m:
            continue
        ar = m.get("ann_return", 0.0)
        sh = m.get("sharpe", 0.0)
        md = m.get("max_drawdown", 0.0)
        wr = m.get("win_rate", 0.0)
        n = m.get("n_months", 0)
        alpha = ar - bnh_ret if strat != "Buy_and_Hold" else 0.0
        alpha_str = f" ({alpha:+.1%} alpha)" if strat != "Buy_and_Hold" else ""
        lines.append(
            f"| {strat} | {ar:+.1%}{alpha_str} | {sh:.2f} | {md:.1%} | {wr:.1%} | {n} |\n"
        )

    lines.append("\n")

    # Strategy descriptions
    lines.append("## Strategy Descriptions\n")
    lines.append(
        "### 1. Covered Call\n"
        "- Hold 100 virtual shares, sell 1 call at ~0.30 delta, 30 DTE each month.\n"
        "- Premium received: BS call price using realized 30-day vol.\n"
        "- P&L = stock return + premium collected - call payout at expiry.\n"
        "- Underperforms BnH in strong bull markets; outperforms in flat/mild environments.\n\n"
    )
    lines.append(
        "### 2. Cash-Secured Put (CSP)\n"
        "- Sell 1 put at ~0.30 delta (OTM) each month, 30 DTE.\n"
        "- Cash reserve earns risk-free rate (5% annualized).\n"
        "- P&L = premium + cash interest - put assignment loss.\n"
        "- Economically equivalent to Covered Call via put-call parity (same return profile).\n\n"
    )
    lines.append(
        "### 3. Bull Call Spread\n"
        "- Buy ATM call (~0.50 delta), sell OTM call (~0.20 delta), 30 DTE.\n"
        "- Net debit = long call - short call.\n"
        "- Max gain = (K_short - K_long) - debit at K_short; max loss = debit.\n"
        "- Directional bet with defined risk; outperforms in moderate up-moves.\n\n"
    )

    # Premium analysis
    lines.append("## Premium Analysis\n")
    lines.append("| Strategy | Avg Monthly Premium | Avg Win Rate |\n")
    lines.append("|:---------|--------------------:|-------------:|\n")
    for strat in ["Covered_Call", "Cash_Secured_Put"]:
        all_p: List[float] = []
        for t in all_premiums.get(strat, {}):
            all_p.extend(all_premiums[strat][t])
        m = portfolio.get(strat, {})
        wr = m.get("win_rate", 0.0)
        avg = float(np.mean(all_p)) if all_p else 0.0
        lines.append(f"| {strat} | ${avg:.2f}/share | {wr:.1%} |\n")
    lines.append("\n")

    # Per-ticker table
    if not r_df.empty:
        lines.append("## Per-Ticker Results\n")
        for strat in ["Covered_Call", "Cash_Secured_Put", "Bull_Call_Spread"]:
            sub = r_df[r_df["strategy"] == strat].sort_values(
                "ann_return", ascending=False
            )
            if sub.empty:
                continue
            lines.append(f"### {strat}\n")
            lines.append(
                "| Ticker | Ann Return | BnH Return | Alpha | Sharpe | Win Rate |\n"
                "|:------|-----------:|-----------:|------:|-------:|---------:|\n"
            )
            for _, row in sub.iterrows():
                lines.append(
                    f"| {row['ticker']} | {row['ann_return']:+.1%} | "
                    f"{row['bnh_ann_return']:+.1%} | {row['alpha_vs_bnh']:+.1%} | "
                    f"{row['sharpe']:.2f} | {row['win_rate']:.1%} |\n"
                )
            lines.append("\n")

    # Methodology notes
    lines.append("## Methodology & Limitations\n")
    lines.append(
        "- **IV proxy**: 30-day realized volatility used as implied volatility. "
        "Actual options premiums include a variance risk premium (~2-4 vol points), "
        "so realized-vol pricing may *underestimate* premium income.\n"
        "- **Strike snapping**: Strikes snapped to nearest $2.50 / $5 increment.\n"
        "- **No transaction costs**: Bid-ask spread, commissions not modeled. "
        "Estimate 0.1-0.3% drag per month for realistic results.\n"
        "- **No early assignment**: European-style exercise assumed.\n"
        "- **Monthly roll only**: 21-trading-day periods; no intra-month adjustments.\n"
        "- **Equal-weight portfolio**: All tickers weighted equally within each strategy.\n"
    )
    lines.append("\n---\n")
    lines.append("_Generated by canyon_final_v9_step90_options_backtest.py_\n")

    report_path = ROOT / "options_backtest_report.md"
    with open(report_path, "w") as fh:
        fh.writelines(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canyon v9 Options Strategy Backtest (Step 90)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="Number of top-scoring tickers to simulate (default: 20)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=2,
        metavar="N",
        help="Years of historical data to use (default: 2)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_backtest(top_n=args.top, years=args.years)
