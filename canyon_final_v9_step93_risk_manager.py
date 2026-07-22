#!/usr/bin/env python3
"""
Canyon v9 — Step 93: Risk Manager
===================================
Standalone portfolio risk engine.  No scipy.  No live trading.
pandas + numpy only (yfinance only for price fetch fallback).

Inputs
------
  sp500_price_cache.csv   — wide price history (Date index, ticker columns)
  daily_final.csv         — current portfolio (ticker, weight columns)
  trade_journal.csv       — Kelly estimation per strategy_type (optional)
  alpha_scores.csv        — position context (optional)

Outputs
-------
  risk_report.csv         — per-ticker risk table
  risk_report.md          — markdown summary

Usage
-----
  python3 canyon_final_v9_step93_risk_manager.py
  python3 canyon_final_v9_step93_risk_manager.py --top 30 --confidence 0.99 --years 3
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── File paths ────────────────────────────────────────────────────────────────
PRICE_CACHE_CSV   = ROOT / "sp500_price_cache.csv"
PORTFOLIO_CSV     = ROOT / "daily_final.csv"
TRADE_JOURNAL_CSV = ROOT / "trade_journal.csv"
ALPHA_SCORES_CSV  = ROOT / "alpha_scores.csv"
OUT_CSV           = ROOT / "risk_report.csv"
OUT_MD            = ROOT / "risk_report.md"

# ── Risk thresholds ───────────────────────────────────────────────────────────
HHI_CONCENTRATION_THRESHOLD  = 0.20
VAR_HIGH_THRESHOLD            = 0.02   # 1-day 95% VaR > 2%
CORR_HIGH_THRESHOLD           = 0.85   # pair correlation flag
KELLY_OVERWEIGHT_MULTIPLIER   = 1.50   # current_weight > kelly * 1.5 → flag
DEFAULT_KELLY                 = 0.10   # if trade_journal not found
ANN_FACTOR                    = 252


# =============================================================================
# Core math helpers (no scipy)
# =============================================================================

def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Full Kelly = (win_rate / avg_loss - (1 - win_rate) / avg_win) * avg_win
    Returns HALF Kelly, capped at 0.25.

    Parameters
    ----------
    win_rate : fraction of winning trades (0–1)
    avg_win  : average gain as a positive fraction (e.g. 0.05 = 5%)
    avg_loss : average loss as a positive fraction (e.g. 0.03 = 3%)
    """
    if avg_loss <= 0 or avg_win <= 0:
        return DEFAULT_KELLY
    full_kelly = (win_rate / avg_loss - (1.0 - win_rate) / avg_win) * avg_win
    half_kelly = full_kelly / 2.0
    return float(min(max(half_kelly, 0.0), 0.25))


def portfolio_historical_var(
    weights: dict,
    returns_df: pd.DataFrame,
    confidence: float = 0.95,
) -> float:
    """
    Historical VaR (1-day).

    portfolio_return_t = sum(w_i * r_i_t)
    VaR = -percentile(portfolio_returns, 1 - confidence)

    Returns a positive number representing the loss at the confidence level.
    """
    tickers = [t for t in weights if t in returns_df.columns]
    if not tickers:
        return np.nan
    w = np.array([weights[t] for t in tickers])
    r = returns_df[tickers].dropna(how="all")
    r = r.fillna(0.0)
    port_rets = r.values @ w
    cutoff = np.percentile(port_rets, (1.0 - confidence) * 100.0)
    return float(-cutoff)


def portfolio_cvar(
    weights: dict,
    returns_df: pd.DataFrame,
    confidence: float = 0.95,
) -> float:
    """
    CVaR / Expected Shortfall = mean of returns strictly below the VaR threshold.

    Returns a positive number.
    """
    tickers = [t for t in weights if t in returns_df.columns]
    if not tickers:
        return np.nan
    w = np.array([weights[t] for t in tickers])
    r = returns_df[tickers].dropna(how="all")
    r = r.fillna(0.0)
    port_rets = r.values @ w
    cutoff = np.percentile(port_rets, (1.0 - confidence) * 100.0)
    tail = port_rets[port_rets <= cutoff]
    if len(tail) == 0:
        return float(-cutoff)
    return float(-tail.mean())


def correlation_matrix(
    returns_df: pd.DataFrame,
    min_periods: int = 60,
) -> pd.DataFrame:
    """
    30-day rolling correlation snapshot of the provided holdings.
    Uses the last 30 valid rows of returns.
    """
    recent = returns_df.iloc[-30:] if len(returns_df) >= 30 else returns_df
    return recent.corr(min_periods=min_periods)


def herfindahl_index(weights: dict) -> float:
    """
    HHI = sum(w_i^2).  HHI > 0.25 is considered concentrated.
    """
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    normalized = {k: v / total for k, v in weights.items()}
    return float(sum(v ** 2 for v in normalized.values()))


# =============================================================================
# Data loaders
# =============================================================================

def load_prices(years: int = 2) -> pd.DataFrame:
    """
    Load sp500_price_cache.csv (wide format).
    Falls back to yfinance if file missing.
    """
    if PRICE_CACHE_CSV.exists():
        prices = pd.read_csv(PRICE_CACHE_CSV, index_col=0, parse_dates=True)
        prices = prices.sort_index()
        cutoff = prices.index[-1] - timedelta(days=int(years * 365.25))
        prices = prices[prices.index >= cutoff]
        return prices
    print("  [warn] sp500_price_cache.csv not found — attempting yfinance fallback")
    try:
        import yfinance as yf
        end_dt   = datetime.today()
        start_dt = end_dt - timedelta(days=int(years * 365.25) + 60)
        raw = yf.download(
            "SPY",
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        prices = raw[["Close"]] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        prices.columns = ["SPY"]
        return prices
    except Exception as exc:
        print(f"  [warn] yfinance fallback failed: {exc}")
        return pd.DataFrame()


def load_portfolio() -> pd.DataFrame:
    """
    Load daily_final.csv.  Returns DataFrame with [ticker, weight] columns.
    Normalizes weights to sum to 1.
    """
    if not PORTFOLIO_CSV.exists():
        print(f"  [warn] {PORTFOLIO_CSV.name} not found — returning empty portfolio")
        return pd.DataFrame(columns=["ticker", "weight"])
    df = pd.read_csv(PORTFOLIO_CSV)
    # Detect weight column
    weight_col = None
    for candidate in ["weight_pct", "weight", "current_weight"]:
        if candidate in df.columns:
            weight_col = candidate
            break
    if weight_col is None:
        print("  [warn] No weight column found in daily_final.csv")
        return pd.DataFrame(columns=["ticker", "weight"])
    if "ticker" not in df.columns:
        print("  [warn] No ticker column in daily_final.csv")
        return pd.DataFrame(columns=["ticker", "weight"])
    out = df[["ticker", weight_col]].copy()
    out = out.rename(columns={weight_col: "weight"})
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    # If weights look like percentages (e.g. 4.49 for 4.49%), normalize to fractions
    if out["weight"].max() > 2.0:
        out["weight"] = out["weight"] / 100.0
    total = out["weight"].sum()
    if total > 0:
        out["weight"] = out["weight"] / total
    return out.reset_index(drop=True)


def load_trade_journal() -> pd.DataFrame | None:
    """Load trade_journal.csv if it exists.  Returns None otherwise."""
    if not TRADE_JOURNAL_CSV.exists():
        # Also try canyon_trade_journal.csv
        alt = ROOT / "canyon_trade_journal.csv"
        if alt.exists():
            return pd.read_csv(alt)
        return None
    return pd.read_csv(TRADE_JOURNAL_CSV)


def load_alpha_scores() -> pd.DataFrame:
    """Load alpha_scores.csv for position context."""
    if ALPHA_SCORES_CSV.exists():
        return pd.read_csv(ALPHA_SCORES_CSV)
    return pd.DataFrame()


# =============================================================================
# Kelly estimation from trade journal
# =============================================================================

def estimate_kelly_by_strategy(journal: pd.DataFrame) -> dict[str, float]:
    """
    Compute Kelly fraction per strategy_type from trade_journal history.

    Expected columns: strategy_type (or signal_grade), pnl_pct (or pnl / return_pct)
    Returns dict: strategy_type -> half_kelly fraction
    """
    result: dict[str, float] = {}

    # Normalise column names
    df = journal.copy()
    # strategy_type column
    strat_col = None
    for c in ["strategy_type", "signal_grade", "strategy", "type"]:
        if c in df.columns:
            strat_col = c
            break
    # pnl column
    pnl_col = None
    for c in ["pnl_pct", "return_pct", "pnl", "return"]:
        if c in df.columns:
            pnl_col = c
            break
    if strat_col is None or pnl_col is None:
        return result

    df[pnl_col] = pd.to_numeric(df[pnl_col], errors="coerce")
    df = df.dropna(subset=[pnl_col])
    if len(df) == 0:
        return result

    for strat, grp in df.groupby(strat_col):
        pnls = grp[pnl_col].values
        if len(pnls) < 5:
            continue
        wins  = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0.5
        avg_win  = float(wins.mean())   if len(wins)   > 0 else 0.01
        avg_loss = float(-losses.mean()) if len(losses) > 0 else 0.01
        kf = kelly_fraction(win_rate, avg_win, avg_loss)
        result[str(strat)] = kf

    return result


# =============================================================================
# Marginal VaR / variance contribution
# =============================================================================

def compute_marginal_var(
    weights: dict,
    returns_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    Marginal VaR approximation using covariance:
        marginal_var_i = w_i * sum_j(cov(r_i, r_j) * w_j) / portfolio_vol
                       = w_i * (Sigma @ w)_i / portfolio_vol

    Returns (marginal_var_series, var_contrib_pct_series) both indexed by ticker.
    marginal_var:      dollar-vol contribution per unit weight (positive = risk contribution)
    var_contrib_pct:   fraction of total portfolio variance attributed to each position
    """
    tickers = [t for t in weights if t in returns_df.columns]
    if not tickers:
        empty = pd.Series(dtype=float)
        return empty, empty
    w = np.array([weights[t] for t in tickers])
    r = returns_df[tickers].dropna(how="all").fillna(0.0)
    cov = r.cov().values
    port_var = float(w @ cov @ w)
    port_vol = np.sqrt(port_var) if port_var > 0 else 1e-9
    # Component variance contribution = w_i * (Sigma @ w)_i
    contributions = w * (cov @ w)
    # Normalise to portfolio variance so they sum to 1
    var_pct = contributions / port_var if port_var > 0 else contributions
    marginal = contributions / port_vol  # marginal VaR (dollar-vol per unit weight)
    return pd.Series(
        {
            tickers[i]: float(marginal[i])
            for i in range(len(tickers))
        }
    ), pd.Series(
        {
            tickers[i]: float(var_pct[i])
            for i in range(len(tickers))
        }
    )


# =============================================================================
# Risk flags
# =============================================================================

def compute_risk_flags(
    report_df: pd.DataFrame,
    corr_mat: pd.DataFrame,
    hhi: float,
    var_1d: float,
    confidence: float,
) -> list[str]:
    """Collect all risk flags and return as list of strings."""
    flags: list[str] = []

    # 1. Overweight vs Kelly
    for _, row in report_df.iterrows():
        if (
            pd.notna(row.get("kelly_weight"))
            and row["kelly_weight"] > 0
            and pd.notna(row.get("current_weight"))
            and row["current_weight"] > row["kelly_weight"] * KELLY_OVERWEIGHT_MULTIPLIER
        ):
            flags.append(
                f"OVERWEIGHT vs Kelly  — {row['ticker']}: "
                f"current={row['current_weight']:.3f}  kelly={row['kelly_weight']:.3f}"
            )

    # 2. Concentration
    if hhi > HHI_CONCENTRATION_THRESHOLD:
        flags.append(f"CONCENTRATION RISK  — HHI={hhi:.4f} (>{HHI_CONCENTRATION_THRESHOLD})")

    # 3. High VaR
    if pd.notna(var_1d) and var_1d > VAR_HIGH_THRESHOLD:
        flags.append(
            f"HIGH VAR  — 1-day {int(confidence*100)}% VaR = {var_1d:.3%} (>{VAR_HIGH_THRESHOLD:.1%})"
        )

    # 4. Correlated pairs in top-10
    if not corr_mat.empty:
        tickers = corr_mat.columns.tolist()
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                c = corr_mat.iloc[i, j]
                if pd.notna(c) and abs(c) > CORR_HIGH_THRESHOLD:
                    flags.append(
                        f"CORRELATED PAIR  — {tickers[i]} / {tickers[j]}: corr={c:.3f}"
                    )

    return flags


# =============================================================================
# Main report builder
# =============================================================================

def run_risk_report(
    top_n: int = 20,
    confidence: float = 0.95,
    years: int = 2,
) -> int:
    """
    Build risk_report.csv and risk_report.md.

    Returns 0 on success, 1 on error.
    """
    total_steps = 9
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*62}")
    print(f"  Canyon v9 — Step 93 Risk Manager     {run_ts}")
    print(f"{'='*62}\n")

    # ── [1/9] Load price history ──────────────────────────────────────────────
    print(f"[1/{total_steps}] Loading price cache ({years}y)  …")
    prices = load_prices(years=years)
    if prices.empty:
        print("  [ERROR] No price data available.  Aborting.")
        return 1
    print(f"         {prices.shape[1]} tickers  |  {len(prices)} trading days")

    # ── [2/9] Compute daily log returns ──────────────────────────────────────
    print(f"[2/{total_steps}] Computing daily log returns  …")
    returns = np.log(prices / prices.shift(1)).iloc[1:]
    returns = returns.replace([np.inf, -np.inf], np.nan)

    # ── [3/9] Load portfolio weights ──────────────────────────────────────────
    print(f"[3/{total_steps}] Loading portfolio from {PORTFOLIO_CSV.name}  …")
    portfolio = load_portfolio()
    if portfolio.empty:
        print("  [ERROR] Portfolio is empty.  Aborting.")
        return 1
    # Restrict to top_n positions by weight
    portfolio = portfolio.sort_values("weight", ascending=False).head(top_n)
    weights_raw = dict(zip(portfolio["ticker"], portfolio["weight"]))
    total_w = sum(weights_raw.values())
    weights: dict[str, float] = {k: v / total_w for k, v in weights_raw.items()} if total_w > 0 else weights_raw
    print(f"         {len(weights)} positions  |  total weight = {total_w:.4f}")

    # ── [4/9] VaR and CVaR ───────────────────────────────────────────────────
    print(f"[4/{total_steps}] Computing VaR / CVaR (confidence={confidence:.0%})  …")
    var_1d   = portfolio_historical_var(weights, returns, confidence)
    cvar_1d  = portfolio_cvar(weights, returns, confidence)
    var_ann  = var_1d * np.sqrt(ANN_FACTOR) if pd.notna(var_1d) else np.nan
    print(f"         1-day VaR  = {var_1d:.4%}")
    print(f"         1-day CVaR = {cvar_1d:.4%}")
    print(f"         Ann. VaR   = {var_ann:.4%}")

    # ── [5/9] Per-ticker marginal VaR ─────────────────────────────────────────
    print(f"[5/{total_steps}] Computing marginal VaR per ticker  …")
    marginal_var_series, var_contrib_pct_series = compute_marginal_var(weights, returns)

    # ── [6/9] HHI concentration ──────────────────────────────────────────────
    print(f"[6/{total_steps}] Computing HHI concentration  …")
    hhi = herfindahl_index(weights)
    print(f"         HHI = {hhi:.4f}  ({'CONCENTRATED' if hhi > HHI_CONCENTRATION_THRESHOLD else 'OK'})")

    # ── [7/9] Kelly estimation ───────────────────────────────────────────────
    print(f"[7/{total_steps}] Estimating Kelly fractions  …")
    journal = load_trade_journal()
    alpha   = load_alpha_scores()

    kelly_by_strategy: dict[str, float] = {}
    if journal is not None and not journal.empty:
        kelly_by_strategy = estimate_kelly_by_strategy(journal)
        print(f"         Trade journal loaded: {len(journal)} rows  "
              f"|  {len(kelly_by_strategy)} strategies with Kelly estimates")
    else:
        print(f"         trade_journal.csv not found — using default Kelly = {DEFAULT_KELLY:.2f}")

    # Map tickers to strategies via alpha_scores if available
    ticker_strategy: dict[str, str] = {}
    if not alpha.empty and "ticker" in alpha.columns:
        for col in ["strategy_type", "signal", "signal_grade", "top_signal"]:
            if col in alpha.columns:
                ticker_strategy = dict(zip(alpha["ticker"], alpha[col].astype(str)))
                break

    # Determine kelly_weight per ticker
    def _kelly_for_ticker(ticker: str) -> float:
        strat = ticker_strategy.get(ticker, "")
        if strat and strat in kelly_by_strategy:
            return kelly_by_strategy[strat]
        if kelly_by_strategy:
            # Use mean of all strategies as fallback
            return float(np.mean(list(kelly_by_strategy.values())))
        return DEFAULT_KELLY

    # ── [8/9] Correlation matrix (top 10) ────────────────────────────────────
    print(f"[8/{total_steps}] Computing correlation matrix (top 10 holdings)  …")
    top10 = list(weights.keys())[:10]
    top10_in_returns = [t for t in top10 if t in returns.columns]
    corr_mat = pd.DataFrame()
    if top10_in_returns:
        corr_mat = correlation_matrix(returns[top10_in_returns])

    # ── [9/9] Build outputs ───────────────────────────────────────────────────
    print(f"[9/{total_steps}] Building risk_report.csv and risk_report.md  …")

    rows = []
    for ticker, w in weights.items():
        kw        = _kelly_for_ticker(ticker)
        kelly_max = min(kw * 2.0, 0.25)   # max = 2× half-Kelly, capped at 25%
        mvc       = float(marginal_var_series.get(ticker, np.nan))
        vcp       = float(var_contrib_pct_series.get(ticker, np.nan))
        rows.append({
            "ticker":         ticker,
            "current_weight": round(w, 6),
            "kelly_weight":   round(kw, 6),
            "kelly_max":      round(kelly_max, 6),
            "var_contrib_pct": round(vcp * 100, 4) if pd.notna(vcp) else np.nan,
            "marginal_var":   round(mvc, 8) if pd.notna(mvc) else np.nan,
        })

    report_df = pd.DataFrame(rows)
    report_df.to_csv(OUT_CSV, index=False)
    print(f"         Saved → {OUT_CSV.name}")

    # Risk flags
    flags = compute_risk_flags(report_df, corr_mat, hhi, var_1d, confidence)

    # ── Markdown report ────────────────────────────────────────────────────────
    md_lines: list[str] = []
    md_lines.append(f"# Canyon v9 — Risk Report  ({run_ts})\n")
    md_lines.append("## Portfolio Summary\n")
    md_lines.append(f"| Metric | Value |")
    md_lines.append(f"|--------|-------|")
    md_lines.append(f"| 1-day VaR ({int(confidence*100)}%) | {var_1d:.4%} |")
    md_lines.append(f"| 1-day CVaR ({int(confidence*100)}%) | {cvar_1d:.4%} |")
    md_lines.append(f"| Annualised VaR | {var_ann:.4%} |")
    md_lines.append(f"| HHI (concentration) | {hhi:.4f} |")
    md_lines.append(f"| Positions analysed | {len(weights)} |")
    md_lines.append(f"| Lookback window | {years} years |")
    md_lines.append("")

    # Risk flags section
    md_lines.append("## Risk Flags\n")
    if flags:
        for f in flags:
            md_lines.append(f"- **{f}**")
    else:
        md_lines.append("_No risk flags triggered._")
    md_lines.append("")

    # Per-ticker table
    md_lines.append("## Per-Ticker Risk Table\n")
    md_lines.append(
        "| Ticker | Weight | Kelly Wt | Kelly Max | VaR Contrib % | Marginal VaR | Status |"
    )
    md_lines.append(
        "|--------|--------|----------|-----------|--------------|-------------|--------|"
    )
    for _, row in report_df.iterrows():
        ticker  = row["ticker"]
        cw      = f"{row['current_weight']:.3%}"
        kw      = f"{row['kelly_weight']:.3%}"
        km      = f"{row['kelly_max']:.3%}"
        vcp_str = f"{row['var_contrib_pct']:.2f}%" if pd.notna(row['var_contrib_pct']) else "—"
        mvc_str = f"{row['marginal_var']:.6f}"  if pd.notna(row['marginal_var'])   else "—"
        # Flag status
        status_parts = []
        if (
            pd.notna(row["kelly_weight"])
            and row["kelly_weight"] > 0
            and row["current_weight"] > row["kelly_weight"] * KELLY_OVERWEIGHT_MULTIPLIER
        ):
            status_parts.append("OW")
        if not status_parts:
            status_parts.append("OK")
        status = " / ".join(status_parts)
        md_lines.append(f"| {ticker} | {cw} | {kw} | {km} | {vcp_str} | {mvc_str} | {status} |")
    md_lines.append("")

    # Correlation matrix (top 10)
    if not corr_mat.empty:
        md_lines.append("## Correlation Matrix (Top 10 Holdings, 30-day)\n")
        cols = corr_mat.columns.tolist()
        md_lines.append("| | " + " | ".join(cols) + " |")
        md_lines.append("|---" * (len(cols) + 1) + "|")
        for row_ticker in cols:
            vals = " | ".join(
                f"{corr_mat.loc[row_ticker, c]:.2f}" if pd.notna(corr_mat.loc[row_ticker, c]) else "—"
                for c in cols
            )
            md_lines.append(f"| {row_ticker} | {vals} |")
        md_lines.append("")

    # Kelly vs current weight comparison
    md_lines.append("## Kelly vs Current Weight Comparison\n")
    md_lines.append("Positions where current weight exceeds 1.5× Kelly are flagged as OVERWEIGHT.\n")
    md_lines.append("| Ticker | Current | Kelly | Ratio | Flag |")
    md_lines.append("|--------|---------|-------|-------|------|")
    for _, row in report_df.iterrows():
        cw   = row["current_weight"]
        kw   = row["kelly_weight"]
        ratio = cw / kw if kw > 0 else float("inf")
        flag  = "OVERWEIGHT" if ratio > KELLY_OVERWEIGHT_MULTIPLIER else ""
        md_lines.append(
            f"| {row['ticker']} | {cw:.3%} | {kw:.3%} | {ratio:.2f}x | {flag} |"
        )
    md_lines.append("")

    # Footer
    md_lines.append("---")
    md_lines.append(f"_Generated by Canyon v9 Step 93 Risk Manager — {run_ts}_")

    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"         Saved → {OUT_MD.name}")

    # ── Print summary box ──────────────────────────────────────────────────────
    box_width = 62
    print(f"\n{'─'*box_width}")
    print(f"  RISK SUMMARY")
    print(f"{'─'*box_width}")
    print(f"  1-day VaR  ({int(confidence*100)}%)  : {var_1d:.4%}")
    print(f"  1-day CVaR ({int(confidence*100)}%)  : {cvar_1d:.4%}")
    print(f"  Annual VaR          : {var_ann:.4%}")
    print(f"  HHI                 : {hhi:.4f}")
    print(f"  Positions           : {len(weights)}")
    print(f"  Lookback            : {years} years  |  {len(returns)} days")
    print(f"{'─'*box_width}")
    if flags:
        print(f"  RISK FLAGS ({len(flags)})")
        for fl in flags:
            print(f"    ! {fl}")
    else:
        print("  No risk flags triggered.")
    print(f"{'─'*box_width}")
    print(f"  Outputs: {OUT_CSV.name}   {OUT_MD.name}")
    print(f"{'─'*box_width}\n")

    return 0


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canyon v9 Step 93 — Portfolio Risk Manager"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="Number of top positions to analyse (default: 20)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="VaR / CVaR confidence level (default: 0.95)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=2,
        help="Lookback window in years for price history (default: 2)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rc = run_risk_report(
            top_n=args.top,
            confidence=args.confidence,
            years=args.years,
        )
        return rc
    except Exception as exc:
        print(f"\n[ERROR] Unhandled exception: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
