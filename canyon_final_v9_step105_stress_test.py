#!/usr/bin/env python3
"""
canyon_final_v9_step105_stress_test.py
=======================================
Canyon v9 — Step 105: Portfolio Stress Test Engine

Estimates portfolio P&L under predefined stress scenarios using
historical beta (OLS, last 252 days) and sector correlations.
No scipy.  pandas + numpy only.

Inputs:
  daily_final.csv      — current portfolio (ticker, weight_pct or weight)
  sp500_price_cache.csv — wide price history (Date index, ticker columns)
  sector_map.csv       — ticker→sector mapping

Outputs:
  stress_test_results.csv   — scenario × metric table
  stress_test_report.md     — summary + per-scenario detail + portfolio chars

Usage:
  python canyon_final_v9_step105_stress_test.py
  python canyon_final_v9_step105_stress_test.py --portfolio-value 500000
  python canyon_final_v9_step105_stress_test.py --scenarios all --portfolio-value 1000000
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# File paths
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO_CSV   = ROOT / "daily_final.csv"
PRICE_CACHE_CSV = ROOT / "sp500_price_cache.csv"
SECTOR_MAP_CSV  = ROOT / "sector_map.csv"

OUT_CSV = ROOT / "stress_test_results.csv"
OUT_MD  = ROOT / "stress_test_report.md"

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

LOOKBACK          = 252          # trading days for beta computation
BETA_CLIP_LOW     = -2.0
BETA_CLIP_HIGH    = 4.0
SPY_TICKER        = "SPY"
DEFAULT_BETA      = 1.0          # fallback when price history is unavailable
DEFAULT_PF_VALUE  = 100_000      # USD

# ─────────────────────────────────────────────────────────────────────────────
# Stress scenarios
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "Market Correction -10%",
        "spy_shock": -0.10,
        "vix_spike": 25,
        "sector_shocks": {},
        "description": "Typical market correction of 10%",
    },
    {
        "name": "Bear Market -20%",
        "spy_shock": -0.20,
        "vix_spike": 35,
        "sector_shocks": {
            "Technology": -0.05,
            "Consumer Discretionary": -0.05,
        },
        "description": "Bear market decline 20% + additional pressure on tech stocks",
    },
    {
        "name": "VIX Spike to 40 (Crisis)",
        "spy_shock": -0.15,
        "vix_spike": 40,
        "sector_shocks": {
            "Financials": -0.08,
            "Real Estate": -0.06,
        },
        "description": "VIX surges to 40, financial crisis scenario",
    },
    {
        "name": "Tech Sector Selloff -20%",
        "spy_shock": -0.05,
        "vix_spike": 22,
        "sector_shocks": {
            "Technology": -0.20,
            "Communication Services": -0.12,
        },
        "description": "Tech sector standalone selloff -20%",
    },
    {
        "name": "Rate Shock +100bps",
        "spy_shock": -0.06,
        "vix_spike": 20,
        "sector_shocks": {
            "Real Estate": -0.10,
            "Utilities": -0.08,
            "Financials": +0.04,
        },
        "description": "Sudden interest rate rise of 100 basis points",
    },
    {
        "name": "Soft Landing Rally +10%",
        "spy_shock": +0.10,
        "vix_spike": 13,
        "sector_shocks": {
            "Technology": +0.05,
            "Financials": +0.03,
        },
        "description": "Soft landing, market rally 10%",
    },
]


# =============================================================================
# [1/4] Data loading helpers
# =============================================================================

def load_portfolio() -> pd.DataFrame:
    """
    Load daily_final.csv and return a DataFrame with columns:
      ticker, weight   (weight normalised to sum=1)

    Handles both weight_pct (0-100 scale) and weight (0-1 scale).
    Keeps only rows with action containing BUY or STEADY (skips SELL / REDUCE).
    """
    if not PORTFOLIO_CSV.exists():
        print(f"  [ERROR] Portfolio file not found: {PORTFOLIO_CSV}")
        sys.exit(1)

    df = pd.read_csv(PORTFOLIO_CSV)

    # ── Resolve weight column ──────────────────────────────────────────────
    if "weight_pct" in df.columns:
        df["weight"] = pd.to_numeric(df["weight_pct"], errors="coerce") / 100.0
    elif "weight" in df.columns:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    else:
        print("  [ERROR] daily_final.csv must have 'weight_pct' or 'weight' column.")
        sys.exit(1)

    # ── Filter to active long positions ───────────────────────────────────
    if "action" in df.columns:
        mask = df["action"].str.upper().str.contains("BUY|STEADY|HOLD", na=False)
        df = df[mask].copy()

    df = df.dropna(subset=["ticker", "weight"])
    df["ticker"] = df["ticker"].str.upper().str.strip()
    df = df[df["weight"] > 0].copy()

    if df.empty:
        print("  [ERROR] No active long positions found in daily_final.csv.")
        sys.exit(1)

    # ── Normalise weights to sum = 1 ──────────────────────────────────────
    total = df["weight"].sum()
    df["weight"] = df["weight"] / total

    return df[["ticker", "weight"]].reset_index(drop=True)


def load_prices() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load sp500_price_cache.csv (wide format: Date index, ticker columns).
    Returns (prices_df, spy_returns).
    prices_df  — log returns, shape (dates, tickers)
    spy_returns — SPY log return series (same index)
    """
    if not PRICE_CACHE_CSV.exists():
        print(f"  [ERROR] Price cache not found: {PRICE_CACHE_CSV}")
        sys.exit(1)

    raw = pd.read_csv(PRICE_CACHE_CSV, index_col=0, parse_dates=True)
    raw = raw.sort_index()

    # compute log returns; drop first row (NaN)
    log_ret = np.log(raw / raw.shift(1)).iloc[1:]
    log_ret = log_ret.replace([np.inf, -np.inf], np.nan)

    if SPY_TICKER not in log_ret.columns:
        print(f"  [WARN] SPY not found in price cache — using market proxy (mean of all tickers).")
        spy_ret = log_ret.mean(axis=1)
    else:
        spy_ret = log_ret[SPY_TICKER].dropna()

    return log_ret, spy_ret


def load_sector_map(portfolio_tickers: list[str]) -> dict[str, str]:
    """
    Load sector_map.csv.  Falls back to daily_final.csv 'sector' column,
    then to 'Unknown' if neither source has the mapping.
    Returns {ticker: sector}.
    """
    sector: dict[str, str] = {}

    if SECTOR_MAP_CSV.exists():
        sm = pd.read_csv(SECTOR_MAP_CSV)
        sm.columns = [c.lower().strip() for c in sm.columns]
        if "ticker" in sm.columns and "sector" in sm.columns:
            for _, row in sm.iterrows():
                t = str(row["ticker"]).upper().strip()
                sector[t] = str(row["sector"]).strip()

    # supplement from daily_final.csv sector column
    if PORTFOLIO_CSV.exists():
        pf = pd.read_csv(PORTFOLIO_CSV)
        if "sector" in pf.columns and "ticker" in pf.columns:
            for _, row in pf.iterrows():
                t = str(row["ticker"]).upper().strip()
                if t not in sector and pd.notna(row.get("sector")):
                    sector[t] = str(row["sector"]).strip()

    # fill missing
    for t in portfolio_tickers:
        if t not in sector:
            sector[t] = "Unknown"

    return sector


# =============================================================================
# [2/4] Beta computation and position shock
# =============================================================================

def compute_beta(
    ticker_returns: pd.Series,
    spy_returns: pd.Series,
    lookback: int = LOOKBACK,
) -> float:
    """
    OLS beta = cov(ticker, SPY) / var(SPY) over last `lookback` days.
    Pure numpy: np.cov()[0,1] / np.var(SPY, ddof=1).
    Beta is clipped to [BETA_CLIP_LOW, BETA_CLIP_HIGH].
    Returns DEFAULT_BETA on insufficient data.
    """
    # align on common index, take last `lookback` rows
    combined = pd.concat([ticker_returns, spy_returns], axis=1).dropna()
    combined.columns = ["ticker", "spy"]

    if len(combined) < 30:
        return DEFAULT_BETA

    window = combined.tail(lookback)
    t_arr = window["ticker"].to_numpy(dtype=float)
    s_arr = window["spy"].to_numpy(dtype=float)

    spy_var = float(np.var(s_arr, ddof=1))
    if spy_var == 0.0:
        return DEFAULT_BETA

    cov_mat = np.cov(t_arr, s_arr, ddof=1)
    cov_ts  = float(cov_mat[0, 1])
    beta    = cov_ts / spy_var

    return float(np.clip(beta, BETA_CLIP_LOW, BETA_CLIP_HIGH))


def estimate_position_shock(
    ticker: str,
    weight: float,
    beta: float,
    sector: str,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    """
    Estimated P&L = weight * (beta * spy_shock + sector_shocks.get(sector, 0))

    Returns dict with keys:
      ticker, weight, beta, sector,
      spy_component, sector_component, total_shock_pct
    """
    spy_shock    = scenario["spy_shock"]
    sector_extra = scenario.get("sector_shocks", {}).get(sector, 0.0)

    spy_component    = weight * beta * spy_shock
    sector_component = weight * sector_extra
    total_shock_pct  = spy_component + sector_component

    return {
        "ticker":           ticker,
        "weight":           weight,
        "beta":             beta,
        "sector":           sector,
        "spy_component":    spy_component,
        "sector_component": sector_component,
        "total_shock_pct":  total_shock_pct,
    }


# =============================================================================
# [3/4] Scenario runner and portfolio characteristics
# =============================================================================

def run_scenario(
    portfolio: pd.DataFrame,
    betas: dict[str, float],
    sectors: dict[str, str],
    scenario: dict[str, Any],
    portfolio_value: float = DEFAULT_PF_VALUE,
) -> dict[str, Any]:
    """
    Run one scenario across the full portfolio.

    Returns dict with keys:
      scenario_name, portfolio_shock_pct, portfolio_shock_dollar,
      worst_3_positions, best_3_positions,
      sector_breakdown: {sector: total_shock_pct}
    """
    position_results: list[dict[str, Any]] = []

    for _, row in portfolio.iterrows():
        ticker = row["ticker"]
        weight = row["weight"]
        beta   = betas.get(ticker, DEFAULT_BETA)
        sector = sectors.get(ticker, "Unknown")

        pos = estimate_position_shock(ticker, weight, beta, sector, scenario)
        position_results.append(pos)

    df_pos = pd.DataFrame(position_results)

    portfolio_shock_pct    = float(df_pos["total_shock_pct"].sum())
    portfolio_shock_dollar = portfolio_shock_pct * portfolio_value

    # sort by total_shock_pct for worst/best
    df_sorted = df_pos.sort_values("total_shock_pct")

    def _fmt_pos(r: pd.Series) -> str:
        return f"{r['ticker']} ({r['total_shock_pct']*100:+.2f}%)"

    worst_3 = [_fmt_pos(r) for _, r in df_sorted.head(3).iterrows()]
    best_3  = [_fmt_pos(r) for _, r in df_sorted.tail(3).iloc[::-1].iterrows()]

    # sector breakdown: sum of total_shock_pct per sector
    sector_breakdown: dict[str, float] = (
        df_pos.groupby("sector")["total_shock_pct"].sum().to_dict()
    )

    return {
        "scenario_name":          scenario["name"],
        "description":            scenario.get("description", ""),
        "spy_shock":              scenario["spy_shock"],
        "vix_spike":              scenario["vix_spike"],
        "portfolio_shock_pct":    portfolio_shock_pct,
        "portfolio_shock_dollar": portfolio_shock_dollar,
        "worst_3_positions":      worst_3,
        "best_3_positions":       best_3,
        "sector_breakdown":       sector_breakdown,
        "position_detail":        df_pos,
    }


def compute_portfolio_characteristics(
    portfolio: pd.DataFrame,
    betas: dict[str, float],
    sectors: dict[str, str],
) -> dict[str, Any]:
    """
    Compute summary stats for the portfolio:
      - weighted_beta
      - sector_weights: {sector: sum_weight}
      - max_position_weight
      - top3_sector_concentration: [(sector, weight), ...]
    """
    portfolio = portfolio.copy()
    portfolio["beta"]   = portfolio["ticker"].map(betas).fillna(DEFAULT_BETA)
    portfolio["sector"] = portfolio["ticker"].map(sectors).fillna("Unknown")

    weighted_beta = float((portfolio["weight"] * portfolio["beta"]).sum())
    max_pos_weight = float(portfolio["weight"].max())

    sector_weights: dict[str, float] = (
        portfolio.groupby("sector")["weight"].sum()
        .sort_values(ascending=False)
        .to_dict()
    )

    top3_sector = list(sector_weights.items())[:3]

    return {
        "weighted_beta":           weighted_beta,
        "sector_weights":          sector_weights,
        "max_position_weight":     max_pos_weight,
        "top3_sector_concentration": top3_sector,
        "n_positions":             len(portfolio),
    }


# =============================================================================
# [4/4] Report writers and main
# =============================================================================

def _pct(v: float, decimals: int = 2) -> str:
    return f"{v*100:+.{decimals}f}%"


def write_csv(results: list[dict[str, Any]]) -> None:
    rows = []
    for r in results:
        # flatten worst/best into strings
        worst_str = " | ".join(r["worst_3_positions"]) if r["worst_3_positions"] else ""
        best_str  = " | ".join(r["best_3_positions"])  if r["best_3_positions"]  else ""

        # identify sector at highest risk (most negative shock)
        sb = r["sector_breakdown"]
        if sb:
            sector_at_risk = min(sb, key=sb.get)
        else:
            sector_at_risk = "N/A"

        rows.append({
            "scenario":              r["scenario_name"],
            "description":           r["description"],
            "spy_shock_pct":         r["spy_shock"] * 100,
            "vix_level":             r["vix_spike"],
            "portfolio_return_pct":  r["portfolio_shock_pct"] * 100,
            "portfolio_dollar_pnl":  r["portfolio_shock_dollar"],
            "sector_at_risk":        sector_at_risk,
            "worst_positions":       worst_str,
            "best_positions":        best_str,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"  Saved → {OUT_CSV.name}")


def write_report(
    results: list[dict[str, Any]],
    chars: dict[str, Any],
    portfolio_value: float,
    run_ts: str,
) -> None:
    lines: list[str] = []

    lines.append(f"# Canyon v9 — Step 105: Portfolio Stress Test Report")
    lines.append(f"*Generated: {run_ts} | Portfolio value: ${portfolio_value:,.0f}*\n")

    # ── Portfolio characteristics ────────────────────────────────────────
    lines.append("## Portfolio Characteristics\n")
    lines.append(f"- **Positions**: {chars['n_positions']}")
    lines.append(f"- **Weighted beta**: {chars['weighted_beta']:.3f}")
    lines.append(f"- **Max single position**: {chars['max_position_weight']*100:.1f}%")

    lines.append("\n**Top 3 Sector Concentrations:**\n")
    lines.append("| Sector | Weight |")
    lines.append("|--------|--------|")
    for sec, wt in chars["top3_sector_concentration"]:
        lines.append(f"| {sec} | {wt*100:.1f}% |")

    lines.append("\n**All Sector Weights:**\n")
    lines.append("| Sector | Weight |")
    lines.append("|--------|--------|")
    for sec, wt in chars["sector_weights"].items():
        lines.append(f"| {sec} | {wt*100:.1f}% |")

    # ── Summary table ────────────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Scenario Summary\n")
    lines.append(
        "| Scenario | SPY Shock | VIX Level | Est. Portfolio Return | "
        "Dollar P&L | Sector at Risk |"
    )
    lines.append("|----------|-----------|-----------|----------------------|"
                 "------------|----------------|")

    for r in results:
        sb = r["sector_breakdown"]
        sector_at_risk = min(sb, key=sb.get) if sb else "N/A"
        lines.append(
            f"| {r['scenario_name']} "
            f"| {_pct(r['spy_shock'])} "
            f"| {r['vix_spike']} "
            f"| **{_pct(r['portfolio_shock_pct'])}** "
            f"| ${r['portfolio_shock_dollar']:+,.0f} "
            f"| {sector_at_risk} |"
        )

    # ── Per-scenario detail ──────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Per-Scenario Detail\n")

    for r in results:
        lines.append(f"### {r['scenario_name']}")
        lines.append(f"*{r['description']}*\n")
        lines.append(f"- SPY shock: **{_pct(r['spy_shock'])}** | VIX target: **{r['vix_spike']}**")
        lines.append(f"- Portfolio estimated return: **{_pct(r['portfolio_shock_pct'])}**")
        lines.append(f"- Dollar P&L: **${r['portfolio_shock_dollar']:+,.0f}**\n")

        df_pos = r["position_detail"].sort_values("total_shock_pct")

        # Top 5 losers
        losers = df_pos.head(5)
        lines.append("**Top 5 Losers:**\n")
        lines.append("| Ticker | Weight | Beta | Sector | Est. Return |")
        lines.append("|--------|--------|------|--------|-------------|")
        for _, row in losers.iterrows():
            lines.append(
                f"| {row['ticker']} "
                f"| {row['weight']*100:.1f}% "
                f"| {row['beta']:.2f} "
                f"| {row['sector']} "
                f"| {_pct(row['total_shock_pct'])} |"
            )

        # Top 5 gainers
        gainers = df_pos.tail(5).iloc[::-1]
        lines.append("\n**Top 5 Gainers:**\n")
        lines.append("| Ticker | Weight | Beta | Sector | Est. Return |")
        lines.append("|--------|--------|------|--------|-------------|")
        for _, row in gainers.iterrows():
            lines.append(
                f"| {row['ticker']} "
                f"| {row['weight']*100:.1f}% "
                f"| {row['beta']:.2f} "
                f"| {row['sector']} "
                f"| {_pct(row['total_shock_pct'])} |"
            )

        # Sector breakdown
        lines.append("\n**Sector Breakdown:**\n")
        lines.append("| Sector | Est. Contribution |")
        lines.append("|--------|-------------------|")
        sb_sorted = sorted(r["sector_breakdown"].items(), key=lambda x: x[1])
        for sec, shock in sb_sorted:
            lines.append(f"| {sec} | {_pct(shock)} |")

        lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved → {OUT_MD.name}")


def print_comparison_table(results: list[dict[str, Any]], portfolio_value: float) -> None:
    """Print clean comparison table to stdout."""
    col_w = [30, 10, 8, 14, 14]
    headers = ["Scenario", "SPY Shock", "VIX", "Portf. Return", "Dollar P&L"]

    sep = "  " + "  ".join("-" * w for w in col_w)
    hdr = "  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_w))

    print(sep)
    print(hdr)
    print(sep)

    for r in results:
        cells = [
            r["scenario_name"][:col_w[0]].ljust(col_w[0]),
            _pct(r["spy_shock"]).ljust(col_w[1]),
            str(r["vix_spike"]).ljust(col_w[2]),
            _pct(r["portfolio_shock_pct"]).ljust(col_w[3]),
            f"${r['portfolio_shock_dollar']:+,.0f}".ljust(col_w[4]),
        ]
        print("  " + "  ".join(cells))

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_stress_test(portfolio_value: float = DEFAULT_PF_VALUE) -> None:
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*60}")
    print(f"  Canyon v9 — Step 105: Portfolio Stress Test Engine")
    print(f"  {run_ts}   Portfolio value: ${portfolio_value:,.0f}")
    print(f"{'='*60}\n")

    # ── [1/4] Load portfolio ───────────────────────────────────────────────
    print("[1/4] Loading portfolio and price history ...")
    portfolio = load_portfolio()
    tickers   = portfolio["ticker"].tolist()
    print(f"  Portfolio: {len(tickers)} positions | "
          f"Weight sum: {portfolio['weight'].sum():.4f}")

    # ── Price history & SPY returns ─────────────────────────────────────
    log_ret, spy_ret = load_prices()
    print(f"  Price history: {len(log_ret)} trading days × "
          f"{len(log_ret.columns)} tickers")

    # ── Sector map ──────────────────────────────────────────────────────
    sectors = load_sector_map(tickers)

    # ── [2/4] Compute beta for each ticker ───────────────────────────────
    print("\n[2/4] Computing betas (last 252 days OLS) ...")
    betas: dict[str, float] = {}
    missing_price = []

    for ticker in tickers:
        if ticker in log_ret.columns:
            betas[ticker] = compute_beta(log_ret[ticker], spy_ret)
        else:
            betas[ticker] = DEFAULT_BETA
            missing_price.append(ticker)

    if missing_price:
        print(f"  [WARN] {len(missing_price)} tickers without price data "
              f"(using beta={DEFAULT_BETA}): {', '.join(missing_price[:10])}"
              + (" ..." if len(missing_price) > 10 else ""))

    beta_vals = list(betas.values())
    print(f"  Beta range: [{min(beta_vals):.2f}, {max(beta_vals):.2f}] | "
          f"Median: {float(np.median(beta_vals)):.2f}")

    # ── Portfolio characteristics ─────────────────────────────────────
    chars = compute_portfolio_characteristics(portfolio, betas, sectors)
    print(f"  Weighted portfolio beta: {chars['weighted_beta']:.3f}")
    print(f"  Max position weight: {chars['max_position_weight']*100:.1f}%")
    top3 = ", ".join(f"{s} ({w*100:.0f}%)"
                     for s, w in chars["top3_sector_concentration"])
    print(f"  Top sectors: {top3}")

    # ── [3/4] Run all scenarios ───────────────────────────────────────────
    print(f"\n[3/4] Running {len(SCENARIOS)} stress scenarios ...")
    results: list[dict[str, Any]] = []

    for sc in SCENARIOS:
        r = run_scenario(portfolio, betas, sectors, sc, portfolio_value)
        results.append(r)
        sign = "+" if r["portfolio_shock_pct"] >= 0 else ""
        print(f"  {r['scenario_name']:<35} → "
              f"{sign}{r['portfolio_shock_pct']*100:.2f}%  "
              f"(${r['portfolio_shock_dollar']:+,.0f})")

    # ── [4/4] Write outputs ───────────────────────────────────────────────
    print(f"\n[4/4] Writing outputs ...")
    write_csv(results)
    write_report(results, chars, portfolio_value, run_ts)

    # ── Print comparison table ────────────────────────────────────────
    print("\n── Scenario Comparison Table ─────────────────────────────────")
    print_comparison_table(results, portfolio_value)

    print(f"\n  Done.  {OUT_CSV.name} | {OUT_MD.name}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canyon v9 Step 105 — Portfolio Stress Test Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenarios",
        default="all",
        choices=["all"],
        help="Which scenario set to run (default: all)",
    )
    parser.add_argument(
        "--portfolio-value",
        type=float,
        default=DEFAULT_PF_VALUE,
        metavar="USD",
        help=f"Total portfolio value in USD for dollar P&L estimates "
             f"(default: {DEFAULT_PF_VALUE:,})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_stress_test(portfolio_value=args.portfolio_value)
    sys.exit(0)


if __name__ == "__main__":
    main()
