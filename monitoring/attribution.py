"""
W47: Performance Attribution Daily Report
==========================================
Decomposes daily portfolio P&L into:
  1. Factor attribution  — P&L from Barra factor exposures
  2. Sector attribution  — P&L from sector over/underweights
  3. Signal attribution  — which signals contributed to today's winners/losers
  4. Stock selection     — residual alpha beyond factor/sector effects

Brinson-Hood-Beebower (1986) attribution:
  Total Active Return = Allocation Effect + Selection Effect + Interaction Effect
  where benchmark = SPY (or equal-weight S&P 500)

Outputs:
  attribution_daily.csv    — daily factor/sector/signal attribution
  attribution_history.csv  — rolling attribution since inception

Usage:
    from monitoring.attribution import run_attribution_report
    run_attribution_report()
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent

BENCHMARK = "SPY"  # benchmark for active return attribution


def _load_portfolio_weights() -> pd.Series:
    """Load current (yesterday's) portfolio weights."""
    for fname in ("daily_picks.csv", "bl_weights.csv"):
        p = ROOT / fname
        if p.exists():
            try:
                df = pd.read_csv(p)
                if "ticker" in df.columns and "weight" in df.columns:
                    w = df.set_index("ticker")["weight"].dropna()
                    return w[w > 0] / (w[w > 0].sum() + 1e-9)
            except Exception:
                pass
    return pd.Series(dtype=float)


def _load_yesterday_returns(prices: pd.DataFrame) -> pd.Series:
    """Get yesterday's single-day returns from price cache."""
    if len(prices) < 2:
        return pd.Series(dtype=float)
    return (prices.iloc[-1] / prices.iloc[-2] - 1).dropna()


def _load_sector_map() -> pd.Series:
    """Load ticker → sector mapping."""
    try:
        from data.sp500_constituents import build_constituent_history, get_sector_map
        history = build_constituent_history()
        return get_sector_map(history, pd.Timestamp.today())
    except Exception:
        pass
    # Fallback: from alpha_scores.csv
    p = ROOT / "alpha_scores.csv"
    if p.exists():
        df = pd.read_csv(p)
        if "ticker" in df.columns and "sector" in df.columns:
            return df.set_index("ticker")["sector"].dropna()
    return pd.Series(dtype=str)


def compute_factor_attribution(
    weights: pd.Series,
    daily_returns: pd.Series,
    factor_exposures: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Factor attribution using Barra factor exposures.

    Portfolio return = Σ_k (factor_exposure_k × factor_return_k) + specific_return
    """
    barra_path = ROOT / "barra_factor_exposures.csv"
    if factor_exposures is None and barra_path.exists():
        factor_exposures = pd.read_csv(barra_path, index_col=0)

    if factor_exposures is None or factor_exposures.empty:
        return {}

    common = weights.index.intersection(factor_exposures.index).intersection(daily_returns.index)
    if len(common) == 0:
        return {}

    w = weights[common].values
    B = factor_exposures.loc[common].values
    r = daily_returns[common].values

    # Portfolio factor exposures
    port_factor_exp = B.T @ w  # (K,)

    # Factor returns: estimated via cross-sectional regression of r on B
    try:
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression(fit_intercept=True)
        reg.fit(B, r)
        factor_returns = reg.coef_  # (K,)
    except Exception:
        return {}

    factor_names = factor_exposures.columns.tolist()
    attribution = {}
    for k, fname in enumerate(factor_names):
        attribution[f"factor_{fname}"] = round(float(port_factor_exp[k] * factor_returns[k]), 6)

    # Residual (specific return)
    explained  = float(port_factor_exp @ factor_returns)
    port_ret   = float(w @ r)
    attribution["specific_return"] = round(port_ret - explained, 6)
    attribution["total_return"]    = round(port_ret, 6)

    return attribution


def compute_sector_attribution(
    weights: pd.Series,
    daily_returns: pd.Series,
    sector_map: pd.Series,
    benchmark_weights: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Brinson-Hood-Beebower sector attribution.

    Active return = allocation + selection
    """
    common = weights.index.intersection(daily_returns.index)
    if len(common) == 0:
        return pd.DataFrame()

    df = pd.DataFrame({
        "weight":  weights.reindex(common).fillna(0.0),
        "return":  daily_returns.reindex(common).fillna(0.0),
        "sector":  sector_map.reindex(common).fillna("Unknown"),
    })

    # Equal-weight benchmark (simplified)
    if benchmark_weights is None:
        benchmark_weights = pd.Series(1.0 / len(common), index=common)

    df["bm_weight"] = benchmark_weights.reindex(common).fillna(1.0 / len(common))

    port_ret = float((df["weight"] * df["return"]).sum())
    bm_ret   = float((df["bm_weight"] * df["return"]).sum())

    rows = []
    for sector, group in df.groupby("sector"):
        port_w  = float(group["weight"].sum())
        bm_w    = float(group["bm_weight"].sum())
        sect_ret = float((group["return"] * group["weight"]).sum()) / (port_w + 1e-9)
        bm_ret_s = float((group["return"] * group["bm_weight"]).sum()) / (bm_w + 1e-9)

        allocation  = (port_w - bm_w) * (bm_ret_s - bm_ret)
        selection   = bm_w * (sect_ret - bm_ret_s)
        interaction = (port_w - bm_w) * (sect_ret - bm_ret_s)

        rows.append({
            "sector":       sector,
            "port_weight":  round(port_w, 4),
            "bm_weight":    round(bm_w, 4),
            "port_return":  round(sect_ret * 100, 4),
            "bm_return":    round(bm_ret_s * 100, 4),
            "allocation":   round(allocation * 100, 5),
            "selection":    round(selection * 100, 5),
            "interaction":  round(interaction * 100, 5),
            "active_return": round((allocation + selection + interaction) * 100, 5),
        })

    return pd.DataFrame(rows).sort_values("active_return", ascending=False)


def run_attribution_report(
    output_path: Optional[Path] = None,
    history_path: Optional[Path] = None,
) -> dict:
    """
    W47: Run daily attribution report.

    Returns dict with factor_attribution, sector_attribution, top_contributors.
    """
    if output_path  is None: output_path  = ROOT / "attribution_daily.csv"
    if history_path is None: history_path = ROOT / "attribution_history.csv"

    today_str = datetime.today().strftime("%Y-%m-%d")
    print(f"\n[Attribution] Performance attribution — {today_str}")

    # Load data
    weights = _load_portfolio_weights()
    if weights.empty:
        print("  No portfolio weights found")
        return {}

    for fname in ("sp500_price_cache_8yr.csv", "sp500_price_cache.csv"):
        p = ROOT / fname
        if p.exists():
            prices = pd.read_csv(p, index_col=0, parse_dates=True)
            break
    else:
        print("  No price cache found")
        return {}

    daily_returns = _load_yesterday_returns(prices)
    spy_ret       = float(daily_returns.get("SPY", 0.0))
    sector_map    = _load_sector_map()

    # Portfolio return
    common = weights.index.intersection(daily_returns.index)
    port_ret = float((weights[common] * daily_returns[common]).sum())
    active_ret = port_ret - spy_ret

    print(f"  Portfolio return: {port_ret:+.3%}")
    print(f"  SPY return:       {spy_ret:+.3%}")
    print(f"  Active return:    {active_ret:+.3%}")

    # Factor attribution
    factor_attr = compute_factor_attribution(weights, daily_returns)
    if factor_attr:
        print(f"\n  Factor attribution:")
        for k, v in sorted(factor_attr.items()):
            if abs(v) > 0.00005:
                print(f"    {k:<25}: {v*100:>+7.4f}bps")

    # Sector attribution
    sector_attr = compute_sector_attribution(weights, daily_returns, sector_map)
    if not sector_attr.empty:
        print(f"\n  Sector attribution (top 5):")
        for _, row in sector_attr.head(5).iterrows():
            print(f"    {row['sector']:<25}: active={row['active_return']:>+.4f}bps "
                  f"(alloc={row['allocation']:>+.4f}, sel={row['selection']:>+.4f})")

    # Top stock contributors
    stock_contrib = (weights[common] * daily_returns[common]).sort_values(ascending=False)
    print(f"\n  Top 5 contributors:")
    for tkr, contrib in stock_contrib.head(5).items():
        print(f"    {tkr:6s}: {contrib*100:>+.4f}%  "
              f"(ret={float(daily_returns.get(tkr, 0))*100:>+.2f}%)")

    print(f"\n  Bottom 5 detractors:")
    for tkr, contrib in stock_contrib.tail(5).items():
        print(f"    {tkr:6s}: {contrib*100:>+.4f}%  "
              f"(ret={float(daily_returns.get(tkr, 0))*100:>+.2f}%)")

    # Save
    summary = {
        "date":       today_str,
        "port_return": round(port_ret, 6),
        "spy_return":  round(spy_ret, 6),
        "active_return": round(active_ret, 6),
        **{k: round(v, 6) for k, v in factor_attr.items() if isinstance(v, float)},
    }
    summary_df = pd.DataFrame([summary])
    if history_path.exists():
        existing = pd.read_csv(history_path)
        existing = existing[existing["date"] != today_str]
        summary_df = pd.concat([existing, summary_df], ignore_index=True)
    summary_df.to_csv(history_path, index=False)

    if not sector_attr.empty:
        sector_attr["date"] = today_str
        sector_attr.to_csv(output_path, index=False)

    print(f"  Saved → {history_path}, {output_path}")
    return {"factor_attribution": factor_attr, "sector_attribution": sector_attr}


if __name__ == "__main__":
    run_attribution_report()
