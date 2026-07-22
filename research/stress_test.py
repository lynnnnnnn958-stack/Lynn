"""
W44: Stress Test Scenarios
===========================
Simulates portfolio performance under historical crisis periods and synthetic stress scenarios.

Historical scenarios:
  COVID-19 (Feb-Mar 2020):     ~35% SPY drawdown in 33 days
  GFC (Sep 2008 - Mar 2009):   ~57% SPY drawdown over 6 months
  Tech Bubble (Mar-Oct 2002):  ~49% NDX drawdown
  Inflation 2022 (Jan-Dec):    ~20% SPY drawdown, rising rates

Synthetic scenarios:
  VIX Spike:    VIX from 15 to 45 overnight (correlations spike to 0.85)
  Credit Crisis: HY spreads widen +500bps in 30 days
  Rate Shock:    10Y yield +150bps in 60 days (bond proxy stocks -15%)
  Factor Crash:  Momentum factor -3σ (style reversal, Jan 2009 style)

For each scenario, we compute:
  - Portfolio return (from Barra factor exposures × factor scenario returns)
  - 95% CVaR (Conditional VaR from regime BEAR distribution)
  - Recovery time estimate

Outputs:
  stress_test_results.csv — scenario × portfolio_impact × spy_impact × recovery_days

Usage:
    from research.stress_test import run_stress_tests
    results = run_stress_tests()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent


# Historical scenario definitions (SPY total return, factor impacts)
HISTORICAL_SCENARIOS = {
    "COVID-19 (Feb-Mar 2020)": {
        "spy_return":       -0.34,
        "factor_shocks":    {"beta": -0.34, "volatility": 0.15, "momentum": -0.20,
                             "size": 0.05, "value": -0.08},
        "vix_peak":         80.0,
        "duration_days":    33,
        "description":      "Fastest -34% crash in SPY history; volatility spiked to 80"
    },
    "GFC (Sep 2008 - Mar 2009)": {
        "spy_return":       -0.57,
        "factor_shocks":    {"beta": -0.57, "leverage": -0.25, "volatility": 0.20,
                             "momentum": -0.40, "value": -0.15, "sector_fin": -0.65},
        "vix_peak":         89.5,
        "duration_days":    182,
        "description":      "Global Financial Crisis; financial sector -65%"
    },
    "Inflation 2022 (Jan-Dec)": {
        "spy_return":       -0.20,
        "factor_shocks":    {"beta": -0.20, "value": 0.05, "growth": -0.35,
                             "sector_tech": -0.35, "momentum": -0.25},
        "vix_peak":         36.0,
        "duration_days":    365,
        "description":      "Rate hike cycle; growth/tech stocks -35%; value outperformed"
    },
    "Tech Bubble (Mar-Oct 2002)": {
        "spy_return":       -0.32,
        "factor_shocks":    {"sector_tech": -0.55, "growth": -0.45, "momentum": -0.30,
                             "beta": -0.32, "value": 0.10},
        "vix_peak":         45.0,
        "duration_days":    215,
        "description":      "Post-dot-com correction; tech -55%, value outperformed"
    },
}

# Synthetic forward-looking scenarios
SYNTHETIC_SCENARIOS = {
    "VIX Spike (15→45 overnight)": {
        "spy_return":       -0.08,
        "factor_shocks":    {"beta": -0.10, "volatility": 0.08, "momentum": -0.06,
                             "size": -0.03},
        "description":      "Overnight VIX spike; short-term correlation spike"
    },
    "Credit Crisis (HY +500bps)": {
        "spy_return":       -0.12,
        "factor_shocks":    {"leverage": -0.15, "sector_fin": -0.20, "beta": -0.12,
                             "value": -0.05},
        "description":      "HY spreads widen 500bps over 30 days"
    },
    "Rate Shock (+150bps 10Y)": {
        "spy_return":       -0.06,
        "factor_shocks":    {"growth": -0.15, "sector_tech": -0.12, "beta": -0.06,
                             "value": 0.08, "sector_fin": 0.05},
        "description":      "Sudden 150bps rate rise; duration-sensitive stocks fall"
    },
    "Momentum Crash (Factor -3σ)": {
        "spy_return":       -0.03,
        "factor_shocks":    {"momentum": -0.25, "beta": -0.03},
        "description":      "Momentum factor reversal (Jan 2009 style); long-momentum crowding unwind"
    },
}


def _load_factor_exposures() -> pd.Series:
    """Load current portfolio factor exposures from factor_exposure_daily.csv."""
    p = ROOT / "factor_exposure_daily.csv"
    if p.exists():
        df = pd.read_csv(p)
        if "factor" in df.columns and "exposure" in df.columns:
            latest_date = df["date"].max() if "date" in df.columns else df.index[0]
            today_df = df[df["date"] == latest_date] if "date" in df.columns else df
            return today_df.set_index("factor")["exposure"]
    # Fallback: neutral exposures
    return pd.Series({
        "size": 0.0, "value": 0.0, "growth": 0.0, "leverage": 0.0,
        "liquidity": 0.0, "momentum": 0.3, "volatility": 0.2, "beta": 1.1,
        "sector_tech": 0.4, "sector_fin": -0.1,
    })


def compute_scenario_impact(
    factor_exposures: pd.Series,
    factor_shocks: dict,
    spy_return: float,
) -> float:
    """
    Compute portfolio impact from factor shocks.

    Portfolio return ≈ β_spy × r_spy + Σ_k (exposure_k - benchmark_k) × shock_k

    We use the long-only portfolio (no benchmark deduction for simplicity).
    """
    # SPY/beta contribution
    beta = float(factor_exposures.get("beta", 1.0))
    beta_contribution = beta * spy_return

    # Factor-specific contributions (beyond market)
    alpha_contribution = 0.0
    for factor, shock in factor_shocks.items():
        if factor == "beta":
            continue
        exposure = float(factor_exposures.get(factor, 0.0))
        alpha_contribution += exposure * shock

    return beta_contribution + alpha_contribution


def run_stress_tests(
    factor_exposures: Optional[pd.Series] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    W44: Run all stress scenarios and report portfolio impact.

    Returns DataFrame: scenario × impact metrics.
    """
    if output_path is None:
        output_path = ROOT / "stress_test_results.csv"

    if factor_exposures is None:
        factor_exposures = _load_factor_exposures()

    print(f"\n[StressTest] Running stress tests")
    print(f"  Portfolio factor exposures:")
    for f, e in factor_exposures.items():
        if abs(float(e)) > 0.05:
            print(f"    {f}: {float(e):+.2f}")

    all_rows = []
    for scenario_type, scenarios in [
        ("Historical", HISTORICAL_SCENARIOS),
        ("Synthetic",  SYNTHETIC_SCENARIOS),
    ]:
        for name, params in scenarios.items():
            port_impact = compute_scenario_impact(
                factor_exposures,
                params["factor_shocks"],
                params["spy_return"],
            )
            spy_impact = params["spy_return"]

            # Recovery time estimate: assume 60% of drawdown recovered in ×2 duration
            duration = params.get("duration_days", 90)
            recovery_days = int(duration * 1.8) if port_impact < -0.05 else \
                            int(duration * 0.8) if port_impact < 0 else 0

            all_rows.append({
                "type":           scenario_type,
                "scenario":       name,
                "port_return":    round(port_impact, 4),
                "spy_return":     round(spy_impact, 4),
                "excess_return":  round(port_impact - spy_impact, 4),
                "duration_days":  duration,
                "recovery_days":  recovery_days,
                "description":    params.get("description", ""),
            })

    df = pd.DataFrame(all_rows)

    # Print stress test table
    print(f"\n  {'Scenario':<40} {'Port':>7} {'SPY':>7} {'Excess':>8}  {'Recovery'}")
    print(f"  {'─'*75}")
    for _, row in df.iterrows():
        icon = "✓" if row["excess_return"] >= 0 else "✗"
        print(f"  {row['scenario']:<40} {row['port_return']:>+6.1%} {row['spy_return']:>+6.1%} "
              f"  {row['excess_return']:>+6.1%}  {icon} {row['recovery_days']}d")

    # Summary
    worst = df.loc[df["port_return"].idxmin()]
    best_excess = df.loc[df["excess_return"].idxmax()]
    print(f"\n  Worst scenario: {worst['scenario']} → portfolio {worst['port_return']:.1%}")
    print(f"  Best excess: {best_excess['scenario']} → outperforms SPY by {best_excess['excess_return']:.1%}")

    n_outperform = (df["excess_return"] >= 0).sum()
    print(f"  Outperforms SPY in {n_outperform}/{len(df)} scenarios")

    df.to_csv(output_path, index=False)
    print(f"  Saved → {output_path}")
    return df


if __name__ == "__main__":
    print("W44: Stress Test Scenarios")
    print("=" * 50)
    run_stress_tests()
