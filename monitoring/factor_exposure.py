"""
W30: Daily Factor Exposure Attribution Report
=============================================
Generates a daily report showing the portfolio's exposures to Barra risk factors.
Helps identify unintended factor bets (e.g., over-exposed to small-cap, high-beta).

Factor exposure calculation:
  portfolio_exposure_k = Σ_i w_i × B_{ik}
  where B_{ik} = stock i's exposure to factor k (from barra_factor_exposures.csv)

Report outputs:
  factor_exposure_daily.csv  — daily time series of factor exposures
  factor_exposure_report.txt — formatted console/file report

Benchmark: S&P 500 equal-weight (exposure = 0 after standardization) serves as
the neutral reference. Deviations from 0 represent active factor bets.

Usage:
    from monitoring.factor_exposure import run_factor_exposure_report
    run_factor_exposure_report()
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent

FACTOR_NAMES = [
    "size", "value", "growth", "leverage", "liquidity",
    "momentum", "volatility", "beta",
    "sector_tech", "sector_fin",
]

# Thresholds for exposure warnings
EXPOSURE_WARN_THRESHOLD = 0.30   # |z-score exposure| > 0.3 standard deviations = notable


def _load_portfolio_weights() -> pd.Series:
    """Load current portfolio weights from daily_picks.csv or alpha_scores.csv."""
    # Try daily_picks first (has explicit weights)
    for fname in ("daily_picks.csv", "bl_weights.csv", "alpha_scores.csv"):
        p = ROOT / fname
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            if "ticker" not in df.columns:
                continue
            if "weight" in df.columns:
                w = df.set_index("ticker")["weight"].dropna()
                w = w[w > 0]
                if len(w) > 0:
                    return w / w.sum()
            elif "alpha_score" in df.columns:
                top25 = df.nlargest(25, "alpha_score")
                w = pd.Series(1.0 / len(top25), index=top25["ticker"])
                return w
        except Exception:
            continue
    return pd.Series(dtype=float)


def _load_factor_exposures() -> pd.DataFrame:
    """Load Barra factor exposure matrix from cache."""
    p = ROOT / "barra_factor_exposures.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0)
    return pd.DataFrame()


def compute_portfolio_factor_exposures(
    weights: pd.Series,
    factor_exposures: pd.DataFrame,
) -> pd.Series:
    """
    Compute portfolio-level factor exposures: w' × B.

    Returns: Series with FACTOR_NAMES as index, values are z-score exposures.
    """
    common = weights.index.intersection(factor_exposures.index)
    if len(common) == 0:
        return pd.Series(0.0, index=FACTOR_NAMES)

    w = weights[common]
    B = factor_exposures.loc[common]

    exposures = {}
    for factor in FACTOR_NAMES:
        if factor in B.columns:
            exposures[factor] = float((w * B[factor]).sum())
        else:
            exposures[factor] = 0.0

    return pd.Series(exposures)


def _exposure_commentary(factor: str, exposure: float) -> str:
    """Generate human-readable commentary for a factor exposure."""
    direction = "long" if exposure > 0 else "short"
    magnitude = abs(exposure)

    commentary_map = {
        "size":       (f"small-cap tilt" if exposure < 0 else "large-cap tilt"),
        "value":      (f"value tilt" if exposure > 0 else "growth tilt"),
        "growth":     (f"high-growth bias" if exposure > 0 else "low-growth/mature"),
        "leverage":   (f"high-leverage exposure" if exposure > 0 else "low-leverage/conservative"),
        "liquidity":  (f"liquid large-caps" if exposure > 0 else "less-liquid midcap"),
        "momentum":   (f"momentum long" if exposure > 0 else "momentum short/reversal"),
        "volatility": (f"high-vol exposure" if exposure > 0 else "low-vol/defensive"),
        "beta":       (f"high-beta / market amplified" if exposure > 0 else "low-beta / defensive"),
        "sector_tech":(f"overweight Technology" if exposure > 0 else "underweight Technology"),
        "sector_fin": (f"overweight Financials" if exposure > 0 else "underweight Financials"),
    }
    return commentary_map.get(factor, f"{direction} {factor}")


def run_factor_exposure_report(
    weights: Optional[pd.Series] = None,
    factor_exposures: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
    append_history: bool = True,
) -> pd.DataFrame:
    """
    Generate factor exposure report for the current portfolio.

    Args:
        weights:          Portfolio weights. Loaded from daily_picks.csv if None.
        factor_exposures: Barra factor exposures. Loaded from cache if None.
        output_path:      Where to save the report.
        append_history:   If True, append to factor_exposure_daily.csv.

    Returns: DataFrame with factor exposures + commentary.
    """
    if output_path is None:
        output_path = ROOT

    if weights is None:
        weights = _load_portfolio_weights()
    if factor_exposures is None:
        factor_exposures = _load_factor_exposures()

    today_str = datetime.today().strftime("%Y-%m-%d")

    print(f"\n[FactorExposure] Portfolio factor exposures — {today_str}")
    print(f"  Portfolio: {len(weights)} positions, weights sum = {weights.sum():.3f}")

    if weights.empty or factor_exposures.empty:
        print("  No portfolio weights or factor exposures available")
        if factor_exposures.empty:
            print("  Run: python risk/barra.py to generate factor exposures")
        return pd.DataFrame()

    # Compute exposures
    port_exposures = compute_portfolio_factor_exposures(weights, factor_exposures)

    rows = []
    print(f"\n  {'Factor':<15} {'Exposure':>8}  {'Commentary'}")
    print(f"  {'─'*60}")

    for factor in FACTOR_NAMES:
        exp = float(port_exposures.get(factor, 0.0))
        commentary = _exposure_commentary(factor, exp)
        flag = " !" if abs(exp) > EXPOSURE_WARN_THRESHOLD else "  "

        print(f"  {factor:<15} {exp:>+8.3f}{flag}  {commentary}")

        rows.append({
            "date":        today_str,
            "factor":      factor,
            "exposure":    round(exp, 4),
            "abs_exposure": round(abs(exp), 4),
            "flag":        abs(exp) > EXPOSURE_WARN_THRESHOLD,
            "commentary":  commentary,
        })

    # Factor risk contribution (if regime covariance available)
    cov_path = ROOT / "regime_cov_blend.csv"
    if cov_path.exists():
        try:
            cov_df = pd.read_csv(cov_path, index_col=0)
            from risk.barra import compute_portfolio_risk
            risk = compute_portfolio_risk(weights)
            if risk:
                print(f"\n  Portfolio risk (Barra):")
                print(f"    Total vol:    {risk.get('total_vol', 0):.1%}")
                print(f"    Factor vol:   {risk.get('factor_vol', 0):.1%} "
                      f"({risk.get('factor_pct', 0):.0%} of total)")
                print(f"    Specific vol: {risk.get('specific_vol', 0):.1%} "
                      f"({risk.get('specific_pct', 0):.0%} of total)")
        except Exception:
            pass

    # Flagged factors
    flagged = [r for r in rows if r["flag"]]
    if flagged:
        print(f"\n  ⚠  Notable exposures (|z| > {EXPOSURE_WARN_THRESHOLD}):")
        for r in flagged:
            print(f"     {r['factor']:<15} {r['exposure']:>+.3f}  — {r['commentary']}")
    else:
        print(f"\n  ✓ All factor exposures within normal range (|z| < {EXPOSURE_WARN_THRESHOLD})")

    report_df = pd.DataFrame(rows)

    # Save daily CSV
    daily_path = output_path / "factor_exposure_daily.csv"
    if append_history and daily_path.exists():
        existing = pd.read_csv(daily_path)
        existing = existing[existing["date"] != today_str]
        combined = pd.concat([existing, report_df], ignore_index=True)
        combined.to_csv(daily_path, index=False)
    else:
        report_df.to_csv(daily_path, index=False)

    print(f"\n  Saved → {daily_path}")
    return report_df


if __name__ == "__main__":
    run_factor_exposure_report()
