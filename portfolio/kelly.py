"""
W33: Fractional Kelly Criterion
=================================
Kelly criterion determines the theoretically optimal fraction of capital to allocate
to each bet to maximise geometric growth rate of the portfolio.

Full Kelly can lead to 50% drawdowns; institutional practice is to use
fractional Kelly (f* × κ, where κ = 0.25 to 0.50).

For a portfolio of N assets (Kelly generalized, Thorp 1997):
  w_kelly = (1/κ) × Σ^{-1} × μ  (same form as MVO with λ = κ)
  where κ is the Kelly fraction (0.25 = quarter-Kelly = institutional default)

For each individual signal/stock:
  f_kelly_i = (IC_i² × SR_i) / σ²_i   (Grinold-Kahn)
  where SR_i = annualised Sharpe ratio, σ_i = vol

Practical implementation:
  1. Estimate per-stock expected return μ from alpha_scores.csv
  2. Estimate covariance Σ from regime_cov_blend.csv
  3. Compute full-Kelly: f* = Σ^{-1} × μ
  4. Scale by κ (default 0.25), cap at max_position

Outputs:
  kelly_weights.csv  — ticker × kelly_weight (fractional Kelly allocations)

Usage:
    from portfolio.kelly import compute_kelly_weights
    weights = compute_kelly_weights(kappa=0.25, top_n=25)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent

KAPPA_DEFAULT  = 0.25   # quarter-Kelly (institutional standard)
MAX_POSITION   = 0.08   # hard cap per position
MIN_POSITION   = 0.0    # no short constraint (long-only)
RF_ANNUAL      = 0.053  # risk-free rate


def _load_expected_returns() -> pd.Series:
    """
    Load expected returns from BL output or alpha_scores.
    Priority: bl_expected_returns.csv > alpha_scores.csv > fallback 0.
    """
    # BL posterior expected returns (most accurate)
    bl_path = ROOT / "bl_expected_returns.csv"
    if bl_path.exists():
        df = pd.read_csv(bl_path)
        if "ticker" in df.columns and "mu_bl" in df.columns:
            return df.set_index("ticker")["mu_bl"]

    # Alpha scores as return proxy
    alpha_path = ROOT / "alpha_scores.csv"
    if alpha_path.exists():
        df = pd.read_csv(alpha_path)
        if "ticker" in df.columns and "mu_override" in df.columns:
            return df.set_index("ticker")["mu_override"]
        if "ticker" in df.columns and "alpha_score" in df.columns:
            # Convert [0, 100] score to expected return range [-20%, +20%]
            alpha = df.set_index("ticker")["alpha_score"]
            return ((alpha - 50) / 50) * 0.20  # [-20%, +20%]

    return pd.Series(dtype=float)


def _load_covariance(tickers: list[str]) -> pd.DataFrame:
    """Load regime-blended covariance, fall back to diagonal."""
    from risk.regime_cov import get_current_covariance
    cov = get_current_covariance(tickers=tickers)
    if not cov.empty:
        return cov

    # Fallback: diagonal 20% vol
    return pd.DataFrame(
        np.diag([0.20 ** 2] * len(tickers)),
        index=tickers, columns=tickers,
    )


def compute_kelly_weights(
    kappa: float = KAPPA_DEFAULT,
    top_n: int = 25,
    max_position: float = MAX_POSITION,
    output_path: Optional[Path] = None,
) -> pd.Series:
    """
    Compute fractional Kelly portfolio weights.

    Args:
        kappa:        Kelly fraction (0.25 = quarter-Kelly).
        top_n:        Maximum portfolio size.
        max_position: Hard cap per position.
        output_path:  Where to save kelly_weights.csv.

    Returns: pd.Series (ticker → fractional Kelly weight, sums to ≤ 1).
    """
    if output_path is None:
        output_path = ROOT / "kelly_weights.csv"

    mu = _load_expected_returns()
    if mu.empty:
        print("  [Kelly] No expected returns available — run BL optimizer or alpha aggregator first")
        return pd.Series(dtype=float)

    # Filter to positive expected returns (long-only Kelly)
    mu = mu - RF_ANNUAL  # excess return above risk-free
    mu_positive = mu[mu > 0].sort_values(ascending=False).head(top_n)

    if mu_positive.empty:
        print("  [Kelly] No stocks with positive excess returns")
        return pd.Series(dtype=float)

    tickers = mu_positive.index.tolist()
    cov = _load_covariance(tickers)
    common = [t for t in tickers if t in cov.index]

    if len(common) < 2:
        print("  [Kelly] Insufficient covariance data")
        return pd.Series(dtype=float)

    mu_c = mu_positive[common].values
    Sigma_c = cov.loc[common, common].values + np.eye(len(common)) * 1e-8

    # Full Kelly: w* = Σ^{-1} μ
    try:
        Sigma_inv = np.linalg.inv(Sigma_c)
        w_full_kelly = Sigma_inv @ mu_c
    except np.linalg.LinAlgError:
        w_full_kelly = mu_c / (np.sum(mu_c) + 1e-9)

    # Apply fractional Kelly: scale by κ
    w_frac_kelly = w_full_kelly * kappa

    # Apply constraints: clip to [0, max_position]
    w_frac_kelly = np.clip(w_frac_kelly, 0.0, max_position)

    # Normalise to sum = 1
    total = w_frac_kelly.sum()
    if total > 1e-9:
        w_frac_kelly /= total
    else:
        w_frac_kelly = np.ones(len(common)) / len(common)

    weights = pd.Series(w_frac_kelly, index=common)
    weights = weights[weights > 1e-4].sort_values(ascending=False)

    # Compute expected Kelly growth rate
    growth_rate = float(mu_c @ w_frac_kelly) - \
                  0.5 * float(w_frac_kelly @ Sigma_c @ w_frac_kelly)

    print(f"  [Kelly] κ={kappa:.2f} fractional Kelly weights computed:")
    print(f"    Positions:      {len(weights)}")
    print(f"    Max weight:     {weights.max():.1%}")
    print(f"    Expected growth: {growth_rate:.2%} annualised (above risk-free)")
    print(f"    Sum of weights: {weights.sum():.3f}")

    # Top positions
    print(f"\n  Top positions:")
    for tkr, w in weights.head(10).items():
        print(f"    {tkr:6s}  {w:.1%}  (μ={mu.get(tkr, 0):.1%})")

    # Save
    out = weights.reset_index()
    out.columns = ["ticker", "kelly_weight"]
    out["kappa"]       = kappa
    out["mu_excess"]   = out["ticker"].map(mu)
    out.to_csv(output_path, index=False)
    print(f"  Saved → {output_path}")

    return weights


if __name__ == "__main__":
    print("W33: Fractional Kelly Criterion")
    print("=" * 40)
    for kappa in [0.50, 0.25, 0.10]:
        print(f"\n=== κ = {kappa} ===")
        w = compute_kelly_weights(kappa=kappa)
        if not w.empty:
            print(f"  HHI (concentration): {(w**2).sum():.4f}")
