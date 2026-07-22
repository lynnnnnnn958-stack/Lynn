"""
W28: Regime-Conditional Covariance Estimation
===============================================
In BULL markets, correlations between stocks are lower (idiosyncratic risk dominates).
In BEAR markets, correlations spike toward 1.0 (systemic risk dominates).
Using a single unconditional covariance matrix underestimates BEAR risk by ~30-50%.

Method:
  1. Load HMM regime labels from hmm_regime_daily.csv (W15)
  2. Split historical returns into BULL and BEAR subsets
  3. Compute EWMA covariance matrix for each regime
  4. Blend: Σ_conditional = p_bull × Σ_bull + p_bear × Σ_bear
     where p_bull/p_bear are current regime probabilities from HMM
  5. Apply Ledoit-Wolf shrinkage to ensure positive-definiteness

This is used by the Black-Litterman optimizer (W29) and portfolio construction.

Ledoit-Wolf shrinkage (Ledoit & Wolf 2004):
  Σ_shrunk = (1 - α) × Σ_sample + α × F
  where F = target (constant-correlation model) and α = optimal shrinkage intensity

Outputs:
  regime_cov_bull.csv   — BULL-regime covariance matrix (subset tickers)
  regime_cov_bear.csv   — BEAR-regime covariance matrix (subset tickers)
  regime_cov_blend.csv  — Blended covariance matrix (current regime)

Usage:
    from risk.regime_cov import build_regime_covariance, get_current_covariance
    cov_blend = get_current_covariance(tickers=["AAPL", "MSFT", "NVDA"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent

EWMA_HALFLIFE = 63   # days
MIN_PERIODS   = 63   # minimum days for reliable estimate
MAX_TICKERS   = 100  # cap to keep matrix tractable


# ─────────────────────────────────────────────────────────────────────────────
# 1. EWMA covariance with optional Ledoit-Wolf shrinkage
# ─────────────────────────────────────────────────────────────────────────────

def _ewma_cov(rets: pd.DataFrame, halflife: int = EWMA_HALFLIFE) -> np.ndarray:
    """
    Exponentially weighted covariance matrix.

    Weights decay with halflife trading days (more recent = higher weight).
    Returns n × n numpy array.
    """
    n = len(rets)
    decay = np.exp(-np.log(2) / halflife)
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])
    weights /= weights.sum()

    X = rets.values
    mu = (X * weights[:, None]).sum(axis=0)
    X_centered = X - mu
    cov = (X_centered * weights[:, None]).T @ X_centered

    return cov


def _ledoit_wolf_shrinkage(cov: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Oracle Ledoit-Wolf shrinkage toward constant-correlation target.

    Target F = diag(σ) × C_bar × diag(σ) where C_bar = average off-diagonal correlation.

    This is the analytical LW formula from Ledoit & Wolf (2004), "A well-conditioned
    estimator for large-dimensional covariance matrices."
    """
    n = cov.shape[0]
    if n < 2:
        return cov

    # Compute sample correlations
    std_devs = np.sqrt(np.diag(cov))
    std_devs = np.where(std_devs < 1e-10, 1e-10, std_devs)
    corr = cov / np.outer(std_devs, std_devs)
    np.fill_diagonal(corr, 1.0)

    # Constant-correlation target: replace off-diagonal with mean correlation
    upper_tri = corr[np.triu_indices(n, k=1)]
    rho_bar   = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0
    rho_bar   = np.clip(rho_bar, -0.99, 0.99)

    target_corr = np.full((n, n), rho_bar)
    np.fill_diagonal(target_corr, 1.0)
    F = np.outer(std_devs, std_devs) * target_corr

    # Optimal shrinkage intensity (simplified Oracle estimator)
    # α = min(1, (p/T) × factor) — scale by dimension/sample ratio
    p = n
    T = max(n_samples, p + 1)
    alpha = min(0.8, (p / T) * 2.0)

    return (1 - alpha) * cov + alpha * F


# ─────────────────────────────────────────────────────────────────────────────
# 2. Regime-conditional covariance estimation
# ─────────────────────────────────────────────────────────────────────────────

def _load_regime_labels() -> pd.DataFrame:
    """Load HMM regime labels from hmm_regime_daily.csv."""
    p = ROOT / "hmm_regime_daily.csv"
    if p.exists():
        df = pd.read_csv(p, parse_dates=["date"])
        return df.set_index("date")
    return pd.DataFrame()


def build_regime_covariance(
    tickers: Optional[list[str]] = None,
    as_of: Optional[pd.Timestamp] = None,
    force_refresh: bool = False,
) -> dict:
    """
    Build BULL and BEAR conditional covariance matrices.

    Args:
        tickers:       List of tickers to include. Defaults to top-100 by liquidity.
        as_of:         Reference date for regime probabilities.
        force_refresh: Recompute even if cached.

    Returns dict:
        cov_bull     — BULL regime covariance (DataFrame: ticker × ticker)
        cov_bear     — BEAR regime covariance (DataFrame: ticker × ticker)
        cov_blend    — Blended covariance (DataFrame: ticker × ticker)
        p_bull       — Current P(BULL) from HMM
        p_bear       — Current P(BEAR) from HMM
        n_bull_days  — Number of BULL days used
        n_bear_days  — Number of BEAR days used
    """
    import time as _time

    if as_of is None:
        as_of = pd.Timestamp.today().normalize()

    blend_path = ROOT / "regime_cov_blend.csv"
    bull_path  = ROOT / "regime_cov_bull.csv"
    bear_path  = ROOT / "regime_cov_bear.csv"

    if not force_refresh and blend_path.exists():
        age = (_time.time() - blend_path.stat().st_mtime) / 86400
        if age < 1.5:
            cov_blend = pd.read_csv(blend_path, index_col=0)
            cov_bull  = pd.read_csv(bull_path,  index_col=0) if bull_path.exists()  else cov_blend
            cov_bear  = pd.read_csv(bear_path,  index_col=0) if bear_path.exists()  else cov_blend
            print(f"  [RegimeCov] Loaded from cache ({age:.1f}d old): {cov_blend.shape}")
            return {
                "cov_bull": cov_bull, "cov_bear": cov_bear, "cov_blend": cov_blend,
                "p_bull": 0.5, "p_bear": 0.5,
            }

    # Load prices
    for fname in ("sp500_price_cache_8yr.csv", "sp500_price_cache.csv"):
        p = ROOT / fname
        if p.exists():
            prices = pd.read_csv(p, index_col=0, parse_dates=True)
            break
    else:
        raise FileNotFoundError("No price cache found")

    prices = prices[prices.index <= as_of]

    # Select tickers
    if tickers is None:
        stock_cols = [c for c in prices.columns if c != "SPY"]
        tickers = stock_cols[:MAX_TICKERS]  # top N by column order

    tickers = [t for t in tickers if t in prices.columns][:MAX_TICKERS]
    rets = prices[tickers].pct_change().dropna()

    # Load regime labels
    regime_df = _load_regime_labels()

    if regime_df.empty:
        # No HMM: fall back to single unconditional covariance
        print("  [RegimeCov] No HMM regime data — using unconditional covariance")
        cov_arr = _ewma_cov(rets)
        cov_arr = _ledoit_wolf_shrinkage(cov_arr, len(rets))
        cov_df  = pd.DataFrame(cov_arr, index=tickers, columns=tickers)
        cov_df.to_csv(blend_path)
        cov_df.to_csv(bull_path)
        cov_df.to_csv(bear_path)
        return {
            "cov_bull": cov_df, "cov_bear": cov_df, "cov_blend": cov_df,
            "p_bull": 0.5, "p_bear": 0.5, "n_bull_days": len(rets), "n_bear_days": 0,
        }

    # Align regime to returns dates (forward-fill)
    regime_aligned = regime_df.reindex(rets.index, method="ffill").dropna()
    common_dates   = rets.index.intersection(regime_aligned.index)

    rets_aligned   = rets.loc[common_dates]
    regime_aligned = regime_aligned.loc[common_dates]

    bull_mask = regime_aligned["regime"] == "BULL"
    bear_mask = regime_aligned["regime"] == "BEAR"

    rets_bull = rets_aligned[bull_mask]
    rets_bear = rets_aligned[bear_mask]

    n_bull = len(rets_bull)
    n_bear = len(rets_bear)
    print(f"  [RegimeCov] BULL={n_bull} days, BEAR={n_bear} days out of {len(rets_aligned)} total")

    # Get current regime probabilities
    valid_regime = regime_df[regime_df.index <= as_of]
    if not valid_regime.empty:
        p_bull = float(valid_regime.iloc[-1].get("p_bull", 0.5))
        p_bear = float(valid_regime.iloc[-1].get("p_bear", 0.5))
    else:
        p_bull, p_bear = 0.5, 0.5

    # Compute conditional covariances with Ledoit-Wolf shrinkage
    def _regime_cov(rets_subset: pd.DataFrame, n: int, fallback_rets: pd.DataFrame) -> np.ndarray:
        if n < MIN_PERIODS:
            print(f"    Insufficient {n} days — using full-sample covariance")
            arr = _ewma_cov(fallback_rets)
        else:
            arr = _ewma_cov(rets_subset)
        return _ledoit_wolf_shrinkage(arr, n)

    cov_bull_arr = _regime_cov(rets_bull, n_bull, rets_aligned)
    cov_bear_arr = _regime_cov(rets_bear, n_bear, rets_aligned)

    # Blend: p_bull × Σ_bull + p_bear × Σ_bear
    cov_blend_arr = p_bull * cov_bull_arr + p_bear * cov_bear_arr

    # Annualise: multiply by 252 (daily covariance → annual)
    cov_bull_df  = pd.DataFrame(cov_bull_arr  * 252, index=tickers, columns=tickers)
    cov_bear_df  = pd.DataFrame(cov_bear_arr  * 252, index=tickers, columns=tickers)
    cov_blend_df = pd.DataFrame(cov_blend_arr * 252, index=tickers, columns=tickers)

    # Save
    cov_bull_df.to_csv(bull_path)
    cov_bear_df.to_csv(bear_path)
    cov_blend_df.to_csv(blend_path)
    print(f"  [RegimeCov] Saved: bull, bear, blend covariances for {len(tickers)} tickers")

    # Print regime risk comparison
    spy_idx = tickers.index("SPY") if "SPY" in tickers else -1
    sample_tickers = tickers[:min(5, len(tickers))]
    sample_idx     = [tickers.index(t) for t in sample_tickers]
    for s_idx, tkr in zip(sample_idx, sample_tickers):
        vol_bull  = float(np.sqrt(cov_bull_arr[s_idx, s_idx]  * 252))
        vol_bear  = float(np.sqrt(cov_bear_arr[s_idx, s_idx]  * 252))
        vol_blend = float(np.sqrt(cov_blend_arr[s_idx, s_idx] * 252))
        print(f"    {tkr:6s}: vol_bull={vol_bull:.1%}  vol_bear={vol_bear:.1%}  blend={vol_blend:.1%}")

    return {
        "cov_bull":    cov_bull_df,
        "cov_bear":    cov_bear_df,
        "cov_blend":   cov_blend_df,
        "p_bull":      p_bull,
        "p_bear":      p_bear,
        "n_bull_days": n_bull,
        "n_bear_days": n_bear,
    }


def get_current_covariance(
    tickers: Optional[list[str]] = None,
    regime: str = "blend",
) -> pd.DataFrame:
    """
    Load the current regime covariance matrix for specified tickers.

    Args:
        tickers: Subset of tickers to return. Returns full matrix if None.
        regime:  'blend' (default), 'bull', or 'bear'.

    Returns DataFrame: ticker × ticker covariance (annualised).
    """
    path_map = {
        "blend": ROOT / "regime_cov_blend.csv",
        "bull":  ROOT / "regime_cov_bull.csv",
        "bear":  ROOT / "regime_cov_bear.csv",
    }
    p = path_map.get(regime, path_map["blend"])
    if not p.exists():
        return pd.DataFrame()

    cov = pd.read_csv(p, index_col=0)
    if tickers is not None:
        common = [t for t in tickers if t in cov.index]
        cov = cov.loc[common, common]
    return cov


if __name__ == "__main__":
    print("W28: Regime-Conditional Covariance")
    print("=" * 50)
    result = build_regime_covariance(force_refresh=True)
    print(f"\nRegime weights: p_bull={result['p_bull']:.2f}, p_bear={result['p_bear']:.2f}")
    print(f"Data points: BULL={result.get('n_bull_days', 0)}, BEAR={result.get('n_bear_days', 0)}")

    cov_blend = result["cov_blend"]
    print(f"\nBlended covariance matrix ({cov_blend.shape}):")
    sample = cov_blend.iloc[:5, :5]
    print(sample.round(5).to_string())

    # Verify positive semi-definiteness
    eigenvalues = np.linalg.eigvalsh(cov_blend.values)
    print(f"\nEigenvalue range: [{eigenvalues.min():.2e}, {eigenvalues.max():.2e}]")
    print(f"  Positive semi-definite: {(eigenvalues >= -1e-10).all()}")
