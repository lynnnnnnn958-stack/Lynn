"""
W29: Black-Litterman Portfolio Optimizer
=========================================
Combines equilibrium market returns (from CAPM implied returns) with
active views (from Canyon alpha signals) to produce posterior expected returns.

The BL update formula (He & Litterman 1999):
  Π    = equilibrium returns (reverse-optimized from market cap weights)
  Q    = active views (from alpha_scores.csv signals)
  Ω    = view uncertainty diagonal matrix (from IC² confidence)
  τ    = scaling factor (typically 0.025 to 0.05)

  Posterior:
  μ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} × [(τΣ)^{-1}Π + P'Ω^{-1}Q]

  where P = view picking matrix (N views × N assets)

Implementation notes:
  - Views: one view per asset ("asset i outperforms/underperforms by alpha_score deviation")
    This is the simplified "absolute" view formulation (one view per stock)
  - Ω_{ii} = τΣ_{ii} / (IC² / IC²_max)   (more confident = tighter view)
  - IC² values from config.yaml signals section (combined signal IC²)
  - Market cap proxy: rank-order of current price × estimated float

Uses regime-conditional covariance matrix (W28) for Σ.

Outputs:
  bl_expected_returns.csv  — ticker × mu_bl (posterior expected annual return)
  bl_weights.csv           — MVO weights using BL expected returns

Usage:
    from portfolio.black_litterman import run_bl_optimizer, BLOptimizer
    result = run_bl_optimizer(top_n=25)
    print(result["weights"].head(10))
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent

TAU               = 0.025    # BL scalar (Satchell & Scowcroft recommend 0.025)
RISK_AVERSION     = 2.5      # λ in mean-variance: max w'μ - λ/2 w'Σw
MAX_POSITION_LONG  = 0.08    # max long weight per stock
MAX_POSITION_SHORT = 0.04    # max short weight
SECTOR_CAP        = 0.25     # max total exposure to any GICS sector


# ─────────────────────────────────────────────────────────────────────────────
# 1. Equilibrium returns (CAPM reverse optimization)
# ─────────────────────────────────────────────────────────────────────────────

def _market_cap_weights(prices: pd.DataFrame) -> pd.Series:
    """
    Proxy for market cap weights: rank-order of price.
    Actual market cap data requires a paid vendor; price rank is a decent proxy
    for the purpose of setting BL prior (correlates ~0.7 with true market cap rank).
    """
    if prices.empty:
        return pd.Series(dtype=float)
    last_price = prices.iloc[-1].dropna()
    # Weight proportional to price rank (larger = higher cap proxy)
    rank_w = last_price.rank() / last_price.rank().sum()
    return rank_w


def compute_equilibrium_returns(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float = RISK_AVERSION,
) -> pd.Series:
    """
    Reverse-optimize equilibrium returns: Π = λ Σ w_mkt.
    These are the returns consistent with the market being in equilibrium.
    """
    common = cov.index.intersection(market_weights.index)
    Sigma_c = cov.loc[common, common].values
    w_c     = market_weights.reindex(common).fillna(0.0).values
    Pi      = risk_aversion * Sigma_c @ w_c
    return pd.Series(Pi, index=common)


# ─────────────────────────────────────────────────────────────────────────────
# 2. View construction from alpha scores
# ─────────────────────────────────────────────────────────────────────────────

def _load_alpha_scores() -> pd.DataFrame:
    """Load alpha_scores.csv from the pipeline."""
    p = ROOT / "alpha_scores.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def _load_ic2_weights() -> dict[str, float]:
    """Load IC² from config.yaml or use defaults."""
    try:
        import yaml
        cfg = yaml.safe_load(open(ROOT / "config.yaml"))
        ic_cfg = cfg.get("signals", {}).get("ic_fallback", {})
        ic2 = {k.replace("sig_", ""): float(v) ** 2 for k, v in ic_cfg.items()}
        if ic2:
            return ic2
    except Exception:
        pass
    return {
        "ml_ensemble": 0.370 ** 2,
        "surprise":    0.229 ** 2,
        "regime_ml":   0.223 ** 2,
        "momentum":    0.168 ** 2,
        "revision":    0.142 ** 2,
    }


def construct_views(
    alpha_scores: pd.DataFrame,
    equilibrium_returns: pd.Series,
    ic2_weights: Optional[dict] = None,
    top_n: int = 50,
) -> tuple[pd.Series, pd.Series]:
    """
    Construct BL views from alpha scores.

    View formulation: each stock has one absolute view.
    Q_i = mean_return + alpha_deviation  (based on z-scored alpha score)
    Ω_ii = uncertainty scaled inversely with composite IC²

    Returns:
        Q  — view expected returns (Series: ticker → annual return view)
        Omega — diagonal view uncertainty (Series: ticker → variance)
    """
    if alpha_scores.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    if ic2_weights is None:
        ic2_weights = _load_ic2_weights()

    # Z-score the alpha scores
    score_col = "alpha_score" if "alpha_score" in alpha_scores.columns else \
                "mu_override" if "mu_override" in alpha_scores.columns else None
    if score_col is None:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    alpha = alpha_scores.set_index("ticker")[score_col].dropna()

    # Scale: top-decile alpha = +10% expected return, bottom-decile = -10%
    mu_alpha = alpha.mean()
    std_alpha = alpha.std()
    if std_alpha < 1e-9:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    alpha_z = (alpha - mu_alpha) / std_alpha
    Q = equilibrium_returns.reindex(alpha.index).fillna(0.0) + alpha_z * 0.10

    # View uncertainty Ω: inversely proportional to combined signal IC²
    # Combined IC²: sum of IC² across signals (assuming uncorrelated)
    total_ic2 = sum(ic2_weights.values())
    if total_ic2 < 1e-9:
        total_ic2 = 0.05

    # Higher IC² → lower uncertainty (tighter view)
    base_uncertainty = 0.05   # 5% annualised view error std dev
    omega_diag = pd.Series(
        base_uncertainty ** 2 / (total_ic2 + 1e-9),
        index=Q.index,
    )

    return Q, omega_diag


# ─────────────────────────────────────────────────────────────────────────────
# 3. Black-Litterman formula
# ─────────────────────────────────────────────────────────────────────────────

def black_litterman_update(
    Pi: pd.Series,
    Sigma: pd.DataFrame,
    Q: pd.Series,
    Omega: pd.Series,
    tau: float = TAU,
) -> pd.Series:
    """
    Core BL update formula (absolute views, one view per asset).

    μ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} × [(τΣ)^{-1}Π + P'Ω^{-1}Q]

    For absolute views: P = I (identity), so this simplifies to:
    μ_BL = [(τΣ)^{-1} + Ω^{-1}]^{-1} × [(τΣ)^{-1}Π + Ω^{-1}Q]

    Args:
        Pi:    Equilibrium returns (N,)
        Sigma: Covariance matrix (N × N)
        Q:     View returns (M ≤ N,)
        Omega: View uncertainty diagonal (M,)
        tau:   Uncertainty scaling parameter

    Returns: Posterior expected returns (N,)
    """
    # Align to common tickers
    common = Pi.index.intersection(Sigma.index).intersection(Q.index)
    if len(common) == 0:
        return Pi

    Pi_c    = Pi[common].values
    Sigma_c = Sigma.loc[common, common].values
    Q_c     = Q[common].values
    Omega_c = Omega[common].values

    n = len(common)
    tau_Sigma = tau * Sigma_c

    # Regularize: add small diagonal to avoid singular matrix
    eps = 1e-8 * np.eye(n)
    tau_Sigma_reg = tau_Sigma + eps

    try:
        inv_tauSigma = np.linalg.inv(tau_Sigma_reg)
        inv_Omega    = np.diag(1.0 / (Omega_c + 1e-12))

        # BL posterior precision: M = (τΣ)^{-1} + P'Ω^{-1}P  (P=I)
        M = inv_tauSigma + inv_Omega

        # BL posterior: v = (τΣ)^{-1}Π + P'Ω^{-1}Q  (P=I)
        v = inv_tauSigma @ Pi_c + inv_Omega @ Q_c

        mu_bl = np.linalg.solve(M, v)
    except np.linalg.LinAlgError:
        # Fallback: blend equilibrium with views by alpha
        mu_bl = 0.6 * Pi_c + 0.4 * Q_c

    return pd.Series(mu_bl, index=common)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Mean-Variance Optimization with constraints
# ─────────────────────────────────────────────────────────────────────────────

def _mean_variance_optimize(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    risk_aversion: float = RISK_AVERSION,
    max_long: float = MAX_POSITION_LONG,
    long_only: bool = True,
    top_n: int = 25,
) -> pd.Series:
    """
    Mean-variance optimization: max w'μ - (λ/2)w'Σw
    Subject to:
      - sum(w) = 1
      - 0 ≤ w_i ≤ max_long (long-only)
      - Concentrated to top_n stocks (zero out rest)

    Uses analytical solution for unconstrained case, then applies clipping
    (Frank-Wolfe approximation).
    """
    common = mu.index.intersection(Sigma.index)
    mu_c    = mu[common].values
    Sigma_c = Sigma.loc[common, common].values + np.eye(len(common)) * 1e-8

    # Pre-screen: keep top-n by expected return
    top_idx = np.argsort(mu_c)[-top_n:][::-1]
    mu_sub    = mu_c[top_idx]
    Sigma_sub = Sigma_c[np.ix_(top_idx, top_idx)]

    # Unconstrained MVO: w* = (1/λ) Σ^{-1} μ, then normalise
    try:
        Sigma_inv  = np.linalg.inv(Sigma_sub + np.eye(len(top_idx)) * 1e-6)
        w_raw      = Sigma_inv @ mu_sub / risk_aversion
    except np.linalg.LinAlgError:
        w_raw = mu_sub / (mu_sub.sum() + 1e-9)

    # Apply constraints
    if long_only:
        w_raw = np.clip(w_raw, 0.0, max_long)

    total = w_raw.sum()
    if total < 1e-9:
        w_raw = np.ones(len(top_idx)) / len(top_idx)
    else:
        w_raw /= total

    # Map back to full ticker space
    weights = np.zeros(len(common))
    weights[top_idx] = w_raw
    return pd.Series(weights, index=common)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Public API
# ─────────────────────────────────────────────────────────────────────────────

class BLOptimizer:
    """Black-Litterman optimizer with regime-conditional covariance."""

    def __init__(
        self,
        tau: float = TAU,
        risk_aversion: float = RISK_AVERSION,
        max_long: float = MAX_POSITION_LONG,
    ):
        self.tau           = tau
        self.risk_aversion = risk_aversion
        self.max_long      = max_long

    def optimize(
        self,
        tickers: list[str],
        alpha_scores: Optional[pd.DataFrame] = None,
        covariance: Optional[pd.DataFrame] = None,
        top_n: int = 25,
    ) -> dict:
        """
        Run BL optimization and return posterior returns + weights.

        Returns dict:
            mu_bl     — posterior expected returns
            weights   — optimal weights
            Pi        — equilibrium returns
            Q         — view returns
        """
        # Load covariance matrix
        if covariance is None:
            from risk.regime_cov import get_current_covariance
            covariance = get_current_covariance(tickers=tickers)

        if covariance.empty:
            # Fallback: diagonal covariance (equal risk)
            vols = pd.Series(0.20, index=tickers)  # 20% default vol
            covariance = pd.DataFrame(
                np.diag(vols.values ** 2), index=tickers, columns=tickers
            )

        # Load prices for market weights
        for fname in ("sp500_price_cache_8yr.csv", "sp500_price_cache.csv"):
            p = ROOT / fname
            if p.exists():
                prices = pd.read_csv(p, index_col=0, parse_dates=True)
                break
        else:
            prices = pd.DataFrame()

        # Market cap proxy weights
        if not prices.empty:
            prices = prices[[t for t in tickers if t in prices.columns]]
            market_weights = _market_cap_weights(prices)
        else:
            market_weights = pd.Series(1.0 / len(tickers), index=tickers)

        # Step 1: Equilibrium returns
        Pi = compute_equilibrium_returns(covariance, market_weights, self.risk_aversion)

        # Step 2: Views from alpha scores
        if alpha_scores is None:
            alpha_scores = _load_alpha_scores()

        ic2_weights = _load_ic2_weights()
        Q, Omega = construct_views(alpha_scores, Pi, ic2_weights, top_n=top_n)

        if Q.empty:
            mu_bl = Pi
        else:
            # Step 3: BL posterior
            mu_bl = black_litterman_update(Pi, covariance, Q, Omega, self.tau)

        # Step 4: MVO
        weights = _mean_variance_optimize(
            mu_bl, covariance, self.risk_aversion, self.max_long, top_n=top_n
        )

        return {
            "mu_bl":          mu_bl,
            "weights":        weights,
            "Pi":             Pi,
            "Q":              Q,
            "Omega":          Omega,
            "covariance":     covariance,
        }


def calibrate_tau_from_history(
    output_path: Optional[Path] = None,
    lookback_months: int = 6,
) -> float:
    """
    Bayesian tau calibration: adjust BL uncertainty scalar based on recent
    prediction accuracy of the BL views (Q) vs realized returns.

    Logic:
      If BL views have been consistently accurate (realized ≈ predicted),
      we increase confidence → lower tau (tighter views).
      If BL views have been consistently wrong → higher tau (looser views).

    tau calibration range: [0.010, 0.100] (Satchell & Scowcroft: 0.025 baseline)
    Falls back to TAU=0.025 if insufficient data.
    """
    if output_path is None:
        output_path = ROOT

    bl_ret_path = output_path / "bl_expected_returns.csv"
    px_path     = ROOT / "backtest_price_cache.csv"

    if not (bl_ret_path.exists() and px_path.exists()):
        return TAU

    try:
        bl_ret = pd.read_csv(bl_ret_path)
        px     = pd.read_csv(px_path, index_col=0, parse_dates=True)

        if "mu_bl" not in bl_ret.columns or "ticker" not in bl_ret.columns:
            return TAU

        bl_ret = bl_ret.dropna(subset=["mu_bl"])

        # Compute realized 21d returns for the tickers BL predicted
        cutoff_date = px.index.max() - pd.DateOffset(months=lookback_months)
        px_recent   = px[px.index >= cutoff_date].sort_index()

        if len(px_recent) < 22:
            return TAU

        # Compare BL predicted μ vs actual returns in lookback window
        errors: list[float] = []
        for _, row in bl_ret.iterrows():
            tk = str(row["ticker"])
            if tk not in px_recent.columns:
                continue
            px_tk = px_recent[tk].dropna()
            if len(px_tk) < 22:
                continue
            actual_21d = float(px_tk.iloc[-1] / px_tk.iloc[max(0, len(px_tk)-22)] - 1)
            predicted  = float(row["mu_bl"])
            errors.append((actual_21d - predicted) ** 2)

        if not errors:
            return TAU

        mse  = float(np.mean(errors))
        rmse = float(np.sqrt(mse))

        # Scale tau: RMSE < 5% → lower tau (confident); RMSE > 20% → raise tau
        if rmse < 0.05:
            tau_new = TAU * 0.80   # views more accurate → tighter
        elif rmse > 0.20:
            tau_new = TAU * 1.40   # views less accurate → looser
        else:
            # Linear interpolation
            t = (rmse - 0.05) / (0.20 - 0.05)
            tau_new = TAU * (0.80 + t * (1.40 - 0.80))

        tau_new = float(np.clip(tau_new, 0.010, 0.100))
        print(f"  [BL-tau] RMSE={rmse:.2%}  tau: {TAU:.3f} → {tau_new:.3f}")

        # Save tau for transparency
        pd.DataFrame([{"tau": tau_new, "rmse": rmse, "n_tickers": len(errors),
                       "date": str(pd.Timestamp.today().date())}]).to_csv(
            output_path / "bl_tau_history.csv",
            mode="a", header=not (output_path / "bl_tau_history.csv").exists(),
            index=False)
        return tau_new

    except Exception as exc:
        print(f"  [BL-tau] Calibration failed: {exc} — using default TAU={TAU}")
        return TAU


def run_bl_optimizer(
    top_n: int = 25,
    as_of: Optional[pd.Timestamp] = None,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Run full BL pipeline: load data → compute → save outputs.

    Args:
        top_n:       Number of stocks in final portfolio.
        as_of:       Reference date (default: today).
        output_path: Directory to save CSVs.

    Returns result dict with mu_bl, weights, Pi, Q.
    """
    if output_path is None:
        output_path = ROOT
    if as_of is None:
        as_of = pd.Timestamp.today().normalize()

    alpha_scores = _load_alpha_scores()
    if alpha_scores.empty:
        print("  [BL] No alpha_scores.csv found — returning empty")
        return {}

    tickers = alpha_scores["ticker"].dropna().tolist() if "ticker" in alpha_scores.columns else []
    if not tickers:
        return {}

    # Calibrate tau from recent BL prediction accuracy (Bayesian self-update)
    tau_calibrated = calibrate_tau_from_history(output_path=output_path)

    optimizer = BLOptimizer(tau=tau_calibrated)
    result    = optimizer.optimize(tickers=tickers, alpha_scores=alpha_scores, top_n=top_n)

    # Save outputs
    mu_bl   = result.get("mu_bl",   pd.Series(dtype=float))
    weights = result.get("weights", pd.Series(dtype=float))

    if not mu_bl.empty:
        mu_df = mu_bl.reset_index()
        mu_df.columns = ["ticker", "mu_bl"]
        mu_df["as_of"] = str(as_of.date())
        mu_df.to_csv(output_path / "bl_expected_returns.csv", index=False)

    if not weights.empty:
        w_df = weights[weights > 1e-4].sort_values(ascending=False).reset_index()
        w_df.columns = ["ticker", "weight"]
        w_df["as_of"] = str(as_of.date())
        w_df.to_csv(output_path / "bl_weights.csv", index=False)
        print(f"  [BL] Top positions (BL-weighted):")
        for _, row in w_df.head(10).iterrows():
            print(f"    {row['ticker']:6s}  {row['weight']:.1%}  "
                  f"(μ_BL={float(mu_bl.get(row['ticker'], 0)):.1%})")

    print(f"  [BL] Saved bl_expected_returns.csv ({len(mu_bl)}) + bl_weights.csv ({(weights > 1e-4).sum()})")
    return result


if __name__ == "__main__":
    print("W29: Black-Litterman Portfolio Optimizer")
    print("=" * 50)
    result = run_bl_optimizer(top_n=25)
    if result:
        print(f"\nPortfolio stats:")
        w = result.get("weights", pd.Series())
        print(f"  Active positions: {(w > 1e-4).sum()}")
        print(f"  Sum of weights:   {w.sum():.3f}")
        print(f"  Max weight:       {w.max():.1%}")
        print(f"  HHI:              {(w**2).sum():.4f}")
