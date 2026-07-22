"""
W15 + W24: Macro-Enhanced HMM Regime Detection
===============================================
Upgrades the existing HMM (which uses only SPY returns + volatility)
to include 4 FRED macro variables as features:

  Feature set:
    1. SPY 21-day return         (price momentum)
    2. SPY 21-day realized vol   (price fear gauge)
    3. VIX z-score               (options fear gauge — from FRED)
    4. HY spread z-score         (credit risk — from FRED)
    5. Yield curve slope         (10Y-2Y, macro cycle — from FRED)
    6. Dollar 60-day momentum    (global risk appetite — from FRED)

  2-state Gaussian HMM:
    State 0 = BULL (low vol, positive momentum, tight spreads)
    State 1 = BEAR (high vol, negative momentum, wide spreads)

Why this matters:
  - Price-only HMM misses credit stress signals (2022 HY spread widening
    predicted equity drawdown weeks before price broke down)
  - VIX spike often leads SPY drawdown by 1-2 weeks
  - Yield curve inversion (TERM < 0) is a 12-month leading indicator

Outputs:
  hmm_regime_daily.csv — daily regime probabilities (bull/bear)

Usage:
    from signals.macro_hmm import fit_macro_hmm, get_regime_on_date
    regime_df = fit_macro_hmm()
    p_bull = get_regime_on_date(regime_df, pd.Timestamp("2024-01-15"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent

# Regime state labels (assigned after fitting by inspecting state means)
BULL_LABEL = "BULL"
BEAR_LABEL = "BEAR"

N_STATES = 2
N_ITER   = 200
RANDOM_STATE = 42


def _build_feature_matrix(
    spy_prices: pd.Series,
    macro_df: Optional[pd.DataFrame] = None,
    lookback: int = 252,
) -> pd.DataFrame:
    """
    Build the HMM feature matrix from price + macro data.

    All features are standardized to zero mean, unit variance before fitting.
    Missing macro values are forward-filled then filled with 0 (neutral).

    Returns DataFrame with DatetimeIndex and feature columns.
    """
    # Price features (always available)
    spy_ret  = spy_prices.pct_change()
    spy_vol  = spy_ret.rolling(21).std() * np.sqrt(252)
    spy_mom  = spy_prices.pct_change(21)

    feats = pd.DataFrame({
        "spy_ret21":  spy_mom,
        "spy_vol21":  spy_vol,
    }, index=spy_prices.index)

    # Macro features (from FRED, W7)
    if macro_df is not None and not macro_df.empty:
        # Align macro to price dates (forward-fill FRED publication gaps)
        macro_aligned = macro_df.reindex(spy_prices.index, method="ffill")

        if "VIX" in macro_aligned.columns:
            # VIX z-score over 252 trading days
            vix = macro_aligned["VIX"]
            feats["vix_zscore"] = (vix - vix.rolling(lookback, min_periods=20).mean()) / \
                                  (vix.rolling(lookback, min_periods=20).std() + 1e-9)

        if "HY" in macro_aligned.columns:
            hy = macro_aligned["HY"]
            feats["hy_zscore"] = (hy - hy.rolling(lookback, min_periods=20).mean()) / \
                                 (hy.rolling(lookback, min_periods=20).std() + 1e-9)

        if "TERM" in macro_aligned.columns:
            feats["term_slope"] = macro_aligned["TERM"]

        if "DOLLAR" in macro_aligned.columns:
            dollar = macro_aligned["DOLLAR"]
            feats["dollar_mom60"] = dollar.pct_change(60)

    # Drop NaN rows (warmup period)
    feats = feats.dropna()

    # Standardize each feature independently (crucial for HMM convergence)
    for col in feats.columns:
        mu, std = feats[col].mean(), feats[col].std()
        feats[col] = (feats[col] - mu) / (std + 1e-9)

    return feats


def _assign_regime_labels(model, feats: pd.DataFrame) -> dict:
    """
    After HMM fitting, determine which state is BULL and which is BEAR.
    BULL state = higher mean SPY return + lower mean VIX z-score.
    Returns {state_idx: label} mapping.
    """
    state_means = model.means_  # shape (n_states, n_features)
    n_feat = state_means.shape[1]

    # Score each state: higher spy_ret21 + lower vix_zscore → more bullish
    scores = []
    for s in range(N_STATES):
        # Feature 0 = spy_ret21: higher is bullish
        score = state_means[s, 0]
        # Feature 2 = vix_zscore (if present): lower is bullish
        if n_feat > 2:
            score -= state_means[s, 2]
        scores.append(score)

    bull_state = int(np.argmax(scores))
    bear_state = 1 - bull_state
    return {bull_state: BULL_LABEL, bear_state: BEAR_LABEL}


def fit_macro_hmm(
    spy_prices: Optional[pd.Series] = None,
    macro_df: Optional[pd.DataFrame] = None,
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fit a 2-state Gaussian HMM using SPY price + FRED macro features.

    Args:
        spy_prices:    SPY daily price series (DatetimeIndex). If None, loads from cache.
        macro_df:      FRED macro DataFrame (DatetimeIndex). If None, loads from cache.
        cache_path:    Where to save/load regime CSV.
        force_refresh: Refit even if cache exists.

    Returns DataFrame with columns:
        date, p_bull, p_bear, regime (BULL/BEAR), regime_score [-1, +1]
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        raise ImportError("hmmlearn required: pip install hmmlearn")

    if cache_path is None:
        cache_path = ROOT / "hmm_regime_daily.csv"

    import time as _time
    if not force_refresh and cache_path.exists():
        age_days = (_time.time() - cache_path.stat().st_mtime) / 86400
        if age_days < 1.5:  # refresh daily
            df = pd.read_csv(cache_path, parse_dates=["date"])
            df = df.set_index("date")
            print(f"  [HMM] Loaded from cache ({age_days:.1f}d old): {len(df)} days")
            return df

    # Load SPY prices if not provided
    if spy_prices is None:
        for fname in ("sp500_price_cache_8yr.csv", "sp500_price_cache.csv"):
            p = ROOT / fname
            if p.exists():
                prices = pd.read_csv(p, index_col=0, parse_dates=True)
                if "SPY" in prices.columns:
                    spy_prices = prices["SPY"].dropna()
                    break
    if spy_prices is None or len(spy_prices) < 100:
        raise ValueError("SPY price data not found or too short")

    # Load macro data if not provided
    if macro_df is None:
        fred_path = ROOT / "fred_macro_daily.csv"
        if fred_path.exists():
            macro_df = pd.read_csv(fred_path, index_col=0, parse_dates=True)

    # Build feature matrix
    feats = _build_feature_matrix(spy_prices, macro_df)
    n_feats = feats.shape[1]

    print(f"  [HMM] Fitting {N_STATES}-state HMM on {len(feats)} days, {n_feats} features: {feats.columns.tolist()}")

    # Fit HMM
    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    X = feats.values
    model.fit(X)

    # Decode: get state probabilities for each day
    log_probs, state_seq = model.decode(X, algorithm="viterbi")
    posteriors = model.predict_proba(X)

    # Assign BULL/BEAR labels based on state characteristics
    label_map = _assign_regime_labels(model, feats)

    bull_state = [k for k, v in label_map.items() if v == BULL_LABEL][0]

    result = pd.DataFrame(index=feats.index)
    result["p_bull"]        = posteriors[:, bull_state]
    result["p_bear"]        = 1 - posteriors[:, bull_state]
    result["regime"]        = np.where(posteriors[:, bull_state] >= 0.5, BULL_LABEL, BEAR_LABEL)
    result["regime_score"]  = result["p_bull"] * 2 - 1  # maps [0,1] → [-1,+1]
    result["n_features"]    = n_feats  # audit: how many features were used

    result.index.name = "date"
    result.to_csv(cache_path)
    print(f"  [HMM] Saved regime history ({len(result)} days) → {cache_path}")

    bull_pct = (result["regime"] == BULL_LABEL).mean()
    print(f"  [HMM] Bull fraction: {bull_pct:.1%}  (current: {result['regime'].iloc[-1]})")
    return result


def get_regime_on_date(
    regime_df: pd.DataFrame,
    as_of: pd.Timestamp,
) -> dict:
    """
    Get regime state for a specific date.

    Returns:
        dict with: regime, p_bull, p_bear, regime_score
    """
    if regime_df.empty:
        return {"regime": "UNKNOWN", "p_bull": 0.5, "p_bear": 0.5, "regime_score": 0.0}

    df = regime_df if isinstance(regime_df.index, pd.DatetimeIndex) else \
         regime_df.set_index("date")

    # Get most recent row on or before as_of
    valid = df[df.index <= as_of]
    if valid.empty:
        return {"regime": "UNKNOWN", "p_bull": 0.5, "p_bear": 0.5, "regime_score": 0.0}

    row = valid.iloc[-1]
    return {
        "regime":       str(row["regime"]),
        "p_bull":       float(row["p_bull"]),
        "p_bear":       float(row["p_bear"]),
        "regime_score": float(row.get("regime_score", row["p_bull"] * 2 - 1)),
    }


def get_regime_weights(
    regime_df: pd.DataFrame,
    as_of: pd.Timestamp,
    base_weights: dict[str, float],
    bull_multipliers: dict[str, float],
    bear_multipliers: dict[str, float],
) -> dict[str, float]:
    """
    Compute regime-conditional signal weights.

    Args:
        base_weights:     Default signal weights (sum to 1)
        bull_multipliers: Scale each signal weight in BULL regime
        bear_multipliers: Scale each signal weight in BEAR regime

    Returns adjusted weights normalized to sum to 1.
    """
    state = get_regime_on_date(regime_df, as_of)
    p_bull = state["p_bull"]
    p_bear = state["p_bear"]

    adjusted = {}
    for sig, w in base_weights.items():
        bull_mult = bull_multipliers.get(sig, 1.0)
        bear_mult = bear_multipliers.get(sig, 1.0)
        adjusted[sig] = w * (p_bull * bull_mult + p_bear * bear_mult)

    total = sum(adjusted.values())
    if total < 1e-9:
        return base_weights
    return {k: v / total for k, v in adjusted.items()}


if __name__ == "__main__":
    print("Fitting macro-enhanced HMM regime detector...")
    regime_df = fit_macro_hmm(force_refresh=True)
    print(f"\nRegime history (last 10 days):")
    print(regime_df.tail(10).to_string())

    # Spot checks
    covid_date = pd.Timestamp("2020-03-23")
    covid_state = get_regime_on_date(regime_df, covid_date)
    print(f"\nRegime on COVID low ({covid_date.date()}): {covid_state['regime']} (p_bull={covid_state['p_bull']:.2f})")
    assert covid_state["regime"] == "BEAR", f"Expected BEAR at COVID low, got {covid_state['regime']}"

    bull_2021 = get_regime_on_date(regime_df, pd.Timestamp("2021-01-15"))
    print(f"Regime on 2021-01-15: {bull_2021['regime']} (p_bull={bull_2021['p_bull']:.2f})")

    print("\n✓ Macro HMM validation passed")
