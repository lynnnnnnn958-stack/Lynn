"""
W19: SHAP-Guided LightGBM Feature Pruning
==========================================
Trains LightGBM on the full 21-feature set used across step66 and step77,
then uses SHAP values to identify the top 12 most predictive features.

Full feature set (21):
  Price momentum (4):  mom_1m, mom_3m, mom_6m, mom_12m_skip1m
  Technical (3):       trend_200, rsi_14, inv_vol
  Rank/regime (3):     rank_mom, rank_trend, spy_regime
  Macro (3):           yc_spread, hyd_proxy, dxy_proxy
  Bear-regime (3):     neg_mom_1m, neg_mom_3m, neg_rsi_dev
  Fundamental (2):     quality_score, rank_sentiment
  Earnings (3):        rank_sue, revision_score, rank_options

SHAP method:
  TreeExplainer computes exact Shapley values (not approximations).
  Each feature's importance = mean(|SHAP value|) across all samples.
  Top-12 features by this metric are retained.

Outputs:
  shap_feature_importance.csv  — all features ranked by mean |SHAP|
  lgb_pruned_features.json     — top-12 feature names for W20 retraining

Usage:
    from signals.lgb_shap import run_shap_pruning, load_pruned_features
    df = run_shap_pruning()
    features = load_pruned_features()   # list of 12 feature names
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent

# Full 21-feature set to evaluate
ALL_FEATURES = [
    # Price momentum (4)
    "mom_1m", "mom_3m", "mom_6m", "mom_12m_skip1m",
    # Technical (3)
    "trend_200", "rsi_14", "inv_vol",
    # Cross-sectional rank / regime (3)
    "rank_mom", "rank_trend", "spy_regime",
    # Macro (3) — yield curve, HY proxy, dollar proxy
    "yc_spread", "hyd_proxy", "dxy_proxy",
    # Bear-regime features (3)
    "neg_mom_1m", "neg_mom_3m", "neg_rsi_dev",
    # Fundamental quality + sentiment (2)
    "quality_score", "rank_sentiment",
    # Earnings (3)
    "rank_sue", "revision_score", "rank_options",
]

TOP_K_FEATURES = 12   # target after pruning
FORWARD_DAYS   = 21   # 1-month forward return (monthly rebalance)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature matrix builder
# ─────────────────────────────────────────────────────────────────────────────

def _load_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute all price-based features for every ticker/date in the price cache."""
    rows = []
    tickers = [c for c in prices.columns if c != "SPY"]
    spy_rets = prices["SPY"].pct_change() if "SPY" in prices.columns else None

    for t in range(252, len(prices) - FORWARD_DAYS, 21):
        date = prices.index[t]

        # SPY regime: fraction of last 63 days where SPY was up
        if spy_rets is not None:
            spy_slice = spy_rets.iloc[t - 63:t]
            spy_regime_val = float((spy_slice > 0).mean())
        else:
            spy_regime_val = 0.5

        for tkr in tickers:
            px = prices[tkr].iloc[:t + 1].dropna()
            if len(px) < 252:
                continue

            p0   = float(px.iloc[-1])
            p1m  = float(px.iloc[-22])   if len(px) > 22  else np.nan
            p3m  = float(px.iloc[-63])   if len(px) > 63  else np.nan
            p6m  = float(px.iloc[-126])  if len(px) > 126 else np.nan
            p12m = float(px.iloc[-252])  if len(px) > 252 else np.nan
            p13m = float(px.iloc[-273])  if len(px) > 273 else np.nan
            p200 = float(px.iloc[-200])  if len(px) > 200 else np.nan

            mom_1m         = p0 / p1m  - 1 if p1m  else np.nan
            mom_3m         = p0 / p3m  - 1 if p3m  else np.nan
            mom_6m         = p0 / p6m  - 1 if p6m  else np.nan
            mom_12m_skip1m = p0 / p13m - 1 if p13m else np.nan  # skip most recent month
            trend_200      = p0 / p200 - 1 if p200 else np.nan

            # RSI-14
            daily_ret = px.pct_change().iloc[-15:]
            gains  = daily_ret.clip(lower=0).mean()
            losses = (-daily_ret.clip(upper=0)).mean()
            rsi_14 = 100 - 100 / (1 + gains / (losses + 1e-9)) if losses > 0 else 100.0

            # Inverse vol (vol is annualised; invert so high = low vol = more bullish)
            vol_21 = float(px.pct_change().iloc[-21:].std() * np.sqrt(252))
            inv_vol = 1.0 / (vol_21 + 1e-6)

            # Bear-specific
            neg_mom_1m  = -mom_1m if not np.isnan(mom_1m) else np.nan
            neg_mom_3m  = -mom_3m if not np.isnan(mom_3m) else np.nan
            neg_rsi_dev = -(rsi_14 - 50)

            rows.append({
                "date":            date,
                "ticker":          tkr,
                "mom_1m":          mom_1m,
                "mom_3m":          mom_3m,
                "mom_6m":          mom_6m,
                "mom_12m_skip1m":  mom_12m_skip1m,
                "trend_200":       trend_200,
                "rsi_14":          rsi_14,
                "inv_vol":         inv_vol,
                "spy_regime":      spy_regime_val,
                "neg_mom_1m":      neg_mom_1m,
                "neg_mom_3m":      neg_mom_3m,
                "neg_rsi_dev":     neg_rsi_dev,
            })

    return pd.DataFrame(rows)


def _add_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sectional rank features per date."""
    def _rank_pct(x: pd.Series) -> pd.Series:
        return x.rank(pct=True) * 100

    ranks = []
    for date, group in df.groupby("date"):
        g = group.copy()
        g["rank_mom"]   = _rank_pct(g["mom_12m_skip1m"])
        g["rank_trend"] = _rank_pct(g["trend_200"])
        ranks.append(g)
    return pd.concat(ranks, ignore_index=True)


def _add_macro_features(df: pd.DataFrame, macro_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Merge macro features onto ticker-date panel."""
    if macro_df is None or macro_df.empty:
        df["yc_spread"]  = 0.0
        df["hyd_proxy"]  = 0.0
        df["dxy_proxy"]  = 0.0
        return df

    macro_feat = pd.DataFrame(index=macro_df.index)
    if "TERM" in macro_df.columns:
        macro_feat["yc_spread"] = macro_df["TERM"]
    if "HY" in macro_df.columns:
        hy = macro_df["HY"]
        macro_feat["hyd_proxy"] = (hy - hy.rolling(252, min_periods=20).mean()) / \
                                  (hy.rolling(252, min_periods=20).std() + 1e-9)
    if "DOLLAR" in macro_df.columns:
        macro_feat["dxy_proxy"] = macro_df["DOLLAR"].pct_change(63)

    macro_feat = macro_feat.fillna(method="ffill")
    df["date"] = pd.to_datetime(df["date"])
    macro_feat.index = pd.to_datetime(macro_feat.index)
    df = df.merge(
        macro_feat.reset_index().rename(columns={"index": "date"}),
        on="date", how="left",
    )
    for c in ["yc_spread", "hyd_proxy", "dxy_proxy"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0)
    return df


def _add_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge pre-computed signal scores (quality, sentiment, earnings, options)."""
    signal_files = {
        "quality_score":  ("fundamental_quality_rank.csv", "quality_score"),
        "rank_sentiment": ("finbert_sentiment.csv",         "rank_sentiment"),
        "rank_sue":       ("earnings_surprise_scores.csv",  "rank_sue"),
        "revision_score": ("earnings_revision_scores.csv",  "revision_score"),
        "rank_options":   ("options_signals.csv",           "rank_options"),
    }
    for feat_name, (fname, col) in signal_files.items():
        p = ROOT / fname
        if p.exists():
            try:
                sig_df = pd.read_csv(p)
                if "ticker" in sig_df.columns and col in sig_df.columns:
                    sig_map = sig_df.set_index("ticker")[col].to_dict()
                    df[feat_name] = df["ticker"].map(sig_map).fillna(50.0)
                else:
                    df[feat_name] = 50.0
            except Exception:
                df[feat_name] = 50.0
        else:
            df[feat_name] = 50.0
    return df


def _add_forward_returns(
    df: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """Add forward 21-day returns, respecting PIT (no lookahead)."""
    fwd_rows = []
    for (date, tkr), group in df.groupby(["date", "ticker"]):
        t_idx = prices.index.get_loc(date) if date in prices.index else None
        if t_idx is None:
            continue
        t_fwd = t_idx + FORWARD_DAYS
        if t_fwd >= len(prices):
            continue
        if tkr not in prices.columns:
            continue
        p0  = prices[tkr].iloc[t_idx]
        pfwd = prices[tkr].iloc[t_fwd]
        if pd.isna(p0) or pd.isna(pfwd) or p0 == 0:
            continue
        fwd_rows.append({"date": date, "ticker": tkr, "forward_ret": pfwd / p0 - 1})

    if not fwd_rows:
        return df.assign(forward_ret=np.nan)

    fwd_df = pd.DataFrame(fwd_rows)
    return df.merge(fwd_df, on=["date", "ticker"], how="left")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SHAP analysis
# ─────────────────────────────────────────────────────────────────────────────

def _run_lgb_shap(panel: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Train LightGBM on the panel and compute SHAP values.

    Returns DataFrame with: feature, mean_abs_shap, rank, shap_values array.
    """
    try:
        import lightgbm as lgb
        import shap
    except ImportError as e:
        raise ImportError(
            f"Required: pip install lightgbm shap\n{e}"
        )

    train = panel.dropna(subset=feature_cols + ["forward_ret"]).copy()

    # Winsorize forward returns at 1%/99% to reduce outlier influence
    q_lo, q_hi = train["forward_ret"].quantile([0.01, 0.99])
    train["forward_ret_w"] = train["forward_ret"].clip(q_lo, q_hi)

    # Fill any remaining NaNs in features with column medians
    for col in feature_cols:
        med = train[col].median()
        train[col] = train[col].fillna(med)

    X = train[feature_cols].values
    y = train["forward_ret_w"].values

    print(f"  [SHAP] Training LightGBM on {len(train):,} samples × {len(feature_cols)} features")

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)

    # SHAP TreeExplainer — exact values for tree models
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)  # shape: (n_samples, n_features)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    result = pd.DataFrame({
        "feature":        feature_cols,
        "mean_abs_shap":  mean_abs_shap,
        "lgb_split_gain": model.feature_importances_,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    result["rank"]    = result.index + 1
    result["keep"]    = result["rank"] <= TOP_K_FEATURES
    result["pct_shap"] = result["mean_abs_shap"] / result["mean_abs_shap"].sum() * 100

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_shap_pruning(
    force_refresh: bool = False,
    shap_output:   Optional[Path] = None,
    json_output:   Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run SHAP feature importance analysis and save pruned feature list.

    Args:
        force_refresh: Recompute even if outputs already exist.
        shap_output:   Path for shap_feature_importance.csv (default: ROOT).
        json_output:   Path for lgb_pruned_features.json (default: ROOT).

    Returns:
        DataFrame with feature importances, sorted by mean |SHAP|.
    """
    import time as _time

    if shap_output is None:
        shap_output = ROOT / "shap_feature_importance.csv"
    if json_output is None:
        json_output = ROOT / "lgb_pruned_features.json"

    if not force_refresh and shap_output.exists() and json_output.exists():
        age = (_time.time() - shap_output.stat().st_mtime) / 86400
        if age < 30:
            print(f"  [SHAP] Loading cached results ({age:.1f}d old)")
            return pd.read_csv(shap_output)

    # Load price data
    for fname in ("sp500_price_cache_8yr.csv", "sp500_price_cache.csv",
                  "backtest_price_cache.csv"):
        p = ROOT / fname
        if p.exists():
            prices = pd.read_csv(p, index_col=0, parse_dates=True)
            print(f"  [SHAP] Loaded {fname}: {prices.shape}")
            break
    else:
        raise FileNotFoundError("No price cache found — run data/extend_history.py first")

    # Load macro data
    macro_df = None
    fred_path = ROOT / "fred_macro_daily.csv"
    if fred_path.exists():
        macro_df = pd.read_csv(fred_path, index_col=0, parse_dates=True)
        print(f"  [SHAP] Loaded FRED macro: {macro_df.shape}")

    # Build feature matrix
    print("  [SHAP] Building price features (this takes ~30s for 8yr data)...")
    panel = _load_price_features(prices)
    panel = _add_rank_features(panel)
    panel = _add_macro_features(panel, macro_df)
    panel = _add_signal_features(panel)
    panel = _add_forward_returns(panel, prices)

    available_feats = [f for f in ALL_FEATURES if f in panel.columns]
    missing_feats   = [f for f in ALL_FEATURES if f not in panel.columns]
    if missing_feats:
        print(f"  [SHAP] Warning: {len(missing_feats)} features unavailable: {missing_feats}")

    print(f"  [SHAP] Panel: {len(panel):,} rows, {len(available_feats)} features available")

    # Run SHAP
    shap_df = _run_lgb_shap(panel, available_feats)

    # Save results
    shap_df.to_csv(shap_output, index=False)
    print(f"  [SHAP] Saved feature importances → {shap_output}")

    # Save pruned feature list
    pruned = shap_df[shap_df["keep"]]["feature"].tolist()
    with open(json_output, "w") as f:
        json.dump({"pruned_features": pruned, "n_features": len(pruned)}, f, indent=2)
    print(f"  [SHAP] Saved pruned {len(pruned)} features → {json_output}")

    # Print summary
    print("\n  SHAP Feature Ranking:")
    print(f"  {'Rank':<5} {'Feature':<22} {'Mean |SHAP|':<14} {'% Total':<10} {'Keep'}")
    print(f"  {'─'*60}")
    for _, row in shap_df.iterrows():
        keep_mark = "✓" if row["keep"] else " "
        print(f"  {int(row['rank']):<5} {row['feature']:<22} "
              f"{row['mean_abs_shap']:.5f}       "
              f"{row['pct_shap']:.1f}%      {keep_mark}")
    print(f"\n  Pruned to {len(pruned)}/{len(available_feats)} features: {pruned}")

    return shap_df


def load_pruned_features() -> list[str]:
    """
    Load the pruned feature list from lgb_pruned_features.json.
    Falls back to ALL_FEATURES[:TOP_K_FEATURES] if file not found.
    """
    p = ROOT / "lgb_pruned_features.json"
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        return data["pruned_features"]
    # Fallback: use first TOP_K_FEATURES from full list
    return ALL_FEATURES[:TOP_K_FEATURES]


if __name__ == "__main__":
    print("W19: SHAP-guided LightGBM feature pruning")
    print("=" * 50)
    df = run_shap_pruning(force_refresh=True)
    pruned = load_pruned_features()
    print(f"\nFinal pruned features ({len(pruned)}):")
    for i, f in enumerate(pruned, 1):
        print(f"  {i:2d}. {f}")
