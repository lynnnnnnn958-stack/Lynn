"""
W45: v9 ↔ v11 Historical Signal Bridge
========================================
Bridges Canyon v9 signal history with v11 signal improvements for long-horizon backtesting.

Problem:
  v9 alpha_scores.csv has daily signal data from ~2022-present.
  v11 uses a walk-forward backtest on the 8-year price cache (2018-2026).
  There's a gap: v11 signals don't cover 2018-2022 because the data sources
  (EDGAR PIT, Form 4, FRED) haven't been backfilled to that period.

Solution:
  1. Use v11 price-based signals for the full 8-year period (price signals don't have the gap)
  2. Augment with v9 signals where they overlap (2022-present)
  3. Fill the 2018-2022 gap with the v9 signal formula (momentum + quality from v9 logic)
  4. Output a unified signal CSV for use in combined backtesting

This is NOT introducing lookahead: the bridge only applies v9's historical
signal formulas (which are PIT-compliant by design) to historical data.

Outputs:
  v9_v11_bridge_signals.csv  — unified signal for 2018-present
  bridge_signal_ic.csv       — IC comparison: v9 vs v11 signals in overlap period

Usage:
    from research.v9_bridge import build_bridge_signals, compute_bridge_ic
    unified = build_bridge_signals()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent


def _load_v9_history() -> pd.DataFrame:
    """Load v9 historical alpha scores (from canyon_output_vault or alpha_scores.csv)."""
    vault = ROOT / "canyon_output_vault"
    all_dfs = []

    if vault.exists():
        for d in sorted(vault.iterdir()):
            p = d / "alpha_scores.csv"
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    if "ticker" in df.columns:
                        # Extract date from folder name (YYYYMMDD_HHMMSS_pre-run)
                        date_str = d.name[:8]
                        df["signal_date"] = pd.Timestamp(date_str)
                        all_dfs.append(df)
                except Exception:
                    pass

    if not all_dfs:
        p = ROOT / "alpha_scores.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["signal_date"] = pd.Timestamp.today().normalize()
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def _load_v11_signals() -> pd.DataFrame:
    """Load v11 full signals (from v11_full_signals.csv)."""
    p = ROOT / "v11_full_signals.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["date"] if "date" in pd.read_csv(p, nrows=0).columns else None)
    return pd.DataFrame()


def _compute_price_bridge_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute price-based bridge signal for 2018-2022 gap period.
    Uses the same formula as v9/step87 momentum + vol signals.

    Returns DataFrame: ticker, date, bridge_score [0-100]
    """
    tickers = [c for c in prices.columns if c != "SPY"]
    spy = prices["SPY"] if "SPY" in prices.columns else prices.iloc[:, 0]

    rows = []
    for t_idx in range(252, len(prices) - 21, 63):  # quarterly snapshots for speed
        date = prices.index[t_idx]
        if date > pd.Timestamp("2022-01-01"):
            break

        p_slice = prices[tickers].iloc[:t_idx + 1]
        spy_slice = spy.iloc[:t_idx + 1]

        if len(p_slice) < 252:
            continue

        last    = p_slice.iloc[-1]
        p1m     = p_slice.iloc[-22]
        p3m     = p_slice.iloc[-63]
        p12m    = p_slice.iloc[-252]
        p13m    = p_slice.iloc[-273] if len(p_slice) > 273 else p_slice.iloc[-252]

        mom_1m   = (last / p1m  - 1).clip(-0.5, 0.5)
        mom_12m  = (last / p13m - 1 - (last / p1m - 1)).clip(-1, 1)
        vol_21   = p_slice.pct_change().iloc[-21:].std()
        inv_vol  = 1.0 / (vol_21 + 0.01)
        spy_ma   = spy_slice.rolling(200).mean().iloc[-1]
        spy_reg  = 1.0 if float(spy_slice.iloc[-1]) > spy_ma else 0.0

        # Cross-sectional z-score
        def _rank_pct(s):
            return s.rank(pct=True) * 100

        score = (
            _rank_pct(mom_12m) * 0.35 +
            _rank_pct(mom_1m)  * 0.15 +
            _rank_pct(inv_vol) * 0.25 +
            pd.Series(spy_reg * 15.0, index=last.index) +
            25.0  # centering
        ).clip(0, 100)

        for tkr in tickers:
            if tkr in score.index and not np.isnan(score[tkr]):
                rows.append({
                    "ticker":       tkr,
                    "date":         date,
                    "bridge_score": round(float(score[tkr]), 2),
                    "source":       "price_bridge",
                })

    return pd.DataFrame(rows)


def build_bridge_signals(
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build unified v9/v11 signal history for 2018-present.

    Returns DataFrame: ticker, date, alpha_score, source.
    """
    if output_path is None:
        output_path = ROOT / "v9_v11_bridge_signals.csv"

    print("\n[Bridge] Building v9 ↔ v11 signal bridge...")

    all_signals = []

    # Phase 1: Price-based bridge for 2018-2022 gap
    for fname in ("sp500_price_cache_8yr.csv", "sp500_price_cache.csv"):
        p = ROOT / fname
        if p.exists():
            prices = pd.read_csv(p, index_col=0, parse_dates=True)
            print(f"  [Bridge] Computing price bridge signals (2018-2022)...")
            bridge_df = _compute_price_bridge_signal(prices)
            if not bridge_df.empty:
                bridge_df = bridge_df.rename(columns={"bridge_score": "alpha_score"})
                all_signals.append(bridge_df)
                print(f"    {len(bridge_df):,} records (price bridge)")
            break

    # Phase 2: v9 historical alpha scores (2022-present)
    v9_df = _load_v9_history()
    if not v9_df.empty and "signal_date" in v9_df.columns:
        v9_clean = v9_df[["ticker", "signal_date", "alpha_score"]].copy()
        v9_clean = v9_clean.rename(columns={"signal_date": "date"})
        v9_clean["source"] = "v9_pipeline"
        v9_clean = v9_clean[v9_clean["date"] >= pd.Timestamp("2022-01-01")]
        if not v9_clean.empty:
            all_signals.append(v9_clean)
            print(f"    {len(v9_clean):,} records (v9 pipeline)")

    # Phase 3: v11 signals (most recent)
    v11_df = _load_v11_signals()
    if not v11_df.empty:
        score_col = "alpha_score" if "alpha_score" in v11_df.columns else \
                    "composite_score" if "composite_score" in v11_df.columns else None
        if score_col and "date" in v11_df.columns:
            v11_clean = v11_df[["ticker", "date", score_col]].copy()
            v11_clean = v11_clean.rename(columns={score_col: "alpha_score"})
            v11_clean["source"] = "v11_pipeline"
            all_signals.append(v11_clean)
            print(f"    {len(v11_clean):,} records (v11 pipeline)")

    if not all_signals:
        print("  [Bridge] No signal data available")
        return pd.DataFrame()

    unified = pd.concat(all_signals, ignore_index=True)
    unified["date"] = pd.to_datetime(unified["date"])

    # Deduplicate: prefer v11 > v9 > bridge for same ticker/date
    source_priority = {"v11_pipeline": 0, "v9_pipeline": 1, "price_bridge": 2}
    unified["priority"] = unified["source"].map(source_priority).fillna(3)
    unified = unified.sort_values(["ticker", "date", "priority"])
    unified = unified.drop_duplicates(["ticker", "date"], keep="first")
    unified = unified.drop(columns=["priority"])

    unified = unified.sort_values(["date", "ticker"]).reset_index(drop=True)
    unified.to_csv(output_path, index=False)

    n_dates   = unified["date"].nunique()
    n_tickers = unified["ticker"].nunique()
    print(f"\n  [Bridge] Unified signal: {len(unified):,} records, {n_dates} dates, {n_tickers} tickers")
    print(f"  Source breakdown:")
    for src, cnt in unified["source"].value_counts().items():
        print(f"    {src}: {cnt:,}")
    print(f"  Saved → {output_path}")
    return unified


def compute_bridge_ic(
    bridge_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute IC for bridge signal vs v11 signal in the overlap period.

    Returns DataFrame showing IC agreement between sources.
    """
    if bridge_df is None:
        p = ROOT / "v9_v11_bridge_signals.csv"
        if not p.exists():
            return pd.DataFrame()
        bridge_df = pd.read_csv(p, parse_dates=["date"])

    for fname in ("sp500_price_cache_8yr.csv", "sp500_price_cache.csv"):
        p = ROOT / fname
        if p.exists():
            prices = pd.read_csv(p, index_col=0, parse_dates=True)
            break
    else:
        return pd.DataFrame()

    tickers = list(bridge_df["ticker"].unique())
    rows = []

    # For each date in the bridge, compute IC vs forward 21-day return
    for date, group in bridge_df.groupby("date"):
        if date not in prices.index:
            continue
        t_idx = prices.index.get_loc(date)
        if t_idx + 21 >= len(prices):
            continue

        common_tkrs = [t for t in group["ticker"] if t in prices.columns]
        if len(common_tkrs) < 30:
            continue

        score = group.set_index("ticker")["alpha_score"][common_tkrs]
        fwd_ret = (prices.loc[prices.index[t_idx + 21], common_tkrs] /
                   prices.loc[date, common_tkrs] - 1).dropna()

        common = score.index.intersection(fwd_ret.index)
        if len(common) < 20:
            continue

        ic, _ = spearmanr(score[common], fwd_ret[common])
        source = group.iloc[0].get("source", "unknown")
        rows.append({"date": date, "source": source, "IC": round(float(ic), 4), "n": len(common)})

    if not rows:
        return pd.DataFrame()

    ic_df = pd.DataFrame(rows)
    for src in ic_df["source"].unique():
        sub = ic_df[ic_df["source"] == src]["IC"]
        print(f"  [Bridge] {src}: mean IC = {sub.mean():.3f}, std = {sub.std():.3f}")

    ic_df.to_csv(ROOT / "bridge_signal_ic.csv", index=False)
    return ic_df


if __name__ == "__main__":
    print("W45: v9 ↔ v11 Signal Bridge")
    print("=" * 50)
    unified = build_bridge_signals()
    if not unified.empty:
        ic_df = compute_bridge_ic(unified)
