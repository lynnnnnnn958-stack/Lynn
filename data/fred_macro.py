"""
W7: FRED Macro Data Fetcher
============================
Fetches 4 key macro time series from the Federal Reserve (FRED).
All series are free with no API key needed (via pandas_datareader).

Series used:
  VIX    — CBOE Volatility Index (daily fear gauge, replaces SPY-vol proxy in v11)
  HY     — ICE BofA High Yield Option-Adjusted Spread (credit risk appetite)
  TERM   — 10-Year minus 2-Year Treasury yield spread (recession signal)
  DOLLAR — Broad dollar index (global risk appetite)

Logic:
  - VIX > 25: high-vol regime (reduce max position weight 30%)
  - HY spread z-score < -1: credit bullish (risk-on)
  - TERM < 0: yield curve inverted (recession risk, reduce long bias)
  - DOLLAR momentum: strong dollar → headwind for multinationals

Usage:
    from data.fred_macro import pull_fred, get_macro_features
    macro = pull_fred(start="2018-01-01")
    feats = get_macro_features(macro, as_of=pd.Timestamp("2024-06-01"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent

SERIES = {
    "VIX":    "VIXCLS",           # CBOE VIX (daily)
    "HY":     "BAMLH0A0HYM2",     # HY OAS spread, bps
    "TERM":   "T10Y2Y",           # 10Y - 2Y Treasury spread
    "DOLLAR": "DTWEXBGS",         # Broad dollar index (goods & services)
}

# FRED has 1-day publication lag; data for day T available on T+1
FRED_LAG_DAYS = 1


def pull_fred(
    start: str = "2018-01-01",
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download FRED macro series and return as daily DataFrame.

    Returns DataFrame with columns: VIX, HY, TERM, DOLLAR
    Index: DatetimeIndex (business days)
    Missing values forward-filled (FRED has occasional weekend/holiday gaps).

    Args:
        start:         Start date string "YYYY-MM-DD"
        cache_path:    CSV path to cache (refreshes if > 1 day old)
        force_refresh: Ignore cache and re-download
    """
    if cache_path is None:
        cache_path = ROOT / "fred_macro_daily.csv"

    import time as _time
    if not force_refresh and cache_path.exists():
        age_days = (_time.time() - cache_path.stat().st_mtime) / 86400
        if age_days < 1.5:  # refresh daily
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    try:
        import pandas_datareader as pdr
    except ImportError:
        print("  [FRED] pandas_datareader not installed. Run: pip install pandas-datareader")
        return _load_cache_or_empty(cache_path)

    frames: dict[str, pd.Series] = {}
    for name, series_id in SERIES.items():
        try:
            s = pdr.get_data_fred(series_id, start=start)
            frames[name] = s.iloc[:, 0]  # strip column name
        except Exception as e:
            print(f"  [FRED] Failed to fetch {name} ({series_id}): {e}")

    if not frames:
        print("  [FRED] All fetches failed — returning cached data if available")
        return _load_cache_or_empty(cache_path)

    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Forward-fill gaps (weekends, FRED publication gaps)
    df = df.ffill().dropna(how="all")

    df.to_csv(cache_path)
    print(f"  [FRED] Saved {len(df)} days → {cache_path}")
    return df


def _load_cache_or_empty(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)
    return pd.DataFrame(columns=list(SERIES.keys()))


def get_macro_features(
    macro: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback: int = 252,
) -> dict[str, float]:
    """
    Compute scalar macro features for use as signal inputs or regime variables.

    PIT: uses macro data strictly up to as_of - FRED_LAG_DAYS.

    Returns dict with:
        vix_level:    Raw VIX level
        vix_zscore:   VIX 1-year z-score (high = fearful market)
        hy_zscore:    HY spread 1-year z-score (high = credit stress)
        term_slope:   10Y-2Y spread level (negative = inverted)
        dollar_mom:   60-day dollar momentum
        is_high_vol:  bool, VIX > 25
        is_inverted:  bool, yield curve inverted
        regime_score: composite [-1, +1] risk-on/off score
    """
    # Apply FRED publication lag
    pit_date = as_of - pd.Timedelta(days=FRED_LAG_DAYS)
    hist = macro[macro.index <= pit_date].tail(lookback)

    if hist.empty or len(hist) < 20:
        return {k: np.nan for k in ["vix_level", "vix_zscore", "hy_zscore",
                                     "term_slope", "dollar_mom", "is_high_vol",
                                     "is_inverted", "regime_score"]}

    features: dict[str, float] = {}

    # VIX
    if "VIX" in hist.columns:
        vix_now  = float(hist["VIX"].iloc[-1])
        vix_mean = float(hist["VIX"].mean())
        vix_std  = float(hist["VIX"].std())
        features["vix_level"]  = vix_now
        features["vix_zscore"] = (vix_now - vix_mean) / (vix_std + 1e-9)
        features["is_high_vol"] = bool(vix_now > 25.0)
    else:
        features.update({"vix_level": np.nan, "vix_zscore": np.nan, "is_high_vol": False})

    # HY spread
    if "HY" in hist.columns:
        hy_now  = float(hist["HY"].iloc[-1])
        hy_mean = float(hist["HY"].mean())
        hy_std  = float(hist["HY"].std())
        features["hy_zscore"] = (hy_now - hy_mean) / (hy_std + 1e-9)
    else:
        features["hy_zscore"] = np.nan

    # Yield curve
    if "TERM" in hist.columns:
        term = float(hist["TERM"].iloc[-1])
        features["term_slope"]  = term
        features["is_inverted"] = bool(term < 0)
    else:
        features.update({"term_slope": np.nan, "is_inverted": False})

    # Dollar momentum (60-day)
    if "DOLLAR" in hist.columns and len(hist) >= 60:
        dollar_60d = float(hist["DOLLAR"].iloc[-1] / hist["DOLLAR"].iloc[-60] - 1)
        features["dollar_mom"] = dollar_60d
    else:
        features["dollar_mom"] = np.nan

    # Composite regime score: +1 = risk-on, -1 = risk-off
    # Low VIX z + Low HY z + Steep curve + Weak dollar = risk-on
    risk_components = []
    if not np.isnan(features.get("vix_zscore", np.nan)):
        risk_components.append(-features["vix_zscore"] / 2)      # high VIX = risk-off
    if not np.isnan(features.get("hy_zscore", np.nan)):
        risk_components.append(-features["hy_zscore"] / 2)       # high HY = risk-off
    if not np.isnan(features.get("term_slope", np.nan)):
        risk_components.append(np.sign(features["term_slope"]))   # positive slope = risk-on
    if not np.isnan(features.get("dollar_mom", np.nan)):
        risk_components.append(-np.sign(features["dollar_mom"]))  # strong dollar = risk-off

    features["regime_score"] = float(np.mean(risk_components)) if risk_components else 0.0
    features["regime_score"] = float(np.clip(features["regime_score"], -1.0, 1.0))

    return features


def get_vix_series(macro: pd.DataFrame, as_of: pd.Timestamp) -> Optional[float]:
    """
    Get latest VIX level as of a date (with FRED lag).
    Returns None if unavailable. Used to replace SPY-vol proxy in v11.
    """
    pit = macro[macro.index <= as_of - pd.Timedelta(days=FRED_LAG_DAYS)]
    if pit.empty or "VIX" not in pit.columns:
        return None
    val = pit["VIX"].dropna()
    return float(val.iloc[-1]) if len(val) > 0 else None


def compute_macro_signals(macro: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float]:
    """
    Return cross-sectionally constant macro overlay signals.
    These are scalar values applied equally to all stocks (regime tilt).
    """
    feats = get_macro_features(macro, as_of)
    return {
        "macro_risk_on":  max(0.0, feats.get("regime_score", 0.0)),
        "macro_risk_off": max(0.0, -feats.get("regime_score", 0.0)),
        "macro_vix":      feats.get("vix_level", 20.0),
        "macro_term":     feats.get("term_slope", 1.0),
        "macro_credit":   -feats.get("hy_zscore", 0.0),  # flipped: high HY z = bearish
    }


if __name__ == "__main__":
    print("Fetching FRED macro data from 2018...")
    macro = pull_fred(start="2018-01-01")
    print(f"\nShape: {macro.shape}")
    print(f"Columns: {macro.columns.tolist()}")
    print(f"\nMost recent 5 rows:")
    print(macro.tail(5).to_string())

    # Spot check: VIX peak during COVID
    covid_peak = pd.Timestamp("2020-03-16")
    vix_covid = macro.loc[macro.index <= covid_peak, "VIX"].iloc[-1]
    print(f"\nVIX on ~{covid_peak.date()}: {vix_covid:.1f}  (historical high: ~82.69)")
    # VIX can vary slightly day-to-day; 82 was the intraday, close was ~82
    assert vix_covid > 75, f"VIX COVID peak should be > 75, got {vix_covid:.1f}"

    # Test macro features
    feats = get_macro_features(macro, as_of=pd.Timestamp("2024-01-01"))
    print(f"\nMacro features as of 2024-01-01:")
    for k, v in feats.items():
        print(f"  {k:20s}: {v}")

    print("\n✓ FRED macro data OK")
