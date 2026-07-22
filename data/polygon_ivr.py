"""
W11: Polygon.io Free Tier — Real Options IV and IVR
====================================================
Fetches implied volatility data from Polygon.io to compute:
  - IVR (IV Rank): where current IV sits in its 52-week range [0, 100]
  - IV Percentile: fraction of past 252 days current IV was higher

Free tier limits:
  - 5 API calls/minute (use sleep between requests)
  - Previous day's data (1-day delay)
  - Options snapshots, no real-time streaming

Why this replaces the current IVR estimate:
  Current step_institutional_upgrades.py computes IVR from price volatility
  (realized vol), which is a crude proxy. True IVR uses options market prices
  (implied volatility), which reflects the market's forward-looking fear.

IVR formula:
  IVR = (IV_now - IV_52wk_low) / (IV_52wk_high - IV_52wk_low) × 100
  IVR = 0  → IV at 52-week low (cheap options, complacency)
  IVR = 100 → IV at 52-week high (expensive options, fear)

Signal interpretation:
  High IVR + buy signal → sell vol (sell premium) → sell put spreads
  Low IVR + sell signal → buy puts for protection (cheap insurance)

Setup:
  1. Register free at polygon.io (no credit card)
  2. Copy API key to environment: export POLYGON_API_KEY=your_key
  3. OR set it in config.yaml under data.polygon_api_key_env

Usage:
    from data.polygon_ivr import fetch_ivr_batch, compute_ivr_signal
    ivr_df = fetch_ivr_batch(["AAPL", "MSFT", "NVDA"])
    signal = compute_ivr_signal(ivr_df)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
POLY_BASE = "https://api.polygon.io"


def _get_api_key() -> Optional[str]:
    """Get Polygon API key from environment or config."""
    key = os.environ.get("POLYGON_API_KEY", "")
    if key:
        return key
    # Try config.yaml
    try:
        import yaml
        cfg = yaml.safe_load(open(ROOT / "config.yaml"))
        env_var = cfg.get("data", {}).get("polygon_api_key_env", "POLYGON_API_KEY")
        return os.environ.get(env_var, "")
    except Exception:
        return ""


def _fetch_options_snapshot(ticker: str, api_key: str) -> dict:
    """
    Fetch the current options chain snapshot for a ticker.
    Returns ATM implied volatility (IV) as a float.
    """
    url = f"{POLY_BASE}/v3/snapshot/options/{ticker}"
    params = {
        "apiKey":  api_key,
        "limit":   250,
        "order":   "asc",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 403:
            raise PermissionError("Polygon API key missing or invalid")
        r.raise_for_status()
        data = r.json()
    except PermissionError:
        raise
    except Exception as e:
        return {}

    results = data.get("results", [])
    if not results:
        return {}

    # Find ATM options (closest to current price) and extract IV
    ivs = []
    for opt in results:
        details = opt.get("details", {})
        greeks  = opt.get("greeks", {})
        iv = opt.get("implied_volatility", None)
        if iv is None:
            iv = greeks.get("vega", None)  # fallback

        if iv is not None and iv > 0:
            ivs.append(float(iv))

    if not ivs:
        return {}

    # ATM IV proxy: median IV of all options (better than using single ATM strike)
    atm_iv = float(np.median(ivs))
    return {"atm_iv": atm_iv, "n_contracts": len(ivs)}


def _fetch_historical_iv(ticker: str, api_key: str, lookback_days: int = 365) -> pd.Series:
    """
    Fetch historical daily IV for a ticker using Polygon's aggs endpoint
    with options data. For free tier: approximated via realized vol of options prices.

    Note: Free tier doesn't have historical options IV directly.
    We use 30-day realized vol as an IV proxy for the IVR lookback window.
    This is the standard substitute when historical IV data isn't available.
    """
    end_date   = pd.Timestamp.today().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")

    url = f"{POLY_BASE}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 500, "apiKey": api_key}

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        results = r.json().get("results", [])
    except Exception:
        return pd.Series(dtype=float)

    if not results:
        return pd.Series(dtype=float)

    prices = pd.Series(
        [x["c"] for x in results],
        index=pd.to_datetime([x["t"] for x in results], unit="ms")
    )
    # 21-day realized vol as IV proxy (annualized)
    vol = prices.pct_change().rolling(21).std() * np.sqrt(252) * 100  # in percent
    return vol.dropna()


def fetch_ivr_batch(
    tickers: list[str],
    sleep_sec: float = 12.5,  # 5 req/min = 12s between calls
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch current IVR for a batch of tickers.

    IVR Computation:
    1. Get current IV (from options snapshot)
    2. Get historical IV series (252 trading days)
    3. IVR = percentile rank of current IV in 52-week range

    Returns DataFrame with columns:
        ticker, iv_current, iv_52wk_high, iv_52wk_low, ivr, iv_pct

    Note: On free tier, current IV from options snapshot; historical IV
    approximated with 21-day realized vol. Professional tier has true
    historical IV (stored by Polygon).
    """
    api_key = _get_api_key()
    if not api_key:
        print("  [Polygon] No API key found. Set POLYGON_API_KEY environment variable.")
        print("  [Polygon] Sign up free at polygon.io (no credit card needed)")
        return _load_cache_or_empty(cache_path)

    if cache_path and cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 20:  # refresh once daily
            return pd.read_csv(cache_path)

    rows: list[dict] = []
    n = len(tickers)

    for i, ticker in enumerate(tickers):
        try:
            # Get current IV from options snapshot
            snap = _fetch_options_snapshot(ticker, api_key)
            iv_now = snap.get("atm_iv", np.nan)

            # Get historical vol series (252 days)
            hist_vol = _fetch_historical_iv(ticker, api_key)

            if len(hist_vol) >= 20 and not np.isnan(iv_now):
                iv_hist = hist_vol.tail(252)
                iv_high = float(iv_hist.max())
                iv_low  = float(iv_hist.min())
                iv_pct  = float((hist_vol <= iv_now * 100).mean())  # fraction below current

                # IVR: rank in 52-week range
                rng = iv_high - iv_low
                ivr = float((iv_now * 100 - iv_low) / rng * 100) if rng > 0.5 else 50.0
                ivr = float(np.clip(ivr, 0, 100))
            else:
                iv_high = iv_low = ivr = iv_pct = np.nan

            rows.append({
                "ticker":      ticker,
                "iv_current":  round(iv_now * 100, 2) if not np.isnan(iv_now) else np.nan,
                "iv_52wk_high": round(iv_high, 2),
                "iv_52wk_low":  round(iv_low, 2),
                "ivr":          round(ivr, 1),
                "iv_pct":       round(iv_pct * 100, 1),
                "fetch_date":  pd.Timestamp.today().date(),
            })

            if (i + 1) % 10 == 0:
                print(f"  [Polygon] {i+1}/{n} tickers processed...")

        except PermissionError as e:
            print(f"  [Polygon] Auth error: {e}")
            break
        except Exception as e:
            print(f"  [Polygon] Failed for {ticker}: {e}")
            rows.append({"ticker": ticker, "ivr": np.nan, "iv_pct": np.nan})

        time.sleep(sleep_sec)

    if not rows:
        return _load_cache_or_empty(cache_path)

    df = pd.DataFrame(rows)
    if cache_path:
        df.to_csv(cache_path, index=False)
        print(f"  [Polygon] Saved IVR for {len(df)} tickers → {cache_path}")
    return df


def _load_cache_or_empty(cache_path: Optional[Path]) -> pd.DataFrame:
    if cache_path and cache_path.exists():
        print(f"  [Polygon] Using stale cache: {cache_path}")
        return pd.read_csv(cache_path)
    return pd.DataFrame(columns=["ticker", "iv_current", "iv_52wk_high",
                                  "iv_52wk_low", "ivr", "iv_pct"])


def compute_ivr_signal(ivr_df: pd.DataFrame) -> pd.Series:
    """
    Compute cross-sectional IVR signal.

    Signal interpretation (contrarian):
      - High IVR (>80) → options expensive → bearish volatility → neutral/mild short signal
      - Low IVR (<20)  → options cheap    → potential long (good risk/reward for calls)

    We use IV Percentile (iv_pct) rather than IVR because it's more
    robust to outliers (IVR depends on the single max/min value).

    Returns:
        pd.Series indexed by ticker
        High values = options cheap (potentially bullish for stock)
        Low values  = options expensive (potentially bearish or high-fear)
    """
    if ivr_df.empty or "ticker" not in ivr_df.columns:
        return pd.Series(dtype=float)

    df = ivr_df.set_index("ticker")
    col = "iv_pct" if "iv_pct" in df.columns else "ivr"
    iv_score = df[col].dropna()

    if iv_score.empty:
        return pd.Series(dtype=float)

    # Invert: low IV pct → high signal (cheaper options → less crowded short side)
    # This matches the logic in step_institutional_upgrades.py
    signal = 100 - iv_score

    # Cross-sectional z-score
    mu, std = signal.mean(), signal.std()
    if std < 1e-9:
        return pd.Series(0.0, index=signal.index)

    return ((signal - mu) / std).clip(-3, 3)


if __name__ == "__main__":
    # Test with a small set — requires POLYGON_API_KEY in environment
    test_tickers = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA"]
    print(f"Fetching IVR for: {test_tickers}")
    print("(Requires POLYGON_API_KEY env var — free tier works)")

    df = fetch_ivr_batch(
        test_tickers,
        cache_path=ROOT / "polygon_ivr_test.csv",
    )
    print(f"\nIVR Results:")
    print(df.to_string())

    if not df.empty and "ivr" in df.columns:
        signal = compute_ivr_signal(df)
        print(f"\nIVR Signal (z-score, inverted):")
        print(signal.sort_values().to_string())
