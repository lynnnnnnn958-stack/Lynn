"""
W38-W39: Slippage Tracker + Almgren-Chriss Calibration
========================================================
W38: Tracks actual execution slippage from paper trading log (execution_log.csv)
     vs Almgren-Chriss market impact model predictions.

W39: Calibrates the AC η parameter from observed slippage data.

Almgren-Chriss (2001) square-root market impact model:
  impact_per_share = η × σ × √(trade_size_shares / ADV_shares)
  total_impact_bps = η × daily_vol_pct × √(adv_fraction) × 10000

where:
  η           = market impact coefficient (to be calibrated from data)
  σ           = daily return volatility (fraction)
  adv_fraction = trade_size / ADV

Default starting value: η = 0.10 (from Almgren et al. 2005 empirical study)

Calibration method:
  observed_slippage_bps = actual_price - arrival_price (in bps)
  Minimise: Σ (observed_slippage - η × σ × √(adv_fraction) × 10000)²

Outputs:
  slippage_log.csv          — ticker, date, predicted_slippage, actual_slippage
  almgren_chriss_params.json — calibrated η value + diagnostics

Usage:
    from monitoring.slippage import run_slippage_analysis, calibrate_ac_model
    df = run_slippage_analysis()
    params = calibrate_ac_model(df)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent

ETA_DEFAULT  = 0.10    # Almgren-Chriss η (starting value)
ADV_PROXY_SHARES = 1_000_000  # S&P 500 stock ADV proxy (1M shares/day average)


def _load_execution_log() -> pd.DataFrame:
    """Load paper trading execution log."""
    p = ROOT / "execution_log.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["submitted"] if "submitted" in pd.read_csv(p, nrows=0).columns else None)
    return pd.DataFrame()


def _load_price_at_time(ticker: str, submit_time: str, prices: pd.DataFrame) -> float:
    """Get price closest to submission time (arrival price)."""
    try:
        date = pd.Timestamp(submit_time).normalize()
        if date in prices.index and ticker in prices.columns:
            return float(prices.loc[date, ticker])
    except Exception:
        pass
    return np.nan


def compute_ac_predicted_slippage(
    qty_shares: int,
    daily_vol: float,
    adv_shares: float = ADV_PROXY_SHARES,
    eta: float = ETA_DEFAULT,
) -> float:
    """
    Compute Almgren-Chriss predicted slippage in basis points.

    impact_bps = η × daily_vol_pct × √(qty/ADV) × 10000
    """
    adv_fraction = qty_shares / (adv_shares + 1)
    impact_bps   = eta * daily_vol * np.sqrt(adv_fraction) * 10000
    return float(impact_bps)


def run_slippage_analysis(
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    W38: Compare actual vs predicted slippage for all executed orders.

    Returns DataFrame with slippage analysis per order.
    """
    if output_path is None:
        output_path = ROOT / "slippage_log.csv"

    exec_log = _load_execution_log()
    if exec_log.empty:
        print("  [Slippage] No execution log found (execution_log.csv)")
        print("  Run paper trades via execution/alpaca_exec.py to generate data")
        # Generate synthetic demo data for testing
        demo_rows = []
        for i, tkr in enumerate(["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]):
            qty = 100
            vol = 0.02 + i * 0.005
            predicted = compute_ac_predicted_slippage(qty, vol / np.sqrt(252))
            actual    = predicted * (0.8 + np.random.rand() * 0.6)  # ±30% noise
            demo_rows.append({
                "ticker":        tkr,
                "qty":           qty,
                "side":          "buy",
                "daily_vol":     round(vol, 4),
                "predicted_bps": round(predicted, 2),
                "actual_bps":    round(actual, 2),
                "error_bps":     round(actual - predicted, 2),
                "source":        "synthetic_demo",
            })
        df = pd.DataFrame(demo_rows)
        df.to_csv(output_path, index=False)
        print(f"  [Slippage] Generated synthetic demo data → {output_path}")
        return df

    # Load price cache for arrival prices
    for fname in ("sp500_price_cache_8yr.csv", "sp500_price_cache.csv"):
        p = ROOT / fname
        if p.exists():
            prices = pd.read_csv(p, index_col=0, parse_dates=True)
            break
    else:
        prices = pd.DataFrame()

    rows = []
    for _, order in exec_log.iterrows():
        tkr = str(order.get("ticker", ""))
        qty = int(order.get("qty", 0))
        side = str(order.get("side", "buy"))

        # Daily vol from price cache
        if tkr in prices.columns and len(prices) >= 21:
            daily_vol = float(prices[tkr].pct_change().iloc[-21:].std())
        else:
            daily_vol = 0.015

        predicted_bps = compute_ac_predicted_slippage(qty, daily_vol)

        # Actual slippage: not available for dry_run / paper orders without fill prices
        actual_bps = np.nan
        if "fill_price" in order and "arrival_price" in order:
            fill_price    = float(order.get("fill_price", 0))
            arrival_price = float(order.get("arrival_price", fill_price))
            if arrival_price > 0:
                actual_bps = (fill_price / arrival_price - 1) * 10000 * \
                             (1 if side == "buy" else -1)

        rows.append({
            "ticker":        tkr,
            "qty":           qty,
            "side":          side,
            "submitted":     order.get("submitted", ""),
            "daily_vol":     round(daily_vol, 5),
            "predicted_bps": round(predicted_bps, 2),
            "actual_bps":    round(actual_bps, 2) if not np.isnan(actual_bps) else np.nan,
            "error_bps":     round(actual_bps - predicted_bps, 2) if not np.isnan(actual_bps) else np.nan,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  [Slippage] Logged {len(df)} orders → {output_path}")

    # Summary stats
    valid = df[df["actual_bps"].notna()]
    if not valid.empty:
        print(f"  Mean predicted: {valid['predicted_bps'].mean():.1f}bps")
        print(f"  Mean actual:    {valid['actual_bps'].mean():.1f}bps")
        print(f"  Mean error:     {valid['error_bps'].mean():.1f}bps")
    return df


def calibrate_ac_model(
    slippage_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> dict:
    """
    W39: Calibrate Almgren-Chriss η from observed slippage data.

    Uses least-squares: min Σ (actual_bps - η × σ × √(adv_frac) × 10000)²

    Returns dict: eta, r2, n_obs, commentary.
    """
    if output_path is None:
        output_path = ROOT / "almgren_chriss_params.json"

    if slippage_df is None:
        slippage_df = pd.read_csv(ROOT / "slippage_log.csv") \
                      if (ROOT / "slippage_log.csv").exists() else pd.DataFrame()

    # Default calibration if no real data
    if slippage_df.empty or "actual_bps" not in slippage_df.columns:
        params = {
            "eta":          ETA_DEFAULT,
            "r2":           np.nan,
            "n_obs":        0,
            "commentary":   "No actual slippage data — using literature default (η=0.10)",
            "calibrated":   False,
        }
        with open(output_path, "w") as f:
            json.dump(params, f, indent=2)
        print(f"  [ACCalib] Using default η={ETA_DEFAULT} (no observation data)")
        return params

    valid = slippage_df[slippage_df["actual_bps"].notna()].copy()
    if len(valid) < 10:
        params = {
            "eta":          ETA_DEFAULT,
            "r2":           np.nan,
            "n_obs":        len(valid),
            "commentary":   f"Insufficient data ({len(valid)} obs) — using default",
            "calibrated":   False,
        }
        with open(output_path, "w") as f:
            json.dump(params, f, indent=2)
        return params

    # Regressor: η × σ × √(qty/ADV) × 10000
    X = valid["daily_vol"].values * np.sqrt(valid["qty"].values / ADV_PROXY_SHARES) * 10000
    y = valid["actual_bps"].values

    # OLS without intercept: η = Σ(X×y) / Σ(X²)
    eta_hat = float(np.sum(X * y) / (np.sum(X ** 2) + 1e-12))
    eta_hat = float(np.clip(eta_hat, 0.01, 1.0))

    # R²
    y_pred  = eta_hat * X
    ss_res  = np.sum((y - y_pred) ** 2)
    ss_tot  = np.sum((y - np.mean(y)) ** 2)
    r2      = float(1 - ss_res / (ss_tot + 1e-12))

    params = {
        "eta":        round(eta_hat, 4),
        "r2":         round(r2, 3),
        "n_obs":      int(len(valid)),
        "mean_pred":  round(float((eta_hat * X).mean()), 2),
        "mean_actual": round(float(y.mean()), 2),
        "commentary": f"η={eta_hat:.3f} calibrated from {len(valid)} paper trade observations",
        "calibrated": True,
    }

    with open(output_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"  [ACCalib] η={eta_hat:.3f} (vs default {ETA_DEFAULT}), R²={r2:.2f}, n={len(valid)}")
    print(f"  Saved → {output_path}")
    return params


if __name__ == "__main__":
    print("W38: Slippage Analysis + W39: AC Calibration")
    print("=" * 50)
    slippage_df = run_slippage_analysis()
    params = calibrate_ac_model(slippage_df)
    print(f"\nAC parameters: {params}")
