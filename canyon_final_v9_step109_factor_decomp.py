#!/usr/bin/env python3
"""
canyon_final_v9_step109_factor_decomp.py
==========================================
Canyon v9 Step 109 — Fama-French 4-Factor Decomposition

Decomposes each holding's returns into:
  MKT  — Market beta (SPY)
  SMB  — Size factor (IWM - SPY)
  HML  — Value factor (IWD - IWF)
  MOM  — Momentum factor (MTUM - SPY)
  Alpha — Intercept after factor removal ("True Alpha")

All math is pure numpy OLS (no scipy).  Factor returns are computed
inline from cached price data; no external factor library needed.

Outputs:
  factor_decomposition.csv   — per-ticker OLS coefficients + True Alpha
  factor_decomp_report.md    — full diagnostic report

Usage:
  python canyon_final_v9_step109_factor_decomp.py               # all holdings
  python canyon_final_v9_step109_factor_decomp.py --lookback 120
  python canyon_final_v9_step109_factor_decomp.py --min-obs 30
  python canyon_final_v9_step109_factor_decomp.py --dry-run
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# Factor proxy tickers
FACTOR_PROXIES = {
    "SPY":  "SPY",   # Market
    "IWM":  "IWM",   # Small-cap leg of SMB
    "IWD":  "IWD",   # Value leg of HML
    "IWF":  "IWF",   # Growth leg of HML
    "MTUM": "MTUM",  # Momentum leg of MOM
}

DEFAULT_LOOKBACK = 252   # trading days for regression window
DEFAULT_MIN_OBS  = 30    # minimum observations required


# ─────────────────────────────────────────────────────────────────────────────
# [1/4] Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_price_matrix(lookback: int) -> pd.DataFrame:
    """
    Load sp500_price_cache.csv (wide: dates as index, tickers as columns).
    Returns the last `lookback + 5` rows of log returns.
    """
    path = ROOT / "sp500_price_cache.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.sort_index().tail(lookback + 5)
        # Convert to log returns
        log_ret = np.log(df / df.shift(1)).iloc[1:]
        return log_ret
    except Exception as e:
        print(f"  [price_matrix] error: {e}")
        return pd.DataFrame()


def _load_holdings() -> list[str]:
    """
    Return tickers to analyse (from daily_picks.csv or alpha_scores.csv).
    """
    for fname in ("daily_picks.csv", "alpha_scores.csv"):
        path = ROOT / fname
        if path.exists():
            try:
                df = pd.read_csv(path)
                if "ticker" in df.columns:
                    return df["ticker"].dropna().drop_duplicates().tolist()
            except Exception:
                pass
    return []


def _load_sector_map() -> dict[str, str]:
    """Return {ticker: sector} from sector_map.csv."""
    path = ROOT / "sector_map.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        if {"ticker", "sector"}.issubset(df.columns):
            return dict(zip(df["ticker"].astype(str), df["sector"].astype(str)))
    except Exception:
        pass
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# [2/4] Factor construction
# ─────────────────────────────────────────────────────────────────────────────

def build_factor_returns(returns: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Build Fama-French 4-factor return series from price matrix.

    Factor definitions:
      MKT = SPY log-return  (excess over zero; risk-free ~0 daily)
      SMB = IWM - SPY
      HML = IWD - IWF
      MOM = MTUM - SPY

    Returns DataFrame with columns [MKT, SMB, HML, MOM], index = dates.
    Returns None if any required proxy is missing.
    """
    required = list(FACTOR_PROXIES.values())
    missing  = [t for t in required if t not in returns.columns]
    if missing:
        print(f"  [factors] missing proxies in price cache: {missing}")
        print("  [factors] will use available proxies only")

    factor_cols: dict[str, pd.Series] = {}

    if "SPY" in returns.columns:
        factor_cols["MKT"] = returns["SPY"]
    else:
        print("  [factors] SPY missing — cannot build MKT factor")
        return None

    if "IWM" in returns.columns:
        factor_cols["SMB"] = returns["IWM"] - returns["SPY"]
    else:
        factor_cols["SMB"] = pd.Series(0.0, index=returns.index)
        print("  [factors] IWM missing — SMB set to zero")

    if "IWD" in returns.columns and "IWF" in returns.columns:
        factor_cols["HML"] = returns["IWD"] - returns["IWF"]
    else:
        factor_cols["HML"] = pd.Series(0.0, index=returns.index)
        print("  [factors] IWD/IWF missing — HML set to zero")

    if "MTUM" in returns.columns:
        factor_cols["MOM"] = returns["MTUM"] - returns["SPY"]
    else:
        factor_cols["MOM"] = pd.Series(0.0, index=returns.index)
        print("  [factors] MTUM missing — MOM set to zero")

    factors = pd.DataFrame(factor_cols).dropna()
    print(f"  [factors] Factor matrix: {factors.shape[0]} observations × 4 factors")
    return factors


# ─────────────────────────────────────────────────────────────────────────────
# [3/4] OLS regression + result packaging
# ─────────────────────────────────────────────────────────────────────────────

def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Pure numpy OLS: y = X @ beta.
    X should already include a constant column.
    Returns (beta, r_squared, residual_std).
    """
    beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    y_hat    = X @ beta
    ss_res   = float(np.sum((y - y_hat) ** 2))
    ss_tot   = float(np.sum((y - y.mean()) ** 2))
    r2       = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    res_std  = float(np.std(y - y_hat, ddof=X.shape[1]))
    return beta, r2, res_std


def decompose_ticker(
    ticker: str,
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    min_obs: int,
) -> Optional[dict]:
    """
    Run Fama-French 4-factor OLS for a single ticker.
    Returns dict of results, or None if insufficient data.
    """
    if ticker not in returns.columns:
        return None

    # Align ticker returns with factor dates
    y_raw = returns[ticker].dropna()
    common_idx = y_raw.index.intersection(factors.index)
    if len(common_idx) < min_obs:
        return None

    y  = y_raw.loc[common_idx].values
    Xf = factors.loc[common_idx].values          # [n, 4]
    # Add constant (intercept = alpha)
    ones = np.ones((len(y), 1))
    X    = np.hstack([ones, Xf])                  # [n, 5]

    beta, r2, res_std = _ols(y, X)

    # beta layout: [intercept, MKT, SMB, HML, MOM]
    alpha_daily = float(beta[0])
    mkt_beta    = float(beta[1])
    smb_beta    = float(beta[2])
    hml_beta    = float(beta[3])
    mom_beta    = float(beta[4])

    # Annualise alpha: assume 252 trading days
    alpha_annual = alpha_daily * 252 * 100   # in percent

    # Factor return contributions (mean factor return × beta × 252 × 100)
    f_means = factors.loc[common_idx].mean().values
    mkt_contrib = float(f_means[0] * mkt_beta * 252 * 100)
    smb_contrib = float(f_means[1] * smb_beta * 252 * 100)
    hml_contrib = float(f_means[2] * hml_beta * 252 * 100)
    mom_contrib = float(f_means[3] * mom_beta * 252 * 100)

    return {
        "ticker":          ticker,
        "n_obs":           len(y),
        "alpha_daily_bps": round(alpha_daily * 10_000, 2),   # daily alpha in bps
        "alpha_annual_pct":round(alpha_annual, 2),           # annualised %
        "mkt_beta":        round(mkt_beta, 3),
        "smb_beta":        round(smb_beta, 3),
        "hml_beta":        round(hml_beta, 3),
        "mom_beta":        round(mom_beta, 3),
        "r2":              round(r2, 4),
        "residual_std_bps":round(res_std * 10_000, 2),
        "mkt_contrib_pct": round(mkt_contrib, 2),
        "smb_contrib_pct": round(smb_contrib, 2),
        "hml_contrib_pct": round(hml_contrib, 2),
        "mom_contrib_pct": round(mom_contrib, 2),
    }


def run_decomposition(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    holdings: list[str],
    min_obs: int,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """
    Run 4-factor decomposition for all holdings.
    Returns a DataFrame sorted by alpha_annual_pct descending.
    """
    results = []
    skipped = 0
    for ticker in holdings:
        rec = decompose_ticker(ticker, returns, factors, min_obs)
        if rec is None:
            skipped += 1
            continue
        rec["sector"] = sector_map.get(ticker, "Unknown")
        results.append(rec)

    if skipped:
        print(f"  [decomp] {skipped} tickers skipped (insufficient data)")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("alpha_annual_pct", ascending=False)
    df = df.reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# [4/4] Report writer + main
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value: float, max_abs: float = 5.0, width: int = 20) -> str:
    """ASCII bar proportional to value."""
    if max_abs == 0:
        return " " * width
    pct  = min(abs(value) / max_abs, 1.0)
    fill = int(pct * width)
    sign = "▲" if value >= 0 else "▼"
    return sign + "█" * fill + "░" * (width - fill)


def write_report(df: pd.DataFrame, lookback: int, dry_run: bool = False) -> str:
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")

    if df.empty:
        md = "# Factor Decomposition Report\n\n_No data available._\n"
        if not dry_run:
            (ROOT / "factor_decomp_report.md").write_text(md, encoding="utf-8")
        return md

    n = len(df)
    avg_r2    = df["r2"].mean()
    avg_alpha = df["alpha_annual_pct"].mean()
    pos_alpha = (df["alpha_annual_pct"] > 0).sum()

    lines = [
        f"# Canyon v9 — Fama-French 4-Factor Decomposition\n\n",
        f"**Date:** {today}  |  **Lookback:** {lookback}d  |  **Holdings:** {n}\n\n",
        f"---\n\n",
        f"## Summary\n\n",
        f"| Metric | Value |\n|--------|-------|\n",
        f"| Tickers decomposed        | {n} |\n",
        f"| Avg R² (factor explained) | {avg_r2:.2%} |\n",
        f"| Avg True Alpha (ann.)     | {avg_alpha:+.2f}% |\n",
        f"| Positive-alpha holdings   | {pos_alpha} / {n} |\n",
        f"\n---\n\n",
        f"## Factor Loadings\n\n",
        f"> MKT=market beta, SMB=size(+small), HML=value(+value), MOM=momentum(+mom)\n\n",
        f"| Ticker | Sector | α ann% | MKT β | SMB β | HML β | MOM β | R² |\n",
        f"|--------|--------|--------|-------|-------|-------|-------|----|  \n",
    ]

    for _, row in df.iterrows():
        alpha_str = f"{row['alpha_annual_pct']:+.2f}%"
        r2_str    = f"{row['r2']:.2%}"
        tick_str  = f"**{row['ticker']}**" if abs(row['alpha_annual_pct']) >= 5 else str(row['ticker'])
        lines.append(
            f"| {tick_str} | {row['sector'][:18]} | {alpha_str} "
            f"| {row['mkt_beta']:.2f} | {row['smb_beta']:.2f} "
            f"| {row['hml_beta']:.2f} | {row['mom_beta']:.2f} | {r2_str} |\n"
        )

    lines += [
        f"\n---\n\n",
        f"## True Alpha Ranking (Top 10)\n\n",
        f"Tickers sorted by annualised True Alpha (intercept after removing 4 factors).\n",
        f"Positive alpha = manager skill / unique edge beyond factor exposure.\n\n",
    ]

    top10 = df.nlargest(10, "alpha_annual_pct")
    max_abs = max(abs(df["alpha_annual_pct"].max()), abs(df["alpha_annual_pct"].min()), 5.0)
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        bar  = _bar(row["alpha_annual_pct"], max_abs)
        lines.append(
            f"{rank:>2}. **{row['ticker']:<6}** {row['alpha_annual_pct']:+6.2f}%  {bar}\n"
        )

    lines += [
        f"\n---\n\n",
        f"## Factor Contribution to Returns (Annualised)\n\n",
        f"| Ticker | MKT contrib | SMB contrib | HML contrib | MOM contrib | α (residual) |\n",
        f"|--------|------------|------------|------------|------------|-------------|\n",
    ]
    for _, row in df.head(20).iterrows():
        lines.append(
            f"| {row['ticker']} "
            f"| {row['mkt_contrib_pct']:+.2f}% "
            f"| {row['smb_contrib_pct']:+.2f}% "
            f"| {row['hml_contrib_pct']:+.2f}% "
            f"| {row['mom_contrib_pct']:+.2f}% "
            f"| {row['alpha_annual_pct']:+.2f}% |\n"
        )

    lines += [
        f"\n---\n\n",
        f"## Interpretation Guide\n\n",
        f"- **MKT β > 1.0** — more volatile than market; β < 1.0 = defensive\n",
        f"- **SMB β > 0** — tilted toward small-cap; < 0 = large-cap bias\n",
        f"- **HML β > 0** — tilted toward value stocks; < 0 = growth bias\n",
        f"- **MOM β > 0** — momentum tilt (following recent winners)\n",
        f"- **True Alpha > 0** — positive after stripping all factor exposures\n",
        f"- **R² > 0.90** — returns are almost entirely explained by factors\n",
        f"- **R² < 0.50** — significant idiosyncratic component (alpha or noise)\n",
    ]

    md = "".join(lines)
    if not dry_run:
        (ROOT / "factor_decomp_report.md").write_text(md, encoding="utf-8")
    return md


def run_factor_decomp(
    lookback: int = DEFAULT_LOOKBACK,
    min_obs:  int = DEFAULT_MIN_OBS,
    dry_run:  bool = False,
) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Canyon v9  Step 109 — Factor Decomposition")
    print("=" * 60)
    print(f"  Lookback: {lookback}d   Min observations: {min_obs}")

    # ── [1/4] Load ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading price data and holdings …")
    returns  = _load_price_matrix(lookback)
    holdings = _load_holdings()
    sectors  = _load_sector_map()

    if returns.empty:
        print("  ERROR: no price data — run step60 first")
        return pd.DataFrame()
    print(f"  Returns matrix: {returns.shape}  Holdings: {len(holdings)}")

    # ── [2/4] Build factors ────────────────────────────────────────────────
    print("\n[2/4] Building factor return series …")
    factors = build_factor_returns(returns)
    if factors is None or factors.empty:
        print("  ERROR: could not build factor series — SPY missing from cache")
        return pd.DataFrame()

    # ── [3/4] Decompose ────────────────────────────────────────────────────
    print(f"\n[3/4] Running OLS decomposition for {len(holdings)} holdings …")
    df = run_decomposition(returns, factors, holdings, min_obs, sectors)
    if df.empty:
        print("  No results — all holdings lacked sufficient history")
        return pd.DataFrame()
    print(f"  Decomposed: {len(df)} tickers  "
          f"  Avg R²: {df['r2'].mean():.2%}  "
          f"  Avg True Alpha: {df['alpha_annual_pct'].mean():+.2f}%")

    # ── [4/4] Write outputs ────────────────────────────────────────────────
    print("\n[4/4] Writing outputs …")
    if not dry_run:
        csv_path = ROOT / "factor_decomposition.csv"
        df.to_csv(csv_path, index=False)
        print(f"  [written] {csv_path}  ({len(df)} rows)")

    md = write_report(df, lookback, dry_run)
    if not dry_run:
        print(f"  [written] {ROOT / 'factor_decomp_report.md'}  ({len(md)} chars)")
    else:
        print("\n  [DRY-RUN] preview:")
        print(md[:1500] + ("\n… [truncated]" if len(md) > 1500 else ""))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Canyon v9 Step 109 — Fama-French 4-Factor Decomposition"
    )
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK,
                        help=f"Regression lookback in trading days (default: {DEFAULT_LOOKBACK})")
    parser.add_argument("--min-obs",  type=int, default=DEFAULT_MIN_OBS,
                        help=f"Minimum observations for regression (default: {DEFAULT_MIN_OBS})")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Preview outputs without writing files")
    args = parser.parse_args()

    df = run_factor_decomp(
        lookback=args.lookback,
        min_obs=args.min_obs,
        dry_run=args.dry_run,
    )

    n = len(df)
    if n > 0:
        avg_alpha = df["alpha_annual_pct"].mean()
        pos = (df["alpha_annual_pct"] > 0).sum()
        print(f"\n✓ Factor decomposition complete — {n} tickers  "
              f"Avg True Alpha: {avg_alpha:+.2f}%  "
              f"Positive alpha: {pos}/{n}\n")
    else:
        print("\n✓ Factor decomposition complete — no results produced\n")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
