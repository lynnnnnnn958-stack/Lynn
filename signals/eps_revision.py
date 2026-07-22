"""
W23: EPS Revision Momentum Signal
===================================
Captures analyst EPS estimate revisions — a well-documented, persistent alpha source
(Womack 1996; Jegadeesh & Kim 2006). The signal logic:

  Revision momentum = change in consensus EPS estimate over 4 weeks
  Positive revisions (upward estimate cuts) → outperform
  Negative revisions (downward cuts)       → underperform

Data source (free):
  SEC EDGAR 8-K Item 2.02 and Item 8.01 filings contain actual EPS and sometimes
  guidance numbers. We extract actual EPS from recent 10-Q/10-K filings to compare
  against prior consensus (naive consensus proxy = last quarter's actual EPS).

Since free analyst consensus data is unavailable without a paid vendor (Bloomberg,
FactSet, Refinitiv), we implement a PIT-compliant surrogate:

  EPS Surprise Momentum = standardised SUE (Standardised Unexpected Earnings)
    SUE_t = (actual_EPS_t - actual_EPS_{t-4q}) / std(EPS changes, last 8q)

  This captures the same drift as analyst revision signals (Bernard & Thomas 1989,
  post-earnings announcement drift lasting 60 days).

  Additional signal: EPS Trend (sequential quarters positive/negative)
    eps_trend_4q = sign of EPS growth across the last 4 quarters

PIT compliance:
  Uses EDGAR PIT fundamentals (edgar_pit_fundamentals.csv, filed date = know_date).
  Never looks at EPS from a future filing date.

Outputs:
  eps_revision_scores.csv — ticker, sue_score, eps_trend, combined_score, revision_score

Usage:
    from signals.eps_revision import compute_eps_revision, get_eps_revision_signal
    df = compute_eps_revision(as_of=pd.Timestamp("2024-03-01"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent

LOOKBACK_QUARTERS = 8   # quarters of EPS history for SUE denominator
MIN_QUARTERS      = 4   # minimum history required


# ─────────────────────────────────────────────────────────────────────────────
# 1. EDGAR PIT EPS loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_pit_eps(as_of: pd.Timestamp) -> pd.DataFrame:
    """
    Load EPS (basic) from edgar_pit_fundamentals.csv, enforcing PIT.

    Returns DataFrame: ticker × [period_end, know_date, eps_basic, shares_out]
    with rows filtered to know_date <= as_of.
    """
    pit_path = ROOT / "edgar_pit_fundamentals.csv"
    if not pit_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(pit_path, parse_dates=["period_end", "know_date"])

    # PIT filter: only use data filed on or before as_of
    df = df[df["know_date"] <= as_of].copy()

    # Keep only EPS Basic rows
    if "concept" in df.columns:
        df = df[df["concept"] == "eps_basic"]
    elif "eps_basic" not in df.columns:
        return pd.DataFrame()

    # Normalise: if concept column exists, pivot; otherwise assume eps_basic column
    if "concept" in df.columns and "value" in df.columns:
        eps_df = df.rename(columns={"value": "eps_basic"})
    else:
        eps_df = df.copy()

    needed = ["ticker", "period_end", "know_date", "eps_basic"]
    for c in needed:
        if c not in eps_df.columns:
            return pd.DataFrame()

    return eps_df[needed].dropna(subset=["eps_basic"]).copy()


def _load_pit_fundamentals_wide(as_of: pd.Timestamp) -> pd.DataFrame:
    """
    Load the wide-format PIT fundamentals (all concepts) from edgar_pit_fundamentals.csv.
    Used as a richer fallback if the file has a wide structure.
    """
    pit_path = ROOT / "edgar_pit_fundamentals.csv"
    if not pit_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(pit_path, parse_dates=["period_end", "know_date"])
    df = df[df["know_date"] <= as_of].copy()

    # If there's a 'concept' column, pivot it wide
    if "concept" in df.columns and "value" in df.columns:
        eps_rows = df[df["concept"] == "eps_basic"].copy()
        eps_rows = eps_rows.rename(columns={"value": "eps_basic"})
        return eps_rows[["ticker", "period_end", "know_date", "eps_basic"]].dropna()

    # Already wide: look for eps_basic column
    if "eps_basic" in df.columns:
        return df[["ticker", "period_end", "know_date", "eps_basic"]].dropna()

    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 2. SUE computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_sue(eps_history: pd.Series) -> float:
    """
    Standardised Unexpected Earnings (Bernard & Thomas 1989).

    SUE = (most_recent_EPS - EPS_4q_ago) / std(EPS_changes_last_8q)

    Positive SUE → analyst expectations likely to be revised up (PEAD).
    """
    if len(eps_history) < MIN_QUARTERS + 1:
        return np.nan

    # Sort by period (oldest to most recent)
    eps = eps_history.sort_index()

    actual   = float(eps.iloc[-1])
    expected = float(eps.iloc[-5]) if len(eps) >= 5 else float(eps.iloc[0])

    # Denominator: std of quarter-over-quarter changes
    changes = eps.diff().dropna()
    if len(changes) >= 2:
        denom = float(changes.std())
    else:
        denom = float(abs(actual - expected)) or 1.0
    denom = max(denom, 1e-6)

    return (actual - expected) / denom


def _compute_eps_trend(eps_history: pd.Series) -> float:
    """
    EPS growth trend across last 4 quarters.

    Returns: fraction of last 4 quarter-over-quarter changes that are positive.
    Range [0, 1]: 1.0 = all 4 quarters improving, 0.0 = all declining.
    """
    eps = eps_history.sort_index()
    changes = eps.diff().dropna()
    if len(changes) < 2:
        return 0.5
    last4 = changes.iloc[-4:]
    return float((last4 > 0).mean())


# ─────────────────────────────────────────────────────────────────────────────
# 3. Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_eps_revision(
    as_of: Optional[pd.Timestamp] = None,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute EPS revision/SUE signal for all tickers in PIT fundamentals.

    Args:
        as_of:      Point-in-time date (default: today). Only uses filed data.
        cache_path: Output CSV path (default: ROOT/eps_revision_scores.csv).

    Returns DataFrame:
        ticker, n_quarters, sue_score, eps_trend, combined_score, revision_score
    """
    if as_of is None:
        as_of = pd.Timestamp.today().normalize()
    if cache_path is None:
        cache_path = ROOT / "eps_revision_scores.csv"

    eps_df = _load_pit_fundamentals_wide(as_of)
    if eps_df.empty:
        # Fallback: try existing earnings_surprise_scores.csv
        fallback = ROOT / "earnings_surprise_scores.csv"
        if fallback.exists():
            print(f"  [EPSRevision] EDGAR PIT not available → using existing {fallback.name}")
            df = pd.read_csv(fallback)
            if "ticker" in df.columns and "rank_sue" in df.columns:
                out = df[["ticker", "rank_sue"]].copy()
                out["sue_score"]      = out["rank_sue"]
                out["eps_trend"]      = 50.0
                out["combined_score"] = out["rank_sue"]
                out["revision_score"] = out["rank_sue"]
                out["source"]         = "fallback_sue_csv"
                out.to_csv(cache_path, index=False)
                return out
        print("  [EPSRevision] No EPS data available — returning empty signal")
        return pd.DataFrame()

    rows = []
    for ticker, group in eps_df.groupby("ticker"):
        # Sort by period_end, use most recent LOOKBACK_QUARTERS entries
        g = group.sort_values("period_end")
        if len(g) < MIN_QUARTERS:
            continue

        # Build EPS time series indexed by period_end
        eps_series = g.set_index("period_end")["eps_basic"].sort_index()
        eps_series = eps_series.iloc[-LOOKBACK_QUARTERS:]

        sue   = _compute_sue(eps_series)
        trend = _compute_eps_trend(eps_series)
        n_q   = len(eps_series)

        rows.append({
            "ticker":       ticker,
            "n_quarters":   n_q,
            "sue_score":    round(sue,   4) if not np.isnan(sue) else np.nan,
            "eps_trend":    round(trend, 4),
            "most_recent_eps": float(eps_series.iloc[-1]),
            "as_of_date":   str(as_of.date()),
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).dropna(subset=["sue_score"])

    # Cross-sectional z-score of SUE
    sue_mu, sue_std = out["sue_score"].mean(), out["sue_score"].std()
    if sue_std > 1e-9:
        out["sue_zscore"] = (out["sue_score"] - sue_mu) / sue_std
    else:
        out["sue_zscore"] = 0.0

    # Combined score: 70% SUE + 30% EPS trend
    out["combined_score"] = (
        0.70 * out["sue_zscore"].clip(-3, 3) / 3.0 * 50 + 50 +  # maps [-3,3] → [0,100]
        0.30 * out["eps_trend"] * 100
    ).clip(0, 100)

    # Cross-rank → revision_score in [0, 100]
    out["revision_score"] = out["combined_score"].rank(pct=True) * 100

    out = out.sort_values("revision_score", ascending=False).reset_index(drop=True)
    out.to_csv(cache_path, index=False)
    print(f"  [EPSRevision] {len(out)} tickers → {cache_path}")
    print(f"  [EPSRevision] Median SUE={out['sue_score'].median():.2f}, "
          f"EPS trend={out['eps_trend'].median():.0%}")
    return out


def get_eps_revision_signal(
    as_of: Optional[pd.Timestamp] = None,
    cache_path: Optional[Path] = None,
) -> pd.Series:
    """
    Load or compute the revision_score as a cross-sectional signal.

    Returns pd.Series: ticker → revision_score [0, 100].
    Higher = more positive EPS revision momentum.
    """
    if cache_path is None:
        cache_path = ROOT / "eps_revision_scores.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path)
        if "ticker" in df.columns and "revision_score" in df.columns:
            return df.set_index("ticker")["revision_score"]

    df = compute_eps_revision(as_of=as_of, cache_path=cache_path)
    if df.empty:
        return pd.Series(dtype=float)
    return df.set_index("ticker")["revision_score"]


if __name__ == "__main__":
    print("W23: EPS Revision Momentum Signal")
    print("=" * 50)
    today = pd.Timestamp.today().normalize()
    df = compute_eps_revision(as_of=today, cache_path=ROOT / "eps_revision_scores.csv")
    if not df.empty:
        print(f"\nTop 15 (highest EPS revision momentum):")
        print(df.head(15)[["ticker", "sue_score", "eps_trend", "revision_score"]].to_string())
        print(f"\nBottom 5 (downward EPS revision):")
        print(df.tail(5)[["ticker", "sue_score", "eps_trend", "revision_score"]].to_string())
    else:
        print("No EPS data available — run data/edgar_pit.py first")
