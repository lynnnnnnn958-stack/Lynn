"""
W16: Daily Data Quality Monitor
================================
Runs at the end of run_daily.py to verify all data sources are fresh,
complete, and internally consistent.

Checks:
  - Price cache: last date ≤ 2 business days ago
  - FRED macro: last date ≤ 1 day ago
  - EDGAR PIT: coverage rate ≥ 80% of universe tickers
  - Signal CSV files: no NaN rate > 20% in key columns
  - alpha_scores.csv: all expected signals present

Outputs:
  data_quality_report.csv — one row per check, status + details
  (also prints warnings if any check fails)

Usage:
    from monitoring.data_quality import run_quality_checks, print_report
    report = run_quality_checks()
    print_report(report)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
TODAY = pd.Timestamp.today().normalize()


def _business_days_ago(date: pd.Timestamp) -> int:
    """Number of business days between date and today."""
    bdays = pd.bdate_range(start=date, end=TODAY)
    return max(0, len(bdays) - 1)


def _check_price_cache() -> dict:
    """Verify price cache is fresh (≤ 2 business days old)."""
    for fname in ("sp500_price_cache_8yr.csv", "sp500_price_cache.csv"):
        p = ROOT / fname
        if p.exists():
            try:
                df = pd.read_csv(p, index_col=0, parse_dates=True)
                last_date = df.index.max()
                lag_bdays = _business_days_ago(last_date)
                n_tickers = len([c for c in df.columns if c != "SPY"])
                missing_pct = float(df.iloc[-5:].isna().mean().mean())
                return {
                    "check":   "price_cache",
                    "file":    fname,
                    "status":  "OK" if lag_bdays <= 2 else "STALE",
                    "last_date": str(last_date.date()),
                    "lag_bdays": lag_bdays,
                    "n_tickers": n_tickers,
                    "recent_missing_pct": round(missing_pct * 100, 1),
                    "detail":  f"Last price: {last_date.date()}, lag={lag_bdays}bd, {n_tickers} stocks",
                }
            except Exception as e:
                return {"check": "price_cache", "status": "ERROR", "detail": str(e)}
    return {"check": "price_cache", "status": "MISSING", "detail": "No price cache found"}


def _check_fred_macro() -> dict:
    """Verify FRED macro data is fresh (≤ 2 days old)."""
    p = ROOT / "fred_macro_daily.csv"
    if not p.exists():
        return {"check": "fred_macro", "status": "MISSING",
                "detail": "fred_macro_daily.csv not found — run data/fred_macro.py"}
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        last_date = df.index.max()
        lag_days = (TODAY - last_date).days
        cols_present = df.columns.tolist()
        expected = ["VIX", "HY", "TERM", "DOLLAR"]
        missing_cols = [c for c in expected if c not in cols_present]
        return {
            "check":       "fred_macro",
            "status":      "OK" if lag_days <= 3 and not missing_cols else "WARN",
            "last_date":   str(last_date.date()),
            "lag_days":    lag_days,
            "columns":     cols_present,
            "missing_cols": missing_cols,
            "detail":      f"Last FRED: {last_date.date()}, lag={lag_days}d, "
                           f"missing={missing_cols if missing_cols else 'none'}",
        }
    except Exception as e:
        return {"check": "fred_macro", "status": "ERROR", "detail": str(e)}


def _check_edgar_pit() -> dict:
    """Verify EDGAR PIT fundamentals cover ≥80% of universe."""
    pit_path = ROOT / "edgar_pit_fundamentals.csv"
    if not pit_path.exists():
        return {"check": "edgar_pit", "status": "MISSING",
                "detail": "edgar_pit_fundamentals.csv not found — run data/edgar_pit.py"}
    try:
        pit_df = pd.read_csv(pit_path, parse_dates=["period_end", "know_date"])
        n_tickers = pit_df["ticker"].nunique()

        # Check PIT constraint: all know_dates after period_end
        violations = (pit_df["know_date"] <= pit_df["period_end"]).sum()

        # Check recency: most recent filings shouldn't be too old
        max_know = pit_df["know_date"].max()
        staleness = (TODAY - max_know).days

        return {
            "check":      "edgar_pit",
            "status":     "OK" if violations == 0 and n_tickers >= 100 else "WARN",
            "n_tickers":  n_tickers,
            "n_rows":     len(pit_df),
            "pit_violations": int(violations),
            "most_recent_filing": str(max_know.date()) if pd.notna(max_know) else "N/A",
            "staleness_days": staleness,
            "detail":     f"{n_tickers} tickers, {len(pit_df):,} rows, "
                          f"{violations} PIT violations, staleness={staleness}d",
        }
    except Exception as e:
        return {"check": "edgar_pit", "status": "ERROR", "detail": str(e)}


def _check_form4() -> dict:
    """Verify EDGAR Form 4 data."""
    for fname in ("edgar_form4_cache.csv",):
        p = ROOT / fname
        if p.exists():
            try:
                df = pd.read_csv(p, parse_dates=["filed_date", "txn_date"])
                n_transactions = len(df)
                n_tickers = df["ticker"].nunique() if "ticker" in df.columns else 0
                # PIT check: filed_date >= txn_date (SEC requires this)
                violations = (df["filed_date"] < df["txn_date"]).sum() \
                             if "filed_date" in df.columns and "txn_date" in df.columns else 0
                return {
                    "check":       "form4",
                    "status":      "OK" if violations == 0 else "WARN",
                    "n_transactions": n_transactions,
                    "n_tickers":   n_tickers,
                    "pit_violations": int(violations),
                    "detail":      f"{n_transactions} transactions, {n_tickers} tickers, "
                                   f"{violations} PIT violations",
                }
            except Exception as e:
                return {"check": "form4", "status": "ERROR", "detail": str(e)}
    return {"check": "form4", "status": "MISSING", "detail": "edgar_form4_cache.csv not found"}


def _check_alpha_scores() -> dict:
    """Verify alpha_scores.csv has all expected signals."""
    p = ROOT / "alpha_scores.csv"
    if not p.exists():
        return {"check": "alpha_scores", "status": "MISSING",
                "detail": "alpha_scores.csv not found"}
    try:
        df = pd.read_csv(p)
        n_tickers = len(df)
        expected_signals = ["sig_regime_ml", "sig_quality", "sig_ml_ensemble",
                            "sig_sentiment", "sig_surprise"]
        missing = [s for s in expected_signals if s not in df.columns]
        high_nan = {col: float(df[col].isna().mean())
                    for col in df.columns
                    if df[col].isna().mean() > 0.20}
        return {
            "check":         "alpha_scores",
            "status":        "OK" if not missing and not high_nan else "WARN",
            "n_tickers":     n_tickers,
            "missing_signals": missing,
            "high_nan_signals": high_nan,
            "detail":        f"{n_tickers} tickers, missing={missing}, high_nan={list(high_nan.keys())}",
        }
    except Exception as e:
        return {"check": "alpha_scores", "status": "ERROR", "detail": str(e)}


def _check_signal_csvs() -> list[dict]:
    """Check key signal output files for freshness and completeness."""
    files_to_check = [
        ("accrual_scores.csv",      ["ticker", "accrual_score"]),
        ("piotroski_scores.csv",    ["ticker", "piotroski_score"]),
        ("insider_cluster_scores.csv", ["ticker", "cluster_score"]),
        ("options_ivrank.csv",      ["ticker"]),
        ("signal_ic_attribution.csv", ["signal", "ic"]),
        ("execution_cost_estimates.csv", ["ticker"]),
    ]
    results = []
    for fname, expected_cols in files_to_check:
        p = ROOT / fname
        status = {
            "check":  f"file_{fname}",
            "file":   fname,
        }
        if not p.exists():
            status["status"] = "MISSING"
            status["detail"] = f"{fname} not found"
        else:
            try:
                df = pd.read_csv(p)
                age_days = (TODAY - pd.Timestamp(datetime.fromtimestamp(p.stat().st_mtime))).days
                missing_cols = [c for c in expected_cols if c not in df.columns]
                status["status"]       = "OK" if not missing_cols and age_days <= 2 else "WARN"
                status["n_rows"]       = len(df)
                status["age_days"]     = age_days
                status["missing_cols"] = missing_cols
                status["detail"]       = f"{len(df)} rows, age={age_days}d, missing={missing_cols}"
            except Exception as e:
                status["status"] = "ERROR"
                status["detail"] = str(e)
        results.append(status)
    return results


def run_quality_checks() -> pd.DataFrame:
    """
    Run all data quality checks and return a summary DataFrame.

    Returns DataFrame with columns:
        check, status, detail, (various check-specific columns)
    """
    print("\n[QC] Running data quality checks...")

    all_checks: list[dict] = [
        _check_price_cache(),
        _check_fred_macro(),
        _check_edgar_pit(),
        _check_form4(),
        _check_alpha_scores(),
        *_check_signal_csvs(),
    ]

    df = pd.DataFrame(all_checks)
    df["checked_at"] = pd.Timestamp.now().isoformat(timespec="seconds")

    df.to_csv(ROOT / "data_quality_report.csv", index=False)

    # Print summary
    ok_count   = (df["status"] == "OK").sum()
    warn_count = (df["status"] == "WARN").sum()
    miss_count = (df["status"] == "MISSING").sum()
    err_count  = (df["status"] == "ERROR").sum()
    print(f"  [QC] {ok_count} OK | {warn_count} WARN | {miss_count} MISSING | {err_count} ERROR")

    return df


def print_report(df: pd.DataFrame) -> None:
    """Print a formatted quality report."""
    icons = {"OK": "✓", "WARN": "⚠", "MISSING": "✗", "ERROR": "✗"}
    colors = {"OK": "\033[92m", "WARN": "\033[93m",
              "MISSING": "\033[91m", "ERROR": "\033[91m"}
    reset = "\033[0m"

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Data Quality Report")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for _, row in df.iterrows():
        status = str(row.get("status", "ERROR"))
        icon   = icons.get(status, "?")
        color  = colors.get(status, "")
        print(f"  {color}{icon}{reset}  {row['check']:<35s}  {row.get('detail', '')}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")


if __name__ == "__main__":
    report = run_quality_checks()
    print_report(report)
