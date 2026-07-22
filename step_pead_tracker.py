#!/usr/bin/env python3
"""
Canyon — step_pead_tracker.py
================================
Post-Earnings Announcement Drift (PEAD) tracker.

For each stock that reported earnings in the last 30 trading days, compute:
  • Earnings surprise magnitude (SUE = standardised unexpected earnings)
  • Cumulative return since earnings date (1d, 5d, 10d, 20d windows)
  • Alpha drift vs SPY over the same window
  • Signal: whether Canyon alpha_score predicted the direction

PEAD is one of the most persistent anomalies in academic finance:
  stocks that beat earnings estimates tend to drift upward for 4-8 weeks,
  and misses drift downward. This step tracks whether Canyon's signals
  capture this effect and how much drift remains to be harvested.

Reads:
  earnings_surprise_scores.csv  or  earnings_calendar.csv
  sp500_price_cache.csv         — daily returns since earnings date
  alpha_scores.csv              — was Canyon bullish before the announcement?

Outputs:
  pead_tracker.csv      — one row per (ticker, window), cumret + alpha_drift
  pead_summary.json     — aggregate stats: PEAD IC, hit rate, avg drift
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT  = Path(__file__).parent
TODAY = datetime.now().strftime("%Y-%m-%d")
NOW   = datetime.now()

GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD = "\033[1m"; RESET  = "\033[0m"

def log(msg): print(f"  {msg}")
def ok(msg):  print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg):print(f"  {YELLOW}⚠{RESET}  {msg}")
def err(msg): print(f"  {RED}✗{RESET}  {msg}")

DRIFT_WINDOWS   = [1, 5, 10, 20]      # trading days post-earnings
LOOKBACK_DAYS   = 30                   # consider earnings up to 30 trading days ago
MIN_SURPRISE_SD = 0.0                  # include all surprises (positive AND negative)


# ── Load earnings surprise data ───────────────────────────────────────────────

def load_surprise_data() -> pd.DataFrame:
    """
    Load recent earnings surprises with ticker, date, SUE, and direction.
    Tries multiple source files.
    """
    candidates = [
        ("earnings_surprise_scores.csv",  {"ticker": "ticker", "date": "earnings_date", "sue": "rank_sue"}),
        ("earnings_revision_scores.csv",  {"ticker": "ticker", "date": "report_date",   "sue": "revision_score"}),
        ("earnings_calendar.csv",         {"ticker": "ticker", "date": "date",          "sue": "surprise_pct"}),
    ]

    for fname, col_map in candidates:
        fpath = ROOT / fname
        if not fpath.exists():
            continue
        try:
            df = pd.read_csv(fpath)
            # Remap columns
            rename = {v: k for k, v in col_map.items() if v in df.columns}
            df = df.rename(columns=rename)

            date_col = col_map["date"] if col_map["date"] in df.columns else "date"
            if date_col not in df.columns and "date" not in df.columns:
                continue

            if "ticker" not in df.columns:
                continue

            # Keep only columns we need
            keep = [c for c in ["ticker", "date", "earnings_date", "sue", "surprise_pct",
                                 "rank_sue", "revision_score"] if c in df.columns]
            df = df[keep].drop_duplicates("ticker")

            # Standardise date
            for dc in ["date", "earnings_date", "report_date"]:
                if dc in df.columns:
                    df["earnings_date"] = pd.to_datetime(df[dc], errors="coerce")
                    break

            # Standardise surprise column
            for sc in ["sue", "rank_sue", "surprise_pct", "revision_score"]:
                if sc in df.columns:
                    df["sue"] = pd.to_numeric(df[sc], errors="coerce")
                    break

            df = df.dropna(subset=["ticker", "earnings_date"])

            # Filter to lookback window
            cutoff = pd.Timestamp(NOW - timedelta(days=LOOKBACK_DAYS * 1.5))
            df = df[df["earnings_date"] >= cutoff]

            if len(df) >= 5:
                ok(f"Loaded surprise data from {fname}: {len(df)} events")
                return df[["ticker", "earnings_date", "sue"]].reset_index(drop=True)
        except Exception as e:
            warn(f"  Could not load {fname}: {e}")
            continue

    # Fallback: use alpha_score_history.csv to find recent high/low movers
    warn("No earnings surprise file found — generating synthetic PEAD from alpha_score_history")
    return _synthetic_from_history()


def _synthetic_from_history() -> pd.DataFrame:
    """
    Create a synthetic PEAD dataset from alpha_score_history.csv by detecting
    large score jumps (proxy for earnings surprise).
    """
    hist_path = ROOT / "alpha_score_history.csv"
    if not hist_path.exists():
        return pd.DataFrame()

    try:
        hist = pd.read_csv(hist_path, parse_dates=["date"])
        if "alpha_score" not in hist.columns or "ticker" not in hist.columns:
            return pd.DataFrame()

        pivot = hist.pivot(index="date", columns="ticker", values="alpha_score")
        pivot = pivot.sort_index()
        # Detect single-day jumps > 15 points in alpha_score as proxy for surprise
        jumps = pivot.diff().abs()
        cutoff = pd.Timestamp(NOW - timedelta(days=LOOKBACK_DAYS * 1.5))
        recent_jumps = jumps[jumps.index >= cutoff]

        rows = []
        for date, row in recent_jumps.iterrows():
            for ticker, val in row.items():
                if val >= 15 and not pd.isna(val):
                    rows.append({
                        "ticker":         ticker,
                        "earnings_date":  pd.Timestamp(date),
                        "sue":            float(pivot.loc[date, ticker]) - 50,  # centre at 0
                    })
        return pd.DataFrame(rows).drop_duplicates("ticker") if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ── Load price returns since earnings date ────────────────────────────────────

def compute_drift(surprise_df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    For each ticker / earnings_date, compute cumulative return over each DRIFT_WINDOWS.
    Also compute SPY-relative drift.
    """
    rets = prices.pct_change().fillna(0)

    # SPY proxy: equal-weighted average
    spy_rets = rets.mean(axis=1)

    rows = []
    for _, r in surprise_df.iterrows():
        ticker    = r["ticker"]
        earn_date = r["earnings_date"]
        sue       = r.get("sue", np.nan)

        if ticker not in rets.columns:
            continue

        # Find the first price date on or after earnings_date
        post_dates = rets.index[rets.index >= earn_date]
        if len(post_dates) == 0:
            continue
        start_date = post_dates[0]

        for window in DRIFT_WINDOWS:
            window_dates = post_dates[:window + 1]  # +1 because first date is day 0
            if len(window_dates) < 2:
                continue

            cumret     = (1 + rets.loc[window_dates, ticker]).prod() - 1
            spy_cumret = (1 + spy_rets.loc[window_dates]).prod() - 1
            alpha_drift = cumret - spy_cumret
            days_elapsed = len(window_dates) - 1

            rows.append({
                "ticker":       ticker,
                "earnings_date": earn_date.strftime("%Y-%m-%d"),
                "sue":          float(sue) if not pd.isna(sue) else None,
                "window_days":  window,
                "days_elapsed": days_elapsed,
                "cumret":       round(float(cumret), 5),
                "spy_cumret":   round(float(spy_cumret), 5),
                "alpha_drift":  round(float(alpha_drift), 5),
                "surprise_dir": "BEAT" if (sue or 0) > 0 else "MISS",
                "drift_dir":    "UP" if cumret > 0 else "DOWN",
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Load Canyon alpha score at earnings date ──────────────────────────────────

def load_canyon_alpha_at_earnings(tickers: list[str], earn_dates: dict[str, pd.Timestamp]) -> dict[str, float]:
    """
    Returns {ticker: alpha_score} as of just before earnings date.
    Uses alpha_score_history.csv if available, else current alpha_scores.csv.
    """
    result = {}

    hist_path = ROOT / "alpha_score_history.csv"
    if hist_path.exists():
        try:
            hist = pd.read_csv(hist_path, parse_dates=["date"])
            pivot = hist.pivot(index="date", columns="ticker", values="alpha_score")
            pivot = pivot.sort_index()
            for tk in tickers:
                if tk not in pivot.columns:
                    continue
                earn_dt = earn_dates.get(tk)
                if earn_dt is None:
                    continue
                # Get score just before earnings
                pre_dates = pivot.index[pivot.index < earn_dt]
                if len(pre_dates) > 0:
                    result[tk] = float(pivot.loc[pre_dates[-1], tk])
            return result
        except Exception:
            pass

    # Fallback: current alpha_scores
    path = ROOT / "alpha_scores.csv"
    if path.exists():
        try:
            df = pd.read_csv(path).set_index("ticker")
            for tk in tickers:
                if tk in df.index:
                    result[tk] = float(df.loc[tk, "alpha_score"])
        except Exception:
            pass
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}Canyon — PEAD Drift Tracker{RESET}  {TODAY}")

    # 1. Load price data
    price_path = ROOT / "sp500_price_cache.csv"
    if not price_path.exists():
        err("sp500_price_cache.csv not found")
        return
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True).sort_index()
    ok(f"Prices: {prices.shape[0]} dates × {prices.shape[1]} tickers")

    # 2. Load earnings surprise events
    surprise_df = load_surprise_data()
    if surprise_df.empty:
        warn("No surprise events — skipping PEAD tracker")
        return
    log(f"Tracking {len(surprise_df)} earnings events …")

    # 3. Compute post-earnings drift
    drift_df = compute_drift(surprise_df, prices)
    if drift_df.empty:
        warn("No drift data computed")
        return

    # 4. Load Canyon pre-earnings alpha scores
    earn_dates = dict(zip(surprise_df["ticker"], surprise_df["earnings_date"]))
    canyon_alpha = load_canyon_alpha_at_earnings(surprise_df["ticker"].tolist(), earn_dates)
    drift_df["canyon_alpha_pre"] = drift_df["ticker"].map(canyon_alpha)
    drift_df["canyon_bullish"]   = drift_df["canyon_alpha_pre"].apply(
        lambda x: "BULL" if (x or 50) > 55 else ("BEAR" if (x or 50) < 45 else "NEUTRAL")
    )

    # Was Canyon's pre-earnings signal directionally correct?
    drift_df["canyon_correct"] = (
        ((drift_df["canyon_bullish"] == "BULL") & (drift_df["cumret"] > 0)) |
        ((drift_df["canyon_bullish"] == "BEAR") & (drift_df["cumret"] < 0))
    )

    # 5. Summary stats
    summary: dict = {"as_of": TODAY, "n_events": len(surprise_df)}
    print(f"\n  {'Window':>8}  {'Avg Drift':>10}  {'Avg SPY':>10}  "
          f"{'Alpha':>10}  {'Hit%':>8}  {'Canyon%':>10}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*10}")

    beats = drift_df[drift_df["surprise_dir"] == "BEAT"]
    misses = drift_df[drift_df["surprise_dir"] == "MISS"]

    window_stats = []
    for w in DRIFT_WINDOWS:
        grp = drift_df[drift_df["window_days"] == w]
        if grp.empty:
            continue
        avg_drift   = grp["cumret"].mean()
        avg_spy     = grp["spy_cumret"].mean()
        avg_alpha   = grp["alpha_drift"].mean()
        hit_rate    = (grp["drift_dir"] == grp["surprise_dir"].map({"BEAT": "UP", "MISS": "DOWN"})).mean()
        canyon_acc  = grp["canyon_correct"].mean()
        drift_c     = GREEN if avg_alpha > 0 else RED
        print(f"  {w:>7}d  {avg_drift*100:>9.2f}%  {avg_spy*100:>9.2f}%  "
              f"{drift_c}{avg_alpha*100:>9.2f}%{RESET}  {hit_rate*100:>7.1f}%  "
              f"{canyon_acc*100:>9.1f}%")
        window_stats.append({
            "window_days":  w,
            "avg_cumret":   round(float(avg_drift), 5),
            "avg_spy_ret":  round(float(avg_spy), 5),
            "avg_alpha":    round(float(avg_alpha), 5),
            "hit_rate":     round(float(hit_rate), 4),
            "canyon_accuracy": round(float(canyon_acc), 4),
            "n":            len(grp),
        })

    summary["by_window"]  = window_stats
    summary["n_beats"]    = int((drift_df["surprise_dir"] == "BEAT").sum() // len(DRIFT_WINDOWS))
    summary["n_misses"]   = int((drift_df["surprise_dir"] == "MISS").sum() // len(DRIFT_WINDOWS))

    # PEAD IC: correlation of SUE vs alpha_drift for 10d window
    w10 = drift_df[drift_df["window_days"] == 10].dropna(subset=["sue", "alpha_drift"])
    if len(w10) >= 5:
        from scipy.stats import spearmanr
        ic, pval = spearmanr(w10["sue"], w10["alpha_drift"])
        summary["pead_ic_10d"]   = round(float(ic), 4)
        summary["pead_ic_pval"]  = round(float(pval), 4)
        color = GREEN if ic > 0.05 else (YELLOW if ic >= 0 else RED)
        ok(f"PEAD IC (SUE vs 10d alpha drift): {color}{ic:.3f}{RESET}  p={pval:.3f}")
    else:
        summary["pead_ic_10d"] = None

    # Save
    out_csv = ROOT / "pead_tracker.csv"
    drift_df.to_csv(out_csv, index=False)
    ok(f"pead_tracker.csv → {len(drift_df)} rows")

    out_json = ROOT / "pead_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    ok(f"pead_summary.json saved")

    print(f"\n{GREEN}✓ PEAD tracker complete — {len(surprise_df)} earnings events analysed{RESET}\n")


if __name__ == "__main__":
    main()
