#!/usr/bin/env python3
"""
Canyon v9  Step 222 — Rolling IC Tracker (Signal Validation)
=============================================================
Systematically measures each signal's predictive power using
rank Information Coefficient (rank-IC) over rolling windows.

WHAT IS IC?
-----------
IC = Spearman rank correlation between today's signal score and
the actual stock return N days later.

    IC = corr_rank( signal_today, return_N_days_later )

    IC > 0.05:  Strong (t-stat likely > 2 after 100+ observations)
    IC > 0.02:  Modest
    IC ≈ 0:     No predictive power (this signal may not be worth using)
    IC < 0:     Perverse (signal goes the wrong way)

IC-IR (Information Ratio of IC):
    IC_IR = mean(IC) / std(IC)
    IC_IR > 0.4: Institutional threshold for using a signal in production

WHY THIS MATTERS
----------------
Currently Canyon has 10 signals. Without IC validation, we don't know
which ones are real vs noise. This step creates a running ledger so
after 6+ months of daily data, we can prune weak signals.

WORKFLOW
--------
Every day this step runs:
1. Looks at signal scores from N days ago (20d, 5d, 1d horizons)
2. Compares to today's actual returns for those tickers
3. Computes rank-IC for each signal
4. Appends to the running IC log
5. After 100+ observations per signal, t-stats become meaningful

INPUTS
------
    alpha_scores.csv              — current signal scores (all 10 signals)
    alpha_score_history.csv       — historical alpha scores (for lag)
    sp500_price_cache.csv         — prices for actual returns
    ic_daily_log.csv              — running log (created if missing)

OUTPUTS
-------
    ic_daily_log.csv              — growing ledger: date, signal, horizon, IC
    ic_summary.csv                — current IC stats per signal
    ic_tracker_report.md          — plain-English scorecard
"""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as t_dist

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

HORIZONS   = [1, 5, 20]          # forward return windows in trading days
MIN_OBS    = 20                   # minimum IC observations for t-stat
IC_STRONG  = 0.05
IC_MODEST  = 0.02
ICIR_THRESHOLD = 0.40

SIGNALS = [
    ("sig_regime_ml",  "Market Regime ML"),
    ("sig_quality",    "Quality (Novy-Marx)"),
    ("sig_momentum",   "Momentum (signal-layer)"),
    ("sig_options",    "Options Flow"),
    ("sig_sentiment",  "News Sentiment"),
    ("sig_insider",    "Insider Activity"),
    ("sig_revision",   "Earnings Revision"),
    ("sig_surprise",   "Earnings Surprise"),
    ("sig_squeeze",    "Short Squeeze"),
    ("sig_ml_ensemble","ML Ensemble"),
    ("alpha_score",    "Combined Alpha Score"),
]


# =============================================================================
# 1. Load data
# =============================================================================

def load_signal_history() -> pd.DataFrame:
    """
    Load historical signal scores.
    First tries alpha_score_history.csv for alpha_score column,
    then falls back to alpha_scores.csv for today's signals.
    """
    # Check if we have per-signal history
    hist_path = ROOT / "alpha_score_history.csv"
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
        return hist.dropna(subset=["date"])
    return pd.DataFrame()


def load_prices() -> pd.DataFrame:
    path = ROOT / "sp500_price_cache.csv"
    if not path.exists():
        return pd.DataFrame()
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    return prices.sort_index()


def load_current_signals() -> pd.DataFrame:
    path = ROOT / "alpha_scores.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.dropna(subset=["ticker"])
    return df


# =============================================================================
# 2. IC calculation for one signal × one date × one horizon
# =============================================================================

def compute_ic_for_date(
    signal_col: str,
    signal_date: pd.Timestamp,
    horizon_days: int,
    alpha_hist: pd.DataFrame,
    prices: pd.DataFrame,
) -> float | None:
    """
    For a given signal and date, compute IC against forward returns.
    Returns rank-IC or None if insufficient data.
    """
    # Get signal scores on signal_date
    if "date" not in alpha_hist.columns:
        return None

    date_mask = alpha_hist["date"].dt.date == signal_date.date()
    snap = alpha_hist[date_mask]

    if snap.empty or signal_col not in snap.columns:
        return None

    snap = snap[["ticker", signal_col]].dropna()
    if len(snap) < 20:
        return None

    # Compute forward return: price at (signal_date + horizon) / price at signal_date
    # Find the actual market days
    dates_in_cache = prices.index
    later_dates = dates_in_cache[dates_in_cache > signal_date]
    if len(later_dates) < horizon_days:
        return None

    eval_date = later_dates[horizon_days - 1]

    # Need signal_date in prices too
    if signal_date not in prices.index:
        # Find closest prior date
        prior = dates_in_cache[dates_in_cache <= signal_date]
        if len(prior) == 0:
            return None
        signal_date_px = prior[-1]
    else:
        signal_date_px = signal_date

    tickers = snap["ticker"].tolist()
    avail   = [t for t in tickers if t in prices.columns]
    if len(avail) < 20:
        return None

    p0 = prices.loc[signal_date_px, avail]
    p1 = prices.loc[eval_date, avail]

    fwd_ret = (p1 / p0 - 1).dropna()

    # Align
    common = snap.set_index("ticker")[signal_col].reindex(fwd_ret.index).dropna()
    fwd_common = fwd_ret[common.index]

    if len(common) < 15:
        return None

    rho, _ = spearmanr(common.values, fwd_common.values)
    return float(rho) if not np.isnan(rho) else None


# =============================================================================
# 3. Today's IC calculation
# =============================================================================

def compute_today_ics(
    alpha_hist: pd.DataFrame,
    prices: pd.DataFrame,
    today: pd.Timestamp,
    horizons: list[int] = HORIZONS,
) -> list[dict]:
    """
    For each (signal, horizon) pair, try to compute IC using data
    from `horizon` days ago vs today's prices.
    """
    results = []
    trading_dates = prices.index
    prior_dates   = trading_dates[trading_dates < today]

    for horizon in horizons:
        if len(prior_dates) < horizon:
            continue
        signal_date = prior_dates[-horizon]

        for sig_col, sig_name in SIGNALS:
            ic = compute_ic_for_date(sig_col, signal_date, horizon, alpha_hist, prices)
            if ic is not None:
                results.append({
                    "date":          today.date(),
                    "signal_col":    sig_col,
                    "signal_name":   sig_name,
                    "horizon_days":  horizon,
                    "ic":            round(ic, 6),
                    "signal_date":   signal_date.date(),
                })

    return results


# =============================================================================
# 4. Summary statistics
# =============================================================================

def compute_ic_summary(log: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean IC, std IC, IC-IR, t-stat, and verdict for each signal.
    """
    if log.empty:
        return pd.DataFrame()

    rows = []
    for (sig_col, sig_name, horizon), grp in log.groupby(["signal_col","signal_name","horizon_days"]):
        ics    = grp["ic"].dropna().values
        n      = len(ics)
        if n < 2:
            continue

        mean_ic = float(np.mean(ics))
        std_ic  = float(np.std(ics, ddof=1))
        ic_ir   = mean_ic / std_ic if std_ic > 1e-8 else 0.0
        t_stat  = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 1e-8 else 0.0

        # Verdict
        if n < MIN_OBS:
            verdict = f"Need {MIN_OBS - n} more obs"
        elif abs(t_stat) < 1.5:
            verdict = "NOT significant — likely noise"
        elif abs(t_stat) < 2.0:
            verdict = "Weak — accumulating evidence"
        elif mean_ic > IC_STRONG and ic_ir > ICIR_THRESHOLD:
            verdict = "✅ STRONG — production-ready"
        elif mean_ic > IC_MODEST:
            verdict = "✅ MODEST — useful signal"
        elif mean_ic < -IC_MODEST:
            verdict = "⚠ PERVERSE — negative IC"
        else:
            verdict = "MARGINAL — keep watching"

        rows.append({
            "signal_col":   sig_col,
            "signal_name":  sig_name,
            "horizon_days": horizon,
            "n_obs":        n,
            "mean_ic":      round(mean_ic, 5),
            "std_ic":       round(std_ic, 5),
            "ic_ir":        round(ic_ir, 3),
            "t_stat":       round(t_stat, 3),
            "pct_positive": round(float((ics > 0).mean()), 3),
            "verdict":      verdict,
        })

    return pd.DataFrame(rows).sort_values(["horizon_days","mean_ic"], ascending=[True, False])


# =============================================================================
# 5. Report writer
# =============================================================================

def write_report(summary: pd.DataFrame, log: pd.DataFrame, today: datetime) -> None:
    lines = [
        "# Canyon v9 — Rolling IC Tracker Report (Step 222)",
        f"Generated: {today.strftime('%Y-%m-%d %H:%M')}\n",
        "## What Is IC?",
        "",
        "IC (Information Coefficient) = Spearman rank correlation between",
        "today's signal score and the actual forward return N days later.",
        "",
        "| IC level | Meaning |",
        "|----------|---------|",
        "| > 0.05 | Strong — institutionally usable |",
        "| 0.02–0.05 | Modest — worth including |",
        "| −0.02–0.02 | Noise — cannot be trusted |",
        "| < −0.02 | Perverse — pointing the wrong way |",
        "",
        "**IC-IR threshold for production use: 0.40** (IC-IR = mean_IC / std_IC)",
        "",
        f"**Total IC observations so far: {len(log)}**",
        "",
    ]

    if summary.empty:
        lines += ["> No IC summary available yet. Need more daily observations."]
    else:
        for horizon in sorted(summary["horizon_days"].unique()):
            sub = summary[summary["horizon_days"] == horizon].copy()
            lines += [
                f"## {horizon}-Day Forward IC",
                "",
                "| Signal | N obs | Mean IC | IC-IR | t-stat | % Positive | Verdict |",
                "|--------|-------|---------|-------|--------|------------|---------|",
            ]
            for _, r in sub.iterrows():
                lines.append(
                    f"| {r['signal_name']} | {r['n_obs']} | {r['mean_ic']:+.4f} | "
                    f"{r['ic_ir']:.2f} | {r['t_stat']:.2f} | {r['pct_positive']*100:.0f}% | "
                    f"{r['verdict']} |"
                )
            lines.append("")

    lines += [
        "## What Happens Over Time",
        "",
        "As this tracker accumulates daily observations:",
        "- After **20 obs**: preliminary verdict available",
        "- After **100 obs** (~5 months): t-stats are meaningful",
        "- After **200 obs** (~10 months): production-grade evidence",
        "",
        "**Keep this running every day.** The daily IC log is the most valuable",
        "long-term asset this system can build.",
        "",
        "## Data note",
        "",
        "⚠️ IC calculated using survivorship-biased price data. Forward returns",
        "are approximate. Treat current IC values as directional indicators only.",
    ]

    (ROOT / "ic_tracker_report.md").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 6. Main
# =============================================================================

def run() -> dict:
    print(f"\n{'='*65}")
    print(f"Canyon v9 — Step 222: Rolling IC Tracker  [{datetime.now():%Y-%m-%d %H:%M:%S}]")
    print(f"{'='*65}")

    today = pd.Timestamp(datetime.now().date())

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1/5] Loading signal history & prices …")
    alpha_hist = load_signal_history()
    prices     = load_prices()

    if prices.empty:
        print("  [ERROR] No price data.")
        return {"status": "error"}

    # We need per-signal history; if not available, use alpha_scores snapshot
    # and just track alpha_score itself as a sanity check
    current_signals = load_current_signals()
    print(f"  Price cache: {len(prices)} days")
    print(f"  Signal history: {len(alpha_hist)} rows "
          f"({'with per-signal cols' if len(alpha_hist.columns) > 3 else 'alpha_score only'})")

    # ── Save today's full signal snapshot ────────────────────────────────────
    snapshot_path = ROOT / "signal_score_history.csv"
    if not current_signals.empty:
        sig_cols = ["ticker","alpha_score"] + [c for c in current_signals.columns
                                                if c.startswith("sig_")]
        avail = [c for c in sig_cols if c in current_signals.columns]
        snap  = current_signals[avail].copy()
        snap["date"] = today.date()

        if snapshot_path.exists():
            old = pd.read_csv(snapshot_path)
            old["date"] = pd.to_datetime(old["date"]).dt.date
            old = old[old["date"] != today.date()]   # remove today if re-run
            full_snap = pd.concat([old, snap], ignore_index=True)
        else:
            full_snap = snap

        full_snap.to_csv(snapshot_path, index=False)
        print(f"  Signal snapshot saved → signal_score_history.csv "
              f"({len(full_snap)} rows, {len(full_snap['date'].unique())} dates)")

        # Merge: per-signal snapshot history + original alpha_score_history
        # (which has more dates but only alpha_score col)
        full_snap["date"] = pd.to_datetime(full_snap["date"])
        if not alpha_hist.empty:
            # Only keep rows from alpha_hist that aren't in full_snap
            snap_dates = set(full_snap["date"].dt.date)
            alpha_extra = alpha_hist[~alpha_hist["date"].dt.date.isin(snap_dates)]
            alpha_hist = pd.concat([alpha_extra, full_snap], ignore_index=True)
        else:
            alpha_hist = full_snap

        print(f"  Combined history: {len(alpha_hist['date'].unique())} unique dates")

    # ── Load or create IC log ─────────────────────────────────────────────────
    print("\n[2/5] Loading IC log …")
    log_path = ROOT / "ic_daily_log.csv"
    if log_path.exists():
        ic_log = pd.read_csv(log_path)
        ic_log["date"] = pd.to_datetime(ic_log["date"]).dt.date
        # Skip if already computed for today
        already_today = ic_log[ic_log["date"] == today.date()]
        print(f"  IC log: {len(ic_log)} rows, "
              f"{len(already_today)} already computed for today")
    else:
        ic_log = pd.DataFrame(columns=[
            "date","signal_col","signal_name","horizon_days","ic","signal_date"
        ])
        print("  IC log: new (first run)")

    # ── Compute ICs: backfill all historical windows + today ──────────────────
    print("\n[3/5] Computing IC (backfill all available dates) …")

    # Already-computed (date, signal_col, horizon_days) combos
    already_done: set = set()
    if not ic_log.empty:
        for _, r in ic_log.iterrows():
            already_done.add((str(r["date"]), str(r["signal_col"]), int(r["horizon_days"])))

    new_rows = []
    trading_dates = prices.index

    # Get all signal dates available in history
    if not alpha_hist.empty and "date" in alpha_hist.columns:
        signal_dates_avail = sorted(alpha_hist["date"].dropna().unique())
    else:
        signal_dates_avail = [today]

    for sig_date in signal_dates_avail:
        sig_date_ts = pd.Timestamp(sig_date)
        for horizon in HORIZONS:
            # Find price date horizon trading days AFTER signal_date
            later_dates = trading_dates[trading_dates > sig_date_ts]
            if len(later_dates) < horizon:
                continue   # not enough future data yet
            eval_date = later_dates[horizon - 1]

            for sig_col, sig_name in SIGNALS:
                key = (str(sig_date_ts.date()), sig_col, horizon)
                if key in already_done:
                    continue

                ic = compute_ic_for_date(sig_col, sig_date_ts, horizon, alpha_hist, prices)
                if ic is not None:
                    new_rows.append({
                        "date":          sig_date_ts.date(),
                        "signal_col":    sig_col,
                        "signal_name":   sig_name,
                        "horizon_days":  horizon,
                        "ic":            round(ic, 6),
                        "signal_date":   sig_date_ts.date(),
                        "eval_date":     eval_date.date(),
                    })

    print(f"  New IC observations: {len(new_rows)} "
          f"(across {len(signal_dates_avail)} signal dates × {len(HORIZONS)} horizons)")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        ic_log = pd.concat([ic_log, new_df], ignore_index=True)
        ic_log["date"] = pd.to_datetime(ic_log["date"]).dt.date
        ic_log.to_csv(log_path, index=False)
        print(f"  ic_daily_log.csv updated: {len(ic_log)} total observations")

    # ── Compute summary ───────────────────────────────────────────────────────
    print("\n[4/5] Computing IC summary …")
    if not ic_log.empty:
        ic_log["ic"] = pd.to_numeric(ic_log["ic"], errors="coerce")
        summary = compute_ic_summary(ic_log)
        summary.to_csv(ROOT / "ic_summary.csv", index=False)
        print(f"  ic_summary.csv: {len(summary)} signal×horizon rows")

        # Print scorecard
        if not summary.empty:
            print(f"\n  {'Signal':25s} {'Horizon':8s} {'N':5s} {'MeanIC':8s} "
                  f"{'ICIR':6s} {'t':6s} {'Verdict'}")
            print(f"  {'─'*85}")
            for _, r in summary.iterrows():
                print(f"  {r['signal_name']:25s} {r['horizon_days']:8.0f}d "
                      f"{r['n_obs']:5.0f} {r['mean_ic']:+8.4f} "
                      f"{r['ic_ir']:6.2f} {r['t_stat']:6.2f}  {r['verdict']}")
    else:
        summary = pd.DataFrame()

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n[5/5] Writing report …")
    write_report(summary, ic_log, datetime.now())
    print(f"  [written] ic_tracker_report.md")
    print(f"  [written] ic_daily_log.csv")
    if not summary.empty:
        print(f"  [written] ic_summary.csv")

    n_sig = len(ic_log["signal_col"].unique()) if not ic_log.empty else 0
    n_obs = len(ic_log)
    print(f"\n  Total: {n_obs} IC observations across {n_sig} signals")
    print(f"  Run this every day. Each run adds {len(new_rows)} observations.")
    print(f"  Need ~100 obs per signal per horizon for meaningful t-stats.")
    print(f"  {'─'*65}\n")

    return {"status": "OK", "n_new": len(new_rows), "n_total": n_obs}


if __name__ == "__main__":
    import sys
    result = run()
    sys.exit(0 if result.get("status") == "OK" else 1)
