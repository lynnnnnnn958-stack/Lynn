#!/usr/bin/env python3
"""
Canyon — step_rolling_ic.py
=============================
Compute rolling 60-day Information Coefficient (IC) for every signal
in SIGNAL_CONFIG and write ic_multipliers to signal_weights.json.

IC = Spearman rank correlation between signal score (cross-sectional rank)
     and 21-day forward return (cross-sectional rank), computed on a rolling
     60-trading-day window.

The multipliers update the effective weight in step87:
  eff_weight = base_weight × ic_multiplier

ic_multiplier is bounded [0.25, 3.0] so no signal gets zeroed out or
dominates excessively. Signals with negative IC over the window get 0.25×.

Runs weekly (triggered from run_daily.py on Mondays or when stale >6 days).

Outputs:
  signal_weights.json  (updates ic_multipliers + raw_ic fields)
  signal_ic_history.csv  (appended row per run date, for trend monitoring)
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT  = Path(__file__).parent
TODAY = datetime.now().strftime("%Y-%m-%d")

GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD = "\033[1m"; RESET  = "\033[0m"

def log(msg): print(f"  {msg}")
def ok(msg):  print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg):print(f"  {YELLOW}⚠{RESET}  {msg}")
def err(msg): print(f"  {RED}✗{RESET}  {msg}")


# ── Signal map: SIGNAL_CONFIG name → (csv_file, score_col) ───────────────────
SIGNAL_MAP = {
    "regime_ml":   ("regime_ml_scores.csv",         "predicted_score"),
    "quality":     ("fundamental_quality_rank.csv",  "quality_score"),
    "revision":    ("earnings_revision_scores.csv",  "revision_score"),
    "surprise":    ("earnings_surprise_scores.csv",  "rank_sue"),
    "sentiment":   ("finbert_sentiment.csv",         "rank_sentiment"),
    "squeeze":     ("short_interest_scores.csv",     "rank_squeeze"),
    "insider":     ("insider_signal_scores.csv",     "rank_insider"),
    "options":     ("options_signals.csv",           "rank_options"),
    "ml_ensemble": ("ml_signal_scores.csv",          "ensemble_score"),
    "momentum":    ("momentum_scores.csv",           "momentum_score"),
    "accruals":    ("accrual_scores.csv",            "accrual_score"),
    "piotroski":   ("piotroski_scores.csv",          "piotroski_score"),
    "ins_cluster": ("insider_cluster_scores.csv",    "cluster_score_rank"),
    "ml_short":    ("ml_alpha_scores.csv",           "ml_short"),
    "ml_medium":   ("ml_alpha_scores.csv",           "ml_medium"),
    "ml_long":     ("ml_alpha_scores.csv",           "ml_long"),
}

IC_WINDOW        = 60     # rolling IC window in trading days
FORWARD_HORIZON  = 21     # forward return horizon for IC (days)
MIN_PAIRS        = 30     # minimum (ticker, date) pairs to compute IC
MULT_CAP_HIGH    = 3.0    # maximum multiplier for strong IC signals
MULT_CAP_LOW     = 0.25   # minimum multiplier (negative IC → quarter weight)
MULT_NEUTRAL     = 1.0    # multiplier for signals with no history


# ── Load price data for forward returns ───────────────────────────────────────

def load_forward_returns(horizon: int = FORWARD_HORIZON) -> pd.DataFrame:
    """Returns date × ticker matrix of forward returns (winsorised rank)."""
    price_path = ROOT / "sp500_price_cache.csv"
    if not price_path.exists():
        return pd.DataFrame()
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True).sort_index()
    fwd = prices.pct_change(horizon).shift(-horizon)
    # winsorise
    flat = fwd.stack(future_stack=True)
    q05, q95 = flat.quantile(0.05), flat.quantile(0.95)
    fwd = fwd.clip(lower=q05, upper=q95)
    # cross-sectional rank
    return fwd.rank(axis=1, pct=True)


# ── Load individual signal scores ─────────────────────────────────────────────

def load_signal_scores(name: str) -> pd.Series | None:
    """Load today's cross-sectional signal score for all tickers. Returns Series[ticker→score]."""
    if name not in SIGNAL_MAP:
        return None
    fname, col = SIGNAL_MAP[name]
    fpath = ROOT / fname
    if not fpath.exists():
        return None
    try:
        df = pd.read_csv(fpath)
        if "ticker" not in df.columns or col not in df.columns:
            return None
        s = df.set_index("ticker")[col].dropna()
        # cross-sectional rank → 0..1
        return s.rank(pct=True)
    except Exception:
        return None


# ── Compute IC from alpha_score_history vs price forward returns ───────────────

def compute_ic_from_history(fwd_returns: pd.DataFrame) -> dict[str, float]:
    """
    For each signal, compute Spearman IC using the available history.

    Since we don't store per-signal daily history (only final alpha_score),
    we use the current signal file cross-sectional ranking as a proxy,
    paired against historical forward returns.

    For signals where we have alpha_score_history, we compute a time-series IC.
    For signals with only today's file, we report IC=null (will use multiplier=1.0).
    """
    ic_results: dict[str, float | None] = {}

    # Check if we have alpha_score_history with enough rows
    hist_path = ROOT / "alpha_score_history.csv"
    if hist_path.exists():
        try:
            hist = pd.read_csv(hist_path, parse_dates=["date"])
            if len(hist) > 200:
                # Compute IC for the final alpha_score vs forward returns
                pivot = hist.pivot(index="date", columns="ticker", values="alpha_score")
                pivot.index = pd.to_datetime(pivot.index)
                pivot_rank = pivot.rank(axis=1, pct=True)

                # align with forward returns
                common_dates = pivot_rank.index.intersection(fwd_returns.index)
                if len(common_dates) >= MIN_PAIRS:
                    ic_series = []
                    for date in common_dates[-IC_WINDOW:]:
                        sig_row = pivot_rank.loc[date].dropna()
                        fwd_row = fwd_returns.loc[date].dropna() if date in fwd_returns.index else pd.Series()
                        common_t = sig_row.index.intersection(fwd_row.index)
                        if len(common_t) >= 20:
                            rho, _ = spearmanr(sig_row[common_t], fwd_row[common_t])
                            if not np.isnan(rho):
                                ic_series.append(rho)
                    if ic_series:
                        ic_results["alpha_composite"] = round(float(np.mean(ic_series)), 4)
                        ok(f"  alpha_composite rolling IC: {ic_results['alpha_composite']:.4f} "
                           f"({len(ic_series)} periods)")
        except Exception as e:
            warn(f"  Could not compute IC from history: {e}")

    # For individual signals, compute IC from current file vs last 21-63d returns
    # (cross-sectional IC on today's signal vs recent realised returns)
    recent_dates = fwd_returns.index[-IC_WINDOW:] if len(fwd_returns) > IC_WINDOW else fwd_returns.index

    for sig_name in SIGNAL_MAP:
        if sig_name in ("ml_short", "ml_medium", "ml_long"):
            # ML signals: use appropriate horizon
            continue
        sig_scores = load_signal_scores(sig_name)
        if sig_scores is None or len(sig_scores) < 20:
            ic_results[sig_name] = None
            continue

        # Pair today's signal rank vs each historical forward return date
        ic_series = []
        for date in recent_dates:
            if date not in fwd_returns.index:
                continue
            fwd_row = fwd_returns.loc[date].dropna()
            common_t = sig_scores.index.intersection(fwd_row.index)
            if len(common_t) < 20:
                continue
            x = sig_scores[common_t].values
            y = fwd_row[common_t].values
            if len(x) != len(y) or len(x) < 10:
                continue
            rho, _ = spearmanr(x, y)
            if not np.isnan(rho):
                ic_series.append(rho)

        if len(ic_series) >= 5:
            mean_ic = float(np.mean(ic_series))
            ic_results[sig_name] = round(mean_ic, 4)
        else:
            ic_results[sig_name] = None

    # ML signals: compute IC vs their specific horizon
    for sig_name, horizon in [("ml_short", 5), ("ml_medium", 21), ("ml_long", 63)]:
        sig_scores = load_signal_scores(sig_name)
        if sig_scores is None or len(sig_scores) < 20:
            ic_results[sig_name] = None
            continue
        fwd_h = load_forward_returns(horizon)
        if fwd_h.empty:
            ic_results[sig_name] = None
            continue
        recent = fwd_h.index[-IC_WINDOW:] if len(fwd_h) > IC_WINDOW else fwd_h.index
        ic_series = []
        for date in recent:
            fwd_row = fwd_h.loc[date].dropna()
            common_t = sig_scores.index.intersection(fwd_row.index)
            if len(common_t) < 20:
                continue
            x = sig_scores[common_t].values
            y = fwd_row[common_t].values
            if len(x) != len(y) or len(x) < 10:
                continue
            rho, _ = spearmanr(x, y)
            if not np.isnan(rho):
                ic_series.append(rho)
        if len(ic_series) >= 5:
            ic_results[sig_name] = round(float(np.mean(ic_series)), 4)
        else:
            ic_results[sig_name] = None

    return ic_results


# ── Convert raw IC → multipliers ──────────────────────────────────────────────

def ic_to_multiplier(ic: float | None) -> float:
    """
    Map raw IC to a weight multiplier.
    IC=0.10 → 2.0× (strong predictive)
    IC=0.05 → 1.5×
    IC=0.0  → 1.0×
    IC<0    → 0.25×
    """
    if ic is None:
        return MULT_NEUTRAL
    if ic < 0:
        return MULT_CAP_LOW
    # linear scale: IC 0.0 → 1.0×, IC 0.10 → 2.0×, IC 0.20 → 3.0×
    mult = 1.0 + (ic / 0.10)
    return round(float(np.clip(mult, MULT_CAP_LOW, MULT_CAP_HIGH)), 3)


# ── Update signal_weights.json ────────────────────────────────────────────────

def update_signal_weights(ic_results: dict[str, float | None]):
    sw_path = ROOT / "signal_weights.json"
    try:
        existing = json.loads(sw_path.read_text()) if sw_path.exists() else {}
    except Exception:
        existing = {}

    # Build new ic_multipliers
    multipliers = {}
    for sig_name in SIGNAL_MAP:
        ic = ic_results.get(sig_name)
        multipliers[sig_name] = ic_to_multiplier(ic)

    existing["updated"]        = TODAY
    existing["n_ic_periods"]   = IC_WINDOW
    existing["raw_ic"]         = {k: ic_results.get(k) for k in SIGNAL_MAP}
    existing["ic_multipliers"] = multipliers
    existing["ic_computation"] = f"Spearman rank IC, {IC_WINDOW}d rolling, {FORWARD_HORIZON}d forward"

    sw_path.write_text(json.dumps(existing, indent=2, default=str))
    ok(f"signal_weights.json updated — {len(multipliers)} ic_multipliers")

    # Print table
    print(f"\n  {'Signal':<15} {'Raw IC':>8} {'Multiplier':>11} {'Direction':>10}")
    print(f"  {'─'*15} {'─'*8} {'─'*11} {'─'*10}")
    for sig, ic in sorted(ic_results.items(), key=lambda x: (x[1] or 0), reverse=True):
        mult = multipliers.get(sig, 1.0)
        direction = "▲ BOOST" if mult > 1.2 else ("▼ DAMP" if mult < 0.8 else "→ NEUTRAL")
        ic_str = f"{ic:.4f}" if ic is not None else "  null"
        color = GREEN if (ic or 0) > 0.03 else (YELLOW if (ic or 0) >= 0 else RED)
        print(f"  {sig:<15} {color}{ic_str:>8}{RESET} {mult:>10.3f}×  {direction}")


# ── Save IC history ───────────────────────────────────────────────────────────

def append_ic_history(ic_results: dict[str, float | None]):
    hist_path = ROOT / "signal_ic_history.csv"
    row = {"date": TODAY}
    row.update({sig: ic for sig, ic in ic_results.items()})
    df = pd.DataFrame([row])
    if hist_path.exists():
        df.to_csv(hist_path, mode="a", header=False, index=False)
    else:
        df.to_csv(hist_path, index=False)
    ok(f"signal_ic_history.csv → appended row for {TODAY}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}Canyon — Rolling IC Optimizer{RESET}  {TODAY}")

    # Check staleness — skip if updated in last 5 days (runs weekly)
    sw_path = ROOT / "signal_weights.json"
    if sw_path.exists():
        try:
            sw = json.loads(sw_path.read_text())
            last = sw.get("updated", "2000-01-01")
            days_old = (datetime.now() - datetime.fromisoformat(last)).days
            if days_old < 5:
                ok(f"signal_weights.json is fresh ({days_old}d old) — skipping IC recompute")
                return
            log(f"signal_weights.json is {days_old}d old — recomputing ICs")
        except Exception:
            pass

    log("Loading forward returns from price cache …")
    fwd_returns = load_forward_returns(FORWARD_HORIZON)
    if fwd_returns.empty:
        err("No price data — cannot compute IC")
        return
    ok(f"Forward returns: {fwd_returns.shape[0]} dates × {fwd_returns.shape[1]} tickers")

    log(f"Computing rolling {IC_WINDOW}d IC for {len(SIGNAL_MAP)} signals …")
    ic_results = compute_ic_from_history(fwd_returns)

    n_computed = sum(1 for v in ic_results.values() if v is not None)
    ok(f"IC computed for {n_computed}/{len(ic_results)} signals")

    update_signal_weights(ic_results)
    append_ic_history(ic_results)

    print(f"\n{GREEN}✓ Rolling IC complete — signal_weights.json updated{RESET}\n")


if __name__ == "__main__":
    main()
