#!/usr/bin/env python3
"""
Canyon v9 — Step 85: Monthly IC Weight Optimizer
=================================================
Computes data-driven signal weights using realized Out-of-Sample IC,
replacing the static literature-based weights in IC_WEIGHTS.

Method: IC²-weighted with Bayesian shrinkage
  1. Compute realized IC for ML sub-signals from wf_oos_predictions.csv
     (ridge, rf, lgbm, ensemble) vs actual 21-day forward returns
  2. Merge with rolling_ic_monitor.csv for other signals
     (fear_vix, google_trends, sec_filing_lag)
  3. For signals with no historical data: use literature-prior ICs with
     a 50% shrinkage weight toward zero (conservative Bayesian estimate)
  4. Final weight_i = IC_i² / sum(IC_j²) — pure IC²-normalized weighting
  5. Apply floor/ceiling: min 0.01, max 0.40 per signal (diversification)
  6. Save ic_weights_optimized.csv (valid for 30 days; recalculate monthly)

Step500 loads this file if it exists and is < 30 days old, using these
weights instead of the hardcoded IC_WEIGHTS dict.

Outputs:
  ic_weights_optimized.csv  — per-signal IC estimates and optimized weights
  ic_optimization_report.md — diagnostic report

Usage:
  python3 canyon_final_v9_step85_ic_optimizer.py
  python3 canyon_final_v9_step85_ic_optimizer.py --refresh
"""
from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent
OUT_CSV    = ROOT / "ic_weights_optimized.csv"
OUT_REPORT = ROOT / "ic_optimization_report.md"

FRESHNESS_DAYS = 30    # only recompute monthly

# ── Literature-based IC priors (conservative estimates from academic research)
# Used when no realized OOS data available; shrunk 50% toward zero
LITERATURE_IC = {
    "ml_score":        0.065,  # CPCV ensemble — computed from WF data
    "factor_score":    0.042,  # Fama-French momentum/quality
    "smart_money":     0.050,  # 13F institutional cluster
    "accruals":        0.080,  # Sloan (1996), Dechow (1995)
    "squeeze":         0.060,  # Concentrated short → reversal
    "sig_fundamental": 0.060,  # XBRL E/P+ROE+FCF+RevGrowth composite (step89)
    "finbert":         0.045,  # FinBERT news sentiment (literature: 0.03-0.06)
    "sig_10k":         0.035,  # MD&A tone delta — Loughran & McDonald (2011)
    "sig_8k":          0.050,  # Earnings 8-K tone (real-time, higher prior than 10-K)
    "sig_insider":     0.055,  # Lakonishok & Lee (2001): 0.04-0.07
    "sig_revision":    0.040,  # Analyst upgrades IC (Chan et al. 1996)
    "sig_options":     0.030,  # Options order flow (Pan & Poteshman 2006)
    "alt_trends":      0.020,  # Google Trends (Da et al. 2011)
    "alt_wiki":        0.018,  # Wikipedia views — lower prior (less evidence)
    "sig_cross_asset": 0.030,  # Intermarket momentum (literature: 0.02-0.04)
    "sig_crowd":       0.030,  # 13F crowding contrarian (Khandani & Lo 2007)
    "sig_pead":        0.070,  # Post-earnings announcement drift (Bernard & Thomas 1989)
}
SHRINKAGE = 0.50   # how much to shrink literature priors toward zero

# Signal name mapping: rolling_ic_monitor → step500 IC_WEIGHTS key
ROLLING_IC_MAP = {
    "ml_ensemble":  "ml_score",
    "google_trends": "alt_trends",
    "fear_vix":     None,          # macro overlay, not in IC_WEIGHTS directly
    "sec_filing_lag": "sig_10k",
}


# ── 1. Compute IC for ML sub-signals ─────────────────────────────────────────

def compute_ml_submodel_ic(lookback_months: int = 18) -> dict[str, float]:
    """
    Compute Spearman IC for each ML sub-model vs realized 21-day forward returns.
    Uses: wf_oos_predictions.csv (signals) + backtest_price_cache.csv (returns).
    """
    wf_path = ROOT / "wf_oos_predictions.csv"
    px_path = ROOT / "backtest_price_cache.csv"

    if not (wf_path.exists() and px_path.exists()):
        return {}

    wf = pd.read_csv(wf_path)
    wf["rebalance_date"] = pd.to_datetime(wf["rebalance_date"], errors="coerce")
    oos = wf[wf["is_oos"] == True].dropna(subset=["rebalance_date"]).copy()

    if oos.empty:
        return {}

    # Only use lookback period
    cutoff = oos["rebalance_date"].max() - pd.DateOffset(months=lookback_months)
    oos = oos[oos["rebalance_date"] >= cutoff]

    # Load prices
    px = pd.read_csv(px_path, index_col=0, parse_dates=True)
    px = px.sort_index()

    # Compute 21-day forward returns per ticker per rebalance_date
    ics: dict[str, list[float]] = {
        "ridge_score": [], "rf_score": [], "lgbm_score": [], "ensemble_score": []
    }

    for date, grp in oos.groupby("rebalance_date"):
        grp = grp.dropna(subset=list(ics.keys()))
        tickers = grp["ticker"].tolist()

        # Find the date in price index (nearest trading day)
        valid_dates = px.index[px.index >= date]
        if len(valid_dates) < 22:
            continue
        t0_idx = px.index.get_loc(valid_dates[0])
        t21_idx = t0_idx + 21
        if t21_idx >= len(px):
            continue

        # Forward returns at 21d horizon
        fwd_rets = {}
        for tk in tickers:
            if tk in px.columns:
                p0  = float(px.iloc[t0_idx][tk])
                p21 = float(px.iloc[t21_idx][tk])
                if p0 > 0 and not (np.isnan(p0) or np.isnan(p21)):
                    fwd_rets[tk] = p21 / p0 - 1

        if len(fwd_rets) < 5:
            continue

        actual = pd.Series(fwd_rets)
        for model in ics:
            if model not in grp.columns:
                continue
            pred = grp.set_index("ticker")[model].reindex(actual.index).dropna()
            act_aligned = actual.reindex(pred.index).dropna()
            common = pred.index.intersection(act_aligned.index)
            if len(common) < 5:
                continue
            ic_val, _ = spearmanr(pred[common], act_aligned[common])
            if not np.isnan(ic_val):
                ics[model].append(float(ic_val))

    # Return mean IC per sub-model
    result = {}
    for model, vals in ics.items():
        if vals:
            result[model] = float(np.mean(vals))
            print(f"  [ML-IC] {model}: mean IC = {result[model]:+.4f}  "
                  f"(n={len(vals)} periods)")
    return result


# ── 2. Load rolling OOS IC monitor ───────────────────────────────────────────

def load_monitor_ic() -> dict[str, float]:
    """Latest OOS IC from rolling_ic_monitor.csv, mapped to step500 signal keys."""
    p = ROOT / "rolling_ic_monitor.csv"
    if not p.exists():
        return {}

    df = pd.read_csv(p)
    oos = df[df["period"] == "OOS"] if "period" in df.columns else df
    latest = oos.sort_values("date").groupby("signal").last()

    result = {}
    for rolling_name, step_key in ROLLING_IC_MAP.items():
        if step_key is None or rolling_name not in latest.index:
            continue
        # Prefer ic_6m (most stable); fall back to ic_3m
        ic = latest.loc[rolling_name].get("ic_6m", np.nan)
        if np.isnan(ic):
            ic = latest.loc[rolling_name].get("ic_3m", np.nan)
        if not np.isnan(ic):
            result[step_key] = float(ic)
            print(f"  [Monitor-IC] {step_key} (from {rolling_name}): {ic:+.4f}")
    return result


# ── 3. Merge ICs, shrink, compute weights ─────────────────────────────────────

def compute_optimized_weights(
    ml_ics:      dict[str, float],
    monitor_ics: dict[str, float],
) -> pd.DataFrame:
    """
    Merge all IC sources → compute IC²-normalized weights with shrinkage.

    Priority:
      1. Realized ML sub-model IC (from wf_oos_predictions.csv)
      2. Realized rolling monitor IC (from rolling_ic_monitor.csv)
      3. Shrunk literature prior (50% toward zero)
    """
    rows = []

    for sig, lit_ic in LITERATURE_IC.items():
        # Source priority
        if sig == "ml_score" and "ensemble_score" in ml_ics:
            ic_est   = ml_ics["ensemble_score"]
            source   = "realized_wf_oos"
            n_months = 18   # approximate
        elif sig in monitor_ics:
            ic_est   = monitor_ics[sig]
            source   = "realized_monitor"
            n_months = 12
        else:
            # Shrink literature prior by 50% (standard Bayesian shrinkage for limited data)
            ic_est   = lit_ic * (1.0 - SHRINKAGE)
            source   = "shrunk_literature"
            n_months = 0

        # IC can be negative for contrarian; take |IC| for weight computation
        # but preserve sign for reporting
        ic_sq = ic_est ** 2

        rows.append({
            "signal":    sig,
            "ic_est":    round(ic_est, 4),
            "ic_sq":     round(ic_sq, 6),
            "source":    source,
            "n_months":  n_months,
            "lit_prior": lit_ic,
        })

    df = pd.DataFrame(rows)

    # IC²-normalized weights with floor=0.01, cap=0.40
    total_ic_sq = df["ic_sq"].sum()
    if total_ic_sq < 1e-9:
        df["weight_raw"] = 1.0 / len(df)
    else:
        df["weight_raw"] = df["ic_sq"] / total_ic_sq

    # Apply floor and cap
    floor, cap = 0.01, 0.40
    df["weight"] = df["weight_raw"].clip(lower=floor, upper=cap)

    # Renormalize after clipping
    total = df["weight"].sum()
    df["weight"] = (df["weight"] / total).round(4)

    # Final renormalize pass (floating point)
    df["weight"] = (df["weight"] / df["weight"].sum()).round(4)
    df.loc[df.index[-1], "weight"] += round(1.0 - df["weight"].sum(), 4)

    df["updated_date"] = datetime.now().strftime("%Y-%m-%d")
    return df.sort_values("weight", ascending=False).reset_index(drop=True)


# ── 4. Main ───────────────────────────────────────────────────────────────────

def run(force: bool = False) -> pd.DataFrame:
    if not force and OUT_CSV.exists():
        age_days = (datetime.now().timestamp() - OUT_CSV.stat().st_mtime) / 86400
        if age_days < FRESHNESS_DAYS:
            print(f"  [IC-Opt] Weights are {age_days:.0f} days old (< {FRESHNESS_DAYS}d) — skipping. "
                  f"Use --refresh to force.")
            return pd.read_csv(OUT_CSV)

    print("\n  [IC-Opt] Computing realized IC for ML sub-models …")
    ml_ics = compute_ml_submodel_ic(lookback_months=18)

    print("\n  [IC-Opt] Loading rolling monitor ICs …")
    monitor_ics = load_monitor_ic()

    print("\n  [IC-Opt] Merging sources + computing IC²-optimal weights …")
    df = compute_optimized_weights(ml_ics, monitor_ics)

    df.to_csv(OUT_CSV, index=False)
    print(f"\n  [IC-Opt] Saved {len(df)} signal weights → {OUT_CSV.name}")
    return df


def write_report(df: pd.DataFrame) -> None:
    total_wf  = (df["source"] == "realized_wf_oos").sum()
    total_mon = (df["source"] == "realized_monitor").sum()
    total_lit = (df["source"] == "shrunk_literature").sum()

    rows = ""
    for _, r in df.iterrows():
        ic_color = "+" if r["ic_est"] >= 0 else ""
        rows += (f"| **{r['signal']}** | {ic_color}{r['ic_est']:.4f} | "
                 f"{r['ic_sq']:.5f} | **{r['weight']:.4f}** | "
                 f"{r['source']} | {r['lit_prior']:.4f} |\n")

    report = f"""# IC Weight Optimization Report — {datetime.now():%Y-%m-%d}

## Summary

| Source | Signals |
|--------|---------|
| Realized OOS (walk-forward) | {total_wf} |
| Realized monitor IC | {total_mon} |
| Shrunk literature prior | {total_lit} |

Shrinkage factor applied to literature priors: **{SHRINKAGE:.0%} toward zero**
IC²-normalization with floor=1% / cap=40% per signal.

## Signal Weights (IC²-Optimal)

| Signal | IC Est. | IC² | Weight | Source | Lit. Prior |
|--------|:-------:|:---:|:------:|--------|:----------:|
{rows}
## Interpretation

- **Realized OOS IC** = computed from actual out-of-sample predictions vs returns.
  These are the most trustworthy and receive full IC² weight.
- **Monitor IC** = from rolling_ic_monitor.csv (6-month rolling OOS Spearman IC).
  High confidence but less data points per signal.
- **Shrunk literature** = academic prior reduced by {SHRINKAGE:.0%}.
  Used when no live data exists. Conservative to avoid overweighting untested signals.

These weights replace IC_WEIGHTS in step500 for the next {FRESHNESS_DAYS} days.
Recalculates automatically on the 1st of each month.
"""
    OUT_REPORT.write_text(report)
    print(f"  [IC-Opt] Report saved → {OUT_REPORT.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monthly IC Weight Optimizer")
    parser.add_argument("--refresh", action="store_true",
                        help="Force recomputation even if < 30 days old")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Canyon v9 — IC Weight Optimizer  [{datetime.now():%Y-%m-%d %H:%M}]")
    print("=" * 60)

    df = run(force=args.refresh)
    if not df.empty:
        write_report(df)
        print(f"\nTop 5 signals by optimized weight:")
        print(df[["signal","ic_est","weight","source"]].head(5).to_string(index=False))

    print("\n" + "=" * 60)
    print("Step 85 Complete")
    print("=" * 60)
