#!/usr/bin/env python3
"""
Canyon v9 — Step 84: Live IC Tracker
======================================
Validate that Step 77 predicted scores actually predict forward returns.

Method:
  IC = Spearman(predicted_score_t,  return_{t → t+H})

  Three horizons computed in parallel:
    5d  — fastest feedback (~1 week)
   10d  — medium confirmation (~2 weeks)
   21d  — primary signal horizon (~1 month)

  Good signal thresholds: IC > 0.03, t-stat > 1.5, IC hit-rate > 55%

Inputs:
  score_history.csv          — Step 77 daily snapshots
  sp500_price_cache.csv      — price data (or backtest_price_cache.csv)

Outputs:
  live_ic_history.csv        — per-period IC rows (all three horizons)
  live_ic_report.md          — summary report (English)
  signal_weights.json        — updated with mean_ic for Step 63 adaptive blend

Survivorship-bias detection:
  Tickers in score_history that are absent from the forward-return
  universe are counted and flagged.  Because only *surviving* stocks
  can be matched, the reported IC is an upper-bound estimate.

Usage:
  python3 canyon_final_v9_step84_live_ic_tracker.py
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent
SNAP_FILE  = ROOT / "score_history.csv"
PRICE_FILE = ROOT / "sp500_price_cache.csv"
PRICE_ALT  = ROOT / "backtest_price_cache.csv"
OUT_IC     = ROOT / "live_ic_history.csv"
OUT_REPORT = ROOT / "live_ic_report.md"
SW_FILE    = ROOT / "signal_weights.json"   # updated with mean_ic for Step 63

HOLD_PERIODS = [5, 10, 21]   # trading days
PRIMARY_H    = 21             # primary IC horizon used for aggregate stats
MIN_TICKERS  = 20             # minimum matched tickers to compute IC


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_scores() -> pd.DataFrame:
    if not SNAP_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(SNAP_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_prices() -> pd.DataFrame:
    for path in (PRICE_FILE, PRICE_ALT):
        if path.exists():
            return pd.read_csv(path, index_col=0, parse_dates=True)
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Survivorship-bias audit
# ─────────────────────────────────────────────────────────────────────────────

def survivorship_audit(scores: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """
    Return dict with counts of tickers in score_history that are absent
    from the price universe.  These are likely delisted / merged stocks.
    Their exclusion from IC calculation biases IC upward.
    """
    scored_tickers  = set(scores["ticker"].dropna().unique())
    price_tickers   = set(prices.columns.tolist())
    missing_tickers = scored_tickers - price_tickers
    return {
        "n_scored":    len(scored_tickers),
        "n_in_prices": len(scored_tickers & price_tickers),
        "n_missing":   len(missing_tickers),
        "pct_missing": round(len(missing_tickers) / max(len(scored_tickers), 1) * 100, 1),
        "missing_sample": sorted(missing_tickers)[:20],   # first 20 for report
    }


# ─────────────────────────────────────────────────────────────────────────────
# IC computation — all three horizons
# ─────────────────────────────────────────────────────────────────────────────

def compute_live_ic(scores: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    For each score date that has complete forward-return data at all three
    horizons, compute cross-sectional Spearman IC.

    Returns a DataFrame with columns:
        score_date, regime, hold_days,
        ic, p_value, n_tickers, long_signals, ic_positive
    """
    prices = prices.ffill()
    scoring_dates = sorted(scores["date"].unique())

    rows = []

    # Pre-compute forward returns for all horizons at once
    fwd_rets = {h: prices.pct_change(h).shift(-h) for h in HOLD_PERIODS}

    for score_date in scoring_dates:
        grp = scores[scores["date"] == score_date]

        # Find nearest trading date in prices
        matching = prices.index[prices.index >= score_date]
        if len(matching) == 0:
            continue
        actual_date = matching[0]

        regime     = grp["regime"].iloc[0] if "regime" in grp.columns else "UNKNOWN"
        long_count = int((grp["signal"] == "LONG").sum()) if "signal" in grp.columns else 0
        s_series   = grp.set_index("ticker")["predicted_score"]

        for h in HOLD_PERIODS:
            # Check sufficient future data exists
            future_dates = prices.index[prices.index > actual_date]
            if len(future_dates) < h:
                continue   # forward window not yet closed

            if actual_date not in fwd_rets[h].index:
                continue

            fwd    = fwd_rets[h].loc[actual_date].dropna()
            common = s_series.index.intersection(fwd.index)

            if len(common) < MIN_TICKERS:
                continue

            s_vals = s_series[common].values.astype(float)
            f_vals = fwd[common].values.astype(float)
            mask   = ~(np.isnan(s_vals) | np.isnan(f_vals))

            if mask.sum() < MIN_TICKERS:
                continue

            ic, pval = stats.spearmanr(s_vals[mask], f_vals[mask])
            if np.isnan(ic):
                continue

            rows.append({
                "score_date":   score_date.strftime("%Y-%m-%d"),
                "regime":       regime,
                "hold_days":    h,
                "ic":           round(float(ic), 4),
                "p_value":      round(float(pval), 4),
                "n_tickers":    int(mask.sum()),
                "long_signals": long_count,
                "ic_positive":  int(ic > 0),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Update signal_weights.json with mean_ic (consumed by Step 63)
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_signal_ic(prices: pd.DataFrame) -> dict[str, float]:
    """
    Compute per-signal IC from alpha_scores.csv (which contains sig_* columns).

    Reads alpha_score_history.csv (multi-day snapshots) and computes
    Spearman IC for each 'sig_<name>' column vs 5-day forward return.
    Returns {signal_name: mean_ic}.  Signals with < 10 periods get 0.0.

    This feeds the fast-feedback loop: signals that predicted poorly recently
    get their weights reduced; signals that predicted well get a bonus.
    Step 87 section 1.6 reads ic_multipliers from signal_weights.json and
    applies them on top of regime weights — so the loop is:
      Step 87 → alpha_score_history.csv → Step 84 → signal_weights.json → Step 87
    """
    hist_path = ROOT / "alpha_score_history.csv"
    if not hist_path.exists():
        return {}

    try:
        hist = pd.read_csv(hist_path)
        if "date" not in hist.columns or hist.empty:
            return {}

        hist["date"] = pd.to_datetime(hist["date"])
        sig_cols = [c for c in hist.columns if c.startswith("sig_")]
        if not sig_cols:
            return {}

        # Forward return horizon: 5 trading days (fast feedback)
        H = 5
        prices_fwd = prices.ffill()
        fwd_rets   = prices_fwd.pct_change(H).shift(-H)

        per_signal_ic: dict[str, list[float]] = {c: [] for c in sig_cols}

        scoring_dates = sorted(hist["date"].unique())
        for score_date in scoring_dates:
            grp = hist[hist["date"] == score_date]

            # Find matching price date
            matching = prices_fwd.index[prices_fwd.index >= score_date]
            if len(matching) == 0:
                continue
            actual_date = matching[0]

            # Check 5-day forward return data exists
            future_dates = prices_fwd.index[prices_fwd.index > actual_date]
            if len(future_dates) < H:
                continue
            if actual_date not in fwd_rets.index:
                continue

            fwd = fwd_rets.loc[actual_date].dropna()

            for col in sig_cols:
                if col not in grp.columns:
                    continue
                s_series = grp.set_index("ticker")[col].dropna()
                common   = s_series.index.intersection(fwd.index)
                if len(common) < 15:
                    continue
                s_vals = s_series[common].values.astype(float)
                f_vals = fwd[common].values.astype(float)
                mask   = ~(np.isnan(s_vals) | np.isnan(f_vals))
                if mask.sum() < 15:
                    continue
                try:
                    from scipy import stats as _sp_stats
                    ic, _ = _sp_stats.spearmanr(s_vals[mask], f_vals[mask])
                    if not np.isnan(ic):
                        per_signal_ic[col].append(float(ic))
                except Exception:
                    pass

        # Compute mean IC per signal
        result: dict[str, float] = {}
        for col, ics in per_signal_ic.items():
            signal_name = col.replace("sig_", "")
            if len(ics) >= 5:
                result[signal_name] = round(float(np.mean(ics)), 4)

        return result

    except Exception as e:
        print(f"  [per-signal IC] error: {e}")
        return {}


def ic_to_multiplier(ic: float) -> float:
    """
    Map a signal's mean IC to a weight multiplier for Step 87.

    IC thresholds (empirically calibrated):
      IC > 0.06 → 1.40  (strong predictor — double down)
      IC > 0.04 → 1.20  (good predictor — boost)
      IC > 0.02 → 1.05  (marginal — slight boost)
      IC  0-0.02 → 1.00  (noise — neutral)
      IC < 0     → 0.75  (negative predictor — reduce significantly)
      IC < -0.04 → 0.50  (strongly negative — halve weight)
    """
    if   ic > 0.06: return 1.40
    elif ic > 0.04: return 1.20
    elif ic > 0.02: return 1.05
    elif ic >= 0:   return 1.00
    elif ic > -0.04: return 0.75
    else:           return 0.50


def update_signal_weights(mean_ic: float,
                          per_signal_ic: dict[str, float] | None = None) -> None:
    """
    Write mean_ic and per-signal IC multipliers to signal_weights.json.

    Consumed by:
      Step 63 (portfolio optimizer): reads mean_ic to set alpha/hist blend ratio
      Step 87 (alpha aggregator): reads ic_multipliers to boost/trim signal weights
    """
    sw: dict = {}
    if SW_FILE.exists():
        try:
            sw = json.loads(SW_FILE.read_text())
        except Exception:
            sw = {}

    sw["mean_ic"]    = round(mean_ic, 4)
    sw["ic_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    if per_signal_ic:
        mults = {sig: ic_to_multiplier(ic) for sig, ic in per_signal_ic.items()}
        sw["ic_multipliers"]  = mults
        sw["per_signal_ic"]   = per_signal_ic
        # Log top/bottom movers
        sorted_mults = sorted(mults.items(), key=lambda x: x[1], reverse=True)
        boosted = [(k, v) for k, v in sorted_mults if v > 1.0][:3]
        reduced = [(k, v) for k, v in sorted_mults if v < 1.0][:3]
        if boosted:
            print(f"  [per-signal] boosted: "
                  + ", ".join(f"{k}×{v:.2f}(IC={per_signal_ic[k]:+.3f})"
                               for k, v in boosted))
        if reduced:
            print(f"  [per-signal] reduced: "
                  + ", ".join(f"{k}×{v:.2f}(IC={per_signal_ic[k]:+.3f})"
                               for k, v in reduced))
    else:
        print("  [per-signal] no per-signal IC computed (need alpha_score_history.csv with sig_* cols)")

    SW_FILE.write_text(json.dumps(sw, indent=2))
    print(f"  signal_weights.json updated  (mean_ic={mean_ic:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────

def write_report(
    ic_df: pd.DataFrame,
    total_days: int,
    surv: dict | None = None,
) -> None:
    now  = datetime.now().strftime("%Y-%m-%d %H:%M")
    need = max(0, PRIMARY_H - total_days)

    if ic_df.empty:
        lines = [
            "# Canyon v9 — Step 84: Live IC Tracker",
            f"Updated: {now}",
            "",
            "## Status",
            f"Accumulated **{total_days}** scoring days; "
            f"need **{need}** more trading days for the first IC.",
            "",
            "Run `python3 canyon_final_v9_step77_regime_ml.py --score` daily "
            "to accumulate data.",
        ]
        OUT_REPORT.write_text("\n".join(lines))
        return

    # Primary horizon stats
    primary = ic_df[ic_df["hold_days"] == PRIMARY_H]
    ics     = primary["ic"].values if not primary.empty else ic_df["ic"].values
    mean_ic = float(ics.mean())
    std_ic  = float(ics.std(ddof=1)) if len(ics) > 1 else 0.0
    t_stat  = mean_ic / (std_ic / len(ics)**0.5 + 1e-10) if len(ics) > 1 else 0.0
    ic_pos  = float(primary["ic_positive"].mean() if not primary.empty
                    else ic_df["ic_positive"].mean()) * 100

    if mean_ic > 0.05 and t_stat > 2.0:
        status = "✅ Strong IC — model validated by live market"
    elif mean_ic > 0.03 and t_stat > 1.5:
        status = "✅ Acceptable IC — signal usable, continue monitoring"
    elif mean_ic > 0:
        status = "⚠️  Weak positive IC — not yet significant, accumulate more data"
    else:
        status = "❌ Negative IC — check signal logic"

    lines = [
        "# Canyon v9 — Step 84: Live IC Report",
        f"Updated: {now}  |  Completed periods: {len(primary) if not primary.empty else len(ic_df)}",
        "",
        "## Summary",
        f"> {status}",
        "",
        "## Core Metrics  (primary horizon = 21d)",
        "| Metric | Value | Threshold |",
        "|--------|-------|-----------|",
        f"| Mean IC | **{mean_ic:.4f}** | >0.03 = signal present |",
        f"| IC Std  | {std_ic:.4f} | lower = more stable |",
        f"| t-stat  | **{t_stat:.2f}** | >1.5 = significant |",
        f"| IC hit-rate | {ic_pos:.0f}% | >55% = consistent |",
        f"| Validated periods | {len(ics)} | >12 = reliable |",
        "",
    ]

    # Multi-horizon summary
    horizon_lines = [
        "## IC by Horizon",
        "| Horizon | Mean IC | t-stat | Periods |",
        "|---------|---------|--------|---------|",
    ]
    for h in HOLD_PERIODS:
        sub = ic_df[ic_df["hold_days"] == h]
        if sub.empty:
            horizon_lines.append(f"| {h}d | — | — | 0 |")
            continue
        h_ics  = sub["ic"].values
        h_mean = float(h_ics.mean())
        h_std  = float(h_ics.std(ddof=1)) if len(h_ics) > 1 else 0.0
        h_t    = h_mean / (h_std / len(h_ics)**0.5 + 1e-10) if len(h_ics) > 1 else 0.0
        flag   = "✅" if h_mean > 0.03 else ("⚠️" if h_mean > 0 else "❌")
        horizon_lines.append(
            f"| {h}d | {h_mean:+.4f} {flag} | {h_t:.2f} | {len(sub)} |"
        )
    horizon_lines.append("")
    lines += horizon_lines

    # Survivorship bias warning
    if surv and surv["n_missing"] > 0:
        pct = surv["pct_missing"]
        sample = ", ".join(surv["missing_sample"][:10])
        lines += [
            "## ⚠️  Survivorship Bias Warning",
            f"**{surv['n_missing']} tickers** ({pct}% of scored universe) appear in "
            f"score_history but are absent from the price universe.  "
            f"These are likely delisted / merged stocks.",
            f"Because only surviving stocks are matched, the reported IC is an "
            f"**upper-bound estimate** — true IC is likely lower.",
            f"",
            f"Missing sample: `{sample}`{'...' if surv['n_missing'] > 10 else ''}",
            "",
        ]
    elif surv:
        lines += [
            "## Survivorship Bias",
            f"All {surv['n_scored']} scored tickers found in price universe.  "
            f"No survivorship bias detected.",
            "",
        ]

    # By regime
    if "regime" in ic_df.columns and not primary.empty:
        by_regime = primary.groupby("regime")["ic"].agg(["mean", "count"]).round(4)
        lines += [
            "## IC by Regime  (21d horizon)",
            "| Regime | Mean IC | Periods |",
            "|--------|---------|---------|",
        ]
        for regime, row in by_regime.iterrows():
            flag = "✅" if row["mean"] > 0.03 else ("⚠️" if row["mean"] > 0 else "❌")
            lines.append(f"| {regime} | {row['mean']:.4f} {flag} | {int(row['count'])} |")
        lines.append("")

    # Per-period detail (primary horizon)
    detail = primary if not primary.empty else ic_df[ic_df["hold_days"] == ic_df["hold_days"].max()]
    lines += [
        "## Period Detail  (21d horizon)",
        "| Score Date | Regime | IC | p-value | Tickers | LONG |",
        "|-----------|--------|----|---------|---------|------|",
    ]
    for _, r in detail.sort_values("score_date", ascending=False).iterrows():
        flag = "✅" if r["ic"] > 0.03 else ("➡️" if r["ic"] > 0 else "❌")
        lines.append(
            f"| {r['score_date']} | {r['regime']} | "
            f"{r['ic']:+.4f} {flag} | {r['p_value']:.3f} | "
            f"{r['n_tickers']} | {r['long_signals']} |"
        )

    OUT_REPORT.write_text("\n".join(lines))
    print(f"  Report: {OUT_REPORT.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("=" * 60)
    print("Canyon v9 — Step 84: Live IC Tracker")
    print("=" * 60)

    print("\n[1/4] Loading score history …")
    scores = load_scores()
    if scores.empty:
        print("  score_history.csv not found")
        print("  Run: python3 canyon_final_v9_step77_regime_ml.py --score")
        return

    total_days = scores["date"].nunique()
    total_recs = len(scores)
    print(f"  {total_days} scoring dates, {total_recs} records")

    if total_days < PRIMARY_H:
        need = PRIMARY_H - total_days
        print(f"\n  ⏳ Need {need} more trading days for the first IC")
        print(f"  Run Step 77 daily — expected first IC in ~{need} trading days")
        write_report(pd.DataFrame(), total_days, surv=None)
        return

    print("\n[2/4] Loading price data …")
    prices = load_prices()
    if prices.empty:
        print("  Price cache not found — run Step 75 or Step 62 first")
        return
    print(f"  {len(prices)} days × {len(prices.columns)} tickers")

    print("\n[3/4] Survivorship bias audit …")
    surv = survivorship_audit(scores, prices)
    print(f"  Scored tickers: {surv['n_scored']}")
    print(f"  In price universe: {surv['n_in_prices']}")
    if surv["n_missing"] > 0:
        print(f"  ⚠️  Missing (likely delisted): {surv['n_missing']} "
              f"({surv['pct_missing']}%)  →  IC is upper-bound")
        if surv["missing_sample"]:
            print(f"     Sample: {', '.join(surv['missing_sample'][:8])}")
    else:
        print("  No survivorship bias detected")

    print(f"\n[4/4] Computing IC  (horizons: {HOLD_PERIODS}d) …")
    ic_df = compute_live_ic(scores, prices)

    if ic_df.empty:
        print(f"  No complete forward-return windows yet "
              f"(need {PRIMARY_H} trading days of future prices)")
        write_report(pd.DataFrame(), total_days, surv=surv)
        return

    ic_df.to_csv(OUT_IC, index=False)

    # Stats by horizon
    for h in HOLD_PERIODS:
        sub = ic_df[ic_df["hold_days"] == h]
        if sub.empty:
            continue
        h_mean = sub["ic"].mean()
        h_t    = (h_mean / (sub["ic"].std(ddof=1) / len(sub)**0.5 + 1e-10)
                  if len(sub) > 1 else 0.0)
        tag = "✅" if h_mean > 0.03 and h_t > 1.5 else ("⚠️" if h_mean > 0 else "❌")
        print(f"  {h:>2}d horizon — IC={h_mean:+.4f}  t={h_t:.2f}  "
              f"n={len(sub)} periods  {tag}")

    # Primary horizon summary
    primary = ic_df[ic_df["hold_days"] == PRIMARY_H]
    if not primary.empty:
        mean_ic = primary["ic"].mean()
        std_ic  = primary["ic"].std(ddof=1) if len(primary) > 1 else 0.0
        t_stat  = (mean_ic / (std_ic / len(primary)**0.5 + 1e-10)
                   if len(primary) > 1 else 0.0)
        print(f"\n  Primary (21d)  mean IC: {mean_ic:+.4f}   t-stat: {t_stat:.2f}")

        if mean_ic > 0.03:
            print("  ✅ Signal valid — model prediction power confirmed")
        elif mean_ic > 0:
            print("  ⚠️  Positive but weak — accumulate more data")
        else:
            print("  ❌ Negative IC — review signal logic")

        # ── Per-signal IC for fast feedback loop ─────────────────────────────
        # Compute Spearman IC per individual signal (from alpha_score_history.csv
        # sig_* columns) and write ic_multipliers → signal_weights.json.
        # Step 87 section 1.6 reads these multipliers to auto-boost/trim weights.
        print("\n  [per-signal IC] computing 5-day per-signal IC …")
        per_sig = compute_per_signal_ic(prices)
        if per_sig:
            print(f"  [per-signal IC] {len(per_sig)} signals computed:")
            for sn, sic in sorted(per_sig.items(), key=lambda x: x[1], reverse=True):
                flag = "✅" if sic > 0.03 else ("⚠️" if sic > 0 else "❌")
                print(f"    {sn:<15s}  IC={sic:+.4f}  mult={ic_to_multiplier(sic):.2f}  {flag}")

        # Push mean_ic + per-signal multipliers to signal_weights.json
        update_signal_weights(mean_ic, per_signal_ic=per_sig)
    else:
        mean_ic  = ic_df["ic"].mean()
        per_sig  = compute_per_signal_ic(prices)
        update_signal_weights(mean_ic, per_signal_ic=per_sig)

    write_report(ic_df, total_days, surv=surv)
    print(f"  Saved: {OUT_IC.name}")

    print(f"\nDone.")
    print("=" * 60)


if __name__ == "__main__":
    main()
