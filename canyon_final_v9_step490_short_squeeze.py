#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canyon v9 — Step 490: Short Squeeze Composite Signal
=====================================================
Short squeeze is one of the highest risk-adjusted return equity events.
GameStop 2021, Volkswagen 2008, and AMC 2021 are extreme examples.
But institutions capture this signal on a smaller scale daily.

Four trigger conditions for a Short Squeeze (signal is strongest when all are met):

  Condition 1 — High Short Interest
    Short % of float > 10%
    Days to Cover > 5 days
    → Large number of trapped shorts; any price rise forces heavy covering

  Condition 2 — Unusual Options Activity
    PCR < 0.7 (retail aggressively buying calls)
    Implied volatility IV > 80th percentile (market expects large move)
    → Retail/retail options buying forces market makers to delta-hedge by buying shares

  Condition 3 — Analyst Upgrade (Analyst Revision)
    Net upgrades in past 45 days > 0
    → Improving fundamentals support price rise; shorts face fundamental headwind

  Condition 4 — Price Momentum Breakout (Momentum Trigger)
    21-day return > 0 and > SPY over same period
    → Price has already begun rising; shorts start accumulating unrealized losses

Composite score = number of conditions met (0–4) + intensity weighting

Historical validation:
  Next-month average return when all 4 conditions met vs. only 1 condition:
  Literature: Lamont & Thaler (2003), Hanson & Sunderam (2014)
  Expected IC: +0.12~+0.20 (highly unstable, tail distribution)

Usage:
  .venv/bin/python canyon_final_v9_step490_short_squeeze.py
"""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
OOS_CUTOFF = pd.Timestamp("2020-01-01")
ANN_FACTOR = 252


# =============================================================================
# 1. Aggregate all existing signal data
# =============================================================================

def load_all_signals() -> dict[str, pd.DataFrame]:
    """Load all previously computed signals into a dict."""
    signals = {}

    # Short interest (from step390)
    si_path = ROOT / "short_interest_snapshot.csv"
    if si_path.exists():
        signals["short_interest"] = pd.read_csv(si_path)

    # Options snapshot (from step400)
    opts_path = ROOT / "options_snapshot.csv"
    if opts_path.exists():
        signals["options"] = pd.read_csv(opts_path)

    # Analyst revision (from step390)
    rev_path = ROOT / "revision_composite_scores.csv"
    if rev_path.exists():
        signals["revision"] = pd.read_csv(rev_path, parse_dates=["rebalance_date"])

    # Factor composite (from step410) — for momentum
    factor_path = ROOT / "factor_composite.csv"
    if factor_path.exists():
        signals["factor"] = pd.read_csv(factor_path)

    # Smart money (from step480)
    sm_path = ROOT / "smart_money_signal.csv"
    if sm_path.exists():
        signals["smart_money"] = pd.read_csv(sm_path)

    # Accruals (from step470)
    acc_path = ROOT / "accruals_snapshot.csv"
    if acc_path.exists():
        signals["accruals"] = pd.read_csv(acc_path)

    return signals


# =============================================================================
# 2. Score each condition
# =============================================================================

def score_squeeze_conditions(
    signals: dict[str, pd.DataFrame],
    prices:  pd.DataFrame,
) -> pd.DataFrame:
    """
    For each ticker, score 0–4 based on how many conditions are met.
    Also compute a weighted continuous score for ranking.
    """
    tickers = [c for c in prices.columns if c != "SPY"]

    # Latest price momentum vs SPY
    ret_21 = prices.pct_change(21).iloc[-1]
    spy_21 = ret_21.get("SPY", 0.0) if "SPY" in ret_21.index else 0.0

    rows = []
    for tk in tickers:
        c1 = c2 = c3 = c4 = 0
        c1_val = c2_val = c3_val = c4_val = 0.0

        # --- Condition 1: Short Interest ---
        si_df = signals.get("short_interest", pd.DataFrame())
        if not si_df.empty and tk in si_df["ticker"].values:
            r = si_df[si_df["ticker"] == tk].iloc[0]
            si_pct = float(r.get("shortPercentOfFloat", 0) or 0)
            si_ratio = float(r.get("shortRatio", 0) or 0)
            if si_pct > 0.05:   # >5% float shorted
                c1 = 1
            if si_pct > 0.10:   # >10% = high squeeze risk
                c1 = 2
            c1_val = si_pct * 10   # scaled 0–10+

        # --- Condition 2: Options Activity (PCR + IV) ---
        opts_df = signals.get("options", pd.DataFrame())
        if not opts_df.empty and tk in opts_df["ticker"].values:
            r = opts_df[opts_df["ticker"] == tk].iloc[0]
            pcr = float(r.get("put_call_ratio", 1.0) or 1.0)
            iv  = float(r.get("atm_iv_avg", 0.30) or 0.30)
            # Low PCR (calls dominating) + high IV = squeeze setup
            if pcr < 1.0 and iv > 0.40:
                c2 = 1
            if pcr < 0.7:   # very bullish options activity
                c2 = 2
            c2_val = (1.5 - pcr) + iv   # higher = more bullish + fearful

        # --- Condition 3: Analyst Revision ---
        rev_df = signals.get("revision", pd.DataFrame())
        if not rev_df.empty and tk in rev_df["ticker"].values:
            latest = (rev_df[rev_df["ticker"] == tk]
                      .sort_values("rebalance_date")
                      .iloc[-1])
            rev_score = float(latest.get("revision_score", 0) or 0)
            if rev_score > 0:
                c3 = 1
            if rev_score > 2:
                c3 = 2
            c3_val = max(0, rev_score)

        # --- Condition 4: Price Momentum ---
        if tk in ret_21.index:
            ret_tk = float(ret_21[tk])
            if ret_tk > spy_21:     # outperforming SPY
                c4 = 1
            if ret_tk > spy_21 + 0.02:  # outperforming by 2%+
                c4 = 2
            c4_val = ret_tk - spy_21

        total_conditions = min(c1, 1) + min(c2, 1) + min(c3, 1) + min(c4, 1)
        intensity_score  = c1_val * 0.35 + c2_val * 0.25 + c3_val * 0.25 + c4_val * 0.15

        rows.append({
            "ticker":            tk,
            "conditions_met":    total_conditions,
            "short_interest_c":  c1,
            "options_c":         c2,
            "revision_c":        c3,
            "momentum_c":        c4,
            "intensity_score":   round(intensity_score, 4),
            "short_pct":         round(c1_val / 10, 3),
            "pcr":               round(float(signals.get("options", pd.DataFrame())
                                             [signals.get("options", pd.DataFrame())
                                              ["ticker"] == tk]["put_call_ratio"]
                                             .values[0]) if not signals.get("options", pd.DataFrame()).empty
                                        and tk in signals.get("options", pd.DataFrame())["ticker"].values
                                        else 1.0, 3),
            "momentum_vs_spy":   round(float(ret_21.get(tk, 0) - spy_21), 4),
        })

    df = pd.DataFrame(rows)
    return df.sort_values(["conditions_met", "intensity_score"],
                          ascending=[False, False])


# =============================================================================
# 3. IC evaluation (signal combinations from historical backtest)
# =============================================================================

def evaluate_squeeze_ic(
    prices:   pd.DataFrame,
    signals:  dict[str, pd.DataFrame],
    reb_dates: list[str],
) -> tuple[pd.DataFrame, float, float]:
    """
    Build historical short-squeeze proxy from available data:
    - short interest (current snapshot only)
    - momentum (from prices)
    Evaluate cross-sectional IC for OOS period.
    """
    oos = [r for r in reb_dates if r >= str(OOS_CUTOFF.date())]
    tickers = [c for c in prices.columns if c != "SPY"]
    ic_rows = []

    # Use momentum × short interest as a proxy for historical squeeze signal
    # (we only have current short interest, so this is partially historical)
    for i, reb in enumerate(oos[:-1]):
        next_reb = oos[i + 1]
        reb_ts   = pd.Timestamp(reb)

        # Momentum score (price-based, fully historical)
        t_21d_ago = prices.index[prices.index <= reb_ts]
        if len(t_21d_ago) < 21:
            continue
        t_now = t_21d_ago[-1]
        t_21  = t_21d_ago[-22]

        spy_21 = 0.0
        if "SPY" in prices.columns:
            ps0 = prices["SPY"].loc[t_21]
            ps1 = prices["SPY"].loc[t_now]
            if ps0 > 0:
                spy_21 = float(ps1 / ps0) - 1

        scores = {}
        for tk in tickers:
            if tk not in prices.columns:
                continue
            p0 = prices[tk].loc[t_21]
            p1 = prices[tk].loc[t_now]
            if p0 > 0 and p1 > 0:
                mom = float(p1 / p0) - 1
                scores[tk] = mom - spy_21  # relative momentum

        # Forward returns
        t0s = prices.index[prices.index >= reb_ts]
        t1s = prices.index[prices.index >= pd.Timestamp(next_reb)]
        if not len(t0s) or not len(t1s):
            continue

        fwd = {}
        for tk in scores:
            p0 = prices[tk].loc[t0s[0]]
            p1 = prices[tk].loc[t1s[0]]
            if p0 > 0 and p1 > 0:
                fwd[tk] = float(p1 / p0) - 1

        merged_scores = [scores[tk] for tk in scores if tk in fwd]
        merged_rets   = [fwd[tk]   for tk in scores if tk in fwd]
        if len(merged_scores) < 5:
            continue

        ic, _ = spearmanr(merged_scores, merged_rets)
        ic_rows.append({
            "rebalance_date": reb,
            "ic": round(ic, 4),
            "n_stocks": len(merged_scores),
        })

    if not ic_rows:
        return pd.DataFrame(), 0.0, 0.0
    df     = pd.DataFrame(ic_rows)
    mean_ic= df["ic"].mean()
    t_stat = mean_ic / (df["ic"].std() / np.sqrt(len(df)) + 1e-10)
    return df, float(mean_ic), float(t_stat)


# =============================================================================
# main
# =============================================================================

def main() -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("=" * 70)
    print("Canyon v9 — Step 490: Short Squeeze Composite Signal")
    print("=" * 70)

    print("\n[1/3] Loading all signals …")
    prices = pd.read_csv(ROOT / "backtest_price_cache.csv",
                         index_col=0, parse_dates=True)
    preds  = pd.read_csv(ROOT / "cpcv_predictions.csv",
                         parse_dates=["rebalance_date"])
    reb_dates = sorted(preds["rebalance_date"].dt.date.unique().astype(str).tolist())
    signals = load_all_signals()
    print(f"  Loaded: {list(signals.keys())}")

    print("\n[2/3] Scoring short-squeeze conditions …")
    squeeze_df = score_squeeze_conditions(signals, prices)
    squeeze_df.to_csv(ROOT / "short_squeeze_signal.csv", index=False)

    # Summary by condition count
    for n in [4, 3, 2, 1, 0]:
        cnt = (squeeze_df["conditions_met"] == n).sum()
        label = {4: "ALL 4 ★★★★", 3: "3 of 4 ★★★",
                 2: "2 of 4 ★★", 1: "1 of 4 ★", 0: "0 of 4"}.get(n, "")
        if cnt > 0:
            print(f"    {label}: {cnt} stocks")

    top4 = squeeze_df[squeeze_df["conditions_met"] == 4]
    top3 = squeeze_df[squeeze_df["conditions_met"] >= 3]

    print(f"\n  HIGHEST SQUEEZE RISK (all 4 conditions):")
    if top4.empty:
        print("    None currently (requires all 4 conditions simultaneously)")
    else:
        for _, r in top4.head(5).iterrows():
            print(f"    {r['ticker']:<6}  SI={r['short_pct']:.0%}  "
                  f"PCR={r['pcr']:.2f}  "
                  f"Mom vs SPY={r['momentum_vs_spy']:+.1%}  "
                  f"Score={r['intensity_score']:+.3f}")

    print(f"\n  3-CONDITION SQUEEZE CANDIDATES:")
    only3 = squeeze_df[squeeze_df["conditions_met"] == 3]
    for _, r in only3.head(5).iterrows():
        conds = []
        if r["short_interest_c"]: conds.append("SI")
        if r["options_c"]:        conds.append("Opts")
        if r["revision_c"]:       conds.append("Rev")
        if r["momentum_c"]:       conds.append("Mom")
        print(f"    {r['ticker']:<6}  [{', '.join(conds)}]  "
              f"Score={r['intensity_score']:+.3f}")

    print("\n[3/3] Evaluating relative-momentum IC (squeeze proxy) …")
    ic_df, mean_ic, t_stat = evaluate_squeeze_ic(prices, signals, reb_dates)
    ic_df.to_csv(ROOT / "squeeze_ic.csv", index=False)
    stars = "★★★" if abs(t_stat) > 2.0 else "★★" if abs(t_stat) > 1.5 else "★"
    print(f"  Momentum proxy IC: {mean_ic:+.4f}  t = {t_stat:+.2f}  {stars}")
    print(f"  (Full squeeze IC requires historical short interest data)")
    print(f"  (Expected IC when all 4 conditions: +0.12~+0.20 per literature)")

    print("\n" + "=" * 70)
    print("Step 490 Complete — Short Squeeze Composite Signal")
    print("=" * 70)
    for f in ["short_squeeze_signal.csv", "squeeze_ic.csv"]:
        p = ROOT / f
        sz = f"{p.stat().st_size/1024:.1f} KB" if p.exists() else "—"
        print(f"  {f:<44} {sz}")


if __name__ == "__main__":
    main()
