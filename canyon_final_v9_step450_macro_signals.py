#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canyon v9 — Step 450: Cross-Asset Macro Signals
================================================
The biggest blind spot of equity signals: not knowing what is happening in the macro environment.

The institutional-standard macro overlay uses three cross-asset signals:

  Signal 1 — Credit Spread
    Proxy: HYG (high-yield bond ETF) vs IEI (intermediate Treasury ETF) price ratio
    Logic: widening credit spread → rising corporate default risk → leading equity decline
         Credit markets perceive credit risk 2-4 weeks ahead of equities
    IC research: Gilchrist & Zakrajsek (2012) — credit spread is the strongest equity crisis predictor

  Signal 2 — Yield Curve Slope
    Proxy: 10Y Treasury yield minus 2Y Treasury yield (2s10s spread)
         Approximated using TNX (10Y yield) and IRX (3M T-Bill)
    Logic: yield curve inversion (short > long) = historically most reliable recession predictor
         Recession averages 15 months after inversion; equity drawdown begins 6-12 months after
    Note: 2022-2023 saw severe inversion; equity market did decline

  Signal 3 — Dollar Strength (Dollar Index)
    Proxy: UUP (dollar bullish ETF) or DXY-equivalent
    Logic: strong dollar → EM + commodities + multinationals with high overseas revenue under pressure
         Weak dollar → global liquidity expansion → risk asset rally
         Highest impact on S&P 500 companies with large international revenue (AAPL 57%, NVDA 70%)

Combined logic:
  Macro score = -0.4 × credit spread Z + 0.3 × yield curve Z + 0.3 × (-dollar Z)
             (negative sign = widening spreads and strong dollar are risk-off factors)

Signal overlay effect:
  High macro score (risk-friendly) → increase equity exposure
  Low macro score (risk-averse)    → reduce exposure, increase hedges

IC evaluation:
  Current-month macro signal → predict next-month market return (Spearman IC)
  Comparison: macro signal vs HMM regime signal incremental information

Output files:
  macro_signals_daily.csv          daily macro signal history
  macro_regime_monthly.csv         monthly macro regime states
  macro_ic_results.csv             IC evaluation results
  macro_composite.csv              current macro composite signal
  macro_report.md                  institutional-grade report

Usage:
  .venv/bin/python canyon_final_v9_step450_macro_signals.py
"""
from __future__ import annotations

import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

OOS_CUTOFF    = pd.Timestamp("2020-01-01")
ANN_FACTOR    = 252
LOOKBACK_DAYS = 252
REQUEST_DELAY = 0.25

# Macro proxies available via yfinance
MACRO_TICKERS = {
    "HYG":  "High-Yield Corporate Bond ETF (credit risk)",
    "IEI":  "3-7Y Treasury ETF (risk-free baseline)",
    "LQD":  "Investment-Grade Corporate Bond ETF",
    "TLT":  "20+ Year Treasury Bond ETF",
    "UUP":  "USD Bullish ETF (dollar index proxy)",
    "GLD":  "Gold ETF (risk-off / inflation)",
    "SPY":  "S&P 500 benchmark",
}
START_DATE = "2015-01-01"


# =============================================================================
# 1. Fetch macro proxy data
# =============================================================================

def fetch_macro_data() -> pd.DataFrame:
    cache = ROOT / "macro_price_cache.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        print(f"  Macro data: loaded from cache ({len(df)} rows)")
        return df

    print(f"  Downloading macro proxies: {list(MACRO_TICKERS.keys())} …")
    rows = {}
    for tk, desc in MACRO_TICKERS.items():
        try:
            obj = yf.Ticker(tk)
            hist = obj.history(start=START_DATE, auto_adjust=True)
            if not hist.empty:
                rows[tk] = hist["Close"]
                print(f"    {tk}: {len(hist)} rows — {desc}")
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            print(f"    {tk}: error — {e}")

    df = pd.DataFrame(rows).dropna(how="all")
    df.index = df.index.tz_localize(None)
    df.index.name = "date"
    df.to_csv(cache)
    return df


# =============================================================================
# 2. Compute macro signals
# =============================================================================

def compute_macro_signals(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Signal 1: Credit Spread = -(HYG/IEI rolling return spread)
      When HYG underperforms IEI → credit spreads widening → RISK OFF
    Signal 2: Duration Signal = TLT vs SPY relative momentum
      When TLT outperforms (flight to safety) → RISK OFF
    Signal 3: Dollar Signal = -UUP momentum
      When UUP is strong (USD up) → RISK OFF for global equities
    """
    sig = pd.DataFrame(index=macro.index)

    # --- Credit Spread Signal ---
    if "HYG" in macro.columns and "IEI" in macro.columns:
        hyg_ret = macro["HYG"].pct_change(21)  # 21-day rolling return
        iei_ret = macro["IEI"].pct_change(21)
        # Spread = HYG excess return over IEI
        # Negative = HYG underperforming = credit stress = RISK OFF
        sig["credit_spread"] = -(hyg_ret - iei_ret)

    # --- Yield Curve via Treasuries ---
    if "TLT" in macro.columns and "IEI" in macro.columns:
        # TLT = long-duration, IEI = medium-duration
        # TLT/IEI ratio as yield curve proxy
        # Rising TLT/IEI → long rates falling relative to medium → curve steepening/normalization → RISK ON
        tlt_ret = macro["TLT"].pct_change(63)  # 3-month
        iei_ret = macro["IEI"].pct_change(63)
        sig["yield_curve"] = tlt_ret - iei_ret  # positive = flattening = bearish

    # --- Dollar Signal ---
    if "UUP" in macro.columns:
        # USD strength = bad for equity (especially international earners)
        sig["dollar_strength"] = -macro["UUP"].pct_change(21)

    # --- Gold Signal (risk-off indicator) ---
    if "GLD" in macro.columns:
        # Gold rising = flight to safety = risk-off for equities (short-term contrarian)
        sig["gold_flow"] = -macro["GLD"].pct_change(21)

    # Remove NaN rows
    sig = sig.dropna(how="all")
    return sig


# =============================================================================
# 3. Macro composite score
# =============================================================================

def build_macro_composite(signals: pd.DataFrame) -> pd.Series:
    """
    Normalize each signal to z-score (rolling 252-day).
    Macro composite = weighted average of available z-scores.
    Weights: credit_spread 0.40, yield_curve 0.30, dollar -0.30
    (All signals: positive = RISK ON, negative = RISK OFF)
    """
    weights = {
        "credit_spread":   0.40,
        "yield_curve":    -0.30,   # rising long-short spread = flattening = risk-off
        "dollar_strength": 0.30,
        "gold_flow":       0.00,   # informational only, not in composite
    }

    z_signals = {}
    for col in signals.columns:
        if col not in weights:
            continue
        s = signals[col].dropna()
        roll_mu = s.rolling(LOOKBACK_DAYS).mean()
        roll_sd = s.rolling(LOOKBACK_DAYS).std()
        z_signals[col] = (s - roll_mu) / (roll_sd + 1e-10)

    z_df = pd.DataFrame(z_signals)
    composite = pd.Series(0.0, index=z_df.index)
    total_w   = 0.0
    for col, w in weights.items():
        if col in z_df.columns:
            composite += w * z_df[col].fillna(0)
            total_w   += abs(w)
    if total_w > 0:
        composite /= total_w

    return composite.rename("macro_composite")


# =============================================================================
# 4. IC evaluation: macro signals → SPY next-month return
# =============================================================================

def evaluate_macro_ic(
    signals: pd.DataFrame,
    composite: pd.Series,
    macro: pd.DataFrame,
    reb_dates: list[str],
) -> pd.DataFrame:
    """
    Evaluate: macro_composite at rebalance date → SPY return next month
    """
    spy = macro["SPY"].pct_change(21)  # 21-day forward SPY return

    oos_dates = [r for r in reb_dates if r >= str(OOS_CUTOFF.date())]
    rows = []

    for i, reb in enumerate(oos_dates[:-1]):
        next_reb = oos_dates[i + 1]
        reb_ts   = pd.Timestamp(reb)
        nxt_ts   = pd.Timestamp(next_reb)

        # Macro composite at rebalance date
        avail = composite[composite.index <= reb_ts]
        if avail.empty:
            continue
        mc_val = float(avail.iloc[-1])

        # SPY return for the next period
        spy_avail = spy[(spy.index >= reb_ts) & (spy.index <= nxt_ts)]
        if spy_avail.empty:
            continue
        spy_ret = float(spy_avail.iloc[-1])

        rows.append({
            "date":           reb,
            "macro_composite": round(mc_val, 4),
            "spy_ret_next":    round(spy_ret, 6),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).dropna()
    if len(df) < 10:
        return pd.DataFrame()

    ic, pval = spearmanr(df["macro_composite"], df["spy_ret_next"])
    df["period"] = "OOS"

    print(f"  Macro→SPY IC:  {ic:+.4f}  p={pval:.3f}")
    return df


# =============================================================================
# 5. Macro regime classification
# =============================================================================

def classify_macro_regime(composite: pd.Series) -> pd.DataFrame:
    """
    Classify monthly macro regime:
    RISK_ON:  composite > +0.5 std
    NEUTRAL:  -0.5 < composite < +0.5
    RISK_OFF: composite < -0.5 std
    """
    monthly = composite.resample("ME").last()
    rows = []
    for date, val in monthly.items():
        if val > 0.5:
            regime = "RISK_ON"
        elif val < -0.5:
            regime = "RISK_OFF"
        else:
            regime = "NEUTRAL"
        rows.append({"date": date, "macro_score": round(val, 4), "regime": regime})

    return pd.DataFrame(rows)


# =============================================================================
# 6. HMM + Macro combined signal
# =============================================================================

def combine_with_hmm(macro_regime: pd.DataFrame) -> pd.DataFrame:
    """
    Join macro regime with HMM regime.
    When both signal RISK_OFF / Bear → maximum defensive posture.
    When both signal RISK_ON / Bull → maximum offensive posture.
    """
    hmm_file = ROOT / "hmm_regime_monthly.csv"
    if not hmm_file.exists():
        return macro_regime

    hmm = pd.read_csv(hmm_file, parse_dates=["rebalance_date"])
    hmm["date"] = (hmm["rebalance_date"] +
                   pd.offsets.MonthEnd(0)).dt.normalize()

    merged = macro_regime.merge(
        hmm[["date", "regime_label", "exposure"]],
        on="date", how="left"
    )
    merged = merged.rename(columns={"regime_label": "hmm_regime"})

    # Combined signal: macro + HMM
    def _combined_exposure(row):
        m = row.get("regime",     "NEUTRAL")
        h = row.get("hmm_regime", "Bull")
        if m == "RISK_OFF" and h == "Bear":
            return 0.25   # maximum defensive
        elif m == "RISK_OFF" or h == "Bear":
            return 0.60   # one signal risk-off
        elif m == "RISK_ON" and h == "Bull":
            return 1.00   # maximum offensive
        else:
            return 0.80   # neutral

    merged["combined_exposure"] = merged.apply(_combined_exposure, axis=1)
    return merged


# =============================================================================
# 7. Report
# =============================================================================

def generate_report(
    signals: pd.DataFrame,
    composite: pd.Series,
    regime_df: pd.DataFrame,
    ic_df: pd.DataFrame,
    ts: str,
) -> str:
    current_macro = float(composite.iloc[-1]) if len(composite) > 0 else 0.0
    current_regime = "RISK_ON" if current_macro > 0.5 else \
                     "RISK_OFF" if current_macro < -0.5 else "NEUTRAL"

    # IC computation
    if not ic_df.empty:
        ic_val, _ = spearmanr(ic_df["macro_composite"], ic_df["spy_ret_next"])
    else:
        ic_val = 0.0

    lines = [
        "# Canyon v9 — Step 450: Cross-Asset Macro Signals",
        f"Generated: {ts}",
        "",
        "## Current Macro Regime",
        "",
        f"  Macro Composite Score:  {current_macro:+.3f}",
        f"  Macro Regime:           {current_regime}",
        "",
    ]

    if not regime_df.empty and "hmm_regime" in regime_df.columns:
        latest = regime_df.iloc[-1]
        combined = float(latest.get("combined_exposure", 0.8))
        hmm_reg = latest.get("hmm_regime", "—")
        lines += [
            f"  HMM Regime:             {hmm_reg}",
            f"  Combined Exposure:      {combined:.0%}",
            "",
            "  Interpretation:",
        ]
        if combined <= 0.25:
            lines += ["    → MAXIMUM DEFENSIVE: both macro and HMM signal risk-off",
                      "       Reduce gross exposure by 75%"]
        elif combined >= 1.00:
            lines += ["    → MAXIMUM OFFENSIVE: both macro and HMM signal risk-on",
                      "       Deploy full gross exposure"]
        else:
            lines += [f"    → PARTIAL EXPOSURE ({combined:.0%}): mixed signals",
                      "       Scale gross exposure accordingly"]

    lines += [
        "",
        "## Macro Signal Stack",
        "",
        "| Signal | Current Z-Score | Interpretation |",
        "|--------|:---------------:|----------------|",
    ]

    if not signals.empty:
        latest_sig = signals.iloc[-1]
        descs = {
            "credit_spread":   ("Credit Spread (HYG-IEI)",
                                "positive = credit markets healthy = RISK ON"),
            "yield_curve":     ("Yield Curve (TLT-IEI slope)",
                                "negative = inverted = RISK OFF warning"),
            "dollar_strength": ("USD Strength (UUP)",
                                "positive = USD weak = global liquidity ample"),
            "gold_flow":       ("Gold Flow (GLD)",
                                "positive = gold falling = risk appetite up"),
        }
        # Re-compute z-scores for latest
        for col in signals.columns:
            if col not in descs:
                continue
            s = signals[col].dropna()
            roll_mu = s.rolling(LOOKBACK_DAYS).mean()
            roll_sd = s.rolling(LOOKBACK_DAYS).std()
            z = ((s - roll_mu) / (roll_sd + 1e-10)).iloc[-1]
            label, interp = descs[col]
            flag = "↑ RISK ON" if z > 0.5 else "↓ RISK OFF" if z < -0.5 else "→ NEUTRAL"
            lines.append(f"| {label} | {z:+.2f}σ | {flag} |")

    lines += [
        "",
        f"## IC vs SPY Next-Month Return (OOS)",
        "",
        f"  Macro Composite IC: {ic_val:+.4f}",
    ]

    if abs(ic_val) > 0.1:
        lines += [
            "  Interpretation: Strong macro signal — tactical allocation adds value",
        ]
    else:
        lines += [
            "  Interpretation: Modest IC — macro signal is supplementary,",
            "  not a standalone timing tool. Value is in COMBINING with HMM.",
        ]

    lines += [
        "",
        "## Historical Regime Distribution",
        "",
    ]

    if not regime_df.empty:
        oos_regimes = regime_df[regime_df["date"] >= OOS_CUTOFF]
        if not oos_regimes.empty and "regime" in oos_regimes.columns:
            for reg in ["RISK_ON", "NEUTRAL", "RISK_OFF"]:
                pct = (oos_regimes["regime"] == reg).mean()
                if "hmm_regime" in oos_regimes.columns:
                    pass  # skip average
                lines.append(f"  {reg:<12} {pct:.0%} of OOS months")

    lines += [
        "",
        "## Cross-Asset Signal: Institutional Framework",
        "",
        "Top quant funds use credit markets as the leading indicator:",
        "  - Credit spreads widen BEFORE equity selloffs (2-4 week lead)",
        "  - Yield curve inversion precedes recessions by 6-18 months",
        "  - USD strength leads EM/commodity equity weakness by 4-8 weeks",
        "",
        "For Canyon v9, the macro overlay provides:",
        "  1. Regime confirmation for HMM bear signals",
        "  2. Pre-crash warning (credit spread spike + yield curve flat)",
        "  3. Tactical gross exposure adjustment (75%–125% of target)",
    ]

    return "\n".join(lines)


# =============================================================================
# main
# =============================================================================

def main() -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("=" * 70)
    print("Canyon v9 — Step 450: Cross-Asset Macro Signals")
    print("=" * 70)

    print("\n[1/5] Loading base data …")
    preds = pd.read_csv(ROOT / "cpcv_predictions.csv",
                        parse_dates=["rebalance_date"])
    reb_dates = sorted(
        preds["rebalance_date"].dt.date.unique().astype(str).tolist()
    )
    print(f"  {len(reb_dates)} rebalance dates")

    print("\n[2/5] Fetching macro proxy data …")
    macro = fetch_macro_data()
    print(f"  Columns: {list(macro.columns)}")
    print(f"  Date range: {macro.index.min().date()} – {macro.index.max().date()}")

    print("\n[3/5] Computing macro signals …")
    signals = compute_macro_signals(macro)
    print(f"  Signals computed: {list(signals.columns)}")
    composite = build_macro_composite(signals)

    # Current values
    print(f"\n  Current macro composite: {float(composite.iloc[-1]):+.3f}")
    current_regime = ("RISK_ON"  if float(composite.iloc[-1]) > 0.5 else
                      "RISK_OFF" if float(composite.iloc[-1]) < -0.5 else "NEUTRAL")
    print(f"  Current macro regime:    {current_regime}")

    signals.to_csv(ROOT / "macro_signals_daily.csv")
    pd.DataFrame({"date": composite.index, "macro_composite": composite.values}) \
      .to_csv(ROOT / "macro_composite_daily.csv", index=False)

    print("\n[4/5] Evaluating IC …")
    ic_df = evaluate_macro_ic(signals, composite, macro, reb_dates)
    if not ic_df.empty:
        ic_df.to_csv(ROOT / "macro_ic_results.csv", index=False)

    print("\n[5/5] Classifying macro regimes + HMM join …")
    regime_df = classify_macro_regime(composite)
    combined  = combine_with_hmm(regime_df)
    combined.to_csv(ROOT / "macro_regime_monthly.csv", index=False)

    # Show OOS distribution
    oos_reg = combined[combined["date"] >= OOS_CUTOFF]
    if not oos_reg.empty:
        print(f"\n  OOS regime distribution:")
        for reg in ["RISK_ON", "NEUTRAL", "RISK_OFF"]:
            if "regime" in oos_reg.columns:
                pct  = (oos_reg["regime"] == reg).mean()
                print(f"    {reg:<12} {pct:.0%} of OOS months")
        if "combined_exposure" in oos_reg.columns:
            print(f"\n  Average combined exposure: "
                  f"{oos_reg['combined_exposure'].mean():.0%}")

    # Key macro events
    if not combined.empty and "regime" in combined.columns:
        risk_off = combined[combined["regime"] == "RISK_OFF"]
        if not risk_off.empty:
            print(f"\n  RISK_OFF months (macro says reduce exposure):")
            for _, r in risk_off.head(8).iterrows():
                hmm = r.get("hmm_regime", "—")
                exp = r.get("combined_exposure", "—")
                score = r.get("macro_score", 0)
                print(f"    {str(r['date'])[:7]}  score={score:+.2f}  "
                      f"HMM={hmm}  combined_exp={exp if isinstance(exp, str) else f'{exp:.0%}'}")

    report = generate_report(signals, composite, combined, ic_df, ts)
    (ROOT / "macro_report.md").write_text(report)

    # Updated signal stack
    if not ic_df.empty:
        ic_val, _ = spearmanr(ic_df["macro_composite"], ic_df["spy_ret_next"])
    else:
        ic_val = 0.0

    print(f"\n  ── Updated Signal Stack ──")
    print(f"  PEAD:            IC = +0.229  t = +7.32  ★★★  fundamental")
    print(f"  VIX:             IC = +0.223  t = +12.0  ★★★  market timing")
    print(f"  Macro (cross-asset): IC = {ic_val:+.3f}  market-level")
    print(f"  Quality (ROE):   IC = +0.048  t = +1.91  ★★   factor")
    print(f"  Analyst Revision:IC = +0.038  t = +1.66  ★★   fundamental")
    print(f"  Momentum:        IC = +0.042  t = +1.16  ★    factor")

    print("\n" + "=" * 70)
    print("Step 450 Complete — Cross-Asset Macro Signals")
    print("=" * 70)
    for f in ["macro_signals_daily.csv", "macro_composite_daily.csv",
              "macro_regime_monthly.csv", "macro_ic_results.csv",
              "macro_report.md"]:
        p = ROOT / f
        sz = f"{p.stat().st_size/1024:.1f} KB" if p.exists() else "—"
        print(f"  {f:<44} {sz}")


if __name__ == "__main__":
    main()
