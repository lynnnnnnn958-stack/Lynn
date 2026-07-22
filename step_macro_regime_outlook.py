#!/usr/bin/env python3
"""
Canyon — Macro Regime Outlook
==============================
Forward-looking bear probability for the next 4 weeks,
using macro leading indicators that tend to move BEFORE price.

Why this matters:
  The HMM detects the CURRENT price regime (reactive, 1-3 week lag).
  Macro leading indicators signal what is LIKELY COMING (4-12 weeks ahead).
  When HMM and macro diverge, that tells a story: e.g., HMM=BEAR + macro=BENIGN
  suggests a short-term price dip rather than a genuine regime shift.

Indicators (all free data — FRED + yfinance):
  1. Yield curve (T10Y2Y)       — inversion precedes recessions by 12-18 months
  2. Credit spreads (HYG OAS)   — widen before market stress by 4-8 weeks
  3. VIX term structure          — inversion (VIX > VIX3M) = near-term fear spike
  4. SPY trend vs key MAs        — market structural position
  5. Labor market (UNRATE)       — unemployment turning up = late-cycle warning

Output: macro_regime_outlook.json
"""

from __future__ import annotations
import json, warnings
from datetime import datetime, date
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
OUT  = ROOT / "macro_regime_outlook.json"
UA   = {"User-Agent": "canyon-quant research lynnnnnnn958@gmail.com"}

# ── helpers ───────────────────────────────────────────────────────────────────

def fred(series: str, days: int = 150) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    try:
        r   = requests.get(url, headers=UA, timeout=20)
        df  = pd.read_csv(StringIO(r.text), parse_dates=["observation_date"])
        df  = df.set_index("observation_date")
        df[series] = pd.to_numeric(df[series], errors="coerce")
        return df[series].dropna().iloc[-days:]
    except Exception as e:
        print(f"  FRED {series} error: {e}")
        return pd.Series(dtype=float)


def yf_close(ticker: str, period: str = "300d") -> np.ndarray:
    try:
        import yfinance as yf
        h = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        return h["Close"].values.flatten().astype(float)
    except Exception as e:
        print(f"  yfinance {ticker} error: {e}")
        return np.array([])


def score_indicator(value, thresholds: list[tuple[float, float]]) -> float:
    """Map a value to a bear score [0..2] using (threshold, score) breakpoints."""
    for thresh, sc in sorted(thresholds, key=lambda x: x[0]):
        if value <= thresh:
            return sc
    return thresholds[-1][1]


# ── individual signals ────────────────────────────────────────────────────────

def signal_yield_curve() -> dict:
    """
    T10Y2Y: 10y minus 2y Treasury spread.
    Inversion (< 0) is the single most reliable recession leading indicator.
    Lead time: 6-18 months after inversion first appears.
    """
    s = fred("T10Y2Y")
    if s.empty:
        return {"ok": False, "name": "Yield Curve (10Y-2Y)", "value": None}

    now   = float(s.iloc[-1])
    ago4w = float(s.iloc[-21]) if len(s) >= 21 else now
    trend = now - ago4w  # positive = steepening (bullish), negative = flattening (bearish)

    # Bear score [0..2]: inversion is most dangerous
    level_score = score_indicator(now, [
        (-0.75, 2.0),   # deep inversion → very bearish
        (-0.25, 1.5),   # moderate inversion
        ( 0.00, 1.0),   # flat or just-crossed zero
        ( 0.50, 0.3),   # mild positive
        ( 9.99, 0.0),   # steep positive → benign
    ])
    # Trend penalty: rapidly flattening even if still positive
    trend_score = 0.3 if trend < -0.10 else 0.0

    return {
        "ok":          True,
        "name":        "Yield Curve (10Y-2Y)",
        "value":       round(now, 3),
        "unit":        "%",
        "display":     f"{now:+.2f}%",
        "trend":       round(trend, 3),
        "trend_label": "Flattening" if trend < -0.05 else ("Steepening" if trend > 0.05 else "Stable"),
        "bear_score":  round(min(level_score + trend_score, 2.0), 2),
        "max_score":   2.0,
        "interpretation": (
            "Deep inversion — historically precedes recessions by 6-18 months" if now < -0.5 else
            "Inverted — watch closely; inversion averaging 12 months before recessions" if now < 0 else
            "Flat near zero — transition zone; worth monitoring" if now < 0.3 else
            "Positive — normal slope, no recession signal"
        ),
        "watch_for": "Sustained inversion below -0.5% for 3+ months",
        "as_of": str(s.index[-1].date()),
    }


def signal_credit_spreads() -> dict:
    """
    HYG OAS (ICE BofA US High Yield Option-Adjusted Spread).
    Widening spreads = credit stress = leading market indicator by 4-8 weeks.
    """
    s = fred("BAMLH0A0HYM2")
    if s.empty:
        return {"ok": False, "name": "Credit Spreads (HY OAS)", "value": None}

    now   = float(s.iloc[-1])
    ago4w = float(s.iloc[-21]) if len(s) >= 21 else now
    ago8w = float(s.iloc[-42]) if len(s) >= 42 else ago4w
    change_4w = now - ago4w
    change_8w = now - ago8w

    level_score = score_indicator(now, [
        (2.5,  0.0),    # tight → healthy
        (3.5,  0.3),    # normal range
        (4.5,  1.0),    # elevated
        (6.0,  1.5),    # stressed
        (9.99, 2.0),    # crisis level
    ])
    # Momentum: rapidly widening is early warning even if levels still low
    mom_score = (
        0.8 if change_4w > 1.0 else   # sharp widening
        0.4 if change_4w > 0.4 else   # moderate widening
        0.0
    )

    return {
        "ok":          True,
        "name":        "Credit Spreads (HY OAS)",
        "value":       round(now, 2),
        "unit":        "%",
        "display":     f"{now:.2f}%",
        "trend":       round(change_4w, 3),
        "trend_label": (
            f"Widening fast (+{change_4w:.2f}% / 4w)" if change_4w > 0.4 else
            f"Widening (+{change_4w:.2f}% / 4w)"      if change_4w > 0.1 else
            f"Tightening ({change_4w:.2f}% / 4w)"     if change_4w < -0.1 else
            "Stable"
        ),
        "bear_score":  round(min(level_score + mom_score, 2.0), 2),
        "max_score":   2.0,
        "interpretation": (
            "Crisis-level spreads — significant credit stress" if now > 7 else
            "Elevated — market pricing meaningful default risk" if now > 4.5 else
            "Slightly elevated but manageable" if now > 3.5 else
            "Tight spreads — credit market healthy, risk appetite strong"
        ),
        "watch_for": "Spread widening above 4.5% or +1% move in 4 weeks",
        "as_of": str(s.index[-1].date()),
    }


def signal_vix_term_structure() -> dict:
    """
    VIX3M / VIX ratio: when ratio < 1 (backwardation), near-term fear > long-term fear.
    Inversion = market pricing in immediate risk = bear signal.
    """
    vix_arr  = yf_close("^VIX",  "60d")
    vix3m_arr= yf_close("^VIX3M","60d")

    if len(vix_arr) == 0 or len(vix3m_arr) == 0:
        return {"ok": False, "name": "VIX Term Structure", "value": None}

    vix   = float(vix_arr[-1])
    vix3m = float(vix3m_arr[-1])
    ratio = vix3m / vix if vix > 0 else 1.0

    # VIX level score
    vix_score = score_indicator(vix, [
        (15,  0.0),
        (20,  0.3),
        (25,  0.8),
        (30,  1.3),
        (99,  2.0),
    ])
    # Term structure score (ratio < 1 = inverted = bearish)
    ts_score = score_indicator(ratio, [
        (0.90, 1.0),   # deep backwardation
        (0.98, 0.5),   # mild backwardation
        (1.05, 0.0),   # normal
        (9.99, 0.0),   # very calm
    ])

    bear_score = min(vix_score * 0.5 + ts_score, 2.0)

    return {
        "ok":          True,
        "name":        "VIX Term Structure",
        "value":       round(ratio, 3),
        "unit":        "ratio",
        "display":     f"{ratio:.2f}x  (VIX {vix:.1f} / VIX3M {vix3m:.1f})",
        "trend":       None,
        "trend_label": (
            "Backwardation — near-term fear elevated" if ratio < 1.0 else
            "Near flat — slight caution" if ratio < 1.05 else
            "Normal contango — calm market"
        ),
        "bear_score":  round(bear_score, 2),
        "max_score":   2.0,
        "interpretation": (
            f"VIX {vix:.0f} backwardation ({ratio:.2f}x) — near-term stress; market pricing immediate risk" if ratio < 1.0 else
            f"VIX {vix:.0f} normal term structure — no acute fear signal" if vix < 20 else
            f"VIX {vix:.0f} elevated but contango intact — cautious, not crisis"
        ),
        "watch_for": "VIX3M/VIX ratio dropping below 1.0 or VIX spike above 25",
    }


def signal_spy_trend() -> dict:
    """
    SPY position relative to 50d and 200d moving averages.
    Structural bear markets almost always involve crossing below 200d MA.
    """
    arr = yf_close("SPY", "300d")
    if len(arr) < 200:
        return {"ok": False, "name": "SPY Trend", "value": None}

    spy    = float(arr[-1])
    ma50   = float(np.nanmean(arr[-50:]))
    ma200  = float(np.nanmean(arr[-200:]))
    vs200  = spy / ma200 - 1
    vs50   = spy / ma50  - 1
    # Momentum: 20d return and 5d return
    ret20d = float(arr[-1] / arr[-21] - 1) if len(arr) >= 21 else 0.0
    ret5d  = float(arr[-1] / arr[-6]  - 1) if len(arr) >= 6  else 0.0

    # Structural position score
    struct_score = score_indicator(vs200, [
        (-0.10, 2.0),   # deeply below 200MA = bear
        (-0.03, 1.5),
        ( 0.00, 0.8),   # just below 200MA
        ( 0.05, 0.3),   # mildly above
        ( 9.99, 0.0),
    ])
    # 50MA cross adds urgency
    fifty_score = 0.3 if vs50 < 0 else 0.0
    # Strong recent momentum can suppress bear score (showing underlying strength)
    momentum_adj = -0.3 if ret20d > 0.05 else 0.0

    bear_score = min(max(struct_score + fifty_score + momentum_adj, 0), 2.0)

    return {
        "ok":          True,
        "name":        "SPY Trend vs Moving Averages",
        "value":       round(vs200, 4),
        "unit":        "vs 200MA",
        "display":     f"{vs200:+.1%} vs 200MA  ({vs50:+.1%} vs 50MA)",
        "trend":       round(ret20d, 4),
        "trend_label": (
            f"Strong recovery: {ret20d:+.1%} past 20d" if ret20d > 0.03 else
            f"Recent decline: {ret20d:+.1%} past 20d"  if ret20d < -0.03 else
            "Sideways"
        ),
        "bear_score":  round(bear_score, 2),
        "max_score":   2.0,
        "interpretation": (
            "Below 200d MA — structural bear territory; most drawdowns become recoveries from below this line" if vs200 < 0 else
            f"Above 200d MA (+{vs200:.1%}) — structural bull intact"
        ),
        "watch_for": "Break below 200d MA on increasing volume",
    }


def signal_labor_market() -> dict:
    """
    Unemployment rate (UNRATE).
    Rising unemployment is a lagging recession indicator, but a confirmed upturn
    (Sahm Rule: +0.5% from 12-month low) has 100% historical accuracy.
    """
    s = fred("UNRATE")
    if s.empty:
        return {"ok": False, "name": "Labor Market (UNRATE)", "value": None}

    now     = float(s.iloc[-1])
    ago3m   = float(s.iloc[-4])  if len(s) >= 4  else now
    ago12m  = float(s.iloc[-13]) if len(s) >= 13 else now
    low12m  = float(s.iloc[-13:].min()) if len(s) >= 13 else now
    change  = now - ago3m
    sahm    = now - low12m  # Sahm Rule: > 0.5 = recession signal

    level_score = score_indicator(now, [
        (4.0, 0.0),
        (4.5, 0.3),
        (5.0, 0.8),
        (6.0, 1.5),
        (9.9, 2.0),
    ])
    sahm_score = score_indicator(sahm, [
        (0.1, 0.0),
        (0.3, 0.5),
        (0.5, 1.5),   # Sahm threshold — historically reliable
        (9.9, 2.0),
    ])

    bear_score = min(level_score * 0.4 + sahm_score, 2.0)

    return {
        "ok":          True,
        "name":        "Labor Market (UNRATE)",
        "value":       round(now, 2),
        "unit":        "%",
        "display":     f"{now:.1f}%  (Sahm: +{sahm:.2f}%)",
        "trend":       round(change, 2),
        "trend_label": (
            f"Rising {change:+.1f}% in 3m — deteriorating" if change > 0.2 else
            f"Falling {change:+.1f}% in 3m — strengthening" if change < -0.1 else
            "Stable"
        ),
        "bear_score":  round(bear_score, 2),
        "max_score":   2.0,
        "interpretation": (
            f"Sahm Rule triggered (+{sahm:.2f}% from 12m low) — historically 100% recession signal" if sahm >= 0.5 else
            f"Approaching Sahm threshold (+{sahm:.2f}% from 12m low) — worth monitoring" if sahm >= 0.3 else
            f"Labor market healthy — Sahm indicator +{sahm:.2f}% from low"
        ),
        "watch_for": "Sahm Rule trigger: UNRATE rising 0.5% above 12-month low",
        "sahm": round(sahm, 3),
        "as_of": str(s.index[-1].date()),
    }


# ── composite ─────────────────────────────────────────────────────────────────

WEIGHTS = {
    "yield_curve":         0.30,   # strongest long-lead recession signal
    "credit_spreads":      0.25,   # 4-8w lead, very reliable
    "vix_term_structure":  0.20,   # near-term, high frequency
    "spy_trend":           0.15,   # current structural position
    "labor_market":        0.10,   # lagging but high conviction
}


def compute_outlook(signals: dict) -> dict:
    total_weight = 0.0
    weighted_bear = 0.0
    for key, w in WEIGHTS.items():
        sig = signals.get(key, {})
        if not sig.get("ok"):
            continue
        normalized = sig["bear_score"] / sig["max_score"]  # 0-1
        weighted_bear += normalized * w
        total_weight  += w

    if total_weight == 0:
        return {"bear_prob": 0.5, "label": "Unknown", "color": "#888"}

    bear_prob = weighted_bear / total_weight  # 0-1
    # Calibrate: raw score tends to cluster; stretch to usable range
    bear_pct = round(bear_prob * 100, 1)

    if bear_pct < 20:
        label, color = "Low Risk", "#1B6F4A"
    elif bear_pct < 40:
        label, color = "Moderate", "#6BCCA0"
    elif bear_pct < 60:
        label, color = "Elevated", "#B8943F"
    elif bear_pct < 75:
        label, color = "High Risk", "#E07020"
    else:
        label, color = "Crisis", "#B83232"

    return {
        "bear_prob":    bear_pct,
        "label":        label,
        "color":        color,
        "total_weight": round(total_weight, 3),
    }


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"Macro Regime Outlook — {ts}")
    print("=" * 60)

    signals = {
        "yield_curve":        signal_yield_curve(),
        "credit_spreads":     signal_credit_spreads(),
        "vix_term_structure": signal_vix_term_structure(),
        "spy_trend":          signal_spy_trend(),
        "labor_market":       signal_labor_market(),
    }

    for key, sig in signals.items():
        if sig.get("ok"):
            print(f"  {sig['name']:<36} {sig['display']:<30} bear_score={sig['bear_score']:.1f}/2.0")
            print(f"    → {sig['interpretation']}")
        else:
            print(f"  {sig.get('name','?'):<36} [data unavailable]")

    composite = compute_outlook(signals)
    print()
    print(f"  Composite Bear Probability (4-week outlook): {composite['bear_prob']:.1f}%  [{composite['label']}]")
    print()

    result = {
        "as_of":     ts,
        "signals":   signals,
        "composite": composite,
    }

    OUT.write_text(json.dumps(result, indent=2, default=str))
    print(f"  Saved: {OUT.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
