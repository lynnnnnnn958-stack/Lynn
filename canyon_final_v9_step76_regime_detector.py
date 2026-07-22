#!/usr/bin/env python3
"""
Canyon v9 — Step 76: Market Regime Detector
============================================
4-indicator composite:  SPY MA-cross, VIX level, 20-day momentum, RSI(14)
3-state output:         BULL / BEAR / SIDEWAYS
10-day smoothing:       majority-vote rolling window to avoid whipsaws
Looks back to 2000-01-01 (SPY + VIX available from ~1993/2004)

Outputs
-------
regime_history.csv       — daily regime labels + all raw indicators
regime_transitions.csv   — every regime change event
regime_current.json      — today's regime (read by step77)
regime_report.md         — summary statistics

Usage
-----
  python3 canyon_final_v9_step76_regime_detector.py           # full history
  python3 canyon_final_v9_step76_regime_detector.py --today   # append today only
  python3 canyon_final_v9_step76_regime_detector.py --show    # print current regime
"""

import argparse
import json
import warnings
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
OUT_HISTORY     = ROOT / "regime_history.csv"
OUT_TRANSITIONS = ROOT / "regime_transitions.csv"
OUT_CURRENT     = ROOT / "regime_current.json"
OUT_REPORT      = ROOT / "regime_report.md"
CACHE_FILE      = ROOT / "regime_price_cache.csv"

SMOOTH_WINDOW   = 10   # majority-vote days
START_DATE      = "2000-01-01"


# ─────────────────────────────────────────────────────────────
# 1.  DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────

def download_data(start: str = START_DATE, use_cache: bool = True) -> pd.DataFrame:
    """Download SPY + VIX daily OHLCV; return merged DataFrame."""
    cache_ok = False
    if use_cache and CACHE_FILE.exists():
        try:
            cached = pd.read_csv(CACHE_FILE, index_col="Date", parse_dates=True)
            last_date = cached.index.max().date()
            today = date.today()
            if last_date >= today or (today - last_date).days <= 1:
                cache_ok = True
                df = cached
        except Exception:
            pass

    if not cache_ok:
        print("  Downloading SPY + VIX from yfinance …")
        spy = yf.download("SPY", start=start, auto_adjust=True, progress=False)
        vix = yf.download("^VIX", start=start, auto_adjust=True, progress=False)

        # Flatten MultiIndex if present
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)

        spy_close = spy["Close"].rename("spy_close")
        vix_close = vix["Close"].rename("vix")

        df = pd.concat([spy_close, vix_close], axis=1).dropna(subset=["spy_close"])
        df.index.name = "Date"
        df.to_csv(CACHE_FILE)

    return df


# ─────────────────────────────────────────────────────────────
# 2.  INDICATOR CALCULATION
# ─────────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicator columns to df."""
    df = df.copy()
    close = df["spy_close"]

    # Moving averages
    df["ma50"]  = close.rolling(50).mean()
    df["ma200"] = close.rolling(200).mean()

    # 20-day return
    df["ret_20d"] = close.pct_change(20)

    # RSI 14
    df["rsi14"] = rsi(close, 14)

    # Fill VIX gaps with forward-fill (weekends etc.)
    df["vix"] = df["vix"].ffill()

    return df.dropna(subset=["ma50", "ma200", "ret_20d", "rsi14"])


# ─────────────────────────────────────────────────────────────
# 3.  SCORING & CLASSIFICATION
# ─────────────────────────────────────────────────────────────

def score_row(row: pd.Series) -> int:
    """
    Each indicator contributes -1, 0, or +1.
    Total range: -4 … +4
    """
    s = 0

    # 1. MA cross
    if row["ma50"] > row["ma200"]:
        s += 1
    else:
        s -= 1

    # 2. VIX
    if row["vix"] < 20:
        s += 1
    elif row["vix"] > 28:
        s -= 1
    # between 20-28 → 0

    # 3. 20-day momentum
    if row["ret_20d"] > 0.01:
        s += 1
    elif row["ret_20d"] < -0.03:
        s -= 1

    # 4. RSI
    if row["rsi14"] > 55:
        s += 1
    elif row["rsi14"] < 45:
        s -= 1

    return s


def raw_regime(score: int) -> str:
    if score >= 2:
        return "BULL"
    elif score <= -2:
        return "BEAR"
    else:
        return "SIDEWAYS"


def smooth_regime(raw_series: pd.Series, window: int = SMOOTH_WINDOW) -> pd.Series:
    """
    Rolling majority vote over `window` days.
    Encode strings → ints for rolling, decode back after.
    Tie-break order: SIDEWAYS (1) > BULL (2) > BEAR (0) — most conservative.
    """
    from collections import Counter

    ENCODE = {"BEAR": 0, "SIDEWAYS": 1, "BULL": 2}
    DECODE = {0: "BEAR", 1: "SIDEWAYS", 2: "BULL"}

    encoded = raw_series.map(ENCODE).astype(float)

    def majority_int(arr):
        counts = Counter(int(x) for x in arr if not np.isnan(x))
        if not counts:
            return 1  # default SIDEWAYS
        return counts.most_common(1)[0][0]

    smoothed = encoded.rolling(window, min_periods=1).apply(majority_int, raw=True)
    return smoothed.astype(int).map(DECODE)


def classify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["score"]      = df.apply(score_row, axis=1)
    df["raw_regime"] = df["score"].apply(raw_regime)
    df["regime"]     = smooth_regime(df["raw_regime"])
    return df


# ─────────────────────────────────────────────────────────────
# 4.  TRANSITIONS
# ─────────────────────────────────────────────────────────────

def compute_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of regime change events."""
    rows = []
    prev = None
    for dt, row in df.iterrows():
        reg = row["regime"]
        if prev is not None and reg != prev:
            rows.append({
                "date":       dt.date(),
                "from_regime": prev,
                "to_regime":   reg,
                "spy_price":   round(row["spy_close"], 2),
                "vix":         round(row["vix"], 1),
                "score":       int(row["score"]),
            })
        prev = reg
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 5.  STATS & REPORT
# ─────────────────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame) -> dict:
    """Compute per-regime forward 21-day return statistics."""
    df2 = df.copy()
    df2["fwd_21d"] = df2["spy_close"].pct_change(21).shift(-21)
    stats = {}
    for reg in ["BULL", "BEAR", "SIDEWAYS"]:
        sub = df2[df2["regime"] == reg]["fwd_21d"].dropna()
        if len(sub) > 0:
            stats[reg] = {
                "n_days":   len(df2[df2["regime"] == reg]),
                "pct_time": round(len(df2[df2["regime"] == reg]) / len(df2) * 100, 1),
                "fwd_mean": round(sub.mean() * 100, 2),
                "fwd_med":  round(sub.median() * 100, 2),
                "fwd_std":  round(sub.std() * 100, 2),
                "hit_rate": round((sub > 0).mean() * 100, 1),
            }
        else:
            stats[reg] = {"n_days": 0, "pct_time": 0,
                          "fwd_mean": 0, "fwd_med": 0,
                          "fwd_std": 0, "hit_rate": 0}
    return stats


def write_report(df: pd.DataFrame, transitions: pd.DataFrame, stats: dict,
                 current: dict) -> None:
    lines = [
        "# Canyon v9 — Market Regime History Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Current Regime",
        f"**{current['regime']}** (score={current['score']}) as of {current['date']}",
        f"- SPY 50MA={current['ma50']:.2f}  200MA={current['ma200']:.2f}",
        f"- VIX={current['vix']:.1f}  RSI={current['rsi']:.1f}  20d-ret={current['ret_20d']*100:.1f}%",
        "",
        "## Regime Distribution (since 2000)",
    ]

    for reg in ["BULL", "BEAR", "SIDEWAYS"]:
        s = stats[reg]
        lines.append(
            f"- **{reg}**: {s['pct_time']}% of days  "
            f"| fwd-21d avg={s['fwd_mean']:+.2f}%  "
            f"| hit-rate={s['hit_rate']:.1f}%"
        )

    lines += [
        "",
        "## What Each Regime Means for the ML Model",
        "- **BULL**: momentum features dominate → use standard Ridge/RF weights",
        "- **BEAR**: quality + low-vol features dominate → Step 77 switches to bear model",
        "- **SIDEWAYS**: mean-reversion dominates → Step 77 switches to sideways model",
        "",
        "## Last 20 Regime Transitions",
        "| Date | From | To | SPY | VIX | Score |",
        "|------|------|----|-----|-----|-------|",
    ]

    for _, row in transitions.tail(20).iterrows():
        lines.append(
            f"| {row['date']} | {row['from_regime']} | {row['to_regime']} "
            f"| {row['spy_price']} | {row['vix']} | {row['score']} |"
        )

    lines += [
        "",
        "## Historical Bear Periods Captured",
        "These BEAR-regime windows represent the model's ability to detect risk-off environments:",
    ]
    bear_periods = _summarize_periods(df[df["regime"] == "BEAR"])
    for p in bear_periods[-10:]:
        lines.append(f"- {p['start']} → {p['end']}  ({p['days']} days)  "
                     f"SPY peak-to-trough: {p['spt']}")

    OUT_REPORT.write_text("\n".join(lines))
    print(f"  Report: {OUT_REPORT.name}")


def _summarize_periods(df_sub: pd.DataFrame) -> list:
    """Group consecutive days into periods."""
    if df_sub.empty:
        return []
    periods = []
    start = None
    prev_dt = None
    prev_spy = None
    start_spy = None

    for dt, row in df_sub.iterrows():
        if start is None:
            start, start_spy = dt, row["spy_close"]
        elif (dt - prev_dt).days > 5:
            spt = f"{(prev_spy / start_spy - 1) * 100:+.1f}%"
            periods.append({"start": start.date(), "end": prev_dt.date(),
                            "days": (prev_dt - start).days, "spt": spt})
            start, start_spy = dt, row["spy_close"]
        prev_dt = dt
        prev_spy = row["spy_close"]

    if start is not None:
        spt = f"{(prev_spy / start_spy - 1) * 100:+.1f}%"
        periods.append({"start": start.date(), "end": prev_dt.date(),
                        "days": (prev_dt - start).days, "spt": spt})
    return periods


# ─────────────────────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────────────────────

def run_full(start: str = START_DATE) -> pd.DataFrame:
    print("\n╔══════════════════════════════════════════════╗")
    print("║  Canyon v9 — Step 76: Market Regime Detector ║")
    print("╚══════════════════════════════════════════════╝\n")

    print("[1/5] Downloading price data …")
    df_raw = download_data(start=start)
    print(f"  {len(df_raw):,} trading days  ({df_raw.index[0].date()} → {df_raw.index[-1].date()})")

    print("[2/5] Computing indicators …")
    df_ind = build_indicators(df_raw)
    print(f"  Indicators computed on {len(df_ind):,} days after warm-up")

    print("[3/5] Scoring & classifying regimes …")
    df_cls = classify(df_ind)

    # Save regime_history.csv
    out_cols = ["spy_close", "vix", "ma50", "ma200", "ret_20d", "rsi14",
                "score", "raw_regime", "regime"]
    df_cls[out_cols].to_csv(OUT_HISTORY, index=True)
    print(f"  regime_history.csv  ({len(df_cls):,} rows)")

    print("[4/5] Computing transitions & stats …")
    transitions = compute_transitions(df_cls)
    transitions.to_csv(OUT_TRANSITIONS, index=False)
    print(f"  regime_transitions.csv  ({len(transitions)} events)")

    stats = compute_stats(df_cls)
    for reg in ["BULL", "BEAR", "SIDEWAYS"]:
        s = stats[reg]
        print(f"  {reg:10s}: {s['pct_time']:5.1f}% of days | "
              f"fwd-21d avg={s['fwd_mean']:+.2f}% | hit={s['hit_rate']:.1f}%")

    print("[5/5] Writing current regime + report …")
    last = df_cls.iloc[-1]
    current = {
        "date":     str(df_cls.index[-1].date()),
        "regime":   last["regime"],
        "score":    int(last["score"]),
        "ma50":     round(float(last["ma50"]), 2),
        "ma200":    round(float(last["ma200"]), 2),
        "vix":      round(float(last["vix"]), 1),
        "rsi":      round(float(last["rsi14"]), 1),
        "ret_20d":  round(float(last["ret_20d"]), 4),
    }
    OUT_CURRENT.write_text(json.dumps(current, indent=2))
    print(f"  regime_current.json → {current['regime']} (score={current['score']})")

    write_report(df_cls, transitions, stats, current)

    print("\n>>> KEY RESULTS:")
    for reg in ["BULL", "BEAR", "SIDEWAYS"]:
        s = stats[reg]
        print(f"    {reg:10s}  fwd-21d={s['fwd_mean']:+.2f}%  "
              f"hit={s['hit_rate']:.1f}%  ({s['pct_time']:.1f}% of time)")

    print(f"\n    CURRENT REGIME: {current['regime']}  "
          f"(score={current['score']}, date={current['date']})\n")

    return df_cls


def append_today() -> None:
    """Download only today's data and update the CSVs."""
    today_str = date.today().isoformat()

    # Force refresh cache
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()

    df_raw = download_data(start="1999-01-01", use_cache=False)
    df_ind = build_indicators(df_raw)
    df_cls = classify(df_ind)

    last = df_cls.iloc[-1]
    current = {
        "date":     str(df_cls.index[-1].date()),
        "regime":   last["regime"],
        "score":    int(last["score"]),
        "ma50":     round(float(last["ma50"]), 2),
        "ma200":    round(float(last["ma200"]), 2),
        "vix":      round(float(last["vix"]), 1),
        "rsi":      round(float(last["rsi14"]), 1),
        "ret_20d":  round(float(last["ret_20d"]), 4),
    }
    OUT_CURRENT.write_text(json.dumps(current, indent=2))

    out_cols = ["spy_close", "vix", "ma50", "ma200", "ret_20d", "rsi14",
                "score", "raw_regime", "regime"]
    df_cls[out_cols].to_csv(OUT_HISTORY, index=True)

    transitions = compute_transitions(df_cls)
    transitions.to_csv(OUT_TRANSITIONS, index=False)

    print(f"Today ({today_str}): {current['regime']}  score={current['score']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canyon v9 Step 76: Regime Detector")
    parser.add_argument("--today",  action="store_true", help="Append today only (fast)")
    parser.add_argument("--show",   action="store_true", help="Print current regime and exit")
    parser.add_argument("--start",  default=START_DATE,  help="Start date for full run")
    args = parser.parse_args()

    if args.show:
        if OUT_CURRENT.exists():
            c = json.loads(OUT_CURRENT.read_text())
            print(f"\nCurrent regime: {c['regime']}  (score={c['score']})  date={c['date']}")
            print(f"  SPY 50MA={c['ma50']}  200MA={c['ma200']}  VIX={c['vix']}  RSI={c['rsi']}")
        else:
            print("No regime_current.json found. Run without --show first.")
    elif args.today:
        append_today()
    else:
        run_full(start=args.start)
