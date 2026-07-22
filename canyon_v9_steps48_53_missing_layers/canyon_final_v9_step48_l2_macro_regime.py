#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CANYON v9 Step 48 — L2 Macro & Regime Layer

Outputs:
- macro_regime_signals.csv
- macro_regime_report.md
- index_breadth_dashboard.csv
- volatility_regime.csv

No broker. No live order.
Uses yfinance if available. If download fails, writes NO_DATA rows instead of fabricating.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import math
import pandas as pd
import numpy as np

ROOT = Path.cwd()
OUT_SIGNALS = ROOT / "macro_regime_signals.csv"
OUT_REPORT = ROOT / "macro_regime_report.md"
OUT_BREADTH = ROOT / "index_breadth_dashboard.csv"
OUT_VOL = ROOT / "volatility_regime.csv"

MACRO_TICKERS = {
    "SPY": "US large-cap risk benchmark",
    "QQQ": "NASDAQ growth benchmark",
    "IWM": "small-cap breadth / risk appetite",
    "TLT": "long-duration rates proxy",
    "GLD": "gold / real-rate hedge proxy",
    "UUP": "USD strength proxy",
    "^VIX": "equity volatility index",
    "HYG": "credit risk appetite proxy",
}

BREADTH_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "SMH", "SOXX", "XLE", "XLF", "XLV", "XLI", "XLY", "XLP", "XLU", "TLT", "GLD"]


def yf_download(tickers, period="1y"):
    try:
        import yfinance as yf
        data = yf.download(
            tickers,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        return data
    except Exception as e:
        print(f"yfinance download failed: {e}")
        return pd.DataFrame()


def extract_close(data, ticker):
    if data.empty:
        return pd.Series(dtype=float)
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker in data.columns.get_level_values(0):
                s = data[(ticker, "Close")]
            elif "Close" in data.columns.get_level_values(0):
                s = data["Close"][ticker]
            else:
                return pd.Series(dtype=float)
        else:
            s = data["Close"] if "Close" in data.columns else pd.Series(dtype=float)
        return pd.to_numeric(s, errors="coerce").dropna()
    except Exception:
        return pd.Series(dtype=float)


def ret(s, n):
    if len(s) <= n:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-n - 1] - 1)


def ma_state(s, short=20, long=50):
    if len(s) < long:
        return "NO_DATA"
    ma_s = s.rolling(short).mean().iloc[-1]
    ma_l = s.rolling(long).mean().iloc[-1]
    px = s.iloc[-1]
    if px > ma_s > ma_l:
        return "UPTREND"
    if px < ma_s < ma_l:
        return "DOWNTREND"
    return "MIXED"


def realized_vol(s, n=20):
    if len(s) <= n:
        return np.nan
    r = s.pct_change().dropna().tail(n)
    return float(r.std() * math.sqrt(252))


def zscore_last(s, window=252):
    if len(s) < 30:
        return np.nan
    r = s.pct_change().dropna()
    vol = r.rolling(20).std() * math.sqrt(252)
    vv = vol.dropna().tail(window)
    if len(vv) < 20 or vv.std() == 0:
        return np.nan
    return float((vv.iloc[-1] - vv.mean()) / vv.std())


def build_signals():
    data = yf_download(list(MACRO_TICKERS.keys()), period="1y")
    rows = []
    closes = {}
    for t, desc in MACRO_TICKERS.items():
        s = extract_close(data, t)
        closes[t] = s
        rows.append({
            "ticker": t,
            "description": desc,
            "last_close": round(float(s.iloc[-1]), 4) if len(s) else "",
            "ret_5d": ret(s, 5),
            "ret_20d": ret(s, 20),
            "ret_63d": ret(s, 63),
            "trend_state": ma_state(s),
            "realized_vol_20d": realized_vol(s, 20),
            "vol_z_1y": zscore_last(s),
            "data_status": "OK" if len(s) >= 80 else "NO_DATA",
        })
    return pd.DataFrame(rows), closes


def regime_from_signals(df):
    if df.empty:
        return "NO_DATA", "No macro data."

    def get(t, col):
        r = df[df["ticker"] == t]
        if r.empty:
            return np.nan
        return r.iloc[0].get(col, np.nan)

    spy_20 = get("SPY", "ret_20d")
    qqq_20 = get("QQQ", "ret_20d")
    iwm_20 = get("IWM", "ret_20d")
    tlt_20 = get("TLT", "ret_20d")
    uup_20 = get("UUP", "ret_20d")
    vix_20 = get("^VIX", "ret_20d")

    score = 0
    reasons = []
    if pd.notna(spy_20) and spy_20 > 0:
        score += 1; reasons.append("SPY 20d positive")
    if pd.notna(qqq_20) and qqq_20 > spy_20:
        score += 1; reasons.append("QQQ leads SPY")
    if pd.notna(iwm_20) and iwm_20 > 0:
        score += 1; reasons.append("IWM positive breadth")
    if pd.notna(vix_20) and vix_20 < 0:
        score += 1; reasons.append("VIX falling")
    if pd.notna(tlt_20) and tlt_20 < -0.03:
        score -= 1; reasons.append("TLT weak / rates pressure")
    if pd.notna(uup_20) and uup_20 > 0.02:
        score -= 1; reasons.append("USD strength headwind")

    if score >= 3:
        return "RISK_ON", "; ".join(reasons)
    if score <= 0:
        return "RISK_OFF_OR_CHOPPY", "; ".join(reasons)
    return "MIXED_RISK", "; ".join(reasons)


def build_breadth():
    data = yf_download(BREADTH_TICKERS, period="1y")
    rows = []
    above_20 = above_50 = up_20 = total = 0
    for t in BREADTH_TICKERS:
        s = extract_close(data, t)
        if len(s) < 60:
            rows.append({"ticker": t, "close": "", "above_20dma": "", "above_50dma": "", "ret_20d": "", "trend_state": "NO_DATA"})
            continue
        total += 1
        a20 = bool(s.iloc[-1] > s.rolling(20).mean().iloc[-1])
        a50 = bool(s.iloc[-1] > s.rolling(50).mean().iloc[-1])
        r20 = ret(s, 20)
        above_20 += int(a20)
        above_50 += int(a50)
        up_20 += int(pd.notna(r20) and r20 > 0)
        rows.append({
            "ticker": t,
            "close": round(float(s.iloc[-1]), 4),
            "above_20dma": a20,
            "above_50dma": a50,
            "ret_20d": r20,
            "trend_state": ma_state(s),
        })
    df = pd.DataFrame(rows)
    summary = {
        "universe_count": total,
        "above_20dma_pct": above_20 / total if total else np.nan,
        "above_50dma_pct": above_50 / total if total else np.nan,
        "positive_20d_pct": up_20 / total if total else np.nan,
    }
    return df, summary


def build_vol_regime(signals):
    vix = signals[signals["ticker"] == "^VIX"].copy() if not signals.empty else pd.DataFrame()
    if vix.empty:
        rows = [{"metric": "vol_regime", "value": "NO_DATA", "notes": "VIX unavailable"}]
    else:
        r = vix.iloc[0]
        vix_level = r.get("last_close", "")
        vix_ret_20 = r.get("ret_20d", np.nan)
        vol_z = r.get("vol_z_1y", np.nan)
        level = "LOW"
        try:
            lv = float(vix_level)
            if lv >= 25: level = "HIGH"
            elif lv >= 18: level = "MEDIUM"
        except Exception:
            level = "NO_DATA"
        rows = [
            {"metric": "vix_level", "value": vix_level, "notes": level},
            {"metric": "vix_20d_change", "value": vix_ret_20, "notes": "negative is calmer, positive is stress"},
            {"metric": "vol_z_1y", "value": vol_z, "notes": "realized-vol z-score proxy"},
            {"metric": "vol_regime", "value": level, "notes": "rule-based"},
        ]
    return pd.DataFrame(rows)


def pct(x):
    try:
        return f"{float(x):.2%}"
    except Exception:
        return str(x)


def report(signals, breadth, breadth_summary, vol, regime, reasons):
    md = []
    md.append("# Canyon v9 Step 48 — L2 Macro & Regime Report")
    md.append("")
    md.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    md.append("")
    md.append(f"## Regime: **{regime}**")
    md.append("")
    md.append(f"Reasons: {reasons or 'None'}")
    md.append("")
    md.append("## Breadth summary")
    md.append("")
    for k, v in breadth_summary.items():
        md.append(f"- {k}: {pct(v) if 'pct' in k else v}")
    md.append("")
    md.append("## Macro signals")
    md.append("")
    show = signals.copy()
    for c in ["ret_5d", "ret_20d", "ret_63d", "realized_vol_20d"]:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    md.append(show.to_markdown(index=False))
    md.append("")
    md.append("## Breadth dashboard")
    md.append("")
    b = breadth.copy()
    if "ret_20d" in b.columns:
        b["ret_20d"] = pd.to_numeric(b["ret_20d"], errors="coerce").map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    md.append(b.to_markdown(index=False))
    md.append("")
    md.append("## Volatility regime")
    md.append("")
    md.append(vol.to_markdown(index=False))
    md.append("")
    md.append("## Rules")
    md.append("")
    md.append("- L2 is a regime filter, not an entry signal.")
    md.append("- Risk-on allows tactical research, but L8 can still block size.")
    md.append("- Risk-off/choppy means shrink tactical aggression and demand stronger catalysts.")
    md.append("")
    return "\n".join(md)


def main():
    print("=" * 88)
    print("CANYON v9 Step 48 — L2 Macro & Regime")
    print("=" * 88)
    signals, _ = build_signals()
    regime, reasons = regime_from_signals(signals)
    breadth, breadth_summary = build_breadth()
    vol = build_vol_regime(signals)

    signals.to_csv(OUT_SIGNALS, index=False)
    breadth.to_csv(OUT_BREADTH, index=False)
    vol.to_csv(OUT_VOL, index=False)
    OUT_REPORT.write_text(report(signals, breadth, breadth_summary, vol, regime, reasons), encoding="utf-8")

    print(f"Regime: {regime}")
    print(f"Signals: {len(signals)} rows")
    print(f"Breadth: {len(breadth)} rows")
    print("Files generated:")
    print(f"  {OUT_SIGNALS}")
    print(f"  {OUT_REPORT}")
    print(f"  {OUT_BREADTH}")
    print(f"  {OUT_VOL}")


if __name__ == "__main__":
    main()
