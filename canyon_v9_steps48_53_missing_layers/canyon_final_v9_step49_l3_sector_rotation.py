#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CANYON v9 Step 49 — L3 Sector & Theme Rotation Layer

Outputs:
- sector_rotation_scores.csv
- sector_rotation_report.md
- theme_heatmap.csv

No broker. No live order.
Uses yfinance if available; writes NO_DATA rows if download fails.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

ROOT = Path.cwd()
OUT_SCORES = ROOT / "sector_rotation_scores.csv"
OUT_REPORT = ROOT / "sector_rotation_report.md"
OUT_HEATMAP = ROOT / "theme_heatmap.csv"

SECTORS = {
    "XLK": "Technology",
    "SMH": "Semiconductor",
    "SOXX": "Semiconductor",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "IYR": "Real Estate",
    "XLC": "Communication Services",
    "XLB": "Materials",
}
BENCHMARK = "SPY"


def download(tickers):
    try:
        import yfinance as yf
        return yf.download([BENCHMARK] + list(tickers), period="1y", interval="1d", auto_adjust=True, progress=False, group_by="ticker", threads=True)
    except Exception as e:
        print(f"yfinance failed: {e}")
        return pd.DataFrame()


def close(data, ticker):
    if data.empty:
        return pd.Series(dtype=float)
    try:
        if isinstance(data.columns, pd.MultiIndex):
            return pd.to_numeric(data[(ticker, "Close")], errors="coerce").dropna()
        return pd.to_numeric(data["Close"], errors="coerce").dropna()
    except Exception:
        return pd.Series(dtype=float)


def ret(s, n):
    if len(s) <= n:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-n - 1] - 1)


def trend_score(s):
    if len(s) < 100:
        return np.nan
    px = s.iloc[-1]
    ma20 = s.rolling(20).mean().iloc[-1]
    ma50 = s.rolling(50).mean().iloc[-1]
    ma100 = s.rolling(100).mean().iloc[-1]
    score = 0
    score += 1 if px > ma20 else -1
    score += 1 if ma20 > ma50 else -1
    score += 1 if ma50 > ma100 else -1
    return score


def vol_adj_mom(s, n=63):
    if len(s) <= n:
        return np.nan
    r = s.pct_change().dropna()
    mom = ret(s, n)
    vol = r.tail(n).std()
    if vol == 0 or pd.isna(vol):
        return np.nan
    return float(mom / (vol * np.sqrt(n)))


def build_scores():
    data = download(SECTORS.keys())
    spy = close(data, BENCHMARK)
    rows = []
    for t, theme in SECTORS.items():
        s = close(data, t)
        if len(s) < 100 or len(spy) < 100:
            rows.append({
                "ticker": t, "theme": theme, "close": "", "ret_5d": "", "ret_20d": "", "ret_63d": "",
                "relative_20d_vs_spy": "", "relative_63d_vs_spy": "", "trend_score": "", "vol_adj_mom_63d": "",
                "rotation_score": "", "rotation_label": "NO_DATA",
            })
            continue
        r5, r20, r63 = ret(s, 5), ret(s, 20), ret(s, 63)
        spy20, spy63 = ret(spy, 20), ret(spy, 63)
        rel20 = r20 - spy20 if pd.notna(r20) and pd.notna(spy20) else np.nan
        rel63 = r63 - spy63 if pd.notna(r63) and pd.notna(spy63) else np.nan
        tr = trend_score(s)
        vam = vol_adj_mom(s, 63)
        raw = 0
        raw += np.nan_to_num(rel20) * 100
        raw += np.nan_to_num(rel63) * 60
        raw += np.nan_to_num(tr) * 5
        raw += np.nan_to_num(vam) * 10
        label = "LEADER" if raw >= 20 else ("WATCH" if raw >= 8 else ("LAGGARD" if raw <= -8 else "NEUTRAL"))
        rows.append({
            "ticker": t,
            "theme": theme,
            "close": round(float(s.iloc[-1]), 4),
            "ret_5d": r5,
            "ret_20d": r20,
            "ret_63d": r63,
            "relative_20d_vs_spy": rel20,
            "relative_63d_vs_spy": rel63,
            "trend_score": tr,
            "vol_adj_mom_63d": vam,
            "rotation_score": round(raw, 2),
            "rotation_label": label,
        })
    df = pd.DataFrame(rows).sort_values("rotation_score", ascending=False, na_position="last")
    return df


def build_heatmap(scores):
    if scores.empty:
        return scores
    cols = ["ticker", "theme", "ret_5d", "ret_20d", "ret_63d", "relative_20d_vs_spy", "relative_63d_vs_spy", "rotation_score", "rotation_label"]
    return scores[[c for c in cols if c in scores.columns]].copy()


def report(scores, heatmap):
    md = []
    md.append("# Canyon v9 Step 49 — L3 Sector & Theme Rotation Report")
    md.append("")
    md.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    md.append("")
    md.append("## Top leaders")
    md.append("")
    leaders = scores[scores["rotation_label"].isin(["LEADER", "WATCH"])].head(8) if not scores.empty else pd.DataFrame()
    if leaders.empty:
        md.append("_No sector leaders found or data unavailable._")
    else:
        show = leaders.copy()
        for c in ["ret_5d", "ret_20d", "ret_63d", "relative_20d_vs_spy", "relative_63d_vs_spy"]:
            if c in show.columns:
                show[c] = pd.to_numeric(show[c], errors="coerce").map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        md.append(show.to_markdown(index=False))
    md.append("")
    md.append("## Full rotation score")
    md.append("")
    show = scores.copy()
    for c in ["ret_5d", "ret_20d", "ret_63d", "relative_20d_vs_spy", "relative_63d_vs_spy"]:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    md.append(show.to_markdown(index=False) if not show.empty else "_No data._")
    md.append("")
    md.append("## Rules")
    md.append("")
    md.append("- L3 selects where to pay attention, not what to buy.")
    md.append("- A sector leader still needs L2 macro support, L5 event/catalyst, L6 price confirmation, L8 risk sizing.")
    md.append("- If L3 conflicts with L8 concentration warnings, L8 wins.")
    md.append("")
    return "\n".join(md)


def main():
    print("=" * 88)
    print("CANYON v9 Step 49 — L3 Sector & Theme Rotation")
    print("=" * 88)
    scores = build_scores()
    heatmap = build_heatmap(scores)
    scores.to_csv(OUT_SCORES, index=False)
    heatmap.to_csv(OUT_HEATMAP, index=False)
    OUT_REPORT.write_text(report(scores, heatmap), encoding="utf-8")
    print(f"Rows: {len(scores)}")
    if not scores.empty:
        print(scores[["ticker", "theme", "rotation_score", "rotation_label"]].to_string(index=False))
    print("Files generated:")
    print(f"  {OUT_SCORES}")
    print(f"  {OUT_REPORT}")
    print(f"  {OUT_HEATMAP}")


if __name__ == "__main__":
    main()
