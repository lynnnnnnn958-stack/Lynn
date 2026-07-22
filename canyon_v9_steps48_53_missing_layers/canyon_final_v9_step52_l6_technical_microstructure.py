#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CANYON v9 Step 52 — L6 Price / Technical / Microstructure Layer

Outputs:
- technical_signal_matrix.csv
- tactical_candidates.csv
- breakout_reversal_watchlist.csv
- intraday_liquidity_proxy.csv
- technical_microstructure_report.md

Uses yfinance daily data. No broker. No live order.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import math

ROOT = Path.cwd()
OUT_MATRIX = ROOT / "technical_signal_matrix.csv"
OUT_CAND = ROOT / "tactical_candidates.csv"
OUT_BREAK = ROOT / "breakout_reversal_watchlist.csv"
OUT_LIQ = ROOT / "intraday_liquidity_proxy.csv"
OUT_REPORT = ROOT / "technical_microstructure_report.md"


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def universe():
    tickers = set()
    for f in ["universe_master.csv", "options_decision_matrix.csv", "action_cards.csv", "pre_trade_checklist.csv"]:
        df = read_csv(ROOT / f)
        if not df.empty and "ticker" in df.columns:
            tickers.update(df["ticker"].astype(str).str.upper().str.strip())
    return sorted([t for t in tickers if t and t not in {"CASH","TACTICAL_CASH"}])


def download(tickers):
    try:
        import yfinance as yf
        return yf.download(tickers, period="1y", interval="1d", auto_adjust=True, progress=False, group_by="ticker", threads=True)
    except Exception as e:
        print(f"yfinance failed: {e}")
        return pd.DataFrame()


def series(data, ticker, field):
    try:
        if isinstance(data.columns, pd.MultiIndex):
            return pd.to_numeric(data[(ticker, field)], errors="coerce").dropna()
        return pd.to_numeric(data[field], errors="coerce").dropna()
    except Exception:
        return pd.Series(dtype=float)


def hist_df(data, ticker):
    try:
        if isinstance(data.columns, pd.MultiIndex):
            x = data[ticker].copy()
        else:
            x = data.copy()
        cols = ["Open","High","Low","Close","Volume"]
        for c in cols:
            if c not in x.columns:
                x[c] = np.nan
            x[c] = pd.to_numeric(x[c], errors="coerce")
        return x[cols].dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


def rsi(close, n=14):
    if len(close) < n + 2:
        return np.nan
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def atr(df, n=14):
    if len(df) < n + 2:
        return np.nan
    high, low, close = df["High"], df["Low"], df["Close"]
    prev = close.shift(1)
    tr = pd.concat([(high-low), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


def ret(close, n):
    if len(close) <= n:
        return np.nan
    return float(close.iloc[-1] / close.iloc[-n-1] - 1)


def zscore(s, n=60):
    if len(s) < n:
        return np.nan
    x = s.tail(n)
    if x.std() == 0:
        return np.nan
    return float((x.iloc[-1] - x.mean()) / x.std())


def build():
    tickers = universe()
    data = download(tickers)
    rows, liq_rows = [], []
    for t in tickers:
        df = hist_df(data, t)
        if len(df) < 80:
            rows.append({"ticker": t, "data_status": "NO_DATA", "technical_score": 0, "technical_label": "NO_DATA"})
            liq_rows.append({"ticker": t, "liquidity_label": "NO_DATA"})
            continue
        close = df["Close"]
        vol = df["Volume"]
        last = float(close.iloc[-1])
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        ma100 = close.rolling(100).mean().iloc[-1] if len(close) >= 100 else np.nan
        r5, r20, r63 = ret(close, 5), ret(close, 20), ret(close, 63)
        rs = rsi(close)
        a = atr(df)
        atr_pct = a / last if pd.notna(a) and last else np.nan
        vol_z = zscore(vol, 60)
        high20 = close.tail(20).max()
        low20 = close.tail(20).min()
        dist_high20 = last / high20 - 1
        dist_low20 = last / low20 - 1

        score = 0
        reasons = []
        if last > ma20 > ma50:
            score += 25; reasons.append("20/50 trend up")
        if pd.notna(ma100) and ma50 > ma100:
            score += 10; reasons.append("50/100 trend up")
        if pd.notna(r20) and r20 > 0:
            score += 10; reasons.append("20d positive")
        if pd.notna(r63) and r63 > 0:
            score += 10; reasons.append("63d positive")
        if pd.notna(rs) and 45 <= rs <= 70:
            score += 10; reasons.append("RSI constructive")
        elif pd.notna(rs) and rs > 75:
            score -= 10; reasons.append("RSI hot")
        if pd.notna(vol_z) and vol_z > 1:
            score += 10; reasons.append("volume elevated")
        if pd.notna(dist_high20) and dist_high20 > -0.02:
            score += 10; reasons.append("near 20d high")
        if pd.notna(atr_pct) and atr_pct > 0.05:
            score -= 5; reasons.append("ATR high")

        score = max(0, min(100, score))
        label = "TACTICAL_CANDIDATE" if score >= 60 else ("WATCH" if score >= 35 else "NO_TECH_EDGE")

        rows.append({
            "ticker": t,
            "data_status": "OK",
            "close": last,
            "ret_5d": r5,
            "ret_20d": r20,
            "ret_63d": r63,
            "above_20dma": bool(last > ma20),
            "above_50dma": bool(last > ma50),
            "above_100dma": bool(last > ma100) if pd.notna(ma100) else "",
            "rsi14": rs,
            "atr14": a,
            "atr14_pct": atr_pct,
            "volume_z60": vol_z,
            "distance_to_20d_high": dist_high20,
            "distance_to_20d_low": dist_low20,
            "technical_score": score,
            "technical_label": label,
            "reasons": "; ".join(reasons),
        })

        dollar_vol = float((close * vol).tail(20).mean())
        med_vol = float(vol.tail(20).median())
        if dollar_vol > 1e9:
            liq = "HIGH"
        elif dollar_vol > 1e8:
            liq = "MEDIUM"
        else:
            liq = "LOW"
        liq_rows.append({
            "ticker": t,
            "avg_20d_dollar_volume": dollar_vol,
            "median_20d_volume": med_vol,
            "liquidity_label": liq,
            "notes": "Daily proxy only; check live bid-ask manually before paper/live.",
        })

    matrix = pd.DataFrame(rows).sort_values(["technical_score","ticker"], ascending=[False, True])
    liq = pd.DataFrame(liq_rows)
    candidates = matrix[matrix["technical_label"].isin(["TACTICAL_CANDIDATE", "WATCH"])].copy()
    breakout = matrix[(matrix["distance_to_20d_high"].apply(lambda x: pd.notna(x) and x > -0.03)) | (matrix["distance_to_20d_low"].apply(lambda x: pd.notna(x) and x < 0.03))].copy()
    return matrix, candidates, breakout, liq


def report(matrix, candidates, breakout, liq):
    md = []
    md.append("# Canyon v9 Step 52 — L6 Price / Technical / Microstructure Report")
    md.append("")
    md.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    md.append("")
    md.append("## Tactical candidates")
    md.append("")
    md.append(candidates.to_markdown(index=False) if not candidates.empty else "_No tactical candidates._")
    md.append("")
    md.append("## Breakout / reversal watchlist")
    md.append("")
    md.append(breakout.to_markdown(index=False) if not breakout.empty else "_No breakout/reversal watchlist._")
    md.append("")
    md.append("## Liquidity proxy")
    md.append("")
    md.append(liq.to_markdown(index=False) if not liq.empty else "_No liquidity data._")
    md.append("")
    md.append("## Full technical matrix")
    md.append("")
    md.append(matrix.to_markdown(index=False) if not matrix.empty else "_No data._")
    md.append("")
    md.append("## Rules")
    md.append("")
    md.append("- L6 is a timing layer, not a thesis layer.")
    md.append("- Strong technicals still need L2 macro, L5 event, L7 options, and L8 risk approval.")
    md.append("- Liquidity here is daily proxy only; live spread must be checked manually.")
    md.append("")
    return "\n".join(md)


def main():
    print("="*88)
    print("CANYON v9 Step 52 — L6 Price / Technical / Microstructure")
    print("="*88)
    matrix, candidates, breakout, liq = build()
    matrix.to_csv(OUT_MATRIX, index=False)
    candidates.to_csv(OUT_CAND, index=False)
    breakout.to_csv(OUT_BREAK, index=False)
    liq.to_csv(OUT_LIQ, index=False)
    OUT_REPORT.write_text(report(matrix, candidates, breakout, liq), encoding="utf-8")
    print(f"Rows: {len(matrix)}")
    if not matrix.empty:
        print(matrix[["ticker","technical_score","technical_label","rsi14","ret_20d","volume_z60"]].to_string(index=False))
    print("Files generated:")
    print(f"  {OUT_MATRIX}")
    print(f"  {OUT_CAND}")
    print(f"  {OUT_BREAK}")
    print(f"  {OUT_LIQ}")
    print(f"  {OUT_REPORT}")


if __name__ == "__main__":
    main()
