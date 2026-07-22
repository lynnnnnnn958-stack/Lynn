#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CANYON v9 Step 50 — L4 Fundamental / Quality / Valuation Layer

Outputs:
- fundamental_quality_valuation.csv
- fundamental_report.md
- long_term_hold_candidates.csv
- valuation_risk_flags.csv

Uses yfinance if available. No broker, no trading.
For ETFs, marks ETF_NOT_FUNDAMENTAL.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

ROOT = Path.cwd()
OUT_FUND = ROOT / "fundamental_quality_valuation.csv"
OUT_REPORT = ROOT / "fundamental_report.md"
OUT_LONG = ROOT / "long_term_hold_candidates.csv"
OUT_FLAGS = ROOT / "valuation_risk_flags.csv"

ETF_SET = {"SPY","QQQ","IWM","DIA","XLK","SMH","SOXX","XLE","XLF","XLV","XLI","XLY","XLP","XLU","IYR","TLT","GLD","SLV","UUP","HYG","LQD","VNQ","KRE","ARKK","XBI","KWEB"}


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def universe():
    files = ["universe_master.csv", "options_decision_matrix.csv", "action_cards.csv", "pre_trade_checklist.csv"]
    tickers = set()
    for f in files:
        df = read_csv(ROOT / f)
        if not df.empty and "ticker" in df.columns:
            tickers.update(df["ticker"].astype(str).str.upper().str.strip().tolist())
    return sorted([t for t in tickers if t and t not in {"CASH", "TACTICAL_CASH"}])


def get_info(ticker):
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        info = {}
        try:
            info = tk.get_info() or {}
        except Exception:
            info = getattr(tk, "info", {}) or {}
        return info
    except Exception as e:
        print(f"{ticker} info failed: {e}")
        return {}


def val(info, key):
    v = info.get(key, np.nan)
    if v is None:
        return np.nan
    return v


def to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def score_row(r):
    if r["asset_type"] == "ETF":
        return 0, "ETF_NOT_FUNDAMENTAL"
    if r["data_status"] != "OK":
        return 0, "NO_DATA"

    score = 0
    reasons = []

    rg = to_float(r.get("revenue_growth"))
    gm = to_float(r.get("gross_margin"))
    om = to_float(r.get("operating_margin"))
    fcf = to_float(r.get("free_cashflow"))
    de = to_float(r.get("debt_to_equity"))
    fpe = to_float(r.get("forward_pe"))
    peg = to_float(r.get("peg_ratio"))

    if pd.notna(rg) and rg > 0.08:
        score += 20; reasons.append("revenue growth positive")
    elif pd.notna(rg) and rg < 0:
        score -= 15; reasons.append("revenue growth negative")

    if pd.notna(gm) and gm > 0.45:
        score += 15; reasons.append("gross margin strong")
    if pd.notna(om) and om > 0.15:
        score += 15; reasons.append("operating margin solid")
    if pd.notna(fcf) and fcf > 0:
        score += 15; reasons.append("positive free cashflow")
    if pd.notna(de) and de < 100:
        score += 10; reasons.append("debt/equity manageable")
    elif pd.notna(de) and de > 250:
        score -= 10; reasons.append("debt/equity high")

    if pd.notna(fpe) and 0 < fpe < 35:
        score += 10; reasons.append("forward PE not extreme")
    elif pd.notna(fpe) and fpe > 60:
        score -= 10; reasons.append("forward PE expensive")

    if pd.notna(peg) and 0 < peg < 2.5:
        score += 10; reasons.append("PEG acceptable")
    elif pd.notna(peg) and peg > 4:
        score -= 5; reasons.append("PEG expensive")

    score = max(0, min(100, score))
    if score >= 70:
        label = "QUALITY_HOLD_CANDIDATE"
    elif score >= 45:
        label = "FUNDAMENTAL_WATCH"
    elif score > 0:
        label = "TACTICAL_ONLY_OR_WEAK_DATA"
    else:
        label = "NO_FUNDAMENTAL_EDGE"
    return score, f"{label}: " + "; ".join(reasons)


def build():
    rows = []
    for t in universe():
        asset_type = "ETF" if t in ETF_SET else "STOCK"
        if asset_type == "ETF":
            rows.append({"ticker": t, "asset_type": asset_type, "data_status": "ETF_NOT_FUNDAMENTAL", "quality_score": 0, "fundamental_label": "ETF_NOT_FUNDAMENTAL"})
            continue
        info = get_info(t)
        status = "OK" if info else "NO_DATA"
        row = {
            "ticker": t,
            "asset_type": asset_type,
            "data_status": status,
            "company_name": info.get("shortName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": val(info, "marketCap"),
            "revenue_growth": val(info, "revenueGrowth"),
            "gross_margin": val(info, "grossMargins"),
            "operating_margin": val(info, "operatingMargins"),
            "profit_margin": val(info, "profitMargins"),
            "free_cashflow": val(info, "freeCashflow"),
            "debt_to_equity": val(info, "debtToEquity"),
            "trailing_pe": val(info, "trailingPE"),
            "forward_pe": val(info, "forwardPE"),
            "peg_ratio": val(info, "pegRatio"),
            "beta": val(info, "beta"),
            "recommendation_key": info.get("recommendationKey", ""),
            "target_mean_price": val(info, "targetMeanPrice"),
            "current_price": val(info, "currentPrice"),
        }
        score, label = score_row(row)
        row["quality_score"] = score
        row["fundamental_label"] = label
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["quality_score", "ticker"], ascending=[False, True])
    return df


def flags(df):
    rows = []
    if df.empty:
        return pd.DataFrame(columns=["level", "ticker", "message"])
    for _, r in df.iterrows():
        t = r["ticker"]
        if r.get("asset_type") == "ETF":
            continue
        if r.get("data_status") != "OK":
            rows.append({"level": "WARN", "ticker": t, "message": "No yfinance fundamental data."})
        rg = pd.to_numeric(pd.Series([r.get("revenue_growth")]), errors="coerce").iloc[0]
        fpe = pd.to_numeric(pd.Series([r.get("forward_pe")]), errors="coerce").iloc[0]
        de = pd.to_numeric(pd.Series([r.get("debt_to_equity")]), errors="coerce").iloc[0]
        if pd.notna(rg) and rg < 0:
            rows.append({"level": "MEDIUM", "ticker": t, "message": "Revenue growth negative."})
        if pd.notna(fpe) and fpe > 60:
            rows.append({"level": "MEDIUM", "ticker": t, "message": "Forward PE expensive."})
        if pd.notna(de) and de > 250:
            rows.append({"level": "MEDIUM", "ticker": t, "message": "Debt/equity high."})
    return pd.DataFrame(rows, columns=["level", "ticker", "message"])


def report(df, long, flg):
    md = []
    md.append("# Canyon v9 Step 50 — L4 Fundamental / Quality / Valuation Report")
    md.append("")
    md.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    md.append("")
    md.append("## Long-term hold candidates")
    md.append("")
    md.append(long.to_markdown(index=False) if not long.empty else "_No long-term hold candidates._")
    md.append("")
    md.append("## Valuation / quality flags")
    md.append("")
    md.append(flg.to_markdown(index=False) if not flg.empty else "_No flags._")
    md.append("")
    md.append("## Full table")
    md.append("")
    show = df.copy()
    for c in ["revenue_growth","gross_margin","operating_margin","profit_margin"]:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    md.append(show.to_markdown(index=False) if not show.empty else "_No data._")
    md.append("")
    md.append("## Rules")
    md.append("")
    md.append("- L4 is required for long-term hold decisions.")
    md.append("- ETFs are marked ETF_NOT_FUNDAMENTAL here; assess them in L2/L3/L8 instead.")
    md.append("- Tactical trades can exist without L4, but should not become long-term holds.")
    md.append("")
    return "\n".join(md)


def main():
    print("="*88)
    print("CANYON v9 Step 50 — L4 Fundamental / Quality / Valuation")
    print("="*88)
    df = build()
    flg = flags(df)
    long = df[df["fundamental_label"].astype(str).str.contains("QUALITY_HOLD_CANDIDATE", na=False)].copy() if not df.empty else pd.DataFrame()
    df.to_csv(OUT_FUND, index=False)
    long.to_csv(OUT_LONG, index=False)
    flg.to_csv(OUT_FLAGS, index=False)
    OUT_REPORT.write_text(report(df, long, flg), encoding="utf-8")
    print(f"Rows: {len(df)}")
    if not df.empty:
        print(df[["ticker","asset_type","quality_score","fundamental_label"]].to_string(index=False))
    print("Files generated:")
    print(f"  {OUT_FUND}")
    print(f"  {OUT_REPORT}")
    print(f"  {OUT_LONG}")
    print(f"  {OUT_FLAGS}")


if __name__ == "__main__":
    main()
