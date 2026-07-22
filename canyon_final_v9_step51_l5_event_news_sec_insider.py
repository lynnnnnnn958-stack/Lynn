#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CANYON v9 Step 51 — L5 Event / News / SEC / Insider Layer

Outputs:
- sec_event_layer.csv
- evidence_cards.csv
- insider_form4_signals.csv
- news_event_risk.csv
- earnings_calendar_check.csv

Uses yfinance if available. Does not fabricate missing insider/news.
No broker. No live order.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

ROOT = Path.cwd()
OUT_SEC = ROOT / "sec_event_layer.csv"
OUT_EVIDENCE = ROOT / "evidence_cards.csv"
OUT_INSIDER = ROOT / "insider_form4_signals.csv"
OUT_NEWS = ROOT / "news_event_risk.csv"
OUT_EARNINGS = ROOT / "earnings_calendar_check.csv"
OUT_REPORT = ROOT / "event_news_sec_insider_report.md"

ETF_SET = {"SPY","QQQ","IWM","DIA","XLK","SMH","SOXX","XLE","XLF","XLV","XLI","XLY","XLP","XLU","IYR","TLT","GLD","SLV","UUP","HYG","LQD","VNQ","KRE","ARKK","XBI","KWEB"}


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
    return sorted([t for t in tickers if t and t not in {"CASH", "TACTICAL_CASH"}])


def ticker_obj(t):
    try:
        import yfinance as yf
        return yf.Ticker(t)
    except Exception:
        return None


def get_earnings(t, obj):
    if t in ETF_SET:
        return {"ticker": t, "asset_type": "ETF", "earnings_date": "", "earnings_status": "ETF_NO_EARNINGS", "notes": ""}
    if obj is None:
        return {"ticker": t, "asset_type": "STOCK", "earnings_date": "", "earnings_status": "NO_DATA", "notes": "yfinance unavailable"}
    try:
        cal = obj.calendar
        if isinstance(cal, dict):
            date_val = cal.get("Earnings Date", "")
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            # yfinance often returns a dataframe
            if "Earnings Date" in cal.index:
                date_val = cal.loc["Earnings Date"].iloc[0]
            else:
                date_val = ""
        else:
            date_val = ""
        return {"ticker": t, "asset_type": "STOCK", "earnings_date": str(date_val), "earnings_status": "CHECK_MANUALLY" if str(date_val) else "NO_DATE_FROM_YFINANCE", "notes": "Manual confirmation required"}
    except Exception as e:
        return {"ticker": t, "asset_type": "STOCK", "earnings_date": "", "earnings_status": "ERROR", "notes": str(e)[:120]}


def get_news(t, obj):
    rows = []
    if obj is None:
        return [{"ticker": t, "news_status": "NO_DATA", "title": "", "publisher": "", "provider_publish_time": "", "risk_label": "UNKNOWN", "notes": "yfinance unavailable"}]
    try:
        news = getattr(obj, "news", []) or []
        if not news:
            return [{"ticker": t, "news_status": "NO_NEWS_FROM_YFINANCE", "title": "", "publisher": "", "provider_publish_time": "", "risk_label": "UNKNOWN", "notes": ""}]
        for item in news[:5]:
            title = str(item.get("title", ""))
            publisher = str(item.get("publisher", ""))
            ts = item.get("providerPublishTime", "")
            low = title.lower()
            risk = "HIGH" if any(k in low for k in ["lawsuit", "probe", "investigation", "sec", "downgrade", "miss", "warning", "cuts"]) else ("MEDIUM" if any(k in low for k in ["earnings", "guidance", "fed", "rates", "tariff", "chip"]) else "LOW")
            rows.append({"ticker": t, "news_status": "OK", "title": title, "publisher": publisher, "provider_publish_time": ts, "risk_label": risk, "notes": "Headline screen only; manually verify."})
        return rows
    except Exception as e:
        return [{"ticker": t, "news_status": "ERROR", "title": "", "publisher": "", "provider_publish_time": "", "risk_label": "UNKNOWN", "notes": str(e)[:120]}]


def get_sec(t, obj):
    if t in ETF_SET:
        return {"ticker": t, "filing_status": "ETF_NO_SEC_COMPANY_FILINGS", "latest_filing_type": "", "latest_filing_date": "", "filing_count": 0, "notes": ""}
    if obj is None:
        return {"ticker": t, "filing_status": "NO_DATA", "latest_filing_type": "", "latest_filing_date": "", "filing_count": 0, "notes": "yfinance unavailable"}
    try:
        filings = None
        for attr in ["sec_filings", "get_sec_filings"]:
            if hasattr(obj, attr):
                x = getattr(obj, attr)
                filings = x() if callable(x) else x
                break
        if filings is None:
            return {"ticker": t, "filing_status": "NO_SEC_API_AVAILABLE", "latest_filing_type": "", "latest_filing_date": "", "filing_count": 0, "notes": "Manual SEC check required"}
        if isinstance(filings, pd.DataFrame):
            if filings.empty:
                return {"ticker": t, "filing_status": "NO_FILINGS", "latest_filing_type": "", "latest_filing_date": "", "filing_count": 0, "notes": ""}
            r = filings.iloc[0]
            return {"ticker": t, "filing_status": "OK", "latest_filing_type": str(r.get("type", r.get("form", ""))), "latest_filing_date": str(r.get("date", "")), "filing_count": len(filings), "notes": "Manual filing read required"}
        if isinstance(filings, list) and filings:
            r = filings[0]
            return {"ticker": t, "filing_status": "OK", "latest_filing_type": str(r.get("type", r.get("form", ""))), "latest_filing_date": str(r.get("date", "")), "filing_count": len(filings), "notes": "Manual filing read required"}
        return {"ticker": t, "filing_status": "NO_FILINGS", "latest_filing_type": "", "latest_filing_date": "", "filing_count": 0, "notes": ""}
    except Exception as e:
        return {"ticker": t, "filing_status": "ERROR", "latest_filing_type": "", "latest_filing_date": "", "filing_count": 0, "notes": str(e)[:120]}


def get_insider(t, obj):
    if t in ETF_SET:
        return {"ticker": t, "insider_status": "ETF_NO_INSIDERS", "recent_transactions": 0, "net_direction": "", "notes": ""}
    if obj is None:
        return {"ticker": t, "insider_status": "NO_DATA", "recent_transactions": 0, "net_direction": "", "notes": "yfinance unavailable"}
    try:
        ins = None
        for attr in ["insider_transactions", "get_insider_transactions"]:
            if hasattr(obj, attr):
                x = getattr(obj, attr)
                ins = x() if callable(x) else x
                break
        if not isinstance(ins, pd.DataFrame) or ins.empty:
            return {"ticker": t, "insider_status": "NO_DATA_FROM_YFINANCE", "recent_transactions": 0, "net_direction": "", "notes": "Use original SEC Form 4 before relying on this."}
        recent = ins.head(20)
        text = " ".join(map(str, recent.to_string().lower().split()))
        buys = text.count("buy") + text.count("purchase")
        sells = text.count("sell") + text.count("sale")
        if buys > sells:
            direction = "NET_BUY_PROXY"
        elif sells > buys:
            direction = "NET_SELL_PROXY"
        else:
            direction = "MIXED_OR_UNKNOWN"
        return {"ticker": t, "insider_status": "YFINANCE_PROXY", "recent_transactions": len(recent), "net_direction": direction, "notes": "Proxy only; confirm raw Form 4 manually."}
    except Exception as e:
        return {"ticker": t, "insider_status": "ERROR", "recent_transactions": 0, "net_direction": "", "notes": str(e)[:120]}


def build_evidence(sec, news, insider, earnings):
    rows = []
    for t in universe():
        s = sec[sec["ticker"] == t].iloc[0].to_dict() if not sec.empty and t in set(sec["ticker"]) else {}
        e = earnings[earnings["ticker"] == t].iloc[0].to_dict() if not earnings.empty and t in set(earnings["ticker"]) else {}
        i = insider[insider["ticker"] == t].iloc[0].to_dict() if not insider.empty and t in set(insider["ticker"]) else {}
        n = news[news["ticker"] == t].copy() if not news.empty else pd.DataFrame()
        high_news = int((n["risk_label"].astype(str).str.upper() == "HIGH").sum()) if not n.empty and "risk_label" in n.columns else 0
        event_score = 0
        reasons = []
        if s.get("filing_status") == "OK":
            event_score += 15; reasons.append("recent SEC filing proxy exists")
        if high_news > 0:
            event_score -= 25; reasons.append("high-risk news headline")
        if e.get("earnings_status") == "CHECK_MANUALLY":
            event_score -= 10; reasons.append("earnings date needs manual check")
        if "BUY" in str(i.get("net_direction", "")):
            event_score += 15; reasons.append("insider buy proxy")
        elif "SELL" in str(i.get("net_direction", "")):
            event_score -= 10; reasons.append("insider sell proxy")
        label = "EVENT_SUPPORT" if event_score >= 15 else ("EVENT_RISK" if event_score < 0 else "NEUTRAL_OR_NO_EVENT")
        rows.append({"ticker": t, "event_score": event_score, "event_label": label, "reasons": "; ".join(reasons) or "No clear event edge."})
    return pd.DataFrame(rows)


def report(sec, evidence, insider, news, earnings):
    md = []
    md.append("# Canyon v9 Step 51 — L5 Event / News / SEC / Insider Report")
    md.append("")
    md.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    md.append("")
    md.append("## Evidence cards")
    md.append("")
    md.append(evidence.to_markdown(index=False) if not evidence.empty else "_No evidence._")
    md.append("")
    md.append("## Earnings check")
    md.append("")
    md.append(earnings.to_markdown(index=False) if not earnings.empty else "_No earnings data._")
    md.append("")
    md.append("## SEC filing layer")
    md.append("")
    md.append(sec.to_markdown(index=False) if not sec.empty else "_No SEC data._")
    md.append("")
    md.append("## Insider Form 4 proxy")
    md.append("")
    md.append(insider.to_markdown(index=False) if not insider.empty else "_No insider data._")
    md.append("")
    md.append("## News risk")
    md.append("")
    md.append(news.head(50).to_markdown(index=False) if not news.empty else "_No news data._")
    md.append("")
    md.append("## Rules")
    md.append("")
    md.append("- News and insider data from yfinance are proxy data only.")
    md.append("- Raw SEC filings/Form 4 must be opened manually for conviction.")
    md.append("- L5 can block a trade before earnings/news even when options/technical look good.")
    md.append("")
    return "\n".join(md)


def main():
    print("="*88)
    print("CANYON v9 Step 51 — L5 Event / News / SEC / Insider")
    print("="*88)
    sec_rows, insider_rows, earn_rows, news_rows = [], [], [], []
    for t in universe():
        obj = ticker_obj(t)
        sec_rows.append(get_sec(t, obj))
        insider_rows.append(get_insider(t, obj))
        earn_rows.append(get_earnings(t, obj))
        news_rows.extend(get_news(t, obj))

    sec = pd.DataFrame(sec_rows)
    insider = pd.DataFrame(insider_rows)
    earnings = pd.DataFrame(earn_rows)
    news = pd.DataFrame(news_rows)
    evidence = build_evidence(sec, news, insider, earnings)

    sec.to_csv(OUT_SEC, index=False)
    insider.to_csv(OUT_INSIDER, index=False)
    earnings.to_csv(OUT_EARNINGS, index=False)
    news.to_csv(OUT_NEWS, index=False)
    evidence.to_csv(OUT_EVIDENCE, index=False)
    OUT_REPORT.write_text(report(sec, evidence, insider, news, earnings), encoding="utf-8")

    print(f"Tickers: {len(universe())}")
    print(f"Evidence rows: {len(evidence)}")
    print("Files generated:")
    print(f"  {OUT_SEC}")
    print(f"  {OUT_EVIDENCE}")
    print(f"  {OUT_INSIDER}")
    print(f"  {OUT_NEWS}")
    print(f"  {OUT_EARNINGS}")
    print(f"  {OUT_REPORT}")


if __name__ == "__main__":
    main()
