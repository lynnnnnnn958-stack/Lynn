#!/usr/bin/env python3
"""
Canyon Earnings AI — Qualitative 10-Q/10-K Analysis
======================================================
Fetches SEC EDGAR filings for each ticker and uses Claude AI to produce
a qualitative earnings summary per stock — bridging the gap vs. BcallAI's
deep earnings reading capability.

For each ticker it reads:
  - SEC EDGAR company facts (XBRL) for financial metrics
  - yfinance earnings/revenue/guidance data
  - Recent analyst estimates

Claude then synthesizes these into a structured 5-section summary:
  1. Business quality & moat
  2. Earnings quality & trends
  3. Balance sheet health
  4. Growth drivers & risks
  5. Valuation context

Output: earnings_ai_summaries.csv, earnings_ai_report.md

Requirements:
  ANTHROPIC_API_KEY env var must be set for AI summaries.
  Without it, structured metrics are still saved (no narrative summary).

Usage:
  .venv/bin/python step_earnings_ai.py               # top 50 alpha tickers
  CANYON_UNIVERSE=russell1000 .venv/bin/python step_earnings_ai.py
"""

from __future__ import annotations

import json
import os
import time
import warnings
import urllib.request
import urllib.error
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

MAX_TICKERS   = 50     # per run (Claude API calls are paid; keep manageable)
SLEEP_BETWEEN = 2.0    # seconds between tickers
MODEL         = "claude-sonnet-5"
MAX_TOKENS    = 2000


def _claude(prompt: str) -> str:
    """Call Anthropic API. Returns empty string if no API key."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return ""
    payload = {
        "model":      MODEL,
        "max_tokens": MAX_TOKENS,
        "messages":   [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read())
            return resp["content"][0]["text"]
    except Exception as e:
        return f"[AI error: {e}]"


def _safe_float(val, default=None):
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return default


def _edgar_facts(ticker: str) -> dict:
    """Fetch key financial facts from SEC EDGAR company facts API."""
    facts: dict = {}
    try:
        # Get CIK from EDGAR
        search_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2024-01-01&forms=10-Q,10-K"
        cik_url    = f"https://www.sec.gov/cgi-bin/browse-edgar?company=&CIK={ticker}&type=10-Q&dateb=&owner=include&count=1&search_text=&action=getcompany&output=atom"

        # Try the company tickers JSON
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        req = urllib.request.Request(tickers_url, headers={"User-Agent": "Canyon Quant research@canyon.local"})
        with urllib.request.urlopen(req, timeout=10) as r:
            all_co = json.loads(r.read())

        cik = None
        for _, v in all_co.items():
            if v.get("ticker", "").upper() == ticker.upper():
                cik = str(v["cik_str"]).zfill(10)
                facts["company_name"] = v.get("title", "")
                break

        if not cik:
            return facts

        # Fetch company facts
        facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        req2 = urllib.request.Request(facts_url, headers={"User-Agent": "Canyon Quant research@canyon.local"})
        with urllib.request.urlopen(req2, timeout=15) as r:
            data = json.loads(r.read())

        us_gaap = data.get("facts", {}).get("us-gaap", {})

        def _latest(concept: str, unit: str = "USD") -> float | None:
            entries = us_gaap.get(concept, {}).get("units", {}).get(unit, [])
            annual  = [e for e in entries if e.get("form") in ("10-K", "10-Q") and "end" in e]
            if not annual:
                return None
            annual.sort(key=lambda x: x["end"], reverse=True)
            val = annual[0].get("val")
            return float(val) / 1e6 if val is not None else None  # in $M

        facts["revenue_m"]   = _latest("Revenues") or _latest("RevenueFromContractWithCustomerExcludingAssessedTax")
        facts["net_income_m"]= _latest("NetIncomeLoss")
        facts["ebitda_m"]    = _latest("OperatingIncomeLoss")   # approximation
        facts["cash_m"]      = _latest("CashAndCashEquivalentsAtCarryingValue")
        facts["total_debt_m"]= _latest("LongTermDebt")
        facts["total_assets_m"]= _latest("Assets")
        facts["op_cf_m"]     = _latest("NetCashProvidedByUsedInOperatingActivities")
        facts["shares_m"]    = _latest("CommonStockSharesOutstanding", "shares")
        if facts["shares_m"]:
            facts["shares_m"] /= 1e6   # back to millions

    except Exception as e:
        pass

    return facts


def _yf_metrics(ticker: str) -> dict:
    """Fetch key metrics from yfinance."""
    metrics: dict = {}
    try:
        tkr = yf.Ticker(ticker)
        info = tkr.info or {}

        metrics["sector"]           = info.get("sector", "")
        metrics["industry"]         = info.get("industry", "")
        metrics["market_cap_b"]     = _safe_float(info.get("marketCap", 0), 0) / 1e9
        metrics["pe_ttm"]           = _safe_float(info.get("trailingPE"))
        metrics["pe_fwd"]           = _safe_float(info.get("forwardPE"))
        metrics["pb"]               = _safe_float(info.get("priceToBook"))
        metrics["ev_ebitda"]        = _safe_float(info.get("enterpriseToEbitda"))
        metrics["roe"]              = _safe_float(info.get("returnOnEquity"))
        metrics["roa"]              = _safe_float(info.get("returnOnAssets"))
        metrics["gross_margin"]     = _safe_float(info.get("grossMargins"))
        metrics["op_margin"]        = _safe_float(info.get("operatingMargins"))
        metrics["net_margin"]       = _safe_float(info.get("profitMargins"))
        metrics["rev_growth_yoy"]   = _safe_float(info.get("revenueGrowth"))
        metrics["eps_growth_yoy"]   = _safe_float(info.get("earningsGrowth"))
        metrics["current_ratio"]    = _safe_float(info.get("currentRatio"))
        metrics["debt_equity"]      = _safe_float(info.get("debtToEquity"))
        metrics["fcf_yield"]        = _safe_float(info.get("freeCashflow", 0)) / max(
                                        _safe_float(info.get("marketCap", 1), 1), 1)
        metrics["beta"]             = _safe_float(info.get("beta"))
        metrics["52w_high"]         = _safe_float(info.get("fiftyTwoWeekHigh"))
        metrics["52w_low"]          = _safe_float(info.get("fiftyTwoWeekLow"))
        metrics["price"]            = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        metrics["analyst_target"]   = _safe_float(info.get("targetMeanPrice"))
        metrics["analyst_rating"]   = info.get("recommendationKey", "")
        metrics["num_analysts"]     = info.get("numberOfAnalystOpinions", 0)
        metrics["short_float"]      = _safe_float(info.get("shortPercentOfFloat"))
        metrics["description"]      = (info.get("longBusinessSummary") or "")[:400]

        # EPS surprises from earnings history
        try:
            hist = tkr.earnings_history
            if hist is not None and not hist.empty and "epsActual" in hist.columns:
                latest = hist.dropna(subset=["epsActual", "epsEstimate"]).tail(4)
                beats  = int(((latest["epsActual"] - latest["epsEstimate"]) > 0).sum())
                metrics["beats_last_4q"] = beats
                if not latest.empty:
                    last = latest.iloc[-1]
                    metrics["last_eps_surprise_pct"] = _safe_float(
                        (last["epsActual"] - last["epsEstimate"]) / abs(last["epsEstimate"])
                        if last["epsEstimate"] != 0 else None
                    )
        except Exception:
            pass

    except Exception as e:
        pass

    return metrics


def _build_prompt(ticker: str, yf_m: dict, edgar: dict) -> str:
    """Build the Claude prompt for earnings analysis."""
    def fmt(val, pct=False, unit=""):
        if val is None:
            return "N/A"
        if pct:
            return f"{val:.1%}"
        return f"{val:.1f}{unit}"

    sector   = yf_m.get("sector", "")
    industry = yf_m.get("industry", "")
    desc     = yf_m.get("description", "")

    prompt = f"""You are a senior equity analyst. Write a concise 5-section investment research note for {ticker} ({sector} / {industry}).
Use ONLY the data provided below. Do not hallucinate numbers not in the data.

=== FINANCIAL DATA ===
Market cap:        ${fmt(yf_m.get('market_cap_b'))}B
Price:             ${fmt(yf_m.get('price'))}
Analyst target:    ${fmt(yf_m.get('analyst_target'))}  ({yf_m.get('analyst_rating', 'N/A')}, n={yf_m.get('num_analysts', 0)})
P/E TTM / Fwd:     {fmt(yf_m.get('pe_ttm'))}x / {fmt(yf_m.get('pe_fwd'))}x
EV/EBITDA:         {fmt(yf_m.get('ev_ebitda'))}x
P/B:               {fmt(yf_m.get('pb'))}x
FCF yield:         {fmt(yf_m.get('fcf_yield'), pct=True)}
Revenue growth:    {fmt(yf_m.get('rev_growth_yoy'), pct=True)} YoY
EPS growth:        {fmt(yf_m.get('eps_growth_yoy'), pct=True)} YoY
Gross margin:      {fmt(yf_m.get('gross_margin'), pct=True)}
Op margin:         {fmt(yf_m.get('op_margin'), pct=True)}
Net margin:        {fmt(yf_m.get('net_margin'), pct=True)}
ROE / ROA:         {fmt(yf_m.get('roe'), pct=True)} / {fmt(yf_m.get('roa'), pct=True)}
Debt/Equity:       {fmt(yf_m.get('debt_equity'))}x
Current ratio:     {fmt(yf_m.get('current_ratio'))}x
Beta:              {fmt(yf_m.get('beta'))}
Short float:       {fmt(yf_m.get('short_float'), pct=True)}
EPS beats (4Q):    {yf_m.get('beats_last_4q', 'N/A')}/4
Last surprise:     {fmt(yf_m.get('last_eps_surprise_pct'), pct=True)}
Revenue ($M):      {fmt(edgar.get('revenue_m'))}
Net income ($M):   {fmt(edgar.get('net_income_m'))}
Operating CF ($M): {fmt(edgar.get('op_cf_m'))}
Cash ($M):         {fmt(edgar.get('cash_m'))}
Total debt ($M):   {fmt(edgar.get('total_debt_m'))}

Business: {desc}

=== REQUIRED OUTPUT FORMAT ===
Write exactly these 5 sections (1-2 sentences each, be specific with numbers):

**1. Business Quality & Moat**
[What does this company do, what's its competitive advantage]

**2. Earnings Quality & Trends**
[Revenue/EPS growth quality, beat/miss history, margin trajectory]

**3. Balance Sheet Health**
[Debt level, cash position, liquidity assessment]

**4. Growth Drivers & Key Risks**
[What will drive growth, what could go wrong]

**5. Valuation Context**
[Is it cheap/fair/expensive vs. sector peers, price target vs. current]

**6. Synthesis & Action (3 sentences max)**
[Most important near-term catalyst | Biggest downside risk | BUY/HOLD/AVOID + one-line reason]
"""
    return prompt


def analyze_ticker(tkr: str) -> dict | None:
    try:
        yf_m  = _yf_metrics(tkr)
        edgar = _edgar_facts(tkr)

        # Build prompt and get AI summary
        prompt  = _build_prompt(tkr, yf_m, edgar)
        summary = _claude(prompt)

        result = {
            "ticker":               tkr,
            "company_name":         edgar.get("company_name") or yf_m.get("sector", ""),
            "sector":               yf_m.get("sector", ""),
            "industry":             yf_m.get("industry", ""),
            "price":                yf_m.get("price"),
            "market_cap_b":         round(yf_m.get("market_cap_b") or 0, 1),
            "pe_ttm":               yf_m.get("pe_ttm"),
            "pe_fwd":               yf_m.get("pe_fwd"),
            "ev_ebitda":            yf_m.get("ev_ebitda"),
            "gross_margin":         yf_m.get("gross_margin"),
            "op_margin":            yf_m.get("op_margin"),
            "net_margin":           yf_m.get("net_margin"),
            "rev_growth_yoy":       yf_m.get("rev_growth_yoy"),
            "eps_growth_yoy":       yf_m.get("eps_growth_yoy"),
            "roe":                  yf_m.get("roe"),
            "debt_equity":          yf_m.get("debt_equity"),
            "fcf_yield":            yf_m.get("fcf_yield"),
            "beats_last_4q":        yf_m.get("beats_last_4q"),
            "last_eps_surprise_pct":yf_m.get("last_eps_surprise_pct"),
            "analyst_target":       yf_m.get("analyst_target"),
            "analyst_rating":       yf_m.get("analyst_rating"),
            "num_analysts":         yf_m.get("num_analysts"),
            "beta":                 yf_m.get("beta"),
            "short_float":          yf_m.get("short_float"),
            "summary":              summary,
            "has_ai_summary":       bool(summary and not summary.startswith("[AI error")),
            "as_of":                date.today().isoformat(),
        }
        return result

    except Exception as e:
        print(f"    ERROR {tkr}: {e}")
        return None


def main():
    print("=" * 60)
    print(f"  Canyon Earnings AI — {date.today()}")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY", ""))
    print(f"  AI summaries: {'enabled (Claude API)' if has_key else 'DISABLED (set ANTHROPIC_API_KEY)'}")
    print("=" * 60)

    # Load tickers
    universe    = os.environ.get("CANYON_UNIVERSE", "")
    univ_path   = ROOT / f"universe_{universe}.csv" if universe else None
    scores_path = ROOT / "alpha_scores.csv"

    if univ_path and univ_path.exists():
        src     = pd.read_csv(univ_path)
        col     = "ticker" if "ticker" in src.columns else src.columns[0]
        tickers = src[col].dropna().tolist()[:MAX_TICKERS]
        print(f"  Universe: {universe} ({len(tickers)} tickers)")
    elif scores_path.exists():
        scores  = pd.read_csv(scores_path)
        score_col = next((c for c in ["alpha_score", "score"] if c in scores.columns), None)
        if score_col:
            tickers = scores.sort_values(score_col, ascending=False)["ticker"].tolist()[:MAX_TICKERS]
        else:
            tickers = scores["ticker"].tolist()[:MAX_TICKERS]
        print(f"  Universe: S&P 500 top {len(tickers)} by alpha score")
    else:
        print("ERROR: alpha_scores.csv not found")
        return

    # Load existing results to skip tickers already done today
    out_path = ROOT / "earnings_ai_summaries.csv"
    done_today: set[str] = set()
    if out_path.exists():
        existing = pd.read_csv(out_path)
        today_str = date.today().isoformat()
        if "as_of" in existing.columns:
            done_today = set(existing[existing["as_of"] == today_str]["ticker"].tolist())

    todo = [t for t in tickers if t not in done_today]
    print(f"  {len(done_today)} already done today → {len(todo)} remaining\n")

    results = []
    for i, tkr in enumerate(todo, 1):
        print(f"  [{i:3d}/{len(todo)}] {tkr} …", end=" ", flush=True)
        r = analyze_ticker(tkr)
        if r:
            results.append(r)
            ai_flag = "✓ AI" if r.get("has_ai_summary") else "metrics only"
            print(f"ok ({ai_flag})")
        else:
            print("skip")
        time.sleep(SLEEP_BETWEEN)

    if results:
        new_df = pd.DataFrame(results)
        if out_path.exists():
            old_df = pd.read_csv(out_path)
            # Keep old rows for tickers not re-run
            old_df = old_df[~old_df["ticker"].isin(new_df["ticker"].tolist())]
            combined = pd.concat([old_df, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(out_path, index=False)
        ai_count = sum(1 for r in results if r.get("has_ai_summary"))
        print(f"\n  {len(results)} tickers saved ({ai_count} with AI summaries) → {out_path.name}")
    else:
        print("  No new results.")


if __name__ == "__main__":
    main()
