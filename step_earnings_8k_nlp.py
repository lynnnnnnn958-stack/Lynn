#!/usr/bin/env python3
"""
Canyon — Earnings 8-K Press Release NLP
==========================================
Downloads earnings press releases (8-K EX-99.1 exhibits) from SEC EDGAR
for top alpha tickers and uses Claude Sonnet to extract:

  1. Guidance direction  — Raise / Maintain / Lower / Not given
  2. EPS quality score   — 1-10 (cash quality, beat magnitude, sustainability)
  3. Management tone     — 1-10 (1=very cautious/defensive, 10=very confident)
  4. Key risk factor     — single most important downside mentioned
  5. Overall quality     — 1-10 composite earnings quality score

Unlike step_earnings_ai.py (financial metrics → 5-section narrative),
this step processes the ACTUAL 8-K TEXT — the first-hand source that
management uses to frame results before analyst questions.

SEC EDGAR is fully free and explicitly supports programmatic access.

Output:
  earnings_8k_summaries.csv  — ticker + all extracted fields + raw_text excerpt

Usage:
  .venv/bin/python step_earnings_8k_nlp.py            # top 30 alpha tickers
  MAX_TICKERS=10 .venv/bin/python step_earnings_8k_nlp.py
"""

from __future__ import annotations

import json
import os
import re
import time
import html
import warnings
import urllib.request
import urllib.error
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

MAX_TICKERS   = int(os.environ.get("MAX_TICKERS", "30"))
SLEEP_BETWEEN = 2.5   # polite delay between EDGAR requests
MAX_TEXT_CHARS = 6000  # chars of 8-K text sent to Claude

MODEL      = "claude-sonnet-5"
MAX_TOKENS = 1200

UA = "CanyonQuant/1.0 (research; lynnnnnnn958@gmail.com)"


# ── SEC EDGAR helpers ─────────────────────────────────────────────────────────

_CIK_CACHE: dict[str, str] = {}

def _get_cik(ticker: str) -> str | None:
    if ticker in _CIK_CACHE:
        return _CIK_CACHE[ticker]
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        for _, v in data.items():
            if v.get("ticker", "").upper() == ticker.upper():
                cik = str(v["cik_str"]).zfill(10)
                _CIK_CACHE[ticker] = cik
                return cik
    except Exception:
        pass
    return None


def _get_recent_8k_filings(cik: str, days_back: int = 120) -> list[dict]:
    """Return recent 8-K filing accession numbers from EDGAR submissions API."""
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())

        recent = data.get("filings", {}).get("recent", {})
        forms   = recent.get("form", [])
        dates   = recent.get("filingDate", [])
        accnums = recent.get("accessionNumber", [])
        cutoff  = (date.today() - timedelta(days=days_back)).isoformat()

        filings = []
        for form, filed, acc in zip(forms, dates, accnums):
            if form == "8-K" and filed >= cutoff:
                filings.append({"acc": acc.replace("-", ""), "date": filed})
        return filings
    except Exception:
        return []


def _get_ex99_url(cik: str, acc: str) -> str | None:
    """Find the EX-99.1 exhibit URL within an 8-K filing."""
    try:
        idx_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{acc}-index.json"
        req = urllib.request.Request(idx_url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=10) as r:
            idx = json.loads(r.read())

        for doc in idx.get("documents", []):
            dtype = doc.get("type", "")
            dname = doc.get("documentName", "")
            # EX-99.1 is the earnings press release
            if dtype.startswith("EX-99") or "99" in dtype:
                return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{dname}"
    except Exception:
        pass
    return None


def _download_text(url: str) -> str:
    """Download filing document and strip HTML tags."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

    # Strip HTML
    text = html.unescape(raw)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s{3,}", "\n\n", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = text.strip()
    return text[:MAX_TEXT_CHARS]


def _fetch_8k_text(ticker: str) -> tuple[str, str]:
    """
    Try to get the most recent earnings 8-K text for ticker.
    Returns (text, filing_date) or ("", "").
    """
    cik = _get_cik(ticker)
    if not cik:
        return "", ""

    filings = _get_recent_8k_filings(cik)
    if not filings:
        return "", ""

    # Try most recent filings in order
    for filing in filings[:5]:
        acc = filing["acc"]
        url = _get_ex99_url(cik, acc)
        if not url:
            continue
        text = _download_text(url)
        if len(text) > 300:
            return text, filing["date"]
        time.sleep(0.5)

    return "", ""


# ── Claude helpers ────────────────────────────────────────────────────────────

def _claude(prompt: str) -> str:
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
        with urllib.request.urlopen(req, timeout=45) as r:
            return json.loads(r.read())["content"][0]["text"]
    except Exception as e:
        return f"[error: {e}]"


def _build_prompt(ticker: str, text: str) -> str:
    return f"""You are a senior equity analyst reading an earnings press release.
Analyze this {ticker} 8-K earnings press release and return ONLY valid JSON.

PRESS RELEASE TEXT:
{text}

Return ONLY this JSON (no markdown, no explanation):
{{
  "guidance_direction": "<Raise|Maintain|Lower|Not_given>",
  "guidance_detail": "<one sentence: what specific metric was guided, by how much>",
  "eps_quality_score": <1-10: 10=high quality cash earnings, 1=heavy adjustments/one-time items>,
  "eps_quality_note": "<one sentence explaining score>",
  "management_tone": <1-10: 1=very defensive/cautious, 10=very confident/bullish>,
  "tone_evidence": "<specific phrase from the release that best captures tone>",
  "key_risk": "<most important specific downside or uncertainty mentioned>",
  "beat_driver": "<primary reason EPS beat or missed, if stated>",
  "overall_quality": <1-10: composite earnings quality, weight guidance+eps+tone equally>,
  "summary_one_line": "<one crisp analyst sentence summarizing this quarter>"
}}"""


def _parse_claude_json(response: str) -> dict:
    """Extract JSON from Claude response robustly."""
    try:
        return json.loads(response)
    except Exception:
        pass
    # Try to extract JSON block
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {}


# ── Main ──────────────────────────────────────────────────────────────────────

def analyze_ticker(ticker: str) -> dict | None:
    text, filing_date = _fetch_8k_text(ticker)
    if not text:
        return None

    has_key = bool(os.environ.get("ANTHROPIC_API_KEY", ""))
    analysis = {}
    if has_key:
        prompt   = _build_prompt(ticker, text)
        response = _claude(prompt)
        analysis = _parse_claude_json(response)

    result = {
        "ticker":             ticker,
        "filing_date":        filing_date,
        "has_ai_analysis":    bool(analysis),
        "guidance_direction": analysis.get("guidance_direction", ""),
        "guidance_detail":    analysis.get("guidance_detail", ""),
        "eps_quality_score":  analysis.get("eps_quality_score"),
        "eps_quality_note":   analysis.get("eps_quality_note", ""),
        "management_tone":    analysis.get("management_tone"),
        "tone_evidence":      analysis.get("tone_evidence", ""),
        "key_risk":           analysis.get("key_risk", ""),
        "beat_driver":        analysis.get("beat_driver", ""),
        "overall_quality":    analysis.get("overall_quality"),
        "summary_one_line":   analysis.get("summary_one_line", ""),
        "text_excerpt":       text[:500],
        "as_of":              date.today().isoformat(),
    }
    return result


def main():
    print("=" * 60)
    print(f"  Canyon 8-K NLP — {date.today()}")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY", ""))
    print(f"  Model: {MODEL}  |  AI: {'enabled' if has_key else 'DISABLED (no ANTHROPIC_API_KEY)'}")
    print("=" * 60)

    # Load top alpha tickers
    scores_path = ROOT / "alpha_scores.csv"
    if not scores_path.exists():
        print("ERROR: alpha_scores.csv not found")
        return
    scores    = pd.read_csv(scores_path)
    score_col = next((c for c in ["alpha_score", "score"] if c in scores.columns), None)
    if score_col:
        tickers = scores.sort_values(score_col, ascending=False)["ticker"].tolist()[:MAX_TICKERS]
    else:
        tickers = scores["ticker"].tolist()[:MAX_TICKERS]

    # Skip tickers already done today
    out_path = ROOT / "earnings_8k_summaries.csv"
    done_today: set[str] = set()
    if out_path.exists():
        existing  = pd.read_csv(out_path)
        today_str = date.today().isoformat()
        if "as_of" in existing.columns:
            done_today = set(existing[existing["as_of"] == today_str]["ticker"].tolist())

    todo = [t for t in tickers if t not in done_today]
    print(f"  {len(done_today)} already done today → {len(todo)} remaining\n")

    results = []
    for i, tkr in enumerate(todo, 1):
        print(f"  [{i:3d}/{len(todo)}] {tkr} … ", end="", flush=True)
        try:
            r = analyze_ticker(tkr)
            if r:
                results.append(r)
                ai_flag = f"tone={r['management_tone']} guidance={r['guidance_direction']}" if r.get("has_ai_analysis") else "text_only (no API key)"
                print(f"ok ({ai_flag})")
            else:
                print("no 8-K found")
        except Exception as e:
            print(f"error: {e}")
        time.sleep(SLEEP_BETWEEN)

    if results:
        new_df = pd.DataFrame(results)
        if out_path.exists():
            old_df = pd.read_csv(out_path)
            old_df = old_df[~old_df["ticker"].isin(new_df["ticker"].tolist())]
            combined = pd.concat([old_df, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(out_path, index=False)
        ai_count = sum(1 for r in results if r.get("has_ai_analysis"))
        print(f"\n  {len(results)} tickers saved ({ai_count} with AI analysis) → {out_path.name}")
    else:
        print("  No results this run.")


if __name__ == "__main__":
    main()
