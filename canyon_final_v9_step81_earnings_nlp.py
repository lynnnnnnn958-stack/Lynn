#!/usr/bin/env python3
"""
Canyon v9 — Step 81: Earnings 8-K NLP Sentiment
================================================
Downloads SEC 8-K filings (Item 2.02 — Results of Operations / earnings
press releases) and scores management tone with FinBERT.

Why this beats MD&A (step80):
  - 8-K is filed within 4 days of earnings (10-K lags 60-90 days)
  - Earnings press releases contain management's initial reaction
  - Q&A section tone reflects analyst confidence level
  - More real-time signal vs. the annual/quarterly MD&A

Signal logic:
  sig_8k = 0.50 × current_tone_z + 0.50 × sequential_delta_z
  (equal weight: current tone AND how it changed matter equally)

Data: SEC EDGAR 8-K filings, Item 2.02 section
Cache: filings stored in sec_filings_cache/ (permanent per accession)
Freshness: re-processes if output > 14 days old (earnings season = quarterly)

Outputs:
  sec_8k_sentiment.csv    — per-ticker: score, delta, sig_8k
  sec_8k_report.md        — summary report

Usage:
  python3 canyon_final_v9_step81_earnings_nlp.py
  python3 canyon_final_v9_step81_earnings_nlp.py --top 80
  python3 canyon_final_v9_step81_earnings_nlp.py --ticker NVDA --refresh
"""
from __future__ import annotations

import argparse
import json
import re
import ssl
import time
import urllib.request
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent
CACHE_DIR = ROOT / "sec_filings_cache"
CACHE_DIR.mkdir(exist_ok=True)
OUT_CSV    = ROOT / "sec_8k_sentiment.csv"
OUT_REPORT = ROOT / "sec_8k_report.md"
CIK_CACHE  = CACHE_DIR / "company_tickers.json"

SEC_SLEEP       = 0.22
FRESHNESS_DAYS  = 14
DELTA_THR       = 0.08
BULLISH_THR     = 0.12
BEARISH_THR     = -0.12

_HEADERS = {"User-Agent": "CanyonQuant Research canyonquant@research.com",
            "Accept-Encoding": "gzip, deflate"}


# ── HTTP ──────────────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 30) -> Optional[str]:
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            raw = r.read()
            if r.info().get("Content-Encoding") == "gzip":
                import gzip
                raw = gzip.decompress(raw)
            return raw.decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"    [GET] {url[:65]}… → {exc}")
        return None
    finally:
        time.sleep(SEC_SLEEP)


# ── CIK map ───────────────────────────────────────────────────────────────────

def load_cik_map() -> dict[str, str]:
    if CIK_CACHE.exists():
        age = (datetime.now().timestamp() - CIK_CACHE.stat().st_mtime) / 86400
        if age < 7:
            return json.loads(CIK_CACHE.read_text())
    raw = _get("https://www.sec.gov/files/company_tickers.json")
    if not raw:
        return json.loads(CIK_CACHE.read_text()) if CIK_CACHE.exists() else {}
    data = json.loads(raw)
    cik_map = {str(e["ticker"]).upper(): str(e["cik_str"]).zfill(10)
               for e in data.values() if e.get("ticker") and e.get("cik_str")}
    CIK_CACHE.write_text(json.dumps(cik_map))
    return cik_map


# ── 8-K filings ───────────────────────────────────────────────────────────────

def get_recent_8k(cik: str, n: int = 2) -> list[dict]:
    """Get n most recent 8-K filings (filter: Items containing 2.02)."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    raw = _get(url)
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []

    recent  = data.get("filings", {}).get("recent", {})
    forms   = recent.get("form",            [])
    dates   = recent.get("filingDate",      [])
    accnums = recent.get("accessionNumber", [])
    docs    = recent.get("primaryDocument", [])
    items   = recent.get("items",           [])

    results = []
    for form, date, accn, doc, item in zip(forms, dates, accnums, docs, items):
        if form.upper() != "8-K":
            continue
        # Filter for earnings releases: Item 2.02 "Results of Operations"
        item_str = str(item).lower()
        if "2.02" not in item_str and "results of operations" not in item_str:
            continue
        results.append({"form": form, "filing_date": date,
                         "accession_no": accn, "primary_doc": doc, "items": item})
        if len(results) >= n:
            break
    return results


# ── Download + extract earnings text ─────────────────────────────────────────

def fetch_8k_text(cik: str, accession_no: str, primary_doc: str) -> Optional[str]:
    accn_clean = accession_no.replace("-", "")
    cache_file = CACHE_DIR / f"8k_{cik}_{accn_clean}.txt"
    if cache_file.exists() and cache_file.stat().st_size > 500:
        return cache_file.read_text(encoding="utf-8", errors="replace")
    url  = (f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{accn_clean}/{primary_doc}")
    html = _get(url)
    if not html:
        return None
    text = _strip_html(html)
    if len(text) < 200:
        return None
    cache_file.write_text(text, encoding="utf-8")
    return text


def _strip_html(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style"]):
            t.decompose()
        return " ".join(soup.get_text(separator=" ").split())
    except ImportError:
        text = re.sub(r"<[^>]+>", " ", html)
        return " ".join(text.split())


def extract_earnings_text(text: str, max_chars: int = 12_000) -> Optional[str]:
    """Extract the earnings release body from 8-K text."""
    # Look for Item 2.02 section
    patterns = [
        r"Item\s+2\.02[^A-Za-z]{0,20}Results\s+of\s+Operations",
        r"ITEM\s+2\.02[^A-Za-z]{0,20}RESULTS\s+OF\s+OPERATIONS",
        r"Results\s+of\s+Operations\s+and\s+Financial\s+Condition",
        r"Press\s+Release",
        r"PRESS\s+RELEASE",
        r"FOR\s+IMMEDIATE\s+RELEASE",
    ]
    start_pos = None
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            start_pos = m.start()
            break

    if start_pos is None:
        if len(text) < 500:
            return None
        start_pos = 0   # use whole document if no clear section marker

    # End marker: next Item or "Exhibits"
    section = text[start_pos: start_pos + 60_000]
    end_patterns = [
        r"Item\s+[39]\.01", r"ITEM\s+[39]\.01",
        r"Item\s+9\.01", r"Exhibits",
        r"SIGNATURES", r"Signatures",
    ]
    end_pos = len(section)
    for pat in end_patterns:
        m = re.search(pat, section[300:], re.IGNORECASE)
        if m:
            end_pos = min(end_pos, m.start() + 300)

    body = " ".join(section[:end_pos].split())
    return body[:max_chars] if len(body) > 100 else None


# ── FinBERT scoring ───────────────────────────────────────────────────────────

_PIPE: Optional[object] = None


def _load_finbert():
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    try:
        from transformers import pipeline
        _PIPE = pipeline("text-classification", model="ProsusAI/finbert",
                         tokenizer="ProsusAI/finbert",
                         model_kwargs={"cache_dir": str(ROOT / "finbert_model_cache")},
                         truncation=True)
        return _PIPE
    except Exception as exc:
        print(f"  [FinBERT] {exc}")
        return None


def score_text(text: str, chunk_words: int = 400) -> float:
    pipe = _load_finbert()
    if pipe is None:
        return np.nan
    words  = text.split()
    chunks = [" ".join(words[i:i+chunk_words]) for i in range(0, len(words), chunk_words)
              if words[i:i+chunk_words]]
    scores = []
    for chunk in chunks[:10]:
        try:
            out   = pipe(chunk[:1600], truncation=True, max_length=512)[0]
            label = out["label"].lower()
            conf  = float(out["score"])
            scores.append(+conf if label == "positive" else -conf if label == "negative" else 0.0)
        except Exception:
            scores.append(0.0)
    if not scores:
        return np.nan
    weights = [max(0.5, 1.0 - 0.1 * i) for i in range(len(scores))]
    return round(sum(s*w for s,w in zip(scores,weights)) / sum(weights), 6)


# ── Per-ticker processing ─────────────────────────────────────────────────────

def _is_fresh(ticker: str) -> bool:
    if not OUT_CSV.exists():
        return False
    try:
        df  = pd.read_csv(OUT_CSV)
        row = df[df["ticker"] == ticker]
        if row.empty:
            return False
        upd = pd.to_datetime(row.iloc[0].get("updated_date",""), errors="coerce")
        return pd.notna(upd) and (pd.Timestamp.now() - upd).days < FRESHNESS_DAYS
    except Exception:
        return False


def process_ticker(ticker: str, cik: str, refresh: bool = False) -> dict:
    result = {"ticker": ticker, "cik": cik,
              "current_score": np.nan, "prev_score": np.nan,
              "delta": np.nan, "label": "NEUTRAL",
              "current_date": "", "prev_date": "",
              "text_chars": 0, "updated_date": datetime.now().strftime("%Y-%m-%d")}

    if not refresh and _is_fresh(ticker):
        df  = pd.read_csv(OUT_CSV)
        return df[df["ticker"] == ticker].iloc[0].to_dict()

    filings = get_recent_8k(cik, n=2)
    if not filings:
        return result

    cur = filings[0]
    result["current_date"] = cur["filing_date"]
    text = fetch_8k_text(cik, cur["accession_no"], cur["primary_doc"])
    if text:
        body = extract_earnings_text(text)
        if body:
            result["text_chars"]     = len(body)
            result["current_score"]  = score_text(body)

    if len(filings) > 1:
        prv = filings[1]
        result["prev_date"] = prv["filing_date"]
        text_p = fetch_8k_text(cik, prv["accession_no"], prv["primary_doc"])
        if text_p:
            body_p = extract_earnings_text(text_p)
            if body_p:
                result["prev_score"] = score_text(body_p)

    cur_s, prv_s = result["current_score"], result["prev_score"]
    if not np.isnan(cur_s) and not np.isnan(prv_s):
        result["delta"] = round(float(cur_s) - float(prv_s), 6)

    d = result["delta"]
    c = result["current_score"]
    if not (isinstance(d, float) and np.isnan(d)):
        result["label"] = ("BULLISH" if d > DELTA_THR else
                           "BEARISH" if d < -DELTA_THR else "NEUTRAL")
    elif not (isinstance(c, float) and np.isnan(c)):
        result["label"] = ("BULLISH" if c > BULLISH_THR else
                           "BEARISH" if c < BEARISH_THR else "NEUTRAL")
    return result


# ── Batch runner ──────────────────────────────────────────────────────────────

def load_tickers(n: int = 80) -> list[str]:
    for fname in ("alpha_scores.csv", "alpha_scores_v26.csv"):
        p = ROOT / fname
        if p.exists():
            df = pd.read_csv(p)
            if "ticker" in df.columns and "alpha_score" in df.columns:
                return df.nlargest(n, "alpha_score")["ticker"].tolist()
            if "ticker" in df.columns:
                return df["ticker"].head(n).tolist()
    return ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO",
            "JPM","V","UNH","XOM","WMT","MA","LLY","JNJ","HD","MRK"]


def run(tickers: list[str], refresh: bool = False) -> pd.DataFrame:
    cik_map = load_cik_map()
    if not cik_map:
        print("  [8K] CIK map unavailable")
        return pd.DataFrame()

    if OUT_CSV.exists() and not refresh:
        global_age = (datetime.now().timestamp() - OUT_CSV.stat().st_mtime) / 86400
        if global_age < FRESHNESS_DAYS - 2:
            print(f"  [8K] Global output {global_age:.0f}d old — skipping. Use --refresh.")
            return pd.read_csv(OUT_CSV)

    existing = pd.read_csv(OUT_CSV) if OUT_CSV.exists() else pd.DataFrame()
    rows = []
    n = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        cik = cik_map.get(ticker.upper())
        if not cik:
            continue
        print(f"  [{i:3d}/{n}] {ticker:6s} …", end=" ", flush=True)
        row = process_ticker(ticker, cik, refresh=refresh)
        rows.append(row)
        cur  = row.get("current_score", np.nan)
        dlt  = row.get("delta", np.nan)
        cs   = f"{cur:+.3f}" if not (isinstance(cur, float) and np.isnan(cur)) else "—"
        ds   = f"Δ{dlt:+.3f}" if not (isinstance(dlt, float) and np.isnan(dlt)) else ""
        print(f"{cs}  {ds}  [{row.get('label','')}]")

    if not rows:
        return existing if not existing.empty else pd.DataFrame()

    df_new = pd.DataFrame(rows)
    if not existing.empty and "ticker" in existing.columns:
        old = existing[~existing["ticker"].isin(df_new["ticker"])]
        df  = pd.concat([old, df_new], ignore_index=True)
    else:
        df = df_new

    cur_v  = pd.to_numeric(df["current_score"], errors="coerce")
    dlt_v  = pd.to_numeric(df["delta"],         errors="coerce")

    def _z(s): mu, sd = s.mean(), s.std(); return ((s-mu)/(sd+1e-9)).round(4) if sd>1e-9 else s*0
    df["score_z"] = _z(cur_v)
    df["delta_z"] = _z(dlt_v)
    df["sig_8k"]  = (0.50*df["score_z"].fillna(0) + 0.50*df["delta_z"].fillna(0)).round(4)
    df["rank_8k"] = df["sig_8k"].rank(ascending=False, na_option="bottom").astype(int)

    out_cols = ["ticker","current_date","current_score","prev_date","prev_score",
                "delta","label","score_z","delta_z","sig_8k","rank_8k",
                "text_chars","updated_date"]
    df = df[[c for c in out_cols if c in df.columns]]
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  [8K] Saved {len(df)} rows → {OUT_CSV.name}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Earnings 8-K NLP Sentiment")
    parser.add_argument("--ticker",  type=str, help="Single ticker")
    parser.add_argument("--top",     type=int, default=80)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Canyon v9 — Earnings 8-K NLP  [{datetime.now():%Y-%m-%d %H:%M}]")
    print("=" * 60)

    tickers = [args.ticker.upper()] if args.ticker else load_tickers(n=args.top)
    print(f"\nProcessing {len(tickers)} tickers …\n")
    df = run(tickers, refresh=args.refresh)
    if not df.empty:
        valid = df[pd.to_numeric(df.get("current_score", pd.Series()), errors="coerce").notna()]
        print(f"\nSummary: {len(valid)} scored | "
              f"BULLISH={( valid['label']=='BULLISH').sum()} | "
              f"BEARISH={(valid['label']=='BEARISH').sum()}")

    print("\n" + "=" * 60)
    print("Step 81 Complete")
    print("=" * 60)
