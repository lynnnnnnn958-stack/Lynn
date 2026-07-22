"""
W21: EDGAR 8-K Earnings Call → FinBERT Sentiment
==================================================
Fetches 8-K filings (earnings releases + call transcripts) from SEC EDGAR,
runs FinBERT financial sentiment model, and outputs ticker-level sentiment scores.

Data source:
  SEC EDGAR full-text search API (free, no auth):
    https://efts.sec.gov/LATEST/search-index?q="earnings"&dateRange=custom&startdt=...&enddt=...&forms=8-K
  EDGAR submissions API for company-specific 8-K filings.

FinBERT:
  ProsusAI/finbert from HuggingFace (free, no API key needed).
  3 classes: POSITIVE, NEGATIVE, NEUTRAL
  Sentiment score = P(POSITIVE) - P(NEGATIVE), range [-1, +1]

PIT compliance:
  Signal date = filing date (accession date from EDGAR submissions API).
  Never use information beyond the filing date. For each as_of date,
  we look back at filings in the prior lookback_days window.

Graceful degradation:
  - If transformers not installed: returns empty DataFrame
  - If EDGAR rate-limited: returns cached data
  - If 8-K text unavailable: skips that filing

Outputs:
  earnings_call_sentiment.csv — ticker × date × sentiment_score
  (use get_earnings_sentiment() to query for a specific as_of date)

Usage:
    from signals.earnings_call import fetch_8k_sentiment, get_earnings_sentiment
    df = fetch_8k_sentiment(["AAPL", "MSFT"], lookback_days=90)
    scores = get_earnings_sentiment(df, as_of=pd.Timestamp("2024-03-01"))
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent

EDGAR_SUBMISSIONS  = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
EDGAR_FILINGS_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22earnings%22&forms=8-K&dateRange=custom&startdt={start}&enddt={end}"
EDGAR_FILING_URL   = "https://www.sec.gov/Archives/edgar/{path}"
EDGAR_HEADERS      = {"User-Agent": "canyon-quant-research research@canyon.local"}

SLEEP_SEC          = 0.12   # EDGAR rate limit: ~10 req/s max, we use 8 req/s
MAX_FILING_CHARS   = 4000   # truncate 8-K text for FinBERT (512 token limit)

# Sentences most likely to contain forward-looking earnings guidance
_GUIDANCE_PATTERNS = [
    r"revenue.*\$[\d.,]+",
    r"earnings per share",
    r"eps.*\$[\d.,]+",
    r"guidance",
    r"outlook",
    r"expect.*quarter",
    r"full.year.*\d{4}",
    r"operating income",
    r"gross margin",
    r"net income",
]
_GUIDANCE_RE = re.compile("|".join(_GUIDANCE_PATTERNS), re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# 1. EDGAR CIK lookup
# ─────────────────────────────────────────────────────────────────────────────

def _get_cik(ticker: str) -> Optional[int]:
    """Look up CIK for a ticker from EDGAR company_tickers.json."""
    import requests
    try:
        url  = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=10)
        data = resp.json()
        tkr_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == tkr_upper:
                return int(entry["cik_str"])
    except Exception:
        pass
    return None


def _get_recent_8k_filings(
    cik: int,
    start_date: str,
    end_date: str,
    max_filings: int = 5,
) -> list[dict]:
    """
    Get recent 8-K filings for a company from EDGAR submissions API.

    Returns list of dicts with: accession_no, filed_date, primary_doc.
    """
    import requests
    url = EDGAR_SUBMISSIONS.format(cik=cik)
    try:
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    time.sleep(SLEEP_SEC)

    filings = data.get("filings", {}).get("recent", {})
    forms       = filings.get("form", [])
    dates       = filings.get("filingDate", [])
    accessions  = filings.get("accessionNumber", [])
    primary_docs = filings.get("primaryDocument", [])

    results = []
    for form, date, acc, doc in zip(forms, dates, accessions, primary_docs):
        if form not in ("8-K", "8-K/A"):
            continue
        if not (start_date <= date <= end_date):
            continue
        results.append({
            "filed_date":   date,
            "accession_no": acc.replace("-", ""),
            "primary_doc":  doc,
        })
        if len(results) >= max_filings:
            break
    return results


def _fetch_8k_text(cik: int, accession_no: str, primary_doc: str) -> str:
    """Fetch the text of an 8-K filing from EDGAR Archives."""
    import requests
    cik_padded = str(cik).zfill(10)
    acc_dashed = f"{accession_no[:10]}-{accession_no[10:12]}-{accession_no[12:]}"
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no}/{primary_doc}"
    try:
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
        resp.raise_for_status()
        text = resp.text
    except Exception:
        return ""

    time.sleep(SLEEP_SEC)

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 2. Text extraction — earnings-relevant sentences
# ─────────────────────────────────────────────────────────────────────────────

def _extract_guidance_sentences(text: str, max_chars: int = MAX_FILING_CHARS) -> str:
    """
    Extract the most earnings-relevant sentences from 8-K text.

    Prioritises sentences matching guidance patterns (revenue, EPS, outlook).
    Falls back to the first max_chars characters if no patterns match.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    scored = []
    for sent in sentences:
        score = len(_GUIDANCE_RE.findall(sent))
        if score > 0:
            scored.append((score, sent))

    scored.sort(key=lambda x: -x[0])
    top_sentences = [s for _, s in scored[:20]]

    result = " ".join(top_sentences)
    if not result:
        result = text  # fallback: use raw text

    return result[:max_chars]


# ─────────────────────────────────────────────────────────────────────────────
# 3. FinBERT inference
# ─────────────────────────────────────────────────────────────────────────────

def _load_finbert():
    """
    Load FinBERT model (ProsusAI/finbert).
    Returns (pipeline, model_loaded: bool).
    """
    try:
        from transformers import pipeline
        finbert = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
            truncation=True,
            max_length=512,
        )
        return finbert, True
    except Exception as e:
        print(f"  [EarningsCall] FinBERT unavailable: {e}")
        print("  [EarningsCall] Install: pip install transformers torch")
        return None, False


def _score_text(finbert, text: str) -> float:
    """
    Run FinBERT on text chunk, return sentiment score in [-1, +1].
    Score = P(POSITIVE) - P(NEGATIVE).
    """
    if not text or finbert is None:
        return 0.0
    try:
        outputs = finbert(text[:512])
        if isinstance(outputs, list) and outputs:
            scores = {item["label"]: item["score"] for item in outputs[0]}
            pos = scores.get("positive", scores.get("POSITIVE", 0.0))
            neg = scores.get("negative", scores.get("NEGATIVE", 0.0))
            return float(pos - neg)
    except Exception:
        pass
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Public API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_8k_sentiment(
    tickers: list[str],
    lookback_days: int = 90,
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
    sleep_sec: float = SLEEP_SEC,
) -> pd.DataFrame:
    """
    Fetch 8-K filings for given tickers and score with FinBERT.

    Args:
        tickers:       List of S&P 500 tickers.
        lookback_days: How many days back to search for filings.
        cache_path:    Where to save/load the cache CSV.
        force_refresh: Ignore cache and re-fetch.
        sleep_sec:     Sleep between EDGAR requests.

    Returns DataFrame:
        ticker, filed_date, cik, sentiment_score, raw_text_length
    """
    if cache_path is None:
        cache_path = ROOT / "earnings_call_sentiment.csv"

    if not force_refresh and cache_path.exists():
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days < 1.5:
            return pd.read_csv(cache_path, parse_dates=["filed_date"])

    end_date   = pd.Timestamp.today()
    start_date = end_date - pd.Timedelta(days=lookback_days)
    start_str  = start_date.strftime("%Y-%m-%d")
    end_str    = end_date.strftime("%Y-%m-%d")

    finbert, model_loaded = _load_finbert()
    if not model_loaded:
        return pd.DataFrame(columns=["ticker", "filed_date", "sentiment_score"])

    rows = []
    print(f"  [EarningsCall] Fetching 8-K filings for {len(tickers)} tickers "
          f"({start_str} → {end_str})")

    for i, ticker in enumerate(tickers):
        cik = _get_cik(ticker)
        if cik is None:
            continue
        time.sleep(sleep_sec)

        filings = _get_recent_8k_filings(cik, start_str, end_str, max_filings=3)
        for filing in filings:
            raw = _fetch_8k_text(cik, filing["accession_no"], filing["primary_doc"])
            if not raw:
                continue
            text_chunk = _extract_guidance_sentences(raw)
            score      = _score_text(finbert, text_chunk)
            rows.append({
                "ticker":          ticker,
                "filed_date":      pd.Timestamp(filing["filed_date"]),
                "cik":             cik,
                "sentiment_score": round(score, 4),
                "text_length":     len(raw),
            })

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(tickers)} tickers processed, {len(rows)} filings found")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["ticker", "filed_date"]).reset_index(drop=True)
        df.to_csv(cache_path, index=False)
        print(f"  [EarningsCall] Saved {len(df)} filings → {cache_path}")
        pos_pct = (df["sentiment_score"] > 0.05).mean()
        neg_pct = (df["sentiment_score"] < -0.05).mean()
        print(f"  [EarningsCall] Sentiment: {pos_pct:.0%} positive, {neg_pct:.0%} negative")
    else:
        print("  [EarningsCall] No filings found (EDGAR unavailable or no recent 8-Ks)")

    return df


def get_earnings_sentiment(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_days: int = 90,
) -> pd.Series:
    """
    Get cross-sectional earnings call sentiment scores as of a given date.

    PIT: Only uses filings with filed_date <= as_of.
    For each ticker, uses the most recent filing within the lookback window.

    Returns pd.Series: ticker → sentiment_score ([-1, +1]), z-scored cross-sectionally.
    """
    if df.empty or "filed_date" not in df.columns:
        return pd.Series(dtype=float)

    cutoff_start = as_of - pd.Timedelta(days=lookback_days)
    valid = df[
        (df["filed_date"] <= as_of) &
        (df["filed_date"] >= cutoff_start)
    ].copy()

    if valid.empty:
        return pd.Series(dtype=float)

    # Most recent filing per ticker
    latest = valid.sort_values("filed_date").groupby("ticker").last()
    scores = latest["sentiment_score"]

    # Z-score cross-sectionally
    mu, std = scores.mean(), scores.std()
    if std > 1e-9:
        scores = (scores - mu) / std

    return scores


def compute_earnings_signal(
    as_of: pd.Timestamp,
    lookback_days: int = 90,
    cache_path: Optional[Path] = None,
) -> pd.Series:
    """
    Convenience function: load cache and return z-scored sentiment signal for as_of date.
    Returns empty Series if cache not found.
    """
    if cache_path is None:
        cache_path = ROOT / "earnings_call_sentiment.csv"
    if not cache_path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(cache_path, parse_dates=["filed_date"])
    return get_earnings_sentiment(df, as_of=as_of, lookback_days=lookback_days)


if __name__ == "__main__":
    from data.sp500_constituents import get_universe_on_date, build_constituent_history

    print("W21: Fetching 8-K earnings call sentiment via FinBERT")
    history = build_constituent_history()
    tickers = get_universe_on_date(history, pd.Timestamp.today())[:50]  # test on first 50

    df = fetch_8k_sentiment(tickers, lookback_days=90, force_refresh=True)
    print(f"\nFetched {len(df)} filings for {df['ticker'].nunique()} tickers")

    if not df.empty:
        today = pd.Timestamp.today()
        scores = get_earnings_sentiment(df, as_of=today)
        print(f"\nTop bullish (most positive FinBERT sentiment):")
        print(scores.sort_values(ascending=False).head(10))
        print(f"\nTop bearish:")
        print(scores.sort_values().head(10))
