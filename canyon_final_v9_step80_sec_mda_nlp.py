#!/usr/bin/env python3
"""
Canyon v9 — Step 80: SEC 10-K/10-Q MD&A Sentiment Scorer
=========================================================
Extracts the "Management's Discussion & Analysis" section from SEC EDGAR
10-K (annual) and 10-Q (quarterly) filings, scores with FinBERT, and
computes quarter-over-quarter sentiment DELTA.

Why this beats Step 79 (news headlines):
  - News = public information; alpha decays in hours
  - MD&A = filed quarterly; fewer algos process it deeply
  - DELTA in tone (more cautious vs more optimistic) is a documented
    alpha signal: Loughran & McDonald (2011), Cohen et al. (2020)

Signal logic:
  sig_10k = 0.40 × current_tone_z + 0.60 × delta_tone_z
  delta > +0.10 → management became more optimistic → BULLISH
  delta < -0.10 → management became more cautious   → BEARISH

Data source: SEC EDGAR public API (free, no authentication)
  Submissions:  https://data.sec.gov/submissions/CIK{cik}.json
  Filing text:  https://www.sec.gov/Archives/edgar/data/...

Rate limit: SEC allows 10 req/sec; throttled to 5 req/sec here.
Cache: filings cached in sec_filings_cache/ (never re-download same accession)
Freshness: skips tickers updated < CACHE_DAYS_10K days ago (10-K = 30d, 10-Q = 15d)

Outputs:
  sec_mda_sentiment.csv   — one row per ticker: scores, delta, sig_10k
  sec_mda_report.md       — markdown summary, top/bottom 10

Usage:
  python3 canyon_final_v9_step80_sec_mda_nlp.py              # top 50 tickers
  python3 canyon_final_v9_step80_sec_mda_nlp.py --top 100
  python3 canyon_final_v9_step80_sec_mda_nlp.py --ticker AAPL
  python3 canyon_final_v9_step80_sec_mda_nlp.py --refresh    # force re-download
  python3 canyon_final_v9_step80_sec_mda_nlp.py --form 10-Q
"""
from __future__ import annotations

import argparse
import json
import re
import time
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

OUT_CSV    = ROOT / "sec_mda_sentiment.csv"
OUT_REPORT = ROOT / "sec_mda_report.md"
CIK_CACHE  = CACHE_DIR / "company_tickers.json"

# SEC rate limiting (must stay under 10 req/sec)
SEC_SLEEP = 0.22   # ~4.5 req/sec

# Freshness: don't re-process if result is this many days old
CACHE_DAYS_10K = 30
CACHE_DAYS_10Q = 15

# FinBERT signal thresholds
BULLISH_THR = 0.15
BEARISH_THR = -0.15
DELTA_THR   = 0.10

_SEC_HEADERS = {
    "User-Agent": "CanyonQuant Research canyonquant@research.com",
    "Accept-Encoding": "gzip, deflate",
}

# ── HTTP helper ───────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 30) -> Optional[str]:
    import ssl
    import urllib.request
    # Use certifi CA bundle if available (required on macOS stock Python)
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    try:
        req = urllib.request.Request(url, headers=_SEC_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read()
            if resp.info().get("Content-Encoding") == "gzip":
                import gzip
                raw = gzip.decompress(raw)
            return raw.decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"    [GET] {url[:70]}… → {exc}")
        return None
    finally:
        time.sleep(SEC_SLEEP)


# ── 1. CIK mapping ────────────────────────────────────────────────────────────

def load_cik_map(refresh: bool = False) -> dict[str, str]:
    """ticker → zero-padded 10-digit CIK. Cached 7 days."""
    if CIK_CACHE.exists() and not refresh:
        age_days = (datetime.now().timestamp() - CIK_CACHE.stat().st_mtime) / 86400
        if age_days < 7:
            return json.loads(CIK_CACHE.read_text())

    print("  [SEC] Downloading CIK map from EDGAR …")
    raw = _get("https://www.sec.gov/files/company_tickers.json")
    if not raw:
        if CIK_CACHE.exists():
            return json.loads(CIK_CACHE.read_text())
        return {}

    data = json.loads(raw)
    cik_map: dict[str, str] = {}
    for entry in data.values():
        ticker = str(entry.get("ticker", "")).upper().strip()
        cik    = str(entry.get("cik_str", "")).strip().zfill(10)
        if ticker and cik:
            cik_map[ticker] = cik

    CIK_CACHE.write_text(json.dumps(cik_map))
    print(f"  [SEC] CIK map: {len(cik_map)} companies cached")
    return cik_map


# ── 2. Recent filings ─────────────────────────────────────────────────────────

def get_recent_filings(
    cik: str,
    form_type: str = "10-K",
    n: int = 2,
) -> list[dict]:
    """Return n most recent filings of form_type from EDGAR submissions API."""
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

    results = []
    for form, date, accn, doc in zip(forms, dates, accnums, docs):
        if form.upper().startswith(form_type.upper().rstrip("/")):
            results.append({
                "form":        form,
                "filing_date": date,
                "accession_no": accn,
                "primary_doc": doc,
            })
            if len(results) >= n:
                break
    return results


# ── 3. Download filing text ───────────────────────────────────────────────────

def fetch_filing_text(cik: str, accession_no: str, primary_doc: str) -> Optional[str]:
    """Download and cache the primary filing document as plain text."""
    accn_clean = accession_no.replace("-", "")
    cache_file = CACHE_DIR / f"{cik}_{accn_clean}.txt"

    if cache_file.exists() and cache_file.stat().st_size > 2000:
        return cache_file.read_text(encoding="utf-8", errors="replace")

    cik_int = int(cik)
    url = (f"https://www.sec.gov/Archives/edgar/data/"
           f"{cik_int}/{accn_clean}/{primary_doc}")
    html = _get(url)
    if not html:
        return None

    text = _html_to_text(html)
    if len(text) < 500:
        return None

    cache_file.write_text(text, encoding="utf-8")
    return text


def _html_to_text(html: str) -> str:
    """Strip HTML tags. Uses BeautifulSoup if available, regex fallback."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "ix:header", "ix:nonnumeric"]):
            tag.decompose()
        return " ".join(soup.get_text(separator=" ").split())
    except ImportError:
        pass
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
    return " ".join(text.split())


# ── 4. Extract MD&A section ───────────────────────────────────────────────────

# 10-K: Item 7 / 10-Q: Item 2
_MDA_START = [
    r"(?:ITEM|Item)\s+7\.?\s*[\.\:\-–—]?\s*MANAGEMENT.{0,40}DISCUSSION",
    r"(?:ITEM|Item)\s+7\.?\s*[\.\:\-–—]?\s*Management.{0,40}Discussion",
    r"(?:ITEM|Item)\s+2\.?\s*[\.\:\-–—]?\s*MANAGEMENT.{0,40}DISCUSSION",
    r"(?:ITEM|Item)\s+2\.?\s*[\.\:\-–—]?\s*Management.{0,40}Discussion",
]
_MDA_END = [
    r"(?:ITEM|Item)\s+7A\.?\s*[\.\:\-–—]?\s*(?:QUANTITATIVE|Quantitative)",
    r"(?:ITEM|Item)\s+8\.?\s*[\.\:\-–—]?\s*(?:FINANCIAL|Financial)\s+STATEMENTS",
    r"(?:ITEM|Item)\s+8\.?\s*[\.\:\-–—]?\s*(?:FINANCIAL|Financial)\s+Statements",
    r"(?:ITEM|Item)\s+3\.?\s*[\.\:\-–—]?\s*(?:QUANTITATIVE|Quantitative)",  # 10-Q end
    r"(?:ITEM|Item)\s+4\.?\s*[\.\:\-–—]?\s*(?:CONTROLS|Controls)",
]


def extract_mda(text: str, max_chars: int = 18_000) -> Optional[str]:
    """Extract MD&A text from a filing. Returns up to max_chars characters."""
    start_pos = None
    for pat in _MDA_START:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            start_pos = m.start()
            break

    if start_pos is None:
        # broad fallback
        m = re.search(
            r"Management.{0,10}s?\s+Discussion.{0,50}Analysis",
            text, re.IGNORECASE | re.DOTALL
        )
        if m:
            start_pos = m.start()

    if start_pos is None:
        return None

    window = text[start_pos: start_pos + 80_000]

    end_pos = len(window)
    for pat in _MDA_END:
        m = re.search(pat, window[600:], re.IGNORECASE)  # skip past the header
        if m:
            end_pos = min(end_pos, m.start() + 600)

    mda = " ".join(window[:end_pos].split())
    return mda[:max_chars] if len(mda) > 200 else None


# ── 5. FinBERT scoring ────────────────────────────────────────────────────────

_PIPE: Optional[object] = None


def _load_finbert():
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    try:
        from transformers import pipeline
        cache_dir = str(ROOT / "finbert_model_cache")
        _PIPE = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            model_kwargs={"cache_dir": cache_dir},
            truncation=True,
        )
        return _PIPE
    except Exception as exc:
        print(f"    [FinBERT] Cannot load model: {exc}")
        return None


def score_text(text: str, chunk_words: int = 420) -> float:
    """
    Score long text with FinBERT by chunking into ~420-word windows.
    Earlier chunks get slightly higher weight (intro para most forward-looking).
    Returns sentiment in [-1, +1].
    """
    pipe = _load_finbert()
    if pipe is None:
        return np.nan

    words  = text.split()
    chunks = [" ".join(words[i: i + chunk_words])
              for i in range(0, len(words), chunk_words)
              if words[i: i + chunk_words]]

    scores: list[float] = []
    for chunk in chunks[:14]:        # cap at ~5900 words
        try:
            out   = pipe(chunk[:1600], truncation=True, max_length=512)[0]
            label = out["label"].lower()
            conf  = float(out["score"])
            scores.append(+conf if label == "positive" else
                          -conf if label == "negative" else 0.0)
        except Exception:
            scores.append(0.0)

    if not scores:
        return np.nan

    # Position-weighted average: weight[i] = max(0.5, 1.0 - 0.08·i)
    weights = [max(0.5, 1.0 - 0.08 * i) for i in range(len(scores))]
    return round(
        sum(s * w for s, w in zip(scores, weights)) / sum(weights), 6
    )


# ── 6. Per-ticker processing ──────────────────────────────────────────────────

def _is_fresh(ticker: str, cache_days: int) -> bool:
    """Check if ticker already has a fresh result in OUT_CSV."""
    if not OUT_CSV.exists():
        return False
    try:
        df  = pd.read_csv(OUT_CSV)
        row = df[df["ticker"] == ticker]
        if row.empty:
            return False
        updated = pd.to_datetime(row.iloc[0].get("updated_date", ""), errors="coerce")
        if pd.isna(updated):
            return False
        return (pd.Timestamp.now() - updated).days < cache_days
    except Exception:
        return False


def process_ticker(
    ticker: str,
    cik: str,
    form_type: str = "10-K",
    refresh: bool = False,
) -> dict:
    """Download, parse, and score MD&A for one ticker (2 most recent filings)."""
    cache_days = CACHE_DAYS_10K if form_type == "10-K" else CACHE_DAYS_10Q
    if not refresh and _is_fresh(ticker, cache_days):
        df  = pd.read_csv(OUT_CSV)
        row = df[df["ticker"] == ticker].iloc[0]
        return row.to_dict()

    result: dict = {
        "ticker":        ticker,
        "cik":           cik,
        "form_type":     form_type,
        "current_score": np.nan,
        "prev_score":    np.nan,
        "delta":         np.nan,
        "label":         "NEUTRAL",
        "current_date":  "",
        "prev_date":     "",
        "mda_chars":     0,
        "updated_date":  datetime.now().strftime("%Y-%m-%d"),
    }

    # Try preferred form, fall back to 10-Q if 10-K not found
    filings = get_recent_filings(cik, form_type=form_type, n=2)
    if not filings and form_type == "10-K":
        filings = get_recent_filings(cik, form_type="10-Q", n=2)
        if filings:
            result["form_type"] = "10-Q"

    if not filings:
        return result

    # Score current filing
    cur  = filings[0]
    result["current_date"] = cur["filing_date"]
    text = fetch_filing_text(cik, cur["accession_no"], cur["primary_doc"])
    if text:
        mda = extract_mda(text)
        if mda:
            result["mda_chars"]     = len(mda)
            result["current_score"] = score_text(mda)

    # Score previous filing (for delta)
    if len(filings) > 1:
        prv  = filings[1]
        result["prev_date"] = prv["filing_date"]
        text_p = fetch_filing_text(cik, prv["accession_no"], prv["primary_doc"])
        if text_p:
            mda_p = extract_mda(text_p)
            if mda_p:
                result["prev_score"] = score_text(mda_p)

    # Delta and label
    cur_s  = result["current_score"]
    prev_s = result["prev_score"]

    if not (np.isnan(cur_s) or np.isnan(prev_s)):
        result["delta"] = round(float(cur_s) - float(prev_s), 6)

    delta = result["delta"]
    cur_s = result["current_score"]

    if not np.isnan(delta) if not isinstance(delta, float) or not np.isnan(delta) else False:
        result["label"] = ("BULLISH" if delta > DELTA_THR else
                           "BEARISH" if delta < -DELTA_THR else "NEUTRAL")
    elif not (isinstance(cur_s, float) and np.isnan(cur_s)):
        result["label"] = ("BULLISH" if cur_s > BULLISH_THR else
                           "BEARISH" if cur_s < BEARISH_THR else "NEUTRAL")

    return result


# ── 7. Batch runner ───────────────────────────────────────────────────────────

def load_tickers(n: int = 50) -> list[str]:
    """Top-N tickers by alpha score, or fallback universe."""
    for fname in ("alpha_scores.csv", "alpha_scores_v26.csv"):
        p = ROOT / fname
        if p.exists():
            df = pd.read_csv(p)
            if "ticker" in df.columns and "alpha_score" in df.columns:
                return df.nlargest(n, "alpha_score")["ticker"].tolist()
            if "ticker" in df.columns:
                return df["ticker"].head(n).tolist()
    return [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","JPM","V",
        "UNH","XOM","WMT","MA","LLY","JNJ","HD","MRK","ABBV","CVX",
        "AMD","QCOM","INTC","MU","TXN","COST","PEP","KO","PG","TMO",
        "ADBE","CRM","NFLX","PYPL","SBUX","NKE","DIS","VZ","T","IBM",
        "GS","BAC","WFC","C","AXP","BLK","SCHW","USB","PNC","TFC",
    ]


def run(
    tickers: list[str],
    form_type: str = "10-K",
    refresh: bool = False,
) -> pd.DataFrame:
    """Process all tickers and persist results."""
    cik_map = load_cik_map()
    if not cik_map:
        print("  [SEC] CIK map unavailable — check network")
        return pd.DataFrame()

    # Load existing results to merge
    existing = pd.read_csv(OUT_CSV) if OUT_CSV.exists() else pd.DataFrame()

    rows: list[dict] = []
    n = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        cik = cik_map.get(ticker.upper())
        if not cik:
            print(f"  [{i:3d}/{n}] {ticker:6s} — CIK not found, skipped")
            continue

        print(f"  [{i:3d}/{n}] {ticker:6s} …", end=" ", flush=True)
        row = process_ticker(ticker, cik, form_type=form_type, refresh=refresh)
        rows.append(row)

        cur   = row.get("current_score", np.nan)
        delta = row.get("delta", np.nan)
        label = row.get("label", "")
        cur_s = f"{cur:+.3f}" if not (isinstance(cur, float) and np.isnan(cur)) else "—"
        dl_s  = (f"Δ{delta:+.3f}"
                 if not (isinstance(delta, float) and np.isnan(delta)) else "")
        print(f"{cur_s}  {dl_s}  [{label}]")

    if not rows:
        return existing if not existing.empty else pd.DataFrame()

    df_new = pd.DataFrame(rows)

    # Merge: keep new data for updated tickers, preserve old for others
    if not existing.empty and "ticker" in existing.columns:
        old = existing[~existing["ticker"].isin(df_new["ticker"])]
        df  = pd.concat([old, df_new], ignore_index=True)
    else:
        df = df_new

    # Cross-sectional z-score the composite signal
    cur_vals   = pd.to_numeric(df["current_score"], errors="coerce")
    delta_vals = pd.to_numeric(df["delta"],         errors="coerce")

    def _zscore(s: pd.Series) -> pd.Series:
        mu, sd = s.mean(), s.std()
        return ((s - mu) / (sd + 1e-9)).round(4) if sd > 1e-9 else s * 0

    df["score_zscore"] = _zscore(cur_vals)
    df["delta_zscore"] = _zscore(delta_vals)
    # 40% current tone + 60% delta (change in tone is the alpha source)
    df["sig_10k"] = (
        0.40 * df["score_zscore"].fillna(0) +
        0.60 * df["delta_zscore"].fillna(0)
    ).round(4)
    df["rank_10k"] = (
        df["sig_10k"].rank(ascending=False, na_option="bottom").astype(int)
    )

    out_cols = [
        "ticker", "form_type", "current_date", "current_score",
        "prev_date", "prev_score", "delta", "label",
        "score_zscore", "delta_zscore", "sig_10k", "rank_10k",
        "mda_chars", "updated_date",
    ]
    df = df[[c for c in out_cols if c in df.columns]]
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  [SEC] Saved {len(df)} rows → {OUT_CSV.name}")
    return df


# ── 8. Markdown report ────────────────────────────────────────────────────────

def write_report(df: pd.DataFrame) -> None:
    valid = df[pd.to_numeric(df.get("current_score", pd.Series()), errors="coerce").notna()].copy()
    if valid.empty:
        return
    valid = valid.sort_values("sig_10k", ascending=False)

    def _fmt_row(r) -> str:
        cur   = f"{r['current_score']:+.3f}" if pd.notna(r.get("current_score")) else "—"
        prev  = f"{r['prev_score']:+.3f}"    if pd.notna(r.get("prev_score"))    else "—"
        delta = f"{r['delta']:+.3f}"         if pd.notna(r.get("delta"))         else "—"
        return (f"| **{r['ticker']}** | {cur} | {prev} | {delta} | "
                f"{r.get('label','—')} | {r.get('current_date','—')[:10]} |")

    header = ("| Ticker | Current | Prev | Delta | Label | Filed |\n"
              "|--------|:-------:|:----:|:-----:|-------|-------|")
    top10  = "\n".join(_fmt_row(r) for _, r in valid.head(10).iterrows())
    bot10  = "\n".join(_fmt_row(r) for _, r in valid.tail(10).iterrows())

    bullish = (valid["label"] == "BULLISH").sum()
    bearish = (valid["label"] == "BEARISH").sum()
    neutral = (valid["label"] == "NEUTRAL").sum()

    report = f"""# SEC MD&A Sentiment Report — {datetime.now():%Y-%m-%d}

**Tickers with scores:** {len(valid)} / {len(df)}
**BULLISH:** {bullish}  **BEARISH:** {bearish}  **NEUTRAL:** {neutral}

## Top 10 — Management became more optimistic (BULLISH delta)

{header}
{top10}

## Bottom 10 — Management became more cautious (BEARISH delta)

{header}
{bot10}

---
Signal: `sig_10k = 0.40 × current_tone_z + 0.60 × delta_tone_z`
Source: SEC EDGAR 10-K/10-Q MD&A section, scored with FinBERT.
Cached: filings re-used for 30 days (10-K) / 15 days (10-Q).
"""
    OUT_REPORT.write_text(report)
    print(f"  [SEC] Report saved → {OUT_REPORT.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEC 10-K/10-Q MD&A Sentiment Scorer")
    parser.add_argument("--ticker",  type=str,  help="Single ticker to process")
    parser.add_argument("--top",     type=int,  default=50,
                        help="Process top-N tickers by alpha score (default 50)")
    parser.add_argument("--form",    type=str,  default="10-K",
                        choices=["10-K", "10-Q"],
                        help="Filing type (default 10-K)")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-download even if cached")
    args = parser.parse_args()

    # Global freshness gate: skip if output was written < 25 days ago.
    # (10-K filings are quarterly; daily re-processing adds no new signal.)
    if not args.refresh and not args.ticker and OUT_CSV.exists():
        age_days = (datetime.now().timestamp() - OUT_CSV.stat().st_mtime) / 86400
        if age_days < 25:
            print(f"SEC MD&A scores are {age_days:.0f} days old (< 25) — skipping. "
                  f"Run with --refresh to force update.")
            raise SystemExit(0)

    print("=" * 60)
    print(f"Canyon v9 — SEC MD&A NLP  [{datetime.now():%Y-%m-%d %H:%M}]")
    print("=" * 60)

    tickers = [args.ticker.upper()] if args.ticker else load_tickers(n=args.top)
    print(f"\nProcessing {len(tickers)} tickers ({args.form}) …\n")

    df = run(tickers, form_type=args.form, refresh=args.refresh)

    if not df.empty:
        write_report(df)
        valid = df[pd.to_numeric(df.get("current_score", pd.Series()),
                                 errors="coerce").notna()]
        print(f"\nSummary:")
        print(f"  Tickers with MD&A score: {len(valid)} / {len(tickers)}")
        print(f"  BULLISH: {(valid['label']=='BULLISH').sum()}")
        print(f"  BEARISH: {(valid['label']=='BEARISH').sum()}")
        print(f"  NEUTRAL: {(valid['label']=='NEUTRAL').sum()}")
        if "sig_10k" in valid.columns and len(valid) >= 3:
            top3 = valid.nlargest(3, "sig_10k")[
                ["ticker", "sig_10k", "label", "delta"]
            ].to_string(index=False)
            print(f"\n  Top 3 by sig_10k:\n{top3}")

    print("\n" + "=" * 60)
    print("Step 80 Complete")
    print("=" * 60)
