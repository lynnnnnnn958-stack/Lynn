"""
W6: EDGAR Form 4 Insider Transaction Fetcher
=============================================
Fetches real SEC Form 4 filings (insider buy/sell) for all S&P 500 tickers.
This replaces the yfinance institutional holders data (which is NOT Form 4 and
has no transaction-level detail or precise dates).

Key advantages over yfinance:
  - Exact transaction date and price (not just period-end holdings)
  - Insider role: CEO/CFO buys are stronger signal than VP sells
  - True PIT: know_date = filed_date (when SEC published the form)
  - Covers all SEC-reportable insiders (officers, directors, 10%+ owners)

Transaction type codes:
  P = Open market purchase (strongest buy signal)
  S = Open market sale  (strongest sell signal)
  A = Award/grant (option/restricted stock — neutral, don't use)
  F = Tax withholding (forced sale, not informational)
  M = Option exercise (neutral)

SEC API is free; polite rate: ~10 req/sec.

Usage:
    from data.edgar_form4 import fetch_form4_transactions, compute_insider_signal
    df = fetch_form4_transactions(["AAPL", "MSFT"], start_date="2022-01-01")
    signal = compute_insider_signal(df, as_of=pd.Timestamp("2024-01-01"))
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
HEADERS = {"User-Agent": "canyon_quant research lynnnnnnn958@gmail.com"}

# SEC EDGAR full-text search for Form 4 filings
EFTS_URL = (
    "https://efts.sec.gov/LATEST/search-index?"
    "q=%22form-type%22%3A%224%22&"
    "dateRange=custom&startdt={start}&enddt={end}&"
    "hits.hits.total.value=true&hits.hits._source.period_of_report=true"
)

# EDGAR company submissions endpoint (has recent filings list)
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


def _get_cik_map() -> dict[str, str]:
    """Reuse CIK map from edgar_pit.py cache if available."""
    cache = ROOT / "edgar_cik_map.json"
    if cache.exists():
        return json.loads(cache.read_text())
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=HEADERS, timeout=30)
    mapping = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in r.json().values()}
    cache.write_text(json.dumps(mapping))
    return mapping


def _fetch_recent_form4(cik: str, max_filings: int = 200) -> list[dict]:
    """
    Fetch recent Form 4 filings for a company via the submissions API.
    Returns list of {accessionNumber, filingDate, reportDate} dicts.
    """
    url = SUBMISSIONS_URL.format(cik=cik)
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    filings = data.get("filings", {}).get("recent", {})
    if not filings:
        return []

    forms      = filings.get("form", [])
    dates      = filings.get("filingDate", [])
    periods    = filings.get("reportDate", [])
    accessions = filings.get("accessionNumber", [])

    results = []
    for f, d, p, a in zip(forms, dates, periods, accessions):
        if f == "4" and len(results) < max_filings:
            results.append({
                "accession":   a,
                "filed_date":  d,
                "report_date": p,
            })
    return results


def _parse_form4_xml(accession: str, cik: str) -> list[dict]:
    """
    Download and parse a single Form 4 XML filing.
    Returns list of transaction rows.
    """
    # Convert accession to path format: 0001234567-23-001234 → 0001234567/23001234
    acc_clean = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{accession}.xml"

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            # Try the index to find the actual XML filename
            idx_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=4&dateb=&owner=include&count=1&search_text="
            return []
        content = r.text
    except Exception:
        return []

    # Parse relevant XML fields with simple string search (avoids xml dependency issues)
    rows = []
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(content)

        # Namespace-agnostic tag search
        def find_text(elem, tag):
            for child in elem.iter():
                if child.tag.split("}")[-1] == tag:
                    return (child.text or "").strip()
            return ""

        # Insider role
        reporter_name  = find_text(root, "rptOwnerName")
        is_officer     = find_text(root, "isOfficer") == "1"
        is_director    = find_text(root, "isDirector") == "1"
        officer_title  = find_text(root, "officerTitle").upper()

        # CEO/CFO = highest signal quality
        is_c_suite = any(x in officer_title for x in ["CEO", "CFO", "CHIEF EXECUTIVE", "CHIEF FINANCIAL"])

        for txn in root.iter():
            tag = txn.tag.split("}")[-1]
            if tag != "nonDerivativeTransaction":
                continue

            try:
                txn_date = find_text(txn, "transactionDate")
                txn_code = find_text(txn, "transactionCode")
                txn_shares = find_text(txn, "transactionShares")
                txn_price  = find_text(txn, "transactionPricePerShare")
                acquired   = find_text(txn, "transactionAcquiredDisposedCode")

                rows.append({
                    "txn_date":   txn_date,
                    "txn_code":   txn_code,
                    "shares":     float(txn_shares) if txn_shares else np.nan,
                    "price":      float(txn_price)  if txn_price  else np.nan,
                    "acquired":   acquired,   # A=acquired D=disposed
                    "is_officer": is_officer,
                    "is_director": is_director,
                    "is_c_suite": is_c_suite,
                    "reporter":   reporter_name,
                })
            except (ValueError, TypeError):
                continue
    except Exception:
        pass

    return rows


def fetch_form4_transactions(
    tickers: list[str],
    start_date: str = "2018-01-01",
    sleep_sec: float = 0.15,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch Form 4 insider transactions for a list of tickers.

    Returns DataFrame with columns:
        ticker, filed_date, txn_date, txn_code, shares, price,
        acquired, is_officer, is_director, is_c_suite, reporter

    PIT key: use `filed_date` as know_date (not txn_date), because
    the market only learns about the transaction when it's filed.
    Per SEC rules, insiders have 2 business days to file after a transaction.
    """
    if cache_path and cache_path.exists():
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days < 7:
            print(f"  [Form4] Loading from cache ({age_days:.1f}d old)")
            return pd.read_csv(cache_path, parse_dates=["filed_date", "txn_date"])

    cik_map = _get_cik_map()
    all_rows: list[dict] = []
    start_ts = pd.Timestamp(start_date)
    n = len(tickers)

    for i, ticker in enumerate(tickers):
        cik = cik_map.get(ticker.upper())
        if not cik:
            continue

        filings = _fetch_recent_form4(cik)
        for f in filings:
            try:
                filed_ts = pd.Timestamp(f["filed_date"])
            except Exception:
                continue
            if filed_ts < start_ts:
                continue

            txn_rows = _parse_form4_xml(f["accession"], cik)
            for row in txn_rows:
                all_rows.append({
                    "ticker":      ticker.upper(),
                    "filed_date":  f["filed_date"],   # KNOW DATE (PIT)
                    "txn_date":    row.get("txn_date"),
                    "txn_code":    row.get("txn_code"),
                    "shares":      row.get("shares"),
                    "price":       row.get("price"),
                    "acquired":    row.get("acquired"),
                    "is_officer":  row.get("is_officer", False),
                    "is_director": row.get("is_director", False),
                    "is_c_suite":  row.get("is_c_suite", False),
                    "reporter":    row.get("reporter", ""),
                })

        if (i + 1) % 20 == 0:
            print(f"  [Form4] Fetched {i+1}/{n} tickers  ({len(all_rows)} transactions)...")
        time.sleep(sleep_sec)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    for col in ["filed_date", "txn_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.sort_values(["ticker", "filed_date"]).reset_index(drop=True)

    if cache_path:
        df.to_csv(cache_path, index=False)
        print(f"  [Form4] Saved {len(df):,} rows → {cache_path}")

    return df


def compute_insider_signal(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_days: int = 90,
    weight_c_suite: float = 2.0,
) -> pd.Series:
    """
    Compute insider buy/sell net score for each ticker.

    Logic:
      - Only use transactions with filed_date in [as_of - lookback, as_of]  (PIT)
      - Only P (open market purchase) and S (open market sale) — informational trades
      - Score = sum(buy_shares × value × role_weight) - sum(sell_shares × value × role_weight)
      - C-suite trades weighted by weight_c_suite
      - Normalized to cross-sectional z-score

    Returns:
        pd.Series indexed by ticker, values = normalized insider buy score
        Positive = net insider buying (bullish)
        Negative = net insider selling (bearish)
    """
    if df.empty:
        return pd.Series(dtype=float)

    window_start = as_of - pd.Timedelta(days=lookback_days)

    # PIT filter: only use filings known by as_of
    window = df[
        (df["filed_date"] >= window_start) &
        (df["filed_date"] <= as_of) &
        (df["txn_code"].isin(["P", "S"]))   # open market only
    ].copy()

    if window.empty:
        return pd.Series(dtype=float)

    # Compute dollar value of each transaction
    window["dollar_val"] = window["shares"].fillna(0) * window["price"].fillna(0)

    # C-suite trades get higher weight
    window["role_weight"] = np.where(window["is_c_suite"], weight_c_suite, 1.0)

    # Net buy score = purchase value - sale value (weighted by role)
    window["net_score"] = np.where(
        window["txn_code"] == "P",
         window["dollar_val"] * window["role_weight"],   # buy: positive
        -window["dollar_val"] * window["role_weight"],   # sell: negative
    )

    # Aggregate to ticker level
    score = window.groupby("ticker")["net_score"].sum()

    # Cross-sectional normalization: z-score clipped to [-3, 3]
    mu, std = score.mean(), score.std()
    if std < 1e-9:
        return pd.Series(0.0, index=score.index)

    return ((score - mu) / std).clip(-3, 3)


if __name__ == "__main__":
    # Quick test on a few tickers
    test_tickers = ["AAPL", "MSFT", "NVDA", "META", "GOOGL"]
    print(f"Fetching Form 4 for {test_tickers}...")
    df = fetch_form4_transactions(
        test_tickers,
        start_date="2023-01-01",
        cache_path=ROOT / "edgar_form4_test.csv",
    )
    print(f"\nTotal transactions: {len(df)}")
    if not df.empty:
        print(df[["ticker", "filed_date", "txn_code", "shares", "price", "is_c_suite"]].head(10).to_string())

        signal = compute_insider_signal(df, as_of=pd.Timestamp("2024-06-01"))
        print(f"\nInsider signal ({len(signal)} tickers):")
        print(signal.sort_values().head(5).to_string())
        print("...")
        print(signal.sort_values().tail(5).to_string())
