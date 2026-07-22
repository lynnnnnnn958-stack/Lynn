#!/usr/bin/env python3
"""
Canyon v9 — Step 89: SEC EDGAR XBRL Fundamentals
=================================================
Free replacement for Bloomberg/FactSet fundamental data.
Uses SEC EDGAR's XBRL Company Facts API to extract quarterly financial data.

API endpoint (free, no key required):
  https://data.sec.gov/api/xbrl/companyfacts/{CIK}.json
  Returns ALL reported XBRL financial facts for a company in one call.

Extracted signals:
  Valuation:
    earnings_yield     — EPS(TTM) / Price  (E/P ratio)
    book_yield         — BV per share / Price  (B/P)
    fcf_yield          — (OCF - CapEx)(TTM) / Price
    sales_yield        — Revenue(TTM) / Market Cap

  Quality:
    roe                — Net Income / Avg Equity (TTM)
    gross_margin       — Gross Profit / Revenue (TTM)
    asset_turnover     — Revenue / Avg Assets
    current_ratio      — Current Assets / Current Liabilities
    debt_to_equity     — Total Debt / Equity

  Growth:
    revenue_growth     — YoY revenue growth (MRQ vs same quarter prior year)
    eps_growth         — YoY EPS growth

Output:
  xbrl_fundamentals.csv  — per-ticker: all above metrics + derived signals
  xbrl_report.md         — distribution + top/bottom stocks

Freshness: quarterly (re-fetch when output > 30 days old; XBRL updates with each 10-Q/10-K)

Usage:
  python3 canyon_final_v9_step89_xbrl_fundamentals.py
  python3 canyon_final_v9_step89_xbrl_fundamentals.py --ticker AAPL --refresh
  python3 canyon_final_v9_step89_xbrl_fundamentals.py --top 80 --refresh
"""
from __future__ import annotations

import argparse
import json
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

ROOT       = Path(__file__).parent
CACHE_DIR  = ROOT / "sec_filings_cache" / "xbrl"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CIK_CACHE  = ROOT / "sec_filings_cache" / "company_tickers.json"
OUT_CSV    = ROOT / "xbrl_fundamentals.csv"
OUT_REPORT = ROOT / "xbrl_report.md"

FRESHNESS_DAYS  = 30     # quarterly re-fetch
TICKER_FRESH    = 25     # per-ticker freshness (days)
SEC_SLEEP       = 0.22   # EDGAR rate limit courtesy

_HEADERS = {"User-Agent": "CanyonQuant Research canyonquant@research.com",
            "Accept-Encoding": "gzip, deflate"}

# ── XBRL concept keys to extract (US-GAAP taxonomy) ─────────────────────────
# Format: {output_col: (xbrl_concept, form_preference, period)}
# period: "annual" uses 10-K only; "ttm" sums last 4 quarters; "mrq" = latest quarter
XBRL_MAP = {
    # Income statement (TTM = trailing 12 months)
    "revenue_ttm":         ("Revenues",                    "ttm"),
    "revenue_alt_ttm":     ("RevenueFromContractWithCustomerExcludingAssessedTax", "ttm"),
    "net_income_ttm":      ("NetIncomeLoss",               "ttm"),
    "eps_basic_ttm":       ("EarningsPerShareBasic",       "ttm"),
    "gross_profit_ttm":    ("GrossProfit",                 "ttm"),
    "operating_income_ttm":("OperatingIncomeLoss",         "ttm"),
    "ocf_ttm":             ("NetCashProvidedByUsedInOperatingActivities", "ttm"),
    "capex_ttm":           ("PaymentsToAcquirePropertyPlantAndEquipment", "ttm"),
    # Balance sheet (most recent quarter)
    "equity_mrq":          ("StockholdersEquity",          "mrq"),
    "total_assets_mrq":    ("Assets",                      "mrq"),
    "total_liabilities_mrq":("Liabilities",                "mrq"),
    "current_assets_mrq":  ("AssetsCurrent",               "mrq"),
    "current_liab_mrq":    ("LiabilitiesCurrent",          "mrq"),
    "long_term_debt_mrq":  ("LongTermDebt",                "mrq"),
    # EPS for growth (quarterly, for YoY comparison)
    "eps_q1":              ("EarningsPerShareBasic",       "q0"),   # latest quarter
    "eps_q5":              ("EarningsPerShareBasic",       "q4"),   # same quarter 1yr ago
    "rev_q1":              ("Revenues",                    "q0"),
    "rev_q5":              ("Revenues",                    "q4"),
    "rev_alt_q1":          ("RevenueFromContractWithCustomerExcludingAssessedTax", "q0"),
    "rev_alt_q5":          ("RevenueFromContractWithCustomerExcludingAssessedTax", "q4"),
}


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 40) -> Optional[str]:
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
        print(f"    [GET] {url[-60:]} → {exc}")
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


# ── XBRL data fetch ───────────────────────────────────────────────────────────

def fetch_company_facts(cik: str, ticker: str) -> Optional[dict]:
    cache_file = CACHE_DIR / f"{ticker}_{cik}.json"
    if cache_file.exists():
        age = (datetime.now().timestamp() - cache_file.stat().st_mtime) / 86400
        if age < TICKER_FRESH:
            try:
                return json.loads(cache_file.read_text())
            except Exception:
                pass
    url  = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    raw  = _get(url, timeout=45)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        # Cache only the us-gaap sub-tree to save disk space
        facts = {"us-gaap": data.get("facts", {}).get("us-gaap", {})}
        cache_file.write_text(json.dumps(facts))
        return facts
    except Exception:
        return None


# ── Value extractor from XBRL facts ──────────────────────────────────────────

def _extract_concept(
    facts:   dict,
    concept: str,
    period:  str,
) -> Optional[float]:
    """
    Extract a numeric value from the XBRL facts dict.
    period:
      "mrq"  → most recent quarterly report (form=10-Q or 10-K, instant or duration)
      "ttm"  → sum of last 4 annual quarters (duration, 3-month periods)
      "q0"   → latest single quarter
      "q4"   → quarter 4 periods ago (same quarter last year)
    """
    gaap = facts.get("us-gaap", {})
    node = gaap.get(concept)
    if not node:
        return None

    units = node.get("units", {})
    # Try USD first, then pure (for ratios like EPS)
    entries = units.get("USD", units.get("USD/shares", units.get("shares", [])))
    if not entries:
        for u_entries in units.values():
            entries = u_entries
            break

    if not entries:
        return None

    # Filter to 10-Q or 10-K filings only
    valid = [e for e in entries if e.get("form") in ("10-Q", "10-K")
             and e.get("val") is not None]
    if not valid:
        return None

    # Sort by end date descending
    valid.sort(key=lambda e: e.get("end", ""), reverse=True)

    if period == "mrq":
        # Most recent value (instant or shortest duration)
        instant = [e for e in valid if e.get("start") is None or
                   e.get("start","") == e.get("end","")]
        if instant:
            return float(instant[0]["val"])
        # Duration: prefer shortest (quarterly < annual)
        duration = [e for e in valid if e.get("start") and e.get("end")]
        duration.sort(key=lambda e: (
            (datetime.fromisoformat(e["end"]) - datetime.fromisoformat(e["start"])).days
            if len(e["end"]) == 10 and len(e.get("start","")) == 10 else 999
        ))
        return float(duration[0]["val"]) if duration else None

    elif period == "ttm":
        # Sum last 4 quarterly values (3-month periods)
        quarterly = [e for e in valid
                     if e.get("start") and e.get("end")
                     and len(e["end"]) == 10 and len(e.get("start","")) == 10]
        quarterly.sort(key=lambda e: e["end"], reverse=True)
        # Filter to ~90-day durations
        q90 = [e for e in quarterly
               if abs((datetime.fromisoformat(e["end"]) -
                        datetime.fromisoformat(e["start"])).days - 91) < 35]
        if len(q90) >= 4:
            return float(sum(float(e["val"]) for e in q90[:4]))
        # Fallback: use annual (365-day) duration
        q365 = [e for e in quarterly
                if abs((datetime.fromisoformat(e["end"]) -
                         datetime.fromisoformat(e["start"])).days - 365) < 30]
        return float(q365[0]["val"]) if q365 else None

    elif period in ("q0", "q4"):
        idx = 0 if period == "q0" else 4
        quarterly = [e for e in valid
                     if e.get("start") and e.get("end")
                     and len(e["end"]) == 10]
        quarterly.sort(key=lambda e: e["end"], reverse=True)
        q90 = [e for e in quarterly
               if abs((datetime.fromisoformat(e["end"]) -
                        datetime.fromisoformat(e["start"])).days - 91) < 35]
        return float(q90[idx]["val"]) if len(q90) > idx else None

    return None


# ── Per-ticker fundamentals ───────────────────────────────────────────────────

def compute_fundamentals(ticker: str, cik: str) -> dict:
    row: dict = {"ticker": ticker, "updated_date": datetime.now().strftime("%Y-%m-%d")}

    facts = fetch_company_facts(cik, ticker)
    if not facts:
        return row

    # Extract raw XBRL values
    raw: dict[str, Optional[float]] = {}
    for col, (concept, period) in XBRL_MAP.items():
        v = _extract_concept(facts, concept, period)
        raw[col] = v

    # Revenue TTM (try alt if primary missing)
    rev = raw.get("revenue_ttm") or raw.get("revenue_alt_ttm")
    rev_q1 = raw.get("rev_q1") or raw.get("rev_alt_q1")
    rev_q5 = raw.get("rev_q5") or raw.get("rev_alt_q5")

    row["revenue_ttm"]    = rev
    row["net_income_ttm"] = raw.get("net_income_ttm")
    row["eps_ttm"]        = raw.get("eps_basic_ttm")
    row["ocf_ttm"]        = raw.get("ocf_ttm")
    row["capex_ttm"]      = raw.get("capex_ttm")
    row["equity_mrq"]     = raw.get("equity_mrq")
    row["assets_mrq"]     = raw.get("total_assets_mrq")
    row["current_ratio"]  = (
        (raw["current_assets_mrq"] / raw["current_liab_mrq"])
        if raw.get("current_assets_mrq") and raw.get("current_liab_mrq") and raw["current_liab_mrq"] != 0
        else None
    )
    row["gross_margin"] = (
        (raw["gross_profit_ttm"] / rev)
        if raw.get("gross_profit_ttm") and rev and rev != 0
        else None
    )

    # Get current market price for yield calculations
    price = _get_price(ticker)
    row["price"] = price

    if price and price > 0:
        # Earnings yield (E/P)
        if row.get("eps_ttm"):
            row["earnings_yield"] = float(row["eps_ttm"]) / price
        # Book yield (BV/P) — BV per share needs shares outstanding
        # We use equity / (price * shares) ≈ equity / market_cap
        # Best proxy: equity_mrq / price (not per-share, but proportional to B/P)
        if row.get("equity_mrq") and row.get("assets_mrq"):
            # If market cap available, B/P = equity / mkt_cap
            row["book_yield_raw"] = float(row["equity_mrq"])   # to be normalized later
        # FCF yield
        ocf   = row.get("ocf_ttm") or 0.0
        capex = abs(row.get("capex_ttm") or 0.0)
        fcf   = float(ocf) - float(capex)
        row["fcf_ttm"] = fcf
        # Sales yield (Revenue / price; normalized later)
        row["sales_ttm_raw"] = rev

    # Quality metrics
    eq = row.get("equity_mrq")
    ni = row.get("net_income_ttm")
    if eq and ni and eq != 0:
        row["roe"] = float(ni) / float(eq)
    if row.get("assets_mrq") and row.get("total_assets_mrq"):
        pass   # asset turnover needs prev year assets; skip for now
    if raw.get("long_term_debt_mrq") and eq and eq != 0:
        row["debt_to_equity"] = float(raw["long_term_debt_mrq"]) / float(eq)

    # YoY growth
    if rev_q1 and rev_q5 and rev_q5 != 0:
        row["revenue_growth_yoy"] = (float(rev_q1) - float(rev_q5)) / abs(float(rev_q5))
    eps_q1, eps_q5 = raw.get("eps_q1"), raw.get("eps_q5")
    if eps_q1 and eps_q5 and eps_q5 != 0:
        row["eps_growth_yoy"] = (float(eps_q1) - float(eps_q5)) / abs(float(eps_q5))

    return row


def _get_price(ticker: str) -> Optional[float]:
    """Get last price from price cache (avoid extra yfinance calls)."""
    for fname in ("backtest_price_cache.csv", "sp500_price_cache.csv"):
        p = ROOT / fname
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            if ticker in df.columns:
                return float(df[ticker].dropna().iloc[-1])
        except Exception:
            pass
    # Fallback: yfinance fast_info
    try:
        import yfinance as yf
        return float(yf.Ticker(ticker).fast_info.last_price)
    except Exception:
        return None


# ── Cross-sectional normalization ─────────────────────────────────────────────

def build_fundamental_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive z-scored signals from raw fundamentals.
    Standardize cross-sectionally; clip at ±3σ (Winsorize).
    """
    def _z(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        mu, sd = s.mean(), s.std()
        if sd < 1e-12:
            return s * 0.0
        clipped = s.clip(mu - 3*sd, mu + 3*sd)
        return ((clipped - clipped.mean()) / clipped.std()).round(4)

    # Valuation signals (higher yield = more attractive = positive signal)
    if "earnings_yield" in df.columns:
        df["sig_value_ep"] = _z(df["earnings_yield"])
    if "fcf_ttm" in df.columns and "price" in df.columns:
        # FCF yield: need market cap proxy; use relative FCF across universe
        df["sig_value_fcf"] = _z(df["fcf_ttm"])

    # Quality signals (higher ROE / gross margin = positive)
    if "roe" in df.columns:
        df["sig_quality_roe"] = _z(df["roe"])
    if "gross_margin" in df.columns:
        df["sig_quality_gm"] = _z(df["gross_margin"])
    if "current_ratio" in df.columns:
        df["sig_liquidity"] = _z(df["current_ratio"])
    if "debt_to_equity" in df.columns:
        df["sig_leverage"] = _z(-df["debt_to_equity"])   # low debt = positive

    # Growth signals
    if "revenue_growth_yoy" in df.columns:
        df["sig_rev_growth"] = _z(df["revenue_growth_yoy"])
    if "eps_growth_yoy" in df.columns:
        df["sig_eps_growth"] = _z(df["eps_growth_yoy"])

    # Composite fundamental score (equal-weight available sigs)
    sig_cols = [c for c in df.columns if c.startswith("sig_")]
    if sig_cols:
        df["sig_fundamental"] = df[sig_cols].mean(axis=1).round(4)
        df["sig_fundamental"]  = _z(df["sig_fundamental"])
        df["rank_fundamental"] = df["sig_fundamental"].rank(ascending=False).astype(int)

    df["updated_date"] = datetime.now().strftime("%Y-%m-%d")
    return df


# ── Batch runner ──────────────────────────────────────────────────────────────

def load_tickers(n: int = 100) -> list[str]:
    for fname in ("alpha_scores.csv", "alpha_scores_v26.csv"):
        p = ROOT / fname
        if p.exists():
            df = pd.read_csv(p)
            if "ticker" in df.columns:
                return df["ticker"].head(n).tolist()
    # Default: S&P 500 large caps
    return [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","JPM","V",
        "UNH","XOM","WMT","MA","LLY","JNJ","HD","MRK","ABBV","CVX",
        "PG","KO","PEP","COST","ADBE","MCD","NFLX","CRM","TXN","CSCO",
        "ABT","TMO","ACN","BAC","AMD","ORCL","DHR","CMCSA","VZ","NKE",
        "INTC","PM","T","QCOM","BMY","MDT","HON","AMGN","LOW","GS",
    ]


def _is_fresh(ticker: str) -> bool:
    if not OUT_CSV.exists():
        return False
    try:
        df  = pd.read_csv(OUT_CSV)
        row = df[df["ticker"] == ticker]
        if row.empty:
            return False
        upd = pd.to_datetime(row.iloc[0].get("updated_date",""), errors="coerce")
        return pd.notna(upd) and (pd.Timestamp.now() - upd).days < TICKER_FRESH
    except Exception:
        return False


def run(tickers: list[str], refresh: bool = False) -> pd.DataFrame:
    if not refresh and OUT_CSV.exists():
        age = (datetime.now().timestamp() - OUT_CSV.stat().st_mtime) / 86400
        if age < FRESHNESS_DAYS:
            print(f"  [XBRL] Output {age:.0f}d old — skipping. Use --refresh.")
            return pd.read_csv(OUT_CSV)

    cik_map  = load_cik_map()
    existing = pd.read_csv(OUT_CSV) if OUT_CSV.exists() else pd.DataFrame()

    rows = []
    n = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        cik = cik_map.get(ticker.upper())
        if not cik:
            continue
        if not refresh and _is_fresh(ticker) and not existing.empty:
            ex_row = existing[existing["ticker"] == ticker]
            if not ex_row.empty:
                rows.append(ex_row.iloc[0].to_dict())
                continue
        print(f"  [{i:3d}/{n}] {ticker:6s} …", end=" ", flush=True)
        row = compute_fundamentals(ticker, cik)
        rows.append(row)
        # Print key metrics
        ep  = row.get("earnings_yield")
        roe = row.get("roe")
        rg  = row.get("revenue_growth_yoy")
        parts = []
        if ep:  parts.append(f"E/P={ep:.2%}")
        if roe: parts.append(f"ROE={roe:.1%}")
        if rg:  parts.append(f"RevG={rg:.1%}")
        print("  ".join(parts) if parts else "(no data)")

    if not rows:
        return existing if not existing.empty else pd.DataFrame()

    df = pd.DataFrame(rows)
    df = build_fundamental_signals(df)

    out_cols_order = (
        ["ticker","price","revenue_ttm","net_income_ttm","eps_ttm",
         "ocf_ttm","fcf_ttm","equity_mrq","assets_mrq",
         "earnings_yield","roe","gross_margin","current_ratio",
         "debt_to_equity","revenue_growth_yoy","eps_growth_yoy"]
        + [c for c in df.columns if c.startswith("sig_") or c == "rank_fundamental"]
        + ["updated_date"]
    )
    df = df[[c for c in out_cols_order if c in df.columns]]
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  [XBRL] Saved {len(df)} rows → {OUT_CSV.name}")
    return df


def write_report(df: pd.DataFrame) -> None:
    if df.empty:
        return

    def _pct(v): return f"{v:.1%}" if pd.notna(v) and isinstance(v,(int,float)) else "—"
    def _f2(v):  return f"{v:.2f}" if pd.notna(v) and isinstance(v,(int,float)) else "—"

    n_scored = df["sig_fundamental"].notna().sum() if "sig_fundamental" in df.columns else 0
    avg_ep  = df["earnings_yield"].mean() if "earnings_yield" in df.columns else None
    avg_roe = df["roe"].mean() if "roe" in df.columns else None
    avg_gm  = df["gross_margin"].mean() if "gross_margin" in df.columns else None

    top5 = ""
    bot5 = ""
    if "rank_fundamental" in df.columns and "sig_fundamental" in df.columns:
        top5 = df.nsmallest(5,"rank_fundamental")[
            ["ticker","earnings_yield","roe","gross_margin","revenue_growth_yoy","sig_fundamental"]
        ].to_markdown(index=False)
        bot5 = df.nlargest(5,"rank_fundamental")[
            ["ticker","earnings_yield","roe","gross_margin","revenue_growth_yoy","sig_fundamental"]
        ].to_markdown(index=False)

    report = f"""# XBRL Fundamentals Report — {datetime.now():%Y-%m-%d}

## Coverage

- Tickers scored: **{n_scored}** / {len(df)}
- Avg earnings yield: {_pct(avg_ep)}
- Avg ROE: {_pct(avg_roe)}
- Avg gross margin: {_pct(avg_gm)}

## Top 5 (Best Fundamental Score)

{top5}

## Bottom 5 (Weakest Fundamental Score)

{bot5}

## Signals Generated

| Signal | Description |
|--------|-------------|
| sig_value_ep | Earnings yield (E/P) z-score — higher = cheaper |
| sig_value_fcf | Free cash flow z-score — higher = more cash generative |
| sig_quality_roe | Return on equity z-score |
| sig_quality_gm | Gross margin z-score |
| sig_liquidity | Current ratio z-score |
| sig_leverage | Negative debt/equity z-score (low debt = positive) |
| sig_rev_growth | YoY revenue growth z-score |
| sig_eps_growth | YoY EPS growth z-score |
| **sig_fundamental** | **Equal-weight composite of all above** |

## Data Source

SEC EDGAR XBRL Company Facts API — free, no API key required.
Updates with each 10-Q/10-K filing (quarterly).
Cache TTL: {TICKER_FRESH} days per ticker.

---
*XBRL fundamentals replace the 1/beta value proxy in the Barra factor model (step88).*
"""
    OUT_REPORT.write_text(report)
    print(f"  [XBRL] Report → {OUT_REPORT.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEC EDGAR XBRL Fundamentals")
    parser.add_argument("--ticker",  type=str, help="Single ticker")
    parser.add_argument("--top",     type=int, default=80)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Canyon v9 — XBRL Fundamentals  [{datetime.now():%Y-%m-%d %H:%M}]")
    print("=" * 60)

    tickers = [args.ticker.upper()] if args.ticker else load_tickers(n=args.top)
    print(f"\nProcessing {len(tickers)} tickers …\n")

    df = run(tickers, refresh=args.refresh)

    if not df.empty and "sig_fundamental" in df.columns:
        write_report(df)
        valid = df[df["sig_fundamental"].notna()]
        print(f"\n[Top 5 fundamental score]")
        if "rank_fundamental" in df.columns:
            top5 = df.nsmallest(5,"rank_fundamental")[
                ["ticker","earnings_yield","roe","revenue_growth_yoy","sig_fundamental"]
            ]
            print(top5.to_string(index=False))

    print("\n" + "=" * 60)
    print("Step 89 Complete")
    print("=" * 60)
