"""
SEC EDGAR Fundamentals Downloader
Downloads point-in-time annual financial data for S&P 500 stocks.
Uses filing date (not fiscal year end) to ensure no lookahead bias.
Free, no API key required. Rate: 10 req/sec max.
"""

import requests, time, json, pandas as pd, numpy as np
from pathlib import Path

UA = {"User-Agent": "canyon-quant research@example.com"}
OUT = Path("edgar_fundamentals.csv")
RATE_LIMIT = 0.12  # 8 req/sec (below 10 limit)

# ── Step 1: Get ticker → CIK mapping from SEC ──────────────────────────────

def get_cik_map() -> dict:
    """Returns {ticker: zero-padded 10-digit CIK}"""
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=UA, timeout=20)
    data = r.json()
    return {v["ticker"]: str(v["cik_str"]).zfill(10) for v in data.values()}


# ── Step 2: For one CIK, extract point-in-time annual financials ───────────

CONCEPTS = {
    "net_income":  "NetIncomeLoss",
    "op_cf":       "NetCashProvidedByUsedInOperatingActivities",
    "assets":      "Assets",
    "gross_profit":"GrossProfit",
    "revenues":    "Revenues",
    "long_debt":   "LongTermDebt",
}

def fetch_company_facts(cik: str) -> dict | None:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        r = requests.get(url, headers=UA, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def extract_annual_series(facts: dict, concept: str) -> pd.DataFrame:
    """
    Extract annual (FY) 10-K records.
    Returns DataFrame with columns: end, filed, val
    Uses MINIMUM filed date per fiscal year end — point-in-time.
    """
    us = facts.get("facts", {}).get("us-gaap", {})
    if concept not in us:
        return pd.DataFrame()
    rows = us[concept]["units"].get("USD", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Keep only annual 10-K filings
    df = df[df["form"].isin(["10-K", "10-K/A"]) & (df.get("fp", pd.Series()) == "FY")]
    if "fp" in df.columns:
        df = df[df["fp"] == "FY"]
    if df.empty:
        return df
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    # For each fiscal year end, use the EARLIEST filing date (first public disclosure)
    df = df.sort_values("filed").drop_duplicates(subset="end", keep="first")
    return df[["end", "filed", "val"]].reset_index(drop=True)


def compute_fundamentals_for_ticker(ticker: str, cik: str) -> pd.DataFrame:
    """
    Returns DataFrame with one row per fiscal year, columns:
      filed_date (point-in-time: when data became public)
      accrual_ratio = (net_income - op_cf) / assets
      gross_margin = gross_profit / revenues (if available)
      roe_proxy = net_income / assets (ROA as ROE proxy)
      leverage_change = YoY change in long_debt / assets
    """
    facts = fetch_company_facts(cik)
    if facts is None:
        return pd.DataFrame()

    series = {}
    for name, concept in CONCEPTS.items():
        s = extract_annual_series(facts, concept)
        if not s.empty:
            series[name] = s.set_index("end")["val"].rename(name)
            series[f"{name}_filed"] = s.set_index("end")["filed"].rename(f"{name}_filed")

    if "net_income" not in series or "op_cf" not in series or "assets" not in series:
        return pd.DataFrame()

    ni = series["net_income"]
    cf = series["op_cf"]
    as_ = series["assets"]

    # Align on fiscal year end
    idx = ni.index.intersection(cf.index).intersection(as_.index)
    if len(idx) < 2:
        return pd.DataFrame()

    ni, cf, as_ = ni.loc[idx], cf.loc[idx], as_.loc[idx]

    df = pd.DataFrame(index=idx)
    df["ticker"] = ticker

    # Accruals (Sloan 1996): high accruals = low earnings quality = future underperformance
    df["accrual_ratio"] = (ni - cf) / as_.clip(lower=1)

    # Gross profitability (Novy-Marx 2013): high gross margin = future outperformance
    if "gross_profit" in series and "revenues" in series:
        gp = series["gross_profit"].reindex(idx)
        rev = series["revenues"].reindex(idx)
        if rev.isna().all() and "revenues" not in series:
            pass
        else:
            rev = rev.fillna(as_ * 0.5)  # fallback
            df["gross_margin"] = gp / rev.clip(lower=1)
    else:
        df["gross_margin"] = np.nan

    # ROA (profitability)
    df["roa"] = ni / as_.clip(lower=1)

    # Leverage change (higher debt increase → future underperformance)
    if "long_debt" in series:
        debt = series["long_debt"].reindex(idx).fillna(0)
        df["leverage_chg"] = (debt / as_.clip(lower=1)).diff()

    # Filed date = latest of net_income, op_cf, assets filing dates
    # (we need all three to compute accruals → use max of their filing dates)
    ni_filed = series.get("net_income_filed", pd.Series(dtype="datetime64[ns]")).reindex(idx)
    cf_filed = series.get("op_cf_filed", pd.Series(dtype="datetime64[ns]")).reindex(idx)
    as_filed = series.get("assets_filed", pd.Series(dtype="datetime64[ns]")).reindex(idx)

    filed_df = pd.concat([ni_filed, cf_filed, as_filed], axis=1)
    df["filed_date"] = filed_df.max(axis=1)  # latest required component
    df["fiscal_year_end"] = idx

    return df.reset_index(drop=True)


# ── Step 3: Batch download S&P 500 ────────────────────────────────────────

def build_universe(prices_csv="sp500_price_8yr.csv", max_tickers=400) -> list:
    """Use tickers from our existing price data (avoid dead stocks)."""
    prices = pd.read_csv(prices_csv, index_col=0, nrows=1)
    tickers = [c for c in prices.columns if c != "SPY"]
    return tickers[:max_tickers]


def download_all(tickers: list, cik_map: dict) -> pd.DataFrame:
    all_rows = []
    missing = []
    for i, tk in enumerate(tickers):
        cik = cik_map.get(tk)
        if cik is None:
            missing.append(tk)
            continue
        df = compute_fundamentals_for_ticker(tk, cik)
        if not df.empty:
            all_rows.append(df)
            status = f"✓ {len(df)}yr"
        else:
            status = "no data"
        if (i + 1) % 20 == 0 or i < 5:
            print(f"  [{i+1}/{len(tickers)}] {tk}: {status}")
        time.sleep(RATE_LIMIT)

    if missing:
        print(f"  No CIK found for: {missing[:10]}{'...' if len(missing)>10 else ''}")

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Canyon Quant — SEC EDGAR Fundamentals Downloader")
    print("=" * 55)

    print("Fetching CIK map from SEC...")
    cik_map = get_cik_map()
    print(f"  {len(cik_map)} tickers mapped")

    print("Building ticker universe from price data...")
    tickers = build_universe()
    print(f"  {len(tickers)} tickers to process")

    print("Downloading fundamentals (8 req/sec)...")
    data = download_all(tickers, cik_map)

    if data.empty:
        print("ERROR: no data downloaded")
    else:
        data.to_csv(OUT, index=False)
        print(f"\n{'='*55}")
        print(f"Saved: {OUT}")
        print(f"Rows: {len(data)}  Tickers: {data['ticker'].nunique()}")
        print(f"Date range: {data['filed_date'].min()[:10]} → {data['filed_date'].max()[:10]}")
        print(f"\nCoverage check:")
        print(f"  accrual_ratio: {data['accrual_ratio'].notna().sum()}/{len(data)} ({data['accrual_ratio'].notna().mean():.0%})")
        print(f"  gross_margin:  {data['gross_margin'].notna().sum()}/{len(data)} ({data['gross_margin'].notna().mean():.0%})")
        print(f"\nSample (AAPL):")
        print(data[data.ticker=="AAPL"][["fiscal_year_end","filed_date","accrual_ratio","gross_margin","roa"]].tail(5).to_string(index=False))
