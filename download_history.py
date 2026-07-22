"""
BLOCK B: Extend price history to 2000 via yfinance.
Downloads: QQQ, SPY, and all current S&P 500 tickers from 2000-01-01.
Also downloads historical S&P 500 constituent list from GitHub.
Output: sp500_price_25yr.csv, sp500_hist_constituents.csv
"""
import time, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT  = Path(__file__).parent
DELAY = 0.12

# ── Current tickers ───────────────────────────────────────────────────────────
prices_now = pd.read_csv(ROOT / "sp500_price_8yr.csv", index_col=0, parse_dates=True)
tickers    = list(prices_now.columns)   # includes SPY
print(f"Downloading 25-year history for {len(tickers)} tickers (2000-2026)...")

# ── Historical S&P 500 constituents from GitHub ───────────────────────────────
print("\nFetching historical S&P 500 constituent list...")
HIST_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
            "S%26P%20500%20Historical%20Components%20%26%20Changes(08-01-2023).csv")
try:
    UA = {"User-Agent": "canyon-quant research lynnnnnnn958@gmail.com"}
    resp = requests.get(HIST_URL, headers=UA, timeout=20)
    if resp.status_code == 200:
        from io import StringIO
        hist_raw = pd.read_csv(StringIO(resp.text))
        hist_raw.to_csv(ROOT / "sp500_hist_constituents.csv", index=False)
        print(f"  Saved: sp500_hist_constituents.csv ({len(hist_raw)} rows)")
        print(f"  Columns: {hist_raw.columns.tolist()[:5]}")
    else:
        print(f"  Failed: status {resp.status_code} — continuing without")
except Exception as e:
    print(f"  Error: {e}")

# ── Price download in batches ────────────────────────────────────────────────
print(f"\nDownloading price data (2000-01-01 to 2026-06-01)...")
BATCH = 50
all_dfs = []
n_ok, n_fail = 0, 0

for i in range(0, len(tickers), BATCH):
    batch = tickers[i : i + BATCH]
    try:
        df = yf.download(batch, start="2000-01-01", end="2026-06-15",
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            n_fail += len(batch)
            continue
        # Extract Close prices
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"]
        else:
            close = df[["Close"]] if "Close" in df.columns else df
        # Keep only tickers with reasonable data
        good = [c for c in close.columns if close[c].count() > 500]
        close = close[good]
        all_dfs.append(close)
        n_ok += len(good)
        n_fail += len(batch) - len(good)
    except Exception as e:
        n_fail += len(batch)

    if (i // BATCH + 1) % 3 == 0:
        print(f"  [{min(i+BATCH, len(tickers))}/{len(tickers)}]  "
              f"ok={n_ok}  fail={n_fail}")
    time.sleep(DELAY)

if all_dfs:
    combined = pd.concat(all_dfs, axis=1)
    combined = combined.loc["2000-01-01":]
    combined = combined.sort_index()
    # Forward fill up to 5 days (weekends/holidays)
    combined = combined.ffill(limit=5)
    combined.to_csv(ROOT / "sp500_price_25yr.csv")
    print(f"\nSaved: sp500_price_25yr.csv")
    print(f"  Shape: {combined.shape}")
    print(f"  Date range: {combined.index[0].date()} → {combined.index[-1].date()}")
    print(f"  Non-null coverage: {(~combined.isna()).mean().mean():.1%}")
else:
    print("No data downloaded")

print("\nDone.")
