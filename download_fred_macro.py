"""
FRED Macro Data Download
Improvement 4: Better regime detection via yield curve + unemployment
Series:
  T10Y2Y  — 10Y-2Y Treasury spread (inversion = bear signal, 6mo lead)
  UNRATE  — Monthly unemployment rate (rising = bear)
  USREC   — NBER recession binary indicator
  DGS10   — 10-year Treasury yield (risk-free rate level)
  BAMLH0A0HYM2 — ICE BofA HY OAS (credit spread, risk aversion)
"""
import warnings
import numpy as np
import pandas as pd
import requests
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
UA   = {"User-Agent": "canyon-quant research lynnnnnnn958@gmail.com"}

SERIES = {
    "T10Y2Y":          "10Y-2Y yield spread (inversion signal)",
    "UNRATE":          "Unemployment rate (monthly)",
    "USREC":           "NBER recession indicator (0/1)",
    "DGS10":           "10Y Treasury yield",
    "BAMLH0A0HYM2":    "HY credit spread OAS (risk aversion)",
}

print("Downloading FRED macro series...")
frames = {}
for series_id, desc in SERIES.items():
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        resp = requests.get(url, headers=UA, timeout=30)
        if resp.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text), parse_dates=["observation_date"])
            df = df.set_index("observation_date").rename(columns={series_id: series_id})
            df.index.name = "date"
            df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
            frames[series_id] = df
            print(f"  {series_id:<20} {len(df):>5} rows  "
                  f"[{df.index.min().date()} → {df.index.max().date()}]  {desc}")
        else:
            print(f"  {series_id:<20} FAILED status={resp.status_code}")
    except Exception as e:
        print(f"  {series_id:<20} ERROR: {e}")

if frames:
    combined = pd.concat(frames.values(), axis=1)
    combined = combined.sort_index()
    # Forward-fill monthly series to daily
    daily_idx = pd.date_range(combined.index.min(), combined.index.max(), freq="D")
    combined  = combined.reindex(daily_idx).ffill()
    combined.index.name = "date"
    combined.to_csv(ROOT / "fred_macro.csv")
    print(f"\nSaved: fred_macro.csv  shape={combined.shape}")

    # Signal engineering
    print("\nSignal engineering:")
    # 1. Yield curve: flag inversion (T10Y2Y < 0)
    if "T10Y2Y" in combined:
        inverted = (combined["T10Y2Y"] < 0).sum()
        total    = combined["T10Y2Y"].notna().sum()
        print(f"  Yield curve inverted: {inverted}/{total} days "
              f"({inverted/total:.0%})")
        # Periods of inversion historically precede recessions by 6-18 months
        combined["yield_curve_bull"] = (combined["T10Y2Y"] > -0.25).astype(float)
        print(f"  Bull signal (T10Y2Y > -0.25): "
              f"{combined['yield_curve_bull'].mean():.0%} of days")

    # 2. Unemployment momentum: rising = bear
    if "UNRATE" in combined:
        unrate    = combined["UNRATE"]
        unrate_ma = unrate.rolling(90, min_periods=1).mean()
        combined["unrate_rising"] = (unrate > unrate_ma).astype(float)
        print(f"  Unemployment rising (above 90d MA): "
              f"{combined['unrate_rising'].mean():.0%} of days")

    # 3. Credit spread: high = risk-off
    if "BAMLH0A0HYM2" in combined:
        oas      = combined["BAMLH0A0HYM2"]
        oas_75p  = oas.expanding().quantile(0.75)
        combined["credit_stress"] = (oas > oas_75p).astype(float)
        print(f"  Credit stress (OAS > 75th pctile): "
              f"{combined['credit_stress'].mean():.0%} of days")

    combined.to_csv(ROOT / "fred_macro.csv")
    print("\nFRED macro signals saved with engineering columns.")
else:
    print("No data downloaded.")
print("Done.")
