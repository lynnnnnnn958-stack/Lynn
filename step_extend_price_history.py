#!/usr/bin/env python3
"""
step_extend_price_history.py — build a DEEP price history cache (2010 → today)
==============================================================================
The working cache (sp500_price_cache.csv) only holds ~3 years, which is far too
short to test regime models, run walk-forward OOS across 2020/2022, or make any
statistically-powered claim. This downloads ~15 years of daily adjusted closes
for the current universe into a separate deep cache used by the rigorous backtest.

Output: sp500_price_history_deep.csv   (adj close, wide: dates × tickers)
        sp500_volume_history_deep.csv  (volume, for capacity/impact modelling)

Survivorship-bias caveat: this uses the CURRENT S&P 500 universe applied
historically. True point-in-time membership needs a vendor constituent feed;
until then, the rigorous backtest documents this bias explicitly.
"""
from __future__ import annotations
import time, re
from pathlib import Path
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent
START = "2010-01-01"
OUT_PX  = ROOT / "sp500_price_history_deep.csv"
OUT_VOL = ROOT / "sp500_volume_history_deep.csv"


def _universe() -> list[str]:
    p = ROOT / "regime_ml_scores.csv"
    if p.exists():
        tk = pd.read_csv(p)["ticker"].dropna().astype(str).tolist()
    else:
        tk = pd.read_csv(ROOT / "sp500_price_cache.csv", index_col=0).columns.tolist()
    clean = [t.strip() for t in tk if re.fullmatch(r"[A-Z][A-Z.\-]{0,6}", str(t).strip())]
    return list(dict.fromkeys(clean + ["SPY"]))


def main():
    tickers = _universe()
    end = time.strftime("%Y-%m-%d")
    print(f"Downloading deep history {START} → {end} for {len(tickers)} tickers …")

    # incremental: if deep cache exists and is recent enough, only extend forward
    existing = pd.DataFrame()
    start = START
    if OUT_PX.exists() and OUT_PX.stat().st_size > 3:
        existing = pd.read_csv(OUT_PX, index_col=0, parse_dates=True)
        if len(existing):
            last = existing.index.max()
            start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if pd.Timestamp(start) > pd.Timestamp(end):
                print(f"Deep cache already current through {last.date()} — nothing to do")
                return

    batch = 80
    px_parts, vol_parts = [], []
    batches = [tickers[i:i+batch] for i in range(0, len(tickers), batch)]
    for i, b in enumerate(batches, 1):
        try:
            raw = yf.download(b, start=start, end=None, auto_adjust=True,
                              progress=False, threads=True)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                px_parts.append(raw["Close"])
                if "Volume" in raw.columns.get_level_values(0):
                    vol_parts.append(raw["Volume"])
            else:
                px_parts.append(raw[["Close"]].rename(columns={"Close": b[0]}))
                if "Volume" in raw.columns:
                    vol_parts.append(raw[["Volume"]].rename(columns={"Volume": b[0]}))
            print(f"  batch {i}/{len(batches)} ok ({len(b)} tickers)")
        except Exception as e:
            print(f"  batch {i} error: {e}")
        time.sleep(0.4)

    if not px_parts:
        print("No data downloaded — aborting")
        return

    new_px = pd.concat(px_parts, axis=1)
    new_px = new_px.loc[:, ~new_px.columns.duplicated()]
    new_px.index = pd.to_datetime(new_px.index)

    if not existing.empty:
        cols = existing.columns.union(new_px.columns)
        combined = pd.concat([existing.reindex(columns=cols), new_px.reindex(columns=cols)])
    else:
        combined = new_px
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    # keep only valid ticker columns
    combined = combined[[c for c in combined.columns if re.fullmatch(r"[A-Z][A-Z.\-]{0,6}", str(c))]]
    combined.to_csv(OUT_PX)
    print(f"✓ {OUT_PX.name}: {combined.shape[0]} days × {combined.shape[1]} tickers "
          f"({combined.index.min().date()} → {combined.index.max().date()})")

    if vol_parts:
        new_vol = pd.concat(vol_parts, axis=1)
        new_vol = new_vol.loc[:, ~new_vol.columns.duplicated()]
        new_vol.index = pd.to_datetime(new_vol.index)
        new_vol = new_vol[[c for c in new_vol.columns if re.fullmatch(r"[A-Z][A-Z.\-]{0,6}", str(c))]]
        new_vol.sort_index().to_csv(OUT_VOL)
        print(f"✓ {OUT_VOL.name}: {new_vol.shape[0]} days × {new_vol.shape[1]} tickers")


if __name__ == "__main__":
    main()
