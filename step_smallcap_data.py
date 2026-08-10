#!/usr/bin/env python3
"""
step_smallcap_data.py — prices + PIT quarterly EPS for the S&P 600 SmallCap set
==============================================================================
PEAD is arbitraged away in large caps (proven: t=0.26 on the S&P 500). Theory
says the edge lives in small, under-covered names. This assembles a genuine
small-cap dataset so PEAD can be tested where it should actually work.

Outputs:
  smallcap_price_history.csv   (2010→now adjusted close, wide)
  smallcap_eps_pit.csv         (quarterly EPS + filed_date / know_date)
"""
from __future__ import annotations
import time, json
from pathlib import Path
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent
UNIV = ROOT / "sp600_smallcap_universe.csv"
PX_OUT = ROOT / "smallcap_price_history.csv"
EPS_OUT = ROOT / "smallcap_eps_pit.csv"
START = "2010-01-01"


def tickers() -> list[str]:
    return pd.read_csv(UNIV)["ticker"].astype(str).tolist()


def download_prices(tks):
    print(f"Downloading small-cap prices for {len(tks)} names {START}→now …")
    end = time.strftime("%Y-%m-%d")
    parts = []
    for i in range(0, len(tks), 80):
        b = tks[i:i+80]
        try:
            raw = yf.download(b, start=START, end=None, auto_adjust=True,
                              progress=False, threads=True)
            if raw.empty:
                continue
            close = raw["Close"] if hasattr(raw.columns, "levels") else raw[["Close"]].rename(columns={"Close": b[0]})
            parts.append(close)
            print(f"  prices batch {i//80+1} ok")
        except Exception as e:
            print(f"  batch {i//80+1} err: {e}")
        time.sleep(0.4)
    if not parts:
        return
    px = pd.concat(parts, axis=1)
    px = px.loc[:, ~px.columns.duplicated()]
    px.index = pd.to_datetime(px.index)
    px = px[[c for c in px.columns if str(c).replace("-", "").isalpha()]]
    px.sort_index().to_csv(PX_OUT)
    print(f"✓ {PX_OUT.name}: {px.shape[0]} days × {px.shape[1]} tickers")


def download_eps(tks):
    import step_edgar_eps_pit as E
    cmap = E.cik_map()
    print(f"Fetching PIT quarterly EPS for {len(tks)} small-caps …")
    parts, got = [], 0
    for i, tk in enumerate(tks, 1):
        cik = cmap.get(tk)
        if not cik:
            continue
        q = E.quarterly_eps(cik)
        if not q.empty:
            q.insert(0, "ticker", tk); parts.append(q); got += 1
        time.sleep(0.11)
        if i % 100 == 0:
            print(f"  eps {i}/{len(tks)} … {got} with data")
    if parts:
        df = pd.concat(parts, ignore_index=True).sort_values(["ticker", "period_end"])
        df.to_csv(EPS_OUT, index=False)
        print(f"✓ {EPS_OUT.name}: {len(df):,} rows, {got} tickers")


def main():
    tks = tickers()
    download_prices(tks)
    download_eps(tks)
    print("Small-cap dataset ready.")


if __name__ == "__main__":
    main()
