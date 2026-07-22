"""Real-time ETF sector rotation data.
Fetches daily prices for 11 sector ETFs + SPY/QQQ.
Computes 1D / 5D / 1M returns and volume-based flow proxy.
Saves etf_flow_daily.json.
"""
import json, pathlib, time
from datetime import datetime
import pandas as pd
import yfinance as yf

ROOT = pathlib.Path(__file__).parent
t0   = time.time()

ETFS = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLV":  "Health Care",
    "XLY":  "Consumer Disc.",
    "XLC":  "Comm. Services",
    "XLI":  "Industrials",
    "XLP":  "Consumer Staples",
    "XLE":  "Energy",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "XLB":  "Materials",
    "SPY":  "S&P 500",
    "QQQ":  "Nasdaq 100",
}

print(f"[etf_flow_rt] fetching {len(ETFS)} ETFs …")

raw = yf.download(
    list(ETFS.keys()),
    period="3mo",
    auto_adjust=True,
    progress=False,
    threads=True,
)

closes  = raw["Close"]  if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
volumes = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Volume"]]
closes  = closes.dropna(how="all")
volumes = volumes.dropna(how="all")

today_close = closes.iloc[-1]
d1_close    = closes.iloc[-2]   if len(closes) >= 2  else closes.iloc[-1]
d5_close    = closes.iloc[-6]   if len(closes) >= 6  else closes.iloc[0]
d21_close   = closes.iloc[-22]  if len(closes) >= 22 else closes.iloc[0]
d63_close   = closes.iloc[-64]  if len(closes) >= 64 else closes.iloc[0]

# Volume flow proxy: 5-day avg volume vs 20-day avg volume
vol_5d  = volumes.tail(5).mean()
vol_20d = volumes.tail(20).mean()
vol_ratio = (vol_5d / vol_20d).fillna(1.0)

sectors = []
for etf, name in ETFS.items():
    if etf not in closes.columns:
        continue
    px  = float(today_close.get(etf, 0) or 0)
    p1  = float(d1_close.get(etf, px) or px)
    p5  = float(d5_close.get(etf, px) or px)
    p21 = float(d21_close.get(etf, px) or px)
    p63 = float(d63_close.get(etf, px) or px)

    def ret(a, b):
        return round((a / b - 1) * 100, 2) if b > 0 else 0.0

    vr = float(vol_ratio.get(etf, 1.0) or 1.0)
    flow_signal = "INFLOW" if vr > 1.15 else ("OUTFLOW" if vr < 0.85 else "NEUTRAL")

    sectors.append({
        "etf":         etf,
        "name":        name,
        "price":       round(px, 2),
        "ret_1d":      ret(px, p1),
        "ret_5d":      ret(px, p5),
        "ret_1m":      ret(px, p21),
        "ret_3m":      ret(px, p63),
        "vol_ratio":   round(vr, 2),
        "flow_signal": flow_signal,
    })

# Sort by 5-day return (momentum leaders first)
sectors.sort(key=lambda x: x["ret_5d"], reverse=True)

out = {
    "as_of":   str(datetime.now().date()),
    "updated": datetime.now().strftime("%H:%M"),
    "sectors": sectors,
}
with open(ROOT / "etf_flow_daily.json", "w") as f:
    json.dump(out, f)

print(f"  etf_flow_daily.json saved: {len(sectors)} ETFs  ({time.time()-t0:.1f}s)")
for s in sectors:
    flow_tag = f"[{s['flow_signal'][:2]}]"
    print(f"  {s['etf']:5} {s['name']:20} 1D:{s['ret_1d']:+5.2f}%  5D:{s['ret_5d']:+5.2f}%  {flow_tag}")
